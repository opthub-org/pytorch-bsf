"""Explicit CV splits and ensemble composition."""

import math
from fractions import Fraction

import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import extreme_scalings, seeds
from torch_bsf import BezierSpace, Ensemble, FixedControls, Samples
from torch_bsf.model_selection import KFold, cross_validate
from torch_bsf.sampling import sample


def test_splits_and_evidence():
    space = BezierSpace.simplex(1, degree=1)
    points = sample(space, 21, seed=12)
    observations = Samples(points, points.parameters[:, :1])
    state = torch.random.get_rng_state().clone()
    result = cross_validate(space, observations, cv=KFold(4, seed=7))
    assert torch.equal(state, torch.random.get_rng_state())
    assert result.success and result.metrics["pooled_mse"] < 1e-25
    assert sorted(torch.cat([f.test_indices for f in result.folds]).tolist()) == list(range(21))
    for fold in result.folds:
        assert not set(fold.train_indices.tolist()) & set(fold.test_indices.tolist())
    ensemble = Ensemble(result.models)
    assert ensemble.members(points).shape == (4, 21, 1)
    assert torch.allclose(ensemble(points), observations.values)


def test_invalid_splits():
    with pytest.raises(ValueError):
        KFold(5).split(4)


@example(n_samples=2, requested_splits=2, shuffle=False, seed=0)
@example(n_samples=7, requested_splits=3, shuffle=True, seed=0)
@given(n_samples=st.integers(2, 64), requested_splits=st.integers(2, 64), shuffle=st.booleans(), seed=seeds())
def test_kfold_partitions_samples_reproducibly(n_samples, requested_splits, shuffle, seed):
    n_splits = min(n_samples, requested_splits)
    cv = KFold(n_splits, shuffle=shuffle, seed=seed)
    before = torch.random.get_rng_state().clone()
    folds = cv.split(n_samples)
    repeated = cv.split(n_samples)
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(folds) == n_splits
    assert sorted(torch.cat([test for _, test in folds]).tolist()) == list(range(n_samples))
    sizes = [len(test) for _, test in folds]
    assert min(sizes) >= 1 and max(sizes) - min(sizes) <= 1
    for (train, test), (again_train, again_test) in zip(folds, repeated, strict=True):
        assert torch.equal(train, again_train) and torch.equal(test, again_test)
        assert len(train) + len(test) == n_samples
        assert set(train.tolist()).isdisjoint(test.tolist())
        assert sorted(torch.cat([train, test]).tolist()) == list(range(n_samples))
    if not shuffle:
        assert torch.equal(torch.cat([test for _, test in folds]), torch.arange(n_samples))


@pytest.mark.parametrize("zero_weight", [True, False])
def test_nonfinite_validation_is_not_success(zero_weight):
    space = BezierSpace.simplex(1, degree=0)
    params = torch.eye(2, dtype=torch.float64).repeat(2, 1)
    values = [0, 1e200, 0, 1e200] if zero_weight else [1e200, 1e200, -1e200, -1e200]
    observations = Samples(
        params, torch.tensor(values, dtype=torch.float64)[:, None], weights=[1, 0, 1, 0] if zero_weight else None
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False))
    assert result.success == zero_weight
    if zero_weight:
        assert result.metrics["pooled_mse"] == 0
        assert all(f.metrics == {"mse": 0, "mae": 0} for f in result.folds)
    else:
        assert len(result.folds) == 2
        assert all(not torch.isfinite(torch.tensor(f.metrics["mse"])) for f in result.folds)


def test_representable_weighted_error_avoids_square_overflow():

    space = BezierSpace.simplex(1, degree=0)
    params = torch.eye(2, dtype=torch.float64).repeat(2, 1)
    observations = Samples(
        params, torch.tensor([0, 1e200, 0, 1e200], dtype=torch.float64)[:, None], weights=[1, 1e-300, 1, 1e-300]
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]}))
    assert result.success
    # Weight normalization, sqrt, scaling and summation introduce at most 16 roundings.
    assert result.metrics["pooled_mse"] == pytest.approx(1e100, rel=16 * 2**-53)
    for fold in result.folds:
        assert fold.metrics["mae"] == pytest.approx(1e-100, rel=16 * 2**-53)


@pytest.mark.parametrize("value", [1e100, 1e154])
def test_pooled_metrics_avoid_weight_product_overflow(value):

    space = BezierSpace.simplex(1, degree=0)
    params = torch.eye(2, dtype=torch.float64).repeat(2, 1)
    observations = Samples(params, torch.full((4, 1), value, dtype=torch.float64), weights=[1e200] * 4)
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]}))
    assert result.success
    assert result.metrics["pooled_mse"] == pytest.approx(value * value, rel=16 * 2**-53)
    assert result.metrics["mean_mse"] == pytest.approx(value * value, rel=16 * 2**-53)
    assert result.metrics["std_mse"] == 0


def test_validation_overflow_fails_after_successful_training():
    space = BezierSpace.simplex(1, degree=0)
    observations = Samples(torch.eye(2, dtype=torch.float64), torch.tensor([[2.0**700], [-(2.0**700)]], dtype=torch.float64))
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False))
    assert all(f.result.success for f in result.folds)
    assert not result.success


@given(scale=extreme_scalings(450))
def test_weighted_validation_scales_quadratically(scale):
    space = BezierSpace.simplex(1, degree=0)
    observations = Samples(
        torch.eye(2, dtype=torch.float64).repeat(2, 1),
        torch.tensor([1, 2, 3, 4], dtype=torch.float64)[:, None] * scale,
        weights=[1, 3, 2, 4],
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]}))
    assert result.success
    # Normalize the exact power-of-two output scaling; 32u covers roots,
    # products and the fold/pooled reductions of these four dyadic observations.
    assert result.metrics["pooled_mse"] / scale / scale == pytest.approx(9.5, rel=32 * 2**-53)


def test_pooled_metric_preserves_tiny_fold_weight():
    space = BezierSpace.simplex(1, degree=0)
    observations = Samples(
        torch.eye(2, dtype=torch.float64), torch.tensor([[1e150], [0]], dtype=torch.float64), weights=[1e-300, 1e200]
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]}))
    assert result.success
    # Two sqrt weights and four products/reductions fit within 32 float64 roundings.
    assert result.metrics["pooled_mse"] == pytest.approx(1e-200, rel=32 * 2**-53, abs=0)


def test_mae_preserves_tiny_normalized_weight():
    space = BezierSpace.simplex(1, degree=0)
    observations = Samples(
        torch.eye(2, dtype=torch.float64).repeat(2, 1),
        torch.tensor([[1e308], [0], [1e308], [0]], dtype=torch.float64),
        weights=[1e-300, 1e200, 1e-300, 1e200],
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]}))
    assert result.success
    # Root weights preserve the finite product even though weight/total is zero.
    for fold in result.folds:
        assert fold.metrics["mae"] == pytest.approx(1e-192, rel=32 * 2**-53, abs=0)
        assert fold.metrics["mse"] == pytest.approx(1e116, rel=32 * 2**-53)


def test_pooling_uses_the_validated_global_weight_sum():
    count = 11
    maximum = torch.finfo(torch.float64).max
    weights = torch.full((count,), maximum / count, dtype=torch.float64)
    # CPU reduction orders vary; keep the public input valid on each platform.
    while not torch.isfinite(weights.sum()):
        weights = torch.nextafter(weights, torch.zeros_like(weights))
    observations = Samples(
        torch.tensor([[1.0, 0.0]], dtype=torch.float64).repeat(count, 1),
        torch.ones((count, 1), dtype=torch.float64),
        weights=weights,
    )
    result = cross_validate(
        BezierSpace.simplex(1, degree=0), observations, cv=KFold(count, shuffle=False), fixed=FixedControls({0: [0]})
    )
    assert result.success
    # Each residual is exactly one. The reduction of eleven normalized fold
    # contributions permits 32 roundings; a zero pooled score is invalid.
    assert result.metrics["pooled_mse"] == pytest.approx(1.0, rel=32 * 2**-53)


def test_pooled_average_stays_bounded_at_float64_maximum():
    error = float.fromhex("0x1.ffffffffffffep+1023")
    # Frozen binary weights whose separately rounded weighted terms overflow
    # on addition even though a weighted average cannot exceed its inputs.
    weights = [float.fromhex("0x1.5f44296f6fbf6p-1"), float.fromhex("0x1.474361dfd1657p-1")]
    observations = Samples(
        torch.eye(2, dtype=torch.float64), torch.full((2, 1), math.sqrt(error), dtype=torch.float64), weights=weights
    )
    result = cross_validate(
        BezierSpace.simplex(1, degree=0), observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [0]})
    )
    assert result.success
    errors = [fold.metrics["mse"] for fold in result.folds]
    assert min(errors) <= result.metrics["pooled_mse"] <= max(errors)


def test_weighted_metrics_preserve_an_overflowing_residual():
    observations = Samples(
        torch.eye(2, dtype=torch.float64).repeat(2, 1),
        torch.tensor([[-1e308], [1e308], [-1e308], [1e308]], dtype=torch.float64),
        weights=[1e-300, 1e200, 1e-300, 1e200],
    )
    result = cross_validate(
        BezierSpace.simplex(1, degree=0), observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [1e308]})
    )
    assert result.success
    # The residual 2e308 itself is unrepresentable, but its normalized weighted
    # errors are finite. Mantissa/exponent products permit 32 roundings here.
    assert result.metrics["pooled_mse"] == pytest.approx(4e116, rel=32 * 2**-53)
    for fold in result.folds:
        assert fold.metrics["mae"] == pytest.approx(2e-192, rel=32 * 2**-53, abs=0)


def test_overflowing_row_does_not_erase_another_subnormal_residual():
    from torch_bsf.core import Cell

    space = BezierSpace((Cell((), (), (0,)), Cell((), (), (1,))), 2, (0, 1), ())
    tiny, weight = 5e-324, 1e307
    observations = Samples(
        torch.empty((4, 0), dtype=torch.float64),
        torch.tensor([[-1e308], [0.0], [-1e308], [0.0]], dtype=torch.float64),
        cell_ids=[0, 1, 0, 1],
        weights=[tiny, weight, tiny, weight],
    )
    result = cross_validate(space, observations, cv=KFold(2, shuffle=False), fixed=FixedControls({0: [1e308], 1: [tiny]}))
    expected = float(
        (2 * Fraction(1e308) * Fraction(tiny) + Fraction(tiny) * Fraction(weight)) / (Fraction(tiny) + Fraction(weight))
    )
    assert result.success
    # The weighted large residual rounds to 20 minimum subnormal units; the
    # other row contributes one more. Neither is near a rounding tie here.
    assert all(fold.metrics["mae"] == expected for fold in result.folds)
