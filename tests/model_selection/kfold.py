"""Explicit CV splits and ensemble composition."""

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
