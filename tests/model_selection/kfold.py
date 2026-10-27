"""Explicit CV splits and ensemble composition."""

import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import seeds
from torch_bsf import BezierSpace, Ensemble, Samples
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
