"""Explicit CV splits and ensemble composition."""

import pytest
import torch

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
