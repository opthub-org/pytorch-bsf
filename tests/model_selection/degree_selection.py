"""Degree selection keeps all evidence and never implicitly refits."""

import numpy as np
import pytest
import torch

from torch_bsf import BezierSpace, Samples, fit
from torch_bsf.model_selection import KFold, select_degree
from torch_bsf.sampling import sample


def test_complete_search_and_explicit_refit():
    space = BezierSpace.simplex(1, degree=1)
    points = sample(space, 30, seed=7)
    samples = Samples(points, points.parameters[:, :1] ** 2)
    result = select_degree(space, samples, degrees=[0, 1, 2], cv=KFold(3, seed=12))
    assert result.success and result.searched_all and result.best_degree == 2
    assert len(result.candidates) == 3
    assert not hasattr(result, "model")
    assert result.best_space is not None
    model = fit(result.best_space, samples).model
    assert torch.allclose(model(points), samples.values)


def test_unseeded_search_still_reuses_identical_splits():
    space = BezierSpace.simplex(1, degree=1)
    points = sample(space, 20, seed=3)
    result = select_degree(space, Samples(points, points.parameters[:, :1]), degrees=[1, 2, 3], cv=KFold(3))
    reference = result.candidates[0].validation.folds
    for candidate in result.candidates[1:]:
        for actual, expected in zip(candidate.validation.folds, reference, strict=True):
            assert torch.equal(actual.train_indices, expected.train_indices)
            assert torch.equal(actual.test_indices, expected.test_indices)


@pytest.mark.parametrize("multi_cell", [False, True])
def test_numpy_scalar_degree_candidates(multi_cell):
    from torch_bsf import SimploidalSet

    if multi_cell:
        domain = SimploidalSet()
        domain.add_simplex(1)
        domain.add_simplex(1)
        space = BezierSpace.on_set(domain, degree=1)
    else:
        space = BezierSpace.simplex(1, degree=1)
    points = sample(space, 30, seed=7)
    samples = Samples(points, points.parameters[:, :1])
    numpy_result = select_degree(space, samples, degrees=list(np.arange(1, 3)), cv=KFold(3, seed=12))
    reference = select_degree(space, samples, degrees=[1, 2], cv=KFold(3, seed=12))
    assert numpy_result.success == reference.success
    assert [candidate.degree for candidate in numpy_result.candidates] == [1, 2]
    for actual, expected in zip(numpy_result.candidates, reference.candidates, strict=True):
        assert actual.space.compatible(expected.space)
        assert actual.validation.metrics == expected.validation.metrics


def test_unrepresentable_validation_cannot_win_degree_search():
    space = BezierSpace.simplex(1, degree=0)
    params = torch.eye(2, dtype=torch.float64)
    observations = Samples(params, torch.tensor([[2.0**700], [-(2.0**700)]], dtype=torch.float64))
    result = select_degree(space, observations, degrees=[0], cv=KFold(2, shuffle=False))
    assert not result.success and result.best_index is None and result.best_degree is None
    assert len(result.candidates) == 1
    assert not result.candidates[0].validation.success
