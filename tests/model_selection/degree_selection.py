"""Degree selection keeps all evidence and never implicitly refits."""

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
