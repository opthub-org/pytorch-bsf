"""Finite averaging and copied-member contracts."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torch_bsf import BezierModel, BezierSpace, fit_point_cloud
from torch_bsf.ensemble import Ensemble


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["large", "cancel", "subnormal", "ordinary"])
def test_finite_ensemble_means(dtype, kind):
    maximum = torch.finfo(dtype).max
    tiny = torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)).item()
    values, expected = {
        "large": ([maximum, maximum], maximum),
        "cancel": ([maximum, -maximum], 0.0),
        "subnormal": ([tiny] * 4, tiny),
        "ordinary": ([1.0, 3.0], 2.0),
    }[kind]
    space = BezierSpace.simplex(0, degree=0)
    originals = [BezierModel(space, torch.tensor([[x]], dtype=dtype)) for x in values]
    ensemble = Ensemble(originals)
    for member in ensemble.models:
        member.coefficients.requires_grad_()
    points = torch.empty((1, 0), dtype=dtype)
    prediction = ensemble(points)
    torch.testing.assert_close(prediction, torch.tensor([[expected]], dtype=dtype), atol=0, rtol=0)
    prediction.sum().backward()
    for model in ensemble.models:
        torch.testing.assert_close(
            model.coefficients.grad,
            torch.full((1, 1), 0.5, dtype=dtype) if len(values) == 2 else torch.full((1, 1), 0.25, dtype=dtype),
            atol=0,
            rtol=4 * torch.finfo(dtype).eps,
        )
    with torch.no_grad():
        originals[0].coefficients.fill_(0)
    torch.testing.assert_close(ensemble(points), prediction, atol=0, rtol=0)


@given(st.integers(-1000, 1000))
def test_constant_point_cloud_extreme_means(exponent):
    value = 2.0**exponent
    space = BezierSpace.simplex(0, degree=0)
    values = torch.full((4, 1), value, dtype=torch.float64)
    result = fit_point_cloud(space, values).require_success()
    assert result.objective_history == (0.0,)
    torch.testing.assert_close(result.model.coefficients, values[:1], atol=0, rtol=0)


def test_issue_455_constant_point_cloud():
    space = BezierSpace.simplex(0, degree=0)
    result = fit_point_cloud(space, torch.full((2, 1), 1e308, dtype=torch.float64)).require_success()
    assert result.objective_history == (0.0,)
    assert result.model.coefficients.item() == 1e308
