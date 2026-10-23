"""Objective minimization, exact constraints and stopping honesty."""

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, FixedControls, PreconditionedSGD, minimize
from torch_bsf.results import ConvergenceWarning
from torch_bsf.sampling import grid


def test_quadratic_convergence_and_copy():
    space = BezierSpace.simplex(1, degree=2)
    initial = BezierModel(space, torch.zeros(space.n_controls, 1, dtype=torch.float64))

    def target(y, t):
        return (y[:, 0] - 3).square()

    result = minimize(
        initial, target, points=grid(space, 20), method=PreconditionedSGD(max_iter=200, step_size=0.1, gradient_tolerance=1e-7)
    )
    assert result.success
    assert result.objective < 1e-12
    assert torch.count_nonzero(initial.coefficients) == 0
    assert result.model is not initial


def test_budget_and_fixed_values():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, [[0.0], [0.0]])
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, t: (y[:, 0] - 2).square(),
            fixed=FixedControls({0: [1.0]}),
            method=PreconditionedSGD(max_iter=1),
            seed=5,
        )
    assert not result.success and result.status == "iteration_limit"
    assert result.model.coefficients[0].item() == 1
    with pytest.raises(RuntimeError):
        result.require_success()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("preconditioner", ["gram", "none"])
def test_half_precision_metric_solve_is_promoted(dtype, preconditioner):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.zeros(2, 1, dtype=dtype))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: (y[:, 0] - 1).square(),
            points=grid(space, 4),
            method=PreconditionedSGD(max_iter=2, step_size=0.1, preconditioner=preconditioner),
        )
    assert result.model.coefficients.dtype == torch.float32
    assert initial.coefficients.dtype == dtype
    assert torch.isfinite(result.model.coefficients).all()
    assert result.objective < 1
    assert torch.count_nonzero(initial.coefficients) == 0
