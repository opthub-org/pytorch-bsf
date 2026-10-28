"""Objective minimization, exact constraints and stopping honesty."""

import math

import pytest
import torch
from hypothesis import given

from tests.strategies import extreme_scalings
from torch_bsf import BezierModel, BezierSpace, FixedControls, PreconditionedSGD, QuadraticPenalty, minimize
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


@pytest.mark.parametrize("ending", ["all_fixed", "gradient_tolerance", "callback_stop", "iteration_limit"])
def test_large_unpenalized_objective(ending):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.full((2, 1), 1e200, dtype=torch.float64))
    fixed = FixedControls({0: [1e200], 1: [1e200]}) if ending == "all_fixed" else None
    method = PreconditionedSGD(max_iter=1, gradient_tolerance=1e-6 if ending == "gradient_tolerance" else None)

    def run():
        return minimize(
            initial,
            lambda y, p: y[:, 0] if ending == "all_fixed" else y[:, 0] * 0,
            points=grid(space, 4),
            method=method,
            callback=(lambda record, model: True) if ending == "callback_stop" else None,
            fixed=fixed,
        )

    if ending == "iteration_limit":
        with pytest.warns(ConvergenceWarning):
            result = run()
    else:
        result = run()
    assert result.status == ending
    assert math.isfinite(result.objective)
    # Four-term basis evaluation and the mean introduce at most eight float64 roundings.
    assert result.objective == pytest.approx(1e200 if ending == "all_fixed" else 0, rel=8 * 2**-53, abs=0)


@given(scale=extreme_scalings(500))
def test_penalty_scale_invariance(scale):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.tensor([[scale], [-scale]], dtype=torch.float64))
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(ridge=(1 / scale) ** 2, smoothness=(1 / scale) ** 2),
        fixed=FixedControls({0: [scale], 1: [-scale]}),
    )
    # Powers of two scale exactly; ridge is 2 and the single edge contributes 4.
    assert result.objective == 6.0


@pytest.mark.parametrize("ending", ["all_fixed", "gradient_tolerance", "callback_stop", "iteration_limit"])
def test_unrepresentable_objective_raises(ending):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.full((2, 1), 1e200, dtype=torch.float64))
    with pytest.raises(ArithmeticError, match="objective"):
        minimize(
            initial,
            lambda y, p: y[:, 0] * 0,
            points=grid(space, 4),
            penalty=QuadraticPenalty(ridge=1),
            fixed=FixedControls({0: [1e200], 1: [1e200]}) if ending == "all_fixed" else None,
            method=PreconditionedSGD(max_iter=1, gradient_tolerance=1 if ending == "gradient_tolerance" else None),
            callback=(lambda record, model: True) if ending == "callback_stop" else None,
        )


def test_nonfinite_update_raises():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.zeros(2, 1, dtype=torch.float64))
    with pytest.raises(ArithmeticError, match="update"):
        minimize(
            initial,
            lambda y, p: y[:, 0] * 1e100,
            points=grid(space, 4),
            method=PreconditionedSGD(max_iter=1, step_size=1e300, preconditioner="none"),
        )


@pytest.mark.parametrize("ridge,smoothness,expected", [(1e-300, 0, 2e100), (0, 1e-300, 4e100)])
def test_small_weight_avoids_intermediate_square_overflow(ridge, smoothness, expected):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.tensor([[1e200], [-1e200]], dtype=torch.float64))
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(ridge=ridge, smoothness=smoothness),
        fixed=FixedControls({0: [1e200], 1: [-1e200]}),
    )
    # sqrt, scaling, subtraction, square and sum incur at most eight float64 roundings.
    assert result.objective == pytest.approx(expected, rel=8 * 2**-53)


def test_representable_large_loss_mean():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.zeros(2, 1, dtype=torch.float64))
    result = minimize(
        initial, lambda y, p: y[:, 0] * 0 + 1e308, points=grid(space, 4), fixed=FixedControls({0: [0.0], 1: [0.0]})
    )
    assert result.objective == pytest.approx(1e308, rel=8 * 2**-53)


def test_large_constant_controls_have_zero_smoothness():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.full((2, 1), 1e200, dtype=torch.float64))
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(smoothness=1e300),
        fixed=FixedControls({0: [1e200], 1: [1e200]}),
    )
    assert result.objective == 0
