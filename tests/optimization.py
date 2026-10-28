"""Objective minimization, exact constraints and stopping honesty."""

import math

import pytest
import torch
from hypothesis import given

from tests.strategies import extreme_scalings
from torch_bsf import BezierModel, BezierSpace, FixedControls, Points, PreconditionedSGD, QuadraticPenalty, minimize
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


@pytest.mark.parametrize("coefficient,weight", [(1e30, 1e-100), (1e-30, 1e100)])
@pytest.mark.parametrize("kind", ["ridge", "smoothness"])
def test_float32_penalty_weights_are_not_narrowed(coefficient, weight, kind):
    space = BezierSpace.simplex(1, degree=1)
    controls = torch.tensor([[coefficient], [-coefficient]], dtype=torch.float32)
    initial = BezierModel(space, controls)
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(**{kind: weight}),
        fixed=FixedControls({0: controls[0], 1: controls[1]}),
    )
    actual = float(controls[0, 0])
    expected = (2 if kind == "ridge" else 4) * weight * actual * actual
    # Input rounding is captured in actual; eight float64 operations bound evaluation.
    assert result.objective == pytest.approx(expected, rel=8 * 2**-53, abs=0)
    assert result.model.coefficients.dtype == torch.float32


@pytest.mark.parametrize("kind", ["ridge", "smoothness"])
def test_penalty_near_float64_maximum(kind):
    space = BezierSpace.simplex(1, degree=1)
    coefficient = math.sqrt(torch.finfo(torch.float64).max / 2)
    controls = torch.tensor([[coefficient], [0]], dtype=torch.float64)
    initial = BezierModel(space, controls)
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(**{kind: 2}),
        fixed=FixedControls({0: controls[0], 1: controls[1]}),
    )
    expected = 2 * coefficient * coefficient
    assert math.isfinite(expected)
    # Identical multiplication order gives an exact boundary reference.
    assert result.objective == expected


def test_smoothness_with_unrepresentable_difference():
    space = BezierSpace.simplex(1, degree=1)
    controls = torch.tensor([[1e308], [-1e308]], dtype=torch.float64)
    initial = BezierModel(space, controls)
    result = minimize(
        initial,
        lambda y, p: y[:, 0] * 0,
        points=grid(space, 4),
        penalty=QuadraticPenalty(smoothness=1e-320),
        fixed=FixedControls({0: controls[0], 1: controls[1]}),
    )
    # The half-difference is 1e308; a power-of-two rescale introduces no rounding.
    assert result.objective == (1e308 * 1e-320) * 1e308 * 4


@pytest.mark.parametrize("preconditioner,expected_ratio", [("none", 1 / 3), ("gram", 1 / 5)])
def test_float32_large_ridge_keeps_representable_update(preconditioner, expected_ratio):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.full((2, 1), 1e-30, dtype=torch.float32))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: y[:, 0] * 0,
            points=grid(space, 4),
            penalty=QuadraticPenalty(ridge=1e40),
            method=PreconditionedSGD(max_iter=1, step_size=1e-40, preconditioner=preconditioner),
        )
    # Uniform coefficients are an eigenvector of the degree-one Gram with
    # eigenvalue 1/2; identity has eigenvalue 1. Six float32 gradient/update
    # roundings plus float64 solve roundoff are bounded by 32u32.
    torch.testing.assert_close(result.model.coefficients, initial.coefficients * expected_ratio, rtol=32 * 2**-24, atol=0)
    assert result.model.coefficients.dtype == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device is unavailable")
@pytest.mark.parametrize("preconditioner", ["none", "gram"])
@pytest.mark.parametrize("coefficient,weight,step_size", [(1, 0, 0.01), (1e30, 1e-100, 0.01), (1e-30, 1e40, 1e-40)])
def test_mps_optimizer_keeps_device_with_wide_penalties(preconditioner, coefficient, weight, step_size):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.full((2, 1), coefficient, dtype=torch.float32, device="mps"))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: y[:, 0] * 0,
            points=grid(space, 4),
            penalty=QuadraticPenalty(ridge=weight),
            method=PreconditionedSGD(max_iter=1, step_size=step_size, preconditioner=preconditioner),
        )
    assert result.model.coefficients.device.type == "mps"
    assert result.model.coefficients.dtype == torch.float32
    assert math.isfinite(result.objective)
    updated = result.model.coefficients.detach().cpu().double().flatten().tolist()
    expected_objective = sum((weight * value) * value for value in updated)
    assert result.objective == pytest.approx(expected_objective, rel=16 * 2**-53, abs=0)
    metric_eigenvalue = 1 if preconditioner == "none" else 0.5
    ratio = metric_eigenvalue / (metric_eigenvalue + 2 * step_size * weight)
    # Float32 basis/backprop plus transfers and a wide solve: 64u32 allows
    # MPS accumulation differences without hiding a lost update.
    torch.testing.assert_close(result.model.coefficients.cpu(), initial.coefficients.cpu() * ratio, rtol=64 * 2**-24, atol=0)


@pytest.mark.parametrize("preconditioner", ["none", "gram"])
@pytest.mark.parametrize("step_decay", ["constant", "sqrt"])
def test_scale_smoothness_before_accumulating_edges(preconditioner, step_decay):
    from torch_bsf.optimization import gram_matrix

    space = BezierSpace.simplex(1, degree=2)
    controls = torch.tensor([[0.0], [1e-160], [0.0]], dtype=torch.float64)
    initial = BezierModel(space, controls)
    metric = gram_matrix(space) if preconditioner == "gram" else torch.eye(3, dtype=torch.float64)
    laplacian = torch.tensor([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=torch.float64)
    expected = controls.clone()
    for k in range(2):
        alpha = 1e-310 / math.sqrt(k + 1) if step_decay == "sqrt" else 1e-310
        # Zero data gradient makes the regularized update A^-1 M P. Scale the
        # scalar edge weight before forming L so its middle diagonal is finite.
        expected = torch.linalg.solve(metric + ((1e308 * alpha) * 2) * laplacian, metric @ expected)
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: y[:, 0] * 0,
            points=grid(space, 4),
            penalty=QuadraticPenalty(smoothness=1e308),
            method=PreconditionedSGD(max_iter=2, step_size=1e-310, step_decay=step_decay, preconditioner=preconditioner),
        )
    # Two equivalent 3x3 solves/updates and large-gradient multiplication are
    # bounded by 2048 float64 roundings (including the modest Gram condition).
    torch.testing.assert_close(result.model.coefficients, expected, rtol=2048 * 2**-53, atol=0)


@pytest.mark.parametrize("kind", ["maximum", "cancellation", "subnormal"])
def test_finite_extreme_loss_mean_and_gradient(kind):
    from torch_bsf.optimization import _objective

    maximum = torch.finfo(torch.float64).max
    tiny = math.ulp(0.0)
    losses = {"maximum": [maximum] * 3, "cancellation": [maximum, 3 * tiny, -maximum], "subnormal": [tiny] * 3}[kind]
    expected = maximum if kind == "maximum" else tiny
    space = BezierSpace.simplex(0, degree=0)
    initial = BezierModel(space, torch.zeros(1, 1, dtype=torch.float64))
    points = Points(torch.empty(3, 0, dtype=torch.float64))

    def loss(y, p):
        return y[:, 0] + torch.tensor(losses, dtype=torch.float64)

    objective = _objective(initial, loss, points, QuadraticPenalty())
    (gradient,) = torch.autograd.grad(objective, initial.coefficients)
    assert objective.item() == expected
    # The three derivative contributions are 1/3; their rounded sum is one.
    torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=4 * 2**-53, atol=0)
    result = minimize(initial, loss, points=points, method=PreconditionedSGD(gradient_tolerance=1))
    assert result.success and result.status == "gradient_tolerance"
    assert result.objective == expected


def test_finite_update_when_step_product_overflows():
    space = BezierSpace.simplex(0, degree=0)
    initial = BezierModel(space, torch.tensor([[1e308]], dtype=torch.float64))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: torch.where(y[:, 0] > 0, 2 * (y[:, 0] - 1e308), y[:, 0] + 1e308),
            points=Points(torch.empty(1, 0, dtype=torch.float64)),
            method=PreconditionedSGD(max_iter=1, step_size=1e308, preconditioner="none"),
        )
    # The exact update 1e308 - 1e308*2 is representable, though the product is not.
    assert result.model.coefficients.item() == -1e308


def test_large_step_does_not_underflow_direction_before_scaling():
    space = BezierSpace.simplex(0, degree=0)
    initial = BezierModel(space, torch.tensor([[1e-300]], dtype=torch.float64))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            initial,
            lambda y, p: y[:, 0] * 0,
            points=Points(torch.empty(1, 0, dtype=torch.float64)),
            penalty=QuadraticPenalty(ridge=1),
            method=PreconditionedSGD(max_iter=1, step_size=1e100, preconditioner="none"),
        )
    # The exact implicit-ridge result is below the smallest subnormal. Compare
    # in initial-coefficient units to allow 64u of cancellation in the update.
    assert abs(result.model.coefficients.item() / 1e-300) <= 64 * 2**-53
