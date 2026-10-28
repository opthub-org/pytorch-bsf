"""Finite averaging and copied-member contracts."""

from fractions import Fraction

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import extreme_scalings
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_rounded_overflow_threshold(dtype):
    value = torch.tensor(torch.finfo(dtype).max / 3, dtype=dtype)
    space = BezierSpace.simplex(0, degree=0)
    model = BezierModel(space, value.reshape(1, 1))
    prediction = Ensemble([model, model, model])(torch.empty((1, 0), dtype=dtype))
    torch.testing.assert_close(prediction, value.reshape(1, 1), atol=0, rtol=0)


def test_constant_fit_does_not_mistake_squared_underflow_for_exactness():
    space = BezierSpace.simplex(0, degree=0)
    initial = BezierModel(space, torch.zeros((1, 1), dtype=torch.float64))
    values = torch.tensor([[0.0], [1e-200]], dtype=torch.float64)
    result = fit_point_cloud(space, values, initial=initial).require_success()
    # SVD combines two entries and divides by sqrt(2): allow 8 roundings.
    torch.testing.assert_close(
        result.model.coefficients,
        torch.tensor([[5e-201]], dtype=torch.float64),
        atol=0,
        rtol=8 * torch.finfo(torch.float64).eps,
    )
    assert result.control_result is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cancelling_extreme_mean_scaled_loss_has_finite_gradients(dtype):
    maximum = torch.finfo(dtype).max
    space = BezierSpace.simplex(0, degree=0)
    models = [BezierModel(space, torch.tensor([[x]], dtype=dtype)) for x in [maximum, -maximum]]
    ensemble = Ensemble(models)
    prediction = ensemble(torch.empty((1, 0), dtype=dtype))
    assert prediction.item() == 0
    (2 * prediction.sum()).backward()
    for member in ensemble.models:
        torch.testing.assert_close(member.coefficients.grad, torch.ones((1, 1), dtype=dtype), atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype,small", [(torch.float32, 1e-16), (torch.float64, 1e-100), (torch.float32, 1.0), (torch.float64, 1.0)]
)
@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (2, 0, 1)])
def test_scaled_cancellation_preserves_finite_residual(dtype, small, order):
    from torch_bsf._numeric import _finite_mean

    maximum = torch.finfo(dtype).max
    entries = [maximum, -maximum, small]
    values = torch.tensor([entries[i] for i in order], dtype=dtype, requires_grad=True)
    mean = _finite_mean(values, 0)
    # One final division and rounding; permit two dtype epsilons.
    torch.testing.assert_close(mean, values.detach()[order.index(2)] / 3, atol=0, rtol=2 * torch.finfo(dtype).eps)
    (3 * mean).backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), atol=0, rtol=0)


def test_ordinary_mean_avoids_exact_cpu_reduction(monkeypatch):
    import torch_bsf._numeric as numeric

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary groups must remain vectorized")

    monkeypatch.setattr(numeric, "_exact_means", forbidden)
    values = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float64)
    torch.testing.assert_close(numeric._finite_mean(values, 0), values.mean(0), atol=0, rtol=0)


def test_nonfinite_mean_preserves_native_prediction_behavior():
    from torch_bsf._numeric import _finite_mean

    values = torch.tensor([[float("inf"), float("nan"), 1.0], [2.0, 3.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(_finite_mean(values, 0), values.mean(0), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [3, 5, 7, 10, 17, 33])
def test_finite_mean_near_rounded_sum_boundary(dtype, count):
    from torch_bsf._numeric import _finite_mean

    boundary = torch.tensor(torch.finfo(dtype).max / count, dtype=dtype)
    below = torch.nextafter(boundary, torch.tensor(0.0, dtype=dtype))
    values = below.expand(count)
    torch.testing.assert_close(_finite_mean(values, 0), below, atol=0, rtol=0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_float32_mean_avoids_binary64_double_rounding(sign):
    from torch_bsf._numeric import _finite_mean

    values = sign * torch.tensor([8.945040568366156e37, -1.2722663938486925e-26, -1.654059785906659e37], dtype=torch.float32)
    expected = torch.tensor(sign * 2.430326800721439e37, dtype=torch.float32)
    torch.testing.assert_close(_finite_mean(values, 0), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_fraction_rounding_normal_and_subnormal_boundaries(dtype):
    from torch_bsf._numeric import _round_fraction

    info = torch.finfo(dtype)
    unit = Fraction.from_float(info.eps)
    midpoint = Fraction(1) + unit / 2
    perturbation = unit / 2**100
    # At 1, the lower endpoint has an even significand; at its successor the
    # upper endpoint is even. Perturbations must survive before target rounding.
    assert _round_fraction(midpoint, dtype) == 1
    assert _round_fraction(midpoint - perturbation, dtype) == 1
    assert _round_fraction(midpoint + perturbation, dtype) == 1 + info.eps
    assert _round_fraction(Fraction(1) + unit * Fraction(3, 2), dtype) == 1 + 2 * info.eps
    smallest = Fraction.from_float(info.tiny) * unit
    assert _round_fraction(smallest / 2, dtype) == 0
    assert _round_fraction(smallest * Fraction(513, 1024), dtype) == float(smallest)
    assert _round_fraction(-smallest * Fraction(513, 1024), dtype) == -float(smallest)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@given(scale=extreme_scalings(), divisor=st.integers(1, 31), negative=st.booleans())
def test_fraction_rounding_matches_binary_division_property(dtype, scale, divisor, negative):
    from torch_bsf._numeric import _round_fraction

    value = torch.tensor(min(scale, torch.finfo(dtype).max), dtype=dtype)
    if negative:
        value = -value
    exact = Fraction.from_float(value.item()) / divisor
    # Hardware division is an independent reference. Small integer divisors
    # are exact in every format; powers of two span extreme exponents.
    assert _round_fraction(exact, dtype) == (value / divisor).item()
