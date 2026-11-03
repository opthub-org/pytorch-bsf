"""Independent rational-fit, pole-screen and empirical BSF conversion checks."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve.export import export_curve  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveLearningResult  # noqa: E402
from torch_bsf._adaptive_curve.queries import QuerySession  # noqa: E402
from torch_bsf._adaptive_curve.rational import RationalCurveModel, _Rational, fit_rational  # noqa: E402


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _export(model):
    return export_curve(
        CurveLearningResult(
            model.inputs,
            model.values,
            (),
            "fixed_observations",
            {},
            [model.as_spline()],
            (float(model.inputs[0, 0]), float(model.inputs[-1, 0])),
        )
    )


def test_pole_pencil_distinguishes_a_known_real_pole_from_a_polynomial():
    support = np.array([-1.0, 1.0])
    values = np.array([[0.0], [1.0]])
    pole = _Rational(support, values, np.array([1.0, 1.0]), np.zeros(1))
    polynomial = _Rational(support, values, np.array([1.0, -1.0]), np.zeros(1))
    assert pole.has_real_pole(-0.5, 0.5)
    assert not pole.has_real_pole(0.1, 0.9)
    assert not polynomial.has_real_pole(-1, 1)


def test_generic_vector_rational_fit_interpolates_and_exports_without_oracle_calls():
    x = torch.linspace(-1, 1, 29, dtype=torch.float64)[:, None]

    def function(at):
        return torch.cat((1 / (1 + at**2), (at + 0.3) / (1 + at**2), at / (2 + at**2)), 1)

    session = QuerySession(function, [(-1, 1)], budget=len(x))
    session.query(x)
    inputs, values, _ = session.observations()

    def forbidden(_):
        raise AssertionError("model-only fitting and conversion must not query")

    session.oracle = forbidden
    model = fit_rational(inputs, values)
    assert model is not None
    at = torch.linspace(-1, 1, 509, dtype=torch.float64)[:, None]
    native = model.predict(at)
    assert (native - function(at)).abs().max() < 2e-10
    assert (model.predict(inputs) - values).abs().max() < 2e-10
    spline = model.as_spline()
    assert model.as_spline() is spline
    compiled = _export(model)
    assert (compiled.predict(at) - spline.predict(at[:, 0])).abs().max() < 3e-12
    assert (
        torch.linalg.vector_norm(compiled.predict(at) - native, dim=1).max()
        < 3 * model.diagnostics["conversion_checked_tolerance"]
    )
    assert model.diagnostics["conversion_oracle_rows"] == 0
    assert model.diagnostics["conversion_is_certified"] is False
    assert session.cost == len(x)
    json.dumps(model.diagnostics, allow_nan=False)


def test_joint_vector_weights_are_rotation_and_input_unit_equivariant():
    x = torch.linspace(-1, 1, 23, dtype=torch.float64)[:, None]
    y = torch.cat((1 / (1 + x**2), x / (1 + x**2)), 1)
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float64)
    original = fit_rational(x, y)
    changed = fit_rational(x * 3.7 + 12, y @ rotation * 2.3 + 7)
    assert original is not None and changed is not None
    at = torch.linspace(-1, 1, 137, dtype=torch.float64)[:, None]
    assert torch.allclose(changed.predict(at * 3.7 + 12), original.predict(at) @ rotation * 2.3 + 7, atol=2e-10, rtol=2e-10)


def test_distinct_physical_locations_that_collapse_after_scaling_abstain_safely():
    x = torch.tensor([0, 1e-18, 1e-17, 1e-16, 0.1, 0.3, 0.6, 1], dtype=torch.float64)
    y = torch.stack((x, x.square()), 1)
    assert fit_rational(x, y) is None
    from torch_bsf._adaptive_curve.fitting import _fit_prediction

    fallback = _fit_prediction(x, y, 0.0)
    assert torch.isfinite(fallback.predict(torch.tensor([[0.2], [0.8]], dtype=torch.float64))).all()


def test_conversion_does_not_discard_an_observed_narrow_pole_free_peak():
    center, width = 0.123456789, 1e-8
    support = np.array([-1.0, center, 1.0])
    denominator = (support - center) ** 2 + width**2
    weights = np.array([denominator[i] / np.prod(support[i] - np.delete(support, i)) for i in range(len(support))])
    rational = _Rational(support, (width**2 / denominator)[:, None], weights, np.zeros(1))
    assert not rational.has_real_pole(-1, 1)
    x = np.sort(np.r_[np.linspace(-1, 1, 17), center])
    values = rational.predict(x)
    model = RationalCurveModel(torch.from_numpy(x[:, None]), torch.from_numpy(values), {}, rational)
    try:
        spline = model.as_spline()
    except ArithmeticError:
        return  # A declared failure is preferable to losing an observed feature.
    actual_error = (spline.predict(torch.from_numpy(x)) - torch.from_numpy(values)).abs().max()
    assert actual_error <= model.diagnostics["conversion_checked_tolerance"] * 3


def test_constant_data_and_fit_state_remain_finite_and_unchanged():
    x = torch.linspace(-1, 1, 17, dtype=torch.float64)[:, None]
    y = torch.tensor([2.0, -3.0], dtype=torch.float64).expand(len(x), -1).clone()
    originals = (x.clone(), y.clone(), torch.random.get_rng_state().clone(), np.geterr().copy())
    model = fit_rational(x, y)
    assert model is not None
    assert torch.allclose(model.predict(x), y)
    assert torch.allclose(_export(model).predict(x), y, atol=1e-12, rtol=1e-12)
    assert torch.equal(x, originals[0]) and torch.equal(y, originals[1])
    assert torch.equal(torch.random.get_rng_state(), originals[2])
    assert np.geterr() == originals[3]


def test_invalid_observations_rejected_and_insufficient_sample_count_abstains():
    assert fit_rational(torch.arange(7, dtype=torch.float64), torch.ones((7, 1), dtype=torch.float64)) is None
    with pytest.raises(ValueError):
        fit_rational([0, 0, 1, 2, 3, 4, 5, 6], [[0]] * 8)
    with pytest.raises(ValueError):
        fit_rational([0, 1, 2, 3, 4, 5, 6, float("nan")], [[0]] * 8)


@pytest.mark.parametrize(("left", "right"), [(-1.7, 0.3), (-2.3, 0.7), (-0.7, 0.3)])
def test_physical_endpoint_rounding_does_not_break_conversion_or_selection(left, right):
    from torch_bsf._adaptive_curve.fitting import _fit_prediction

    x = torch.linspace(left, right, 17, dtype=torch.float64)
    y = (1 / (1 + x.square()))[:, None]
    model = fit_rational(x, y)
    assert model is not None
    spline = model.as_spline()
    assert spline.knots[0] == x[0] and spline.knots[-1] == x[-1]
    assert torch.allclose(spline.predict(x), model.predict(x[:, None]), atol=2e-10, rtol=2e-10)
    selected = _fit_prediction(x, y, 0.0)
    assert torch.isfinite(selected.as_spline().predict(x)).all()


@pytest.mark.parametrize("which", ["svd", "eigen"])
def test_bounded_numerical_family_failure_keeps_connected_alternative(monkeypatch, which):
    import torch_bsf._adaptive_curve.rational as rational

    def failed(*args, **kwargs):
        raise np.linalg.LinAlgError("fixture: decomposition failed")

    if which == "svd":
        monkeypatch.setattr(np.linalg, "svd", failed)
    else:
        monkeypatch.setattr(rational, "eigvals", failed)
    x = torch.linspace(-1, 1, 17, dtype=torch.float64)
    y = (1 / (1 + x.square()))[:, None]
    assert fit_rational(x, y) is None
