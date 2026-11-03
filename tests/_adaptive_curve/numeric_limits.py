"""Response-unit invariance and bounded-memory physical spline evaluation."""

import math

import pytest
import torch

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve.advanced import _linear_candidate, _press, fit_predictor  # noqa: E402
from torch_bsf._adaptive_curve.export import _compile_result, export_curve  # noqa: E402
from torch_bsf._adaptive_curve.fitting import _fit_prediction  # noqa: E402
from torch_bsf._adaptive_curve.kernel_prediction import refit_prediction  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveLearningResult, StructureLearningResult, _basis, _Spline  # noqa: E402
from torch_bsf._adaptive_curve.rational import fit_rational  # noqa: E402
from torch_bsf._adaptive_curve.response_units import comparison_score, response_scale, restore_score  # noqa: E402
from torch_bsf._adaptive_curve.spectral import _mass, _solve  # noqa: E402
from torch_bsf._adaptive_curve.spectral import fit_predictor as fit_spectral  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
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


@pytest.mark.parametrize("fitter", [fit_predictor, fit_spectral, _fit_prediction])
@pytest.mark.parametrize("value", [1e308, -1e308])
def test_finite_large_constant_does_not_overflow_center_or_solver(fitter, value):
    x = torch.linspace(0, 1, 9, dtype=torch.float64)
    y = torch.full((len(x), 2), value, dtype=torch.float64)
    model = fitter(x, y, noise=0)
    at = torch.linspace(0, 1, 17, dtype=torch.float64)[:, None]
    torch.testing.assert_close(model.predict(at), y[:1].expand(len(at), -1), rtol=1e-15, atol=0)
    torch.testing.assert_close(_export(model).predict(at), y[:1].expand(len(at), -1), rtol=1e-15, atol=0)


def test_nonfinite_intermediate_targets_are_rejected_before_lstsq(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not pass nonfinite intermediate targets to a linear solver.")

    monkeypatch.setattr(torch.linalg, "lstsq", forbidden)
    design = torch.ones((9, 1), dtype=torch.float64)
    values = torch.full((9, 1), math.inf, dtype=torch.float64)
    assert _solve(design, values, torch.ones(9), 1e8) is None
    assert _press(design, values, torch.ones(9), 1e8) is None


@pytest.mark.parametrize("scale", [1e-170, math.ldexp(1.0, -565), 1e150])
def test_tiny_and_large_quadratic_keep_curvature_and_comparable_cv_scores(scale):
    x = torch.linspace(0, 1, 9, dtype=torch.float64)
    model = _fit_prediction(x, x.square()[:, None] * scale, noise=0)
    at = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    torch.testing.assert_close(_export(model).predict(at) / scale, at.square(), rtol=0, atol=2e-14)
    score = comparison_score(model.diagnostics, response_scale(model.values))
    assert score is not None and score < 1e-25
    assert model.diagnostics["cv_response_scale"] != 1


def test_subnormal_responses_retain_resolvable_curvature():
    x = torch.linspace(0, 1, 9, dtype=torch.float64)
    scale = 1e-320
    model = _fit_prediction(x, x.square()[:, None] * scale, noise=0)
    at = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    torch.testing.assert_close(_export(model).predict(at) / scale, at.square(), rtol=0, atol=0.003)


@pytest.mark.parametrize("scale", [1.0, 1e-170, 1e150])
def test_kernel_competes_in_common_units_when_physical_mse_underflows(scale):
    x = torch.linspace(0, 1, 19, dtype=torch.float64)
    normalized = torch.sin(7 * x)[:, None]
    values = normalized * scale
    unit = response_scale(values)
    candidate = _linear_candidate(x, values / unit, _mass(x))
    capacity = {"weighted_cv_mse": candidate.score}
    if unit != 1:
        restore_score(capacity, "weighted_cv_mse", unit)
    native = _compile_result(
        StructureLearningResult(
            x[:, None],
            values,
            (),
            "fixed_observations",
            {"capacity": [capacity]},
            [_Spline(x, 1, values, torch.zeros(1, dtype=torch.float64))],
            (0, 1),
        )
    )
    updated = refit_prediction(native)
    diagnostic = updated.diagnostics["kernel_prediction"][0]
    assert diagnostic["selected"] == "stationary_kernel"
    if scale == 1e-170:
        assert diagnostic["native_observed_cv_mse"] == diagnostic["kernel_observed_cv_mse"] == 0
        assert diagnostic["kernel_normalized_cv_mse"] < diagnostic["native_normalized_cv_mse"]
    at = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
    assert (updated.predict(at) / scale - torch.sin(7 * at)).abs().max() < 2e-5


@pytest.mark.parametrize("scale", [1e-170, 1e150])
def test_rational_conversion_checks_use_normalized_responses(scale):
    x = torch.linspace(0, 1, 17, dtype=torch.float64)
    model = fit_rational(x, (1 / (x + 1.5))[:, None] * scale)
    assert model is not None
    compiled = _export(model)
    at = torch.linspace(0, 1, 257, dtype=torch.float64)[:, None]
    torch.testing.assert_close(compiled.predict(at) / scale, 1 / (at + 1.5), rtol=0, atol=2e-10)
    assert model.diagnostics["normalized_conversion_checked_tolerance"] > 0
    assert model.diagnostics["conversion_response_scale"] != 1


def test_translated_spectral_pieces_follow_represented_physical_knots():
    origin = 1e12
    x = torch.linspace(origin, origin + 1, 33, dtype=torch.float64)
    model = fit_predictor(x, torch.sin(4.6 * (x - origin))[:, None])
    spline = model.as_spline()
    assert len(spline.knots) > 2
    at = torch.unique(torch.cat((torch.linspace(origin, origin + 1, 4097, dtype=torch.float64), spline.knots)))[:, None]
    native, actual = model.predict(at), _export(model).predict(at)
    torch.testing.assert_close(native, torch.sin(4.6 * (at - origin)), rtol=0, atol=2e-14)
    torch.testing.assert_close(actual, native, rtol=0, atol=2e-14)


@pytest.mark.parametrize("degree", [0, 1, 3, 12])
def test_local_spline_matches_global_regression_basis_at_knots_and_endpoints(degree):
    generator = torch.Generator().manual_seed(420)
    knots = torch.tensor([-0.7, -0.4, -0.1, 0.25, 0.9], dtype=torch.float64)
    controls = torch.randn(((len(knots) - 1) * degree + 1, 4), generator=generator, dtype=torch.float64)
    center = torch.randn(4, generator=generator, dtype=torch.float64)
    at = torch.cat((knots, knots[1:-1] - 1e-14, knots[1:-1] + 1e-14, torch.linspace(-1, 1, 137, dtype=torch.float64)))
    spline = _Spline(knots, degree, controls, center)
    torch.testing.assert_close(spline.predict(at), _basis(at, knots, degree) @ controls + center, rtol=0, atol=3e-15)
    assert spline.predict(torch.empty(0, dtype=torch.float64)).shape == (0, 4)


def test_many_piece_prediction_does_not_allocate_the_global_regression_basis(monkeypatch):
    from torch_bsf._adaptive_curve import polynomial

    def forbidden(*args, **kwargs):
        raise AssertionError("Query evaluation must use only the local piece, not a query-by-all-controls matrix.")

    monkeypatch.setattr(polynomial, "_basis", forbidden)
    knots = torch.linspace(0, 1, 1025, dtype=torch.float64)
    controls = torch.sin(torch.linspace(0, 4, 1024 * 12 + 1, dtype=torch.float64))[:, None].expand(-1, 3).clone()
    spline = _Spline(knots, 12, controls, torch.zeros(3, dtype=torch.float64))
    at = torch.linspace(0, 1, 28673, dtype=torch.float64)
    result = spline.predict(at)
    assert result.shape == (len(at), 3) and torch.isfinite(result).all()
    torch.testing.assert_close(result[[0, -1]], controls[[0, -1]], rtol=0, atol=0)
    singleton = _Spline(knots[:1], 0, controls[:1], spline.center)
    torch.testing.assert_close(singleton.predict(at), controls[:1].expand(len(at), -1), rtol=0, atol=0)
