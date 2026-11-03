"""Training-only LOOCV correctness, interpolation and exact BSF export."""

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve.advanced import _cubic_candidate, _direct_loo, _mass, _press, fit_predictor  # noqa: E402
from torch_bsf._adaptive_curve.export import export_curve  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveLearningResult  # noqa: E402
from torch_bsf._adaptive_curve.spectral import _chebyshev  # noqa: E402
from torch_bsf._adaptive_curve.spectral import fit_predictor as fit_v1  # noqa: E402


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


def test_weighted_press_matches_actual_held_row_refits_for_vector_regression():
    generator = torch.Generator().manual_seed(119)
    x = torch.sort(
        torch.cat(
            (torch.tensor([-1.0, 1.0], dtype=torch.float64), torch.rand(23, generator=generator, dtype=torch.float64) * 2 - 1)
        )
    ).values
    y = torch.stack((torch.sin(3 * x), x.square()), 1) + 0.01 * torch.randn(
        (len(x), 2), generator=generator, dtype=torch.float64
    )
    design = _chebyshev(x, 4, -1, 1)
    mass = _mass(x)
    fast = _press(design, y, mass, 1e8)
    direct = _direct_loo(design, y, mass, 1e8)
    assert fast is not None and direct is not None
    assert not fast[3]
    assert fast[1] == pytest.approx(direct[0], rel=2e-12, abs=1e-14)


def test_high_degree_loo_uses_explicit_refits_and_remains_conditioned():
    x = torch.linspace(-1, 1, 25, dtype=torch.float64)
    y = torch.stack((torch.sin(5 * x), x**3), 1)
    result = _press(_chebyshev(x, 18, -1, 1), y, _mass(x), 1e8)
    assert result is not None and result[3]
    assert result[2] <= 1e8
    assert _press(torch.ones((25, 2), dtype=torch.float64), y, _mass(x), 1e8) is None


@pytest.mark.parametrize("kind", ["cubic", "pchip"])
def test_cubic_hermite_controls_match_scipy_interpolant_and_actual_bsf(kind):
    from scipy.interpolate import CubicSpline, PchipInterpolator

    x = torch.tensor([0, 0.03, 0.14, 0.25, 0.56, 0.77, 1.0], dtype=torch.float64)
    y = (x**2 + torch.sin(x * 5))[:, None]
    candidate, _ = _cubic_candidate(kind, x, y, _mass(x))
    scipy_model = (
        CubicSpline(x.numpy(), y.numpy(), axis=0) if kind == "cubic" else PchipInterpolator(x.numpy(), y.numpy(), axis=0)
    )
    at = torch.linspace(0, 1, 201, dtype=torch.float64)
    expected = torch.from_numpy(scipy_model(at.numpy()))
    assert torch.allclose(candidate.predict(at), expected, atol=2e-14, rtol=2e-14)
    from torch_bsf._adaptive_curve.spectral import SpectralCurveModel

    model = SpectralCurveModel(x[:, None], y, {}, candidate, (0, 1))
    assert torch.allclose(_export(model).predict(at[:, None]), expected, atol=3e-14, rtol=3e-14)


def test_loo_can_resolve_analytic_curves_beyond_the_old_threefold_degree_cap():
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)

    def function(t):
        return torch.stack((torch.sin(9.2 * t), torch.cos(7.4 * t), torch.exp(t / 3)), 1)

    fitted = fit_predictor(x, function(x))
    old = fit_v1(x, function(x))
    at = torch.linspace(-1, 1, 307, dtype=torch.float64)[:, None]
    error = (fitted.predict(at) - function(at[:, 0])).abs().max()
    old_error = (old.predict(at) - function(at[:, 0])).abs().max()
    assert error < 1e-9
    assert error < old_error / 100
    assert (_export(fitted).predict(at) - fitted.predict(at)).abs().max() < 2e-12


def test_vector_family_is_rotation_equivariant_and_excludes_channelwise_pchip():
    x = torch.linspace(-1, 1, 39, dtype=torch.float64)
    y = torch.stack((torch.sin(4.6 * x), torch.exp(x / 3)), 1)
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float64)
    original = fit_predictor(x, y)
    changed = fit_predictor(x, y @ rotation)
    at = torch.linspace(-1, 1, 101, dtype=torch.float64)[:, None]
    assert all(row["family"] != "pchip" for row in original.diagnostics["candidates"])
    assert torch.allclose(changed.predict(at), original.predict(at) @ rotation, atol=2e-12, rtol=2e-12)


def test_shape_preserving_scalar_member_does_not_overshoot_monotone_data():
    x = torch.linspace(0, 1, 21, dtype=torch.float64)
    y = torch.sigmoid((x - 0.43) * 40)[:, None]
    candidate, _ = _cubic_candidate("pchip", x, y, _mass(x))
    predicted = candidate.predict(torch.linspace(0, 1, 403, dtype=torch.float64))[:, 0]
    assert (predicted[1:] >= predicted[:-1] - 1e-15).all()
    assert predicted.min() >= y.min() - 1e-15 and predicted.max() <= y.max() + 1e-15


def test_small_sample_and_state_preservation():
    x = torch.tensor([0, 0.4, 1.0], dtype=torch.float64)
    y = x[:, None] ** 2
    state, error_state = torch.random.get_rng_state().clone(), np.geterr().copy()
    fitted = fit_predictor(x, y)
    assert fitted.diagnostics["family"] == "linear_small_sample"
    assert torch.allclose(fitted.predict(x[:, None]), y)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert np.geterr() == error_state


def test_interpolator_numerical_failure_remains_a_reported_rejection(monkeypatch):
    import torch_bsf._adaptive_curve.advanced as advanced

    def failed(*args, **kwargs):
        raise FloatingPointError("fixture: nonfinite cubic coefficients")

    monkeypatch.setattr(advanced, "_interpolator", failed)
    x = torch.linspace(-1, 1, 17, dtype=torch.float64)
    y = torch.stack((x**2, x**3), 1)
    result = fit_predictor(x, y)
    assert result.diagnostics["failed_interpolators"] == [
        {"family": "cubic", "error": "fixture: nonfinite cubic coefficients"}
    ]
    assert torch.allclose(result.predict(x[:, None]), y, atol=1e-12, rtol=1e-12)


def test_scalar_interpolation_condition_is_not_fabricated_as_one():
    x = torch.tensor([0, 0.02, 0.2, 0.31, 0.4, 0.43, 0.46, 0.7, 0.9, 1], dtype=torch.float64)
    y = torch.sigmoid((x - 0.43) * 100)[:, None]
    result = fit_predictor(x, y)
    for row in result.diagnostics["candidates"]:
        if row["family"] in ("pchip", "cubic"):
            assert row["condition"] is None
