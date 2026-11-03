"""Accuracy, stability and exact BSF representation of training-only models."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve.export import export_curve  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveLearningResult  # noqa: E402
from torch_bsf._adaptive_curve.spectral import fit_predictor  # noqa: E402


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _export(model):
    bounds = (float(model.inputs[0, 0]), float(model.inputs[-1, 0]))
    result = CurveLearningResult(model.inputs, model.values, (), "fixed_observations", {}, [model.as_spline()], bounds)
    return export_curve(result)


def test_generic_analytic_vector_curve_has_spectral_accuracy_and_actual_bsf_agreement():
    x = torch.linspace(-1, 1, 61, dtype=torch.float64)
    x[1:-1] += 0.008 * torch.sin(torch.arange(1, 60, dtype=torch.float64))

    def function(t):
        return torch.stack((torch.sin(7.3 * t), torch.exp(0.7 * t), torch.cos(4.2 * t) + t**3), 1)

    model = fit_predictor(x, function(x))
    at = torch.linspace(-1, 1, 307, dtype=torch.float64)[:, None]
    prediction = model.predict(at)
    assert model.diagnostics["kind"] == "chebyshev"
    assert (prediction - function(at[:, 0])).abs().max() < 3e-12
    exported = _export(model)
    assert (exported.predict(at) - prediction).abs().max() < 2e-12
    assert exported.domain.betti_numbers()[0] == 1


@settings(deadline=None)
@given(degree=st.integers(2, 7), shift=st.floats(-4, 4), scale=st.floats(0.2, 5))
def test_generated_polynomials_survive_input_units_and_vector_output_basis(degree, shift, scale):
    x = torch.linspace(-1, 1, 39, dtype=torch.float64)
    x[1:-1] += 0.009 * torch.cos(torch.arange(1, 38, dtype=torch.float64))
    y = torch.stack((x.pow(degree) - 0.3 * x, x.square() + 0.2), 1)
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    original = fit_predictor(x, y)
    changed = fit_predictor(x * scale + shift, y @ rotation * 3.7 + 12)
    at = torch.linspace(-1, 1, 101, dtype=torch.float64)[:, None]
    assert torch.allclose(
        changed.predict(at * scale + shift), original.predict(at) @ rotation * 3.7 + 12, atol=3e-11, rtol=3e-12
    )
    assert original.diagnostics["condition"] <= original.diagnostics["condition_limit"]


def test_clustered_irregular_observations_do_not_force_an_ill_conditioned_high_degree():
    x = torch.cat((torch.linspace(-1, 1, 25, dtype=torch.float64), 0.21 + torch.arange(1, 30, dtype=torch.float64) * 1e-10))
    y = torch.stack((x**4 - 0.4 * x, x**2), 1)
    model = fit_predictor(x, y)
    at = torch.linspace(-1, 1, 201, dtype=torch.float64)[:, None]
    truth = torch.cat((at**4 - 0.4 * at, at**2), 1)
    assert (model.predict(at) - truth).abs().max() < 3e-12
    assert model.diagnostics["condition"] <= 1e8
    assert model.diagnostics["degree"] == 4


def test_orthogonal_64_output_embedding_keeps_accuracy_and_exact_bsf_values():
    generator = torch.Generator().manual_seed(1903)
    basis = torch.linalg.qr(torch.randn((64, 3), generator=generator, dtype=torch.float64)).Q
    x = torch.linspace(-1, 1, 61, dtype=torch.float64)

    def function(t):
        return torch.stack((torch.sin(6.7 * t), torch.exp(t / 3), t**4), 1) @ basis.T

    model = fit_predictor(x, function(x))
    at = torch.linspace(-1, 1, 101, dtype=torch.float64)[:, None]
    assert torch.linalg.vector_norm(model.predict(at) - function(at[:, 0]), dim=1).max() < 4e-12
    assert (_export(model).predict(at) - model.predict(at)).abs().max() < 2e-12


def test_c0_kink_has_continuous_exact_piecewise_export():
    x = torch.linspace(0, 1, 61, dtype=torch.float64)
    y = torch.stack((torch.relu(x - 0.4), torch.abs(x - 0.4)), 1)
    model = fit_predictor(x, y)
    at = torch.linspace(0, 1, 403, dtype=torch.float64)[:, None]
    truth = torch.cat((torch.relu(at - 0.4), torch.abs(at - 0.4)), 1)
    assert (model.predict(at) - truth).abs().max() < 0.015
    assert torch.allclose(_export(model).predict(at), model.predict(at), atol=2e-12, rtol=2e-12)
    spline = model.as_spline()
    for knot in spline.knots[1:-1]:
        near = torch.stack((knot - 1e-10, knot + 1e-10))[:, None]
        assert (model.predict(near)[0] - model.predict(near)[1]).abs().max() < 1e-8


def test_noise_selection_does_not_default_to_interpolating_every_response():
    generator = torch.Generator().manual_seed(811)
    x = torch.linspace(-1, 1, 81, dtype=torch.float64)
    noise = 0.06 * torch.randn((81, 4), generator=generator, dtype=torch.float64)
    y = torch.stack((x, x.square(), x**3, torch.sin(x)), 1) + noise
    model = fit_predictor(x, y, noise=4 * 0.06**2)
    assert model.diagnostics["n_controls_native"] < 20
    assert torch.isfinite(model.predict(x[:, None])).all()


@pytest.mark.parametrize("count", [1, 2, 5])
def test_small_sample_support_and_endpoint_clamping(count):
    x = torch.linspace(0.2, 0.8, count, dtype=torch.float64)
    y = torch.stack((x, 2 * x + 1), 1)
    model = fit_predictor(x, y, bounds=(0, 1))
    assert torch.equal(model.predict(torch.tensor([[0.0], [1.0]], dtype=torch.float64)), y[[0, -1]])
    assert not model.diagnostics["cv_available"]
    assert torch.isfinite(model.predict(torch.empty((0, 1), dtype=torch.float64))).all()


def test_no_input_or_rng_or_decimal_context_mutation():
    from decimal import getcontext

    x = torch.linspace(0, 1, 37, dtype=torch.float64)
    y = torch.stack((torch.sin(6 * x), x**3), 1)
    originals = (x.clone(), y.clone(), torch.random.get_rng_state().clone(), getcontext().copy())
    model = fit_predictor(x, y)
    model.as_spline()
    assert torch.equal(x, originals[0]) and torch.equal(y, originals[1])
    assert torch.equal(torch.random.get_rng_state(), originals[2])
    assert getcontext().prec == originals[3].prec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise": -1},
        {"noise": math.nan},
        {"max_degree": True},
        {"max_pieces": 0},
        {"max_condition": 1},
        {"bounds": (0.2, 1)},
        {"bounds": ((0, 1),)},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_predictor(torch.linspace(0, 1, 7, dtype=torch.float64), torch.ones((7, 2), dtype=torch.float64), **kwargs)


def test_duplicate_nonfinite_and_out_of_bounds_queries_rejected():
    with pytest.raises(ValueError, match="unique"):
        fit_predictor([0, 0, 1], [[0], [0], [1]])
    with pytest.raises(ValueError):
        fit_predictor([0, math.inf], [[0], [1]])
    model = fit_predictor([0.0, 0.123456789123, 1.0], [[0.0], [0.2], [1.0]])
    assert model.inputs[1, 0] == 0.123456789123
    with pytest.raises(ValueError, match="within bounds"):
        model.predict([[1.1]])
