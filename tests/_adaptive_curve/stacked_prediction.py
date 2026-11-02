"""Independent conditional PRESS, support conversion and retained actual BSF."""

import random

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from torch_bsf._adaptive_curve import stacked_prediction as profile  # noqa: E402
from torch_bsf._adaptive_curve.component_kernel import _expanded_nugget, fit_component_kernel  # noqa: E402
from torch_bsf._adaptive_curve.export import _compile_result  # noqa: E402
from torch_bsf._adaptive_curve.kernel import _kernel  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveGap, StructureLearningResult, _Spline  # noqa: E402
from torch_bsf._adaptive_curve.shared_spline import fit_spline_charts  # noqa: E402
from torch_bsf._adaptive_curve.stacked_prediction import (  # noqa: E402
    _blend,
    _c0_residuals,
    _gp_residuals,
    _weight,
    refit_prediction,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def setup_curve():
    x = torch.cat((torch.linspace(-1, -0.02, 12), torch.linspace(0.02, 1, 12))).double()
    y = torch.stack((torch.sin(4 * x), x.square(), torch.cos(2 * x)), 1)
    y[12:] += torch.tensor([0.7, -0.3, 0.1], dtype=torch.float64)
    gaps = (CurveGap(float(x[11]), float(x[12]), 0.8, 0.02, 0),)
    models, diagnostic = fit_spline_charts(x, y, gaps)
    diagnostic["selected_predictor"] = diagnostic["family"]
    return _compile_result(
        StructureLearningResult(x[:, None], y, gaps, "budget_exhausted", {"spline_pooling": diagnostic}, models, (-1, 1))
    )


def test_signed_c0_and_gp_press_use_same_rows_masses_and_replay_recorded_cv():
    native = setup_curve()
    residuals = _c0_residuals(native)
    assert residuals is not None
    errors, mass, validation = residuals
    actual = float((errors[validation].square().sum(1) * mass[validation]).sum() / mass[validation].sum())
    assert actual == pytest.approx(native.diagnostics["spline_pooling"]["weighted_cv_mse"], rel=1e-12, abs=1e-15)
    model = fit_component_kernel(native.inputs[:, 0], native.values, native.gaps)
    assert model is not None
    gp_errors = _gp_residuals(model, native, mass)
    assert gp_errors is not None
    actual = float((gp_errors[validation].square().sum(1) * mass[validation]).sum() / mass[validation].sum())
    assert actual == pytest.approx(model.diagnostics["weighted_cv_mse"], rel=1e-10, abs=1e-14)


def test_convex_cv_weight_is_rotation_invariant_and_optimizes_common_heldout_risk():
    generator = torch.Generator().manual_seed(717)
    first = torch.randn(15, 4, generator=generator, dtype=torch.float64)
    second = -0.4 * first + 0.1 * torch.randn(15, 4, generator=generator, dtype=torch.float64)
    mass = torch.linspace(0.3, 1, 15, dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64)).Q
    weight, score = _weight(first, second, mass)
    other_weight, other_score = _weight(first @ rotation, second @ rotation, mass)
    assert weight == pytest.approx(other_weight, abs=128 * torch.finfo(torch.float64).eps)
    assert score == pytest.approx(other_score, abs=128 * torch.finfo(torch.float64).eps)
    for candidate in torch.linspace(0, 1, 101, dtype=torch.float64):
        actual = float(((first + candidate * (second - first)).square().sum(1) * mass).sum() / mass.sum())
        assert score <= actual + 128 * torch.finfo(torch.float64).eps


def test_mixed_degree_knot_union_blends_exactly_without_sampling_or_refitting():
    first = _Spline(
        torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64),
        2,
        torch.arange(10, dtype=torch.float64).reshape(5, 2),
        torch.ones(2, dtype=torch.float64),
    )
    second = _Spline(
        torch.tensor([0.0, 0.6, 1.0], dtype=torch.float64),
        3,
        torch.arange(14, dtype=torch.float64).reshape(7, 2),
        -torch.ones(2, dtype=torch.float64),
    )
    weight = 0.37
    blend = _blend(first, second, weight)
    query = torch.linspace(0, 1, 201, dtype=torch.float64)
    expected = (1 - weight) * first.predict(query) + weight * second.predict(query)
    torch.testing.assert_close(
        blend.predict(query), expected, rtol=0, atol=float(256 * torch.finfo(torch.float64).eps * expected.abs().max())
    )
    assert blend.degree == 3 and torch.equal(blend.knots, torch.unique(torch.cat((first.knots, second.knots))))


def test_expansion_is_training_only_bounded_and_stops_after_an_interior_best():
    unit = np.linspace(0, 1, 12)
    covariance = _kernel(unit, unit, "rbf", 0.3)
    mass = np.ones(len(unit)) / len(unit)
    validation = np.ones(len(unit), dtype=bool)
    validation[0] = validation[-1] = False
    trend = [np.ones((len(unit), 1))]
    # A pure high-frequency residual has no smooth signal. It exercises the
    # generic large-ridge regime without a named benchmark or test responses.
    values = np.column_stack(((-1.0) ** np.arange(len(unit)), np.sin(17 * unit)))
    best, _, trace = _expanded_nugget({("rbf", 0.3): covariance}, values, trend, mass, validation)
    assert best is not None and trace[-1]["largest_tested_ridge"] <= 1000
    assert len(trace) <= 12
    assert any(row["largest_tested_ridge"] > 0.01 for row in trace)
    if trace[-1]["largest_tested_ridge"] < 1000:
        assert best[1] < trace[-1]["largest_tested_ridge"]


def test_wrapper_preserves_observed_inputs_masks_actual_bsf_and_global_rng():
    native = setup_curve()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    assert isinstance(numpy_state, tuple)
    torch_state = torch.get_rng_state().clone()
    original = native.inputs.clone(), native.values.clone()
    native.diagnostics["cost"] = len(native.inputs)
    actual = refit_prediction(native)
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert isinstance(current, tuple)
    assert current[0] == numpy_state[0] and current[2:] == numpy_state[2:]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert torch.equal(torch_state, torch.get_rng_state())
    assert actual.diagnostics["cost"] == len(native.inputs)
    assert torch.equal(native.inputs, original[0]) and torch.equal(native.values, original[1])
    assert actual.gaps == native.gaps and torch.equal(actual.inputs, native.inputs)
    query = torch.linspace(-1, 1, 151, dtype=torch.float64)[:, None]
    assert torch.equal(torch.isfinite(actual.predict(query)), torch.isfinite(native.predict(query)))
    assert actual._compiled is not None and actual._compiled.domain.betti_numbers()[0] == 2
    diagnostic = actual.diagnostics["stacked_gap_prediction"]
    assert 0 <= diagnostic["stack_weight"] <= 1
    assert diagnostic["stack_cv_mse"] <= diagnostic["c0_common_row_cv_mse"] + 256 * torch.finfo(torch.float64).eps


def test_expanded_fit_is_rotation_equivariant_without_modifying_sources_or_observations():
    native = setup_curve()
    x, y = native.inputs[:, 0], native.values
    original = x.clone(), y.clone()
    generator = torch.Generator().manual_seed(771)
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64)).Q
    first, second = fit_component_kernel(x, y, native.gaps), fit_component_kernel(x, y @ rotation, native.gaps)
    assert first is not None and second is not None
    for key in ("kind", "length", "ridge", "eta", "degree"):
        assert first.diagnostics[key] == second.diagnostics[key]
    query = torch.cat((torch.linspace(-1, -0.02, 21), torch.linspace(0.02, 1, 21))).double()[:, None]
    tolerance = torch.finfo(torch.float64).eps * first.diagnostics["condition"] * 512 * float(y.norm())
    torch.testing.assert_close(first.predict(query) @ rotation, second.predict(query), rtol=0, atol=tolerance, equal_nan=True)
    assert torch.equal(x, original[0]) and torch.equal(y, original[1])


@given(
    first_degree=st.integers(0, 5),
    second_degree=st.integers(0, 6),
    weight=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    seed=st.integers(0, 10000),
)
@settings(max_examples=24, deadline=None)
def test_exact_blend_identity_over_generated_degrees_and_independent_c0_partitions(first_degree, second_degree, weight, seed):
    generator = torch.Generator().manual_seed(seed)
    first = _Spline(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        first_degree,
        torch.randn(2 * first_degree + 1, 3, generator=generator, dtype=torch.float64),
        torch.randn(3, generator=generator, dtype=torch.float64),
    )
    second = _Spline(
        torch.tensor([-1.0, -0.25, 0.5, 1.0], dtype=torch.float64),
        second_degree,
        torch.randn(3 * second_degree + 1, 3, generator=generator, dtype=torch.float64),
        torch.randn(3, generator=generator, dtype=torch.float64),
    )
    query = torch.linspace(-1, 1, 129, dtype=torch.float64)
    expected = (1 - weight) * first.predict(query) + weight * second.predict(query)
    magnitude = max(float(first.controls.abs().max()), float(second.controls.abs().max()), float(expected.abs().max()), 1.0)
    tolerance = torch.finfo(torch.float64).eps * magnitude * 512 * (first_degree + second_degree + 2)
    actual = _blend(first, second, weight)
    torch.testing.assert_close(actual.predict(query), expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("outputs,has_gap,noise", [(1, True, 0), (2, False, 0), (2, True, 0.1)])
def test_existing_scalar_connected_and_estimated_noise_guards_are_identity(outputs, has_gap, noise):
    x = torch.linspace(0, 1, 9, dtype=torch.float64)[:, None]
    native = StructureLearningResult(
        x,
        x.expand(-1, outputs),
        (CurveGap(0.4, 0.6, 1, 0, 0),) if has_gap else (),
        "budget_exhausted",
        {"noise_variance": noise},
        [],
        (0, 1),
    )
    assert profile.refit_prediction(native) is native


def test_no_common_cv_retains_native_without_attempting_new_gp(monkeypatch):
    x = torch.linspace(0, 1, 9, dtype=torch.float64)[:, None]
    y = torch.cat((x, x.square()), 1)
    spline = _Spline(x[:, 0], 1, y, torch.zeros(2, dtype=torch.float64))
    native = _compile_result(
        StructureLearningResult(
            x, y, (), "budget_exhausted", {"spline_pooling": {"selected_predictor": "native_chart_models"}}, [spline], (0, 1)
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A native fallback must not be bypassed by the new converter")

    monkeypatch.setattr(profile, "fit_component_kernel", forbidden)
    result = profile.refit_prediction(native, connected_control=True)
    assert result._compiled is native._compiled
    assert result.diagnostics["stacked_gap_prediction"]["reason"] == "no_common_c0_cv"


def test_no_admissible_gp_preserves_the_original_actual_bsf(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x = torch.linspace(0, 1, 12, dtype=torch.float64)
        y = torch.stack((x, x.square()), 1)
        models, diagnostic = fit_spline_charts(x, y)
        diagnostic["selected_predictor"] = diagnostic["family"]
        native = _compile_result(
            StructureLearningResult(x[:, None], y, (), "budget_exhausted", {"spline_pooling": diagnostic}, models, (0, 1))
        )
        monkeypatch.setattr(profile, "fit_component_kernel", lambda *args, **kwargs: None)
        result = profile.refit_prediction(native, connected_control=True)
        assert result._compiled is native._compiled
        assert result.diagnostics["stacked_gap_prediction"]["reason"] == "no_admissible_gp"
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("failure", ["conversion", "direction"])
def test_failed_conversion_or_reversed_gap_retains_the_existing_actual_model(monkeypatch, failure):
    native = setup_curve()
    model = fit_component_kernel(native.inputs[:, 0], native.values, native.gaps)
    assert model is not None
    monkeypatch.setattr(profile, "fit_component_kernel", lambda *args, **kwargs: model)
    monkeypatch.setattr(profile, "_weight", lambda *args: (1.0, -1.0))

    def failed_conversion(*args):
        if failure == "conversion":
            raise ArithmeticError("test conversion rejection")
        charts = [
            _Spline(chart.knots.clone(), chart.degree, chart.controls.clone(), chart.center.clone())
            for chart in native._models
        ]
        observed = native.values[12] - native.values[11]
        charts[1].center -= 100 * observed
        return charts

    monkeypatch.setattr(profile, "component_splines", failed_conversion)
    actual = refit_prediction(native)
    assert actual._compiled is native._compiled
    assert actual._models is native._models and actual.gaps == native.gaps
    assert actual.diagnostics["stacked_gap_prediction"]["selected_predictor"] == "c0_native_fallback"
    expected = "test conversion rejection" if failure == "conversion" else "retained-gap direction constraint"
    assert expected in actual.diagnostics["stacked_gap_prediction"]["fallback_error"]
    query = torch.linspace(-1, 1, 129, dtype=torch.float64)[:, None]
    torch.testing.assert_close(actual.predict(query), native.predict(query), rtol=0, atol=0, equal_nan=True)
