"""Native C0/PRESS equivalence, shared response jumps, and safe fallback."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve.advanced import _press  # noqa: E402
from torch_bsf._adaptive_curve.export import _compile_result  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveGap, StructureLearningResult, _basis, _Spline  # noqa: E402
from torch_bsf._adaptive_curve.pooled_prediction import gap_contrasts, refit_prediction  # noqa: E402
from torch_bsf._adaptive_curve.shared_spline import _contrast, _score, fit_spline_charts, restrict_spline  # noqa: E402
from torch_bsf._adaptive_curve.spectral import _mass  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_global_c0_design_press_matches_frozen_native_degree5_two_piece_family():
    x = torch.linspace(-1, 1, 64, dtype=torch.float64)
    y = torch.randn((64, 4), generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    knots = x[torch.tensor([0, 32, 63])]
    basis = _basis(x, knots, 5)
    mass, validation = _mass(x), torch.ones(64, dtype=torch.bool)
    validation[0] = validation[-1] = False
    centered = y - y.mean(0)
    native = _press(basis, centered, mass, 1e8)
    design = torch.cat((torch.ones((64, 1), dtype=torch.float64), basis @ _contrast(basis.shape[1])), 1)
    new = _score(design, y, mass, torch.zeros(design.shape[1], dtype=torch.float64), validation)
    assert native is not None and new is not None
    assert new.score == pytest.approx(native[1], rel=1e-11, abs=1e-12)


def test_shared_degree5_two_piece_shape_with_chart_offset_has_exact_actual_bsf():
    x = torch.cat((torch.linspace(-1, -0.02, 32), torch.linspace(0.02, 1, 32))).double()
    knots = x[torch.tensor([0, 32, 63])]
    controls = 0.1 * torch.randn((11, 3), generator=torch.Generator().manual_seed(75), dtype=torch.float64)
    source = _Spline(knots, 5, controls, torch.zeros(3, dtype=torch.float64))
    y = source.predict(x)
    y[32:] += torch.tensor([0.8, 0.2, -0.1], dtype=torch.float64)
    gap = CurveGap(float(x[31]), float(x[32]), 1, 0, 0)
    models, diagnostics = fit_spline_charts(x, y, (gap,))
    result = _compile_result(StructureLearningResult(x[:, None], y, (gap,), "budget_exhausted", {}, models, (-1, 1)))
    query = torch.linspace(-1, 1, 301, dtype=torch.float64)
    reference = source.predict(query)
    reference[query >= gap.right] += torch.tensor([0.8, 0.2, -0.1], dtype=torch.float64)
    valid = (query <= gap.left) | (query >= gap.right)
    predicted = result.predict(query[:, None])
    assert torch.equal(torch.isfinite(predicted).all(1), valid)
    torch.testing.assert_close(predicted[valid], reference[valid], rtol=1e-8, atol=1e-9)
    assert result._compiled is not None and result._compiled.domain.betti_numbers()[0] == 2
    assert diagnostics["degree"] <= 6 and diagnostics["pieces"] <= 4
    contrast = gap_contrasts(result)[0]
    assert contrast["predicted_observed_ratio"] == pytest.approx(1, abs=1e-9)
    assert contrast["direction_cosine"] == pytest.approx(1, abs=1e-9)


def test_rotation_units_and_private_fit_leave_global_random_state_unchanged():
    x = torch.linspace(-1, 1, 48, dtype=torch.float64)
    generator = torch.Generator().manual_seed(311)
    y = torch.stack((x, x.square(), torch.sin(7 * x), (x > 0).double()), 1)
    y += 0.04 * torch.randn(y.shape, generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn((4, 4), generator=generator, dtype=torch.float64)).Q
    gaps = [(float(x[23]), float(x[24]))]
    state = torch.get_rng_state().clone()
    first, _ = fit_spline_charts(x, y, gaps)
    second, _ = fit_spline_charts(x, 0.001 * y @ rotation, gaps)
    assert torch.equal(state, torch.get_rng_state())
    for a, b in zip(first, second, strict=True):
        query = torch.linspace(float(a.knots[0]), float(a.knots[-1]), 51, dtype=torch.float64)
        torch.testing.assert_close(a.predict(query) @ rotation * 0.001, b.predict(query), rtol=1e-8, atol=1e-11)


def test_two_input_components_do_not_certify_a_response_gap():
    x = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)
    y = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    models = [
        _Spline(x[:2], 0, torch.zeros((1, 2), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)),
        _Spline(x[2:], 0, torch.zeros((1, 2), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)),
    ]
    gap = CurveGap(-0.5, 0.5, 1, 0, 0)
    result = _compile_result(StructureLearningResult(x[:, None], y, (gap,), "budget_exhausted", {}, models, (-1, 1)))
    assert result._compiled is not None and result._compiled.domain.betti_numbers()[0] == 2
    contrast = gap_contrasts(result)[0]
    assert contrast["near_coinciding_outputs"] is True
    assert contrast["predicted_endpoint_norm"] == 0 and contrast["direction_cosine"] is None


def test_unsupported_small_chart_fit_retains_actual_native_predictor():
    x = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)
    y = torch.stack((x > 0, x > 0), 1).double()
    models = [
        _Spline(x[:2], 0, torch.zeros((1, 2), dtype=torch.float64), y[0]),
        _Spline(x[2:], 0, torch.zeros((1, 2), dtype=torch.float64), y[-1]),
    ]
    native = _compile_result(
        StructureLearningResult(x[:, None], y, (CurveGap(-0.5, 0.5, 1, 0, 0),), "budget_exhausted", {}, models, (-1, 1))
    )
    actual = refit_prediction(native)
    assert actual._compiled is native._compiled
    assert actual.gaps == native.gaps and "fallback_error" in actual.diagnostics["spline_pooling"]


@pytest.mark.parametrize(
    "outputs,gaps,noise", [(1, (CurveGap(0.4, 0.6, 1, 0, 0),), 0), (2, (), 0), (2, (CurveGap(0.4, 0.6, 1, 0, 0),), 0.1)]
)
def test_existing_scalar_connected_and_noisy_paths_are_exact_identity(outputs, gaps, noise):
    x = torch.tensor([[0.0], [0.4], [0.6], [1.0]], dtype=torch.float64)
    native = StructureLearningResult(x, x.expand(-1, outputs), gaps, "budget_exhausted", {"noise_variance": noise}, [], (0, 1))
    assert refit_prediction(native) is native


def test_ridge_press_matches_independent_leave_row_refits_with_fixed_physical_penalty():
    x = torch.tensor([-1.0, -0.93, -0.45, -0.2, 0.03, 0.11, 0.65, 0.9, 1.0], dtype=torch.float64)
    design = torch.stack((torch.ones_like(x), x, x.square(), x.pow(3)), 1)
    y = torch.stack((torch.sin(2 * x), torch.cos(3 * x)), 1)
    mass = _mass(x)
    validation = torch.ones(len(x), dtype=torch.bool)
    validation[0] = validation[-1] = False
    penalty = x.new_tensor([0.0, 0.001, 0.01, 0.1])
    result = _score(design, y, mass, penalty, validation)
    assert result is not None
    error = 0.0
    for i in range(1, len(x) - 1):
        keep = torch.arange(len(x)) != i
        a, weights, responses = design[keep], mass[keep], y[keep]
        coefficients = torch.linalg.solve(
            a.T @ (weights[:, None] * a) + torch.diag(penalty), a.T @ (weights[:, None] * responses)
        )
        error += float((design[i] @ coefficients - y[i]).square().sum() * mass[i])
    assert result.score == pytest.approx(error / float(mass[validation].sum()), rel=1e-10, abs=1e-13)


def _native_for_selection(score):
    x = torch.cat((torch.linspace(-1, -0.03, 18), torch.linspace(0.03, 1, 18))).double()
    y = torch.stack((x.square(), torch.sin(x), x > 0), 1)
    gap = CurveGap(float(x[17]), float(x[18]), 1, 0, 0)
    models = [_Spline(x[g], 1, y[g], torch.zeros(3, dtype=torch.float64)) for g in (slice(0, 18), slice(18, None))]
    return _compile_result(
        StructureLearningResult(
            x[:, None],
            y,
            (gap,),
            "budget_exhausted",
            {"cost": 36, "trace": [{"cost": 36}], "capacity": [{"weighted_cv_mse": score}] * 2},
            models,
            (-1, 1),
        )
    )


def test_selected_pooling_keeps_observations_trace_mask_and_compiles_actual_bsf_without_queries(monkeypatch):
    from torch_bsf._adaptive_curve.queries import QuerySession

    def fail_query(*args, **kwargs):
        raise AssertionError("Prediction refitting cannot request teacher responses.")

    monkeypatch.setattr(QuerySession, "query", fail_query)
    native = _native_for_selection(1.0)
    inputs, values = native.inputs.clone(), native.values.clone()
    before = native._compiled
    updated = refit_prediction(native)
    assert updated._compiled is not None and updated._compiled is not before
    assert updated.diagnostics["spline_pooling"]["selected_predictor"] == "shared_c0_offsets"
    assert updated.gaps == native.gaps and updated.status == native.status
    assert updated.diagnostics["trace"] == native.diagnostics["trace"] and updated.diagnostics["cost"] == 36
    assert "spline_pooling" not in native.diagnostics
    torch.testing.assert_close(native.inputs, inputs, rtol=0, atol=0)
    torch.testing.assert_close(native.values, values, rtol=0, atol=0)
    points = torch.linspace(-1, 1, 197, dtype=torch.float64)[:, None]
    assert torch.equal(torch.isfinite(updated.predict(points)), torch.isfinite(native.predict(points)))
    assert updated._compiled.domain.betti_numbers()[0] == 2
    assert gap_contrasts(updated)[0]["predicted_observed_ratio"] > 0.9


@pytest.mark.parametrize("score", [0.0, None])
def test_better_or_unavailable_native_validation_retains_existing_actual_operator(score):
    native = _native_for_selection(score)
    updated = refit_prediction(native)
    assert updated._compiled is native._compiled
    assert updated.diagnostics["spline_pooling"]["selected_predictor"] == "native_chart_models"


@st.composite
def _restricted_curves(draw):
    degree = draw(st.integers(0, 5))
    pieces = draw(st.integers(1, 4))
    outputs = draw(st.integers(1, 3))
    count = (degree * pieces + 1) * outputs
    controls = draw(st.lists(st.floats(-2, 2, allow_nan=False, allow_infinity=False), min_size=count, max_size=count))
    left = draw(st.floats(-1, 0.5, allow_nan=False, allow_infinity=False))
    right = draw(st.floats(left + 0.1, 1, allow_nan=False, allow_infinity=False))
    source = _Spline(
        torch.linspace(-1, 1, pieces + 1, dtype=torch.float64),
        degree,
        torch.tensor(controls, dtype=torch.float64).reshape(-1, outputs),
        torch.arange(outputs, dtype=torch.float64),
    )
    return source, left, right


@given(_restricted_curves())
@settings(max_examples=30, deadline=None)
def test_exact_restriction_preserves_random_c0_charts_and_output_offset(case):
    source, left, right = case
    original_controls = source.controls.clone()
    restricted = restrict_spline(source, left, right)
    query = torch.linspace(left, right, 37, dtype=torch.float64)
    torch.testing.assert_close(restricted.predict(query), source.predict(query), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(source.controls, original_controls, rtol=0, atol=0)
    assert float(restricted.knots[0]) == left and float(restricted.knots[-1]) == right


def test_singleton_restriction_keeps_original_value_as_a_vertex():
    source = _Spline(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        1,
        torch.tensor([[1.0, 2.0], [3.0, 5.0], [-1.0, 4.0]], dtype=torch.float64),
        torch.tensor([0.25, -0.5], dtype=torch.float64),
    )
    restricted = restrict_spline(source, 0.25, 0.25)
    query = torch.tensor([0.25], dtype=torch.float64)
    assert restricted.degree == 0 and len(restricted.knots) == 1
    torch.testing.assert_close(restricted.predict(query), source.predict(query), rtol=0, atol=0)
