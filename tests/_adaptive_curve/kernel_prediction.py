"""Stable kernel PRESS, independent KKT references, and actual-BSF fallback."""

from types import SimpleNamespace

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from scipy.linalg import cho_factor, cho_solve

from torch_bsf._adaptive_curve.export import _compile_result
from torch_bsf._adaptive_curve.kernel import _fixed_fit, _kernel, _StableOperator, fit_kernel_predictor
from torch_bsf._adaptive_curve.kernel_prediction import refit_prediction
from torch_bsf._adaptive_curve.polynomial import CurveGap, StructureLearningResult, _Spline
from torch_bsf._adaptive_curve.queries import QuerySession
from torch_bsf._adaptive_curve.spectral import _mass


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _problem(degree=0):
    x = np.linspace(0, 1, 15)
    covariance = _kernel(x, x, "rbf", 0.3535533905932738)
    noise = 1e-10 / (len(x) * _mass(torch.from_numpy(x)).numpy())
    matrix = covariance + np.diag(noise)
    trend = np.column_stack((np.ones(len(x)), 2 * x - 1))[:, : degree + 1]
    return x, covariance, noise, matrix, trend, cho_factor(matrix, lower=True)


def _reference(matrix, trend, values, cross, prediction_trend):
    """Solve the represented GLS saddle system independently at80 digits."""
    p = trend.shape[1]
    saddle = np.block([[matrix, trend], [trend.T, np.zeros((p, p))]])
    output = np.empty((len(cross), values.shape[1]))
    with mp.workdps(80):
        system = mp.matrix(saddle.tolist())
        for column in range(values.shape[1]):
            solution = mp.lu_solve(system, mp.matrix([*values[:, column], *([0] * p)]))
            for i in range(len(cross)):
                output[i, column] = float(mp.fdot([*cross[i], *prediction_trend[i]], list(solution)))
    return output


@pytest.mark.parametrize("degree", [0, 1])
def test_high_condition_constant_linear_trends_match_independent_kkt_without_spurious_residual(degree):
    x, covariance, noise, matrix, trend, factor = _problem(degree)
    values = trend @ np.array([[1.3], [0.7]])[: degree + 1]
    fitted = _fixed_fit(covariance, values, trend, noise)
    assert fitted is not None and 1e10 < fitted.condition < 1e12
    assert _fixed_fit(covariance, values, trend, noise, max_condition=1e6) is None
    q = np.linspace(0, 1, 41)
    polynomial = np.column_stack((np.ones(len(q)), 2 * q - 1))[:, : degree + 1]
    operator = _StableOperator(
        2 * x - 1, "rbf", 0.3535533905932738, degree, fitted, np.zeros(1), 1.0, factor, values - trend @ fitted.beta
    )
    expected = _reference(matrix, trend, values, _kernel(q, x, "rbf", 0.3535533905932738), polynomial)
    gram_condition = np.linalg.cond(trend.T @ cho_solve(factor, trend))
    tolerance = float(256 * len(x) * np.finfo(float).eps * gram_condition * max(1.0, float(np.abs(values).max())))
    np.testing.assert_allclose(operator.predict(2 * q - 1), expected, rtol=0, atol=tolerance)


def test_deleted_row_scores_match_high_precision_joint_mean_refits():
    x, covariance, noise, matrix, trend, _ = _problem()
    values = (np.sin(8 * x) + 0.2 * x)[:, None]
    fitted = _fixed_fit(covariance, values, trend, noise)
    assert fitted is not None
    reference = []
    for i in (2, 7, 11):
        keep = np.arange(len(x)) != i
        predicted = _reference(
            matrix[np.ix_(keep, keep)], trend[keep], values[keep], covariance[i : i + 1, keep], trend[i : i + 1]
        )
        reference.append(float(values[i, 0] - predicted[0, 0]))
    expected = np.asarray(reference)
    tolerance = float(32 * len(x) * np.finfo(float).eps * fitted.condition * np.abs(expected).max())
    np.testing.assert_allclose(fitted.loo_residuals[[2, 7, 11], 0], expected, rtol=0, atol=tolerance)


def test_shared_linear_prediction_respects_output_rotation_at_high_condition():
    x, covariance, noise, matrix, trend, factor = _problem()
    y = np.column_stack((np.sin(8 * x), np.cos(5 * x), x * x, 0.3 * x))
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(717), dtype=torch.float64)).Q.numpy()
    q = np.linspace(0, 1, 41)
    predictions = []
    for values in (y, y @ rotation):
        fitted = _fixed_fit(covariance, values, trend, noise)
        assert fitted is not None
        operator = _StableOperator(
            2 * x - 1, "rbf", 0.3535533905932738, 0, fitted, np.zeros(4), 1.0, factor, values - trend @ fitted.beta
        )
        predictions.append(operator.predict(2 * q - 1))
    cross = _kernel(q, x, "rbf", 0.3535533905932738)
    projected = cho_solve(factor, trend)
    beta_map = np.linalg.solve(trend.T @ projected, projected.T)
    weights = cho_solve(factor, cross.T).T
    functional = weights + (np.ones((len(q), 1)) - weights @ trend) @ beta_map
    tolerance = float(
        256 * len(x) * y.shape[1] * np.finfo(float).eps * max(1.0, float(np.abs(functional).sum(1).max())) * np.abs(y).max()
    )
    np.testing.assert_allclose(predictions[1], predictions[0] @ rotation, rtol=0, atol=tolerance)
    expected = _reference(matrix, trend, y, cross, np.ones((len(q), 1)))
    # The rotation tolerance measures roundings in the shared functional;
    # factorization error against the represented KKT system is separate.
    # Independently reconstruct the cancellation-prone inverse expression
    # that this implementation replaces, as a numerical regression control.
    inverse = cho_solve(factor, np.eye(len(x)))
    naive_projected = inverse @ trend
    naive_beta = np.linalg.solve(trend.T @ naive_projected, naive_projected.T @ y)
    naive_alpha = inverse @ y - naive_projected @ naive_beta
    naive_prediction = cross @ naive_alpha + np.ones((len(q), 1)) @ naive_beta
    stable_error = float(np.abs(predictions[0] - expected).max())
    naive_error = float(np.abs(naive_prediction - expected).max())
    assert naive_error > 100 * max(stable_error, float(np.finfo(float).eps))


def test_selected_operator_compiles_actual_bsf_with_empirical_support_and_c0_checks():
    x = torch.linspace(0, 1, 19, dtype=torch.float64)
    y = torch.sin(7 * x)[:, None]
    model = fit_kernel_predictor(x, y)
    assert model is not None and model.diagnostics["selection_policy"] == "cv"
    chart = model.as_spline()
    assert model.as_spline() is chart
    compiled = _compile_result(StructureLearningResult(x[:, None], y, (), "budget_exhausted", {}, [chart], (0, 1)))
    assert compiled._compiled is not None and compiled._compiled.domain.betti_numbers()[0] == 1
    query = torch.unique(torch.cat((torch.linspace(0, 1, 101, dtype=torch.float64), chart.knots)))[:, None]
    torch.testing.assert_close(
        compiled.predict(query), model.predict(query), rtol=0, atol=model.diagnostics["conversion_checked_tolerance"] * 2
    )
    assert model.diagnostics["conversion_is_certified"] is False and model.diagnostics["conversion_oracle_rows"] == 0


def _native(session):
    inputs, values, _ = session.observations()
    gap = CurveGap(float(inputs[15, 0]), float(inputs[16, 0]), 1.0, 0.01, 0.0)
    models = [
        _Spline(inputs[g, 0], 1, values[g], torch.zeros(1, dtype=torch.float64)) for g in (slice(0, 16), slice(16, None))
    ]
    return _compile_result(
        StructureLearningResult(
            inputs,
            values,
            (gap,),
            "budget_exhausted",
            {"cost": session.cost, "trace": [{"cost": session.cost}], "capacity": [{"weighted_cv_mse": 1.0}] * 2},
            models,
            (-1, 1),
        )
    )


def test_scalar_refit_preserves_gap_masks_charged_budget_observations_rng_and_actual_bsf(monkeypatch):
    session = QuerySession(lambda x: torch.sin(7 * x) + 2 * (x > 0), [[-1, 1]], budget=32)
    session.query(torch.cat((torch.linspace(-1, -0.03, 16), torch.linspace(0.03, 1, 16))).double()[:, None])
    native = _native(session)
    before, state = session.cost, torch.get_rng_state().clone()
    inputs, values = native.inputs.clone(), native.values.clone()

    def forbidden(*args, **options):
        raise AssertionError("Scalar predictor refitting must not query an oracle.")

    monkeypatch.setattr(QuerySession, "query", forbidden)
    updated = refit_prediction(native)
    assert session.cost == before == 32 and torch.equal(state, torch.get_rng_state())
    assert all(row["selected"] == "stationary_kernel" for row in updated.diagnostics["kernel_prediction"])
    assert updated._compiled is not None and updated._compiled.domain.betti_numbers()[0] == 2
    query = torch.linspace(-1, 1, 257, dtype=torch.float64)[:, None]
    assert torch.equal(torch.isfinite(native.predict(query)), torch.isfinite(updated.predict(query)))
    torch.testing.assert_close(native.inputs, inputs, rtol=0, atol=0)
    torch.testing.assert_close(native.values, values, rtol=0, atol=0)
    assert updated.gaps == native.gaps and updated.diagnostics["trace"] == native.diagnostics["trace"]
    assert "kernel_prediction" not in native.diagnostics


def test_failed_operator_conversion_keeps_available_native_bsf(monkeypatch):
    from torch_bsf._adaptive_curve import kernel_prediction

    session = QuerySession(lambda x: x.square(), [[-1, 1]], budget=32)
    session.query(torch.cat((torch.linspace(-1, -0.03, 16), torch.linspace(0.03, 1, 16))).double()[:, None])
    native = _native(session)

    def failure():
        raise ArithmeticError("empirical conversion resource limit")

    monkeypatch.setattr(
        kernel_prediction,
        "fit_kernel_predictor",
        lambda *args, **options: SimpleNamespace(diagnostics={"weighted_cv_mse": 0.0}, as_spline=failure),
    )
    actual = refit_prediction(native)
    assert actual._compiled is native._compiled and actual.gaps == native.gaps
    assert all("resource limit" in row["conversion_rejection"] for row in actual.diagnostics["kernel_prediction"])


@pytest.mark.parametrize("outputs,noise", [(2, 0.0), (1, 0.1)])
def test_vector_and_noisy_results_retain_exact_operator_identity(outputs, noise):
    x = torch.linspace(0, 1, 8, dtype=torch.float64)[:, None]
    original = StructureLearningResult(x, x.expand(-1, outputs), (), "budget_exhausted", {"noise_variance": noise}, [], (0, 1))
    assert refit_prediction(original) is original


def test_small_chart_missing_capacity_and_unsupported_kernel_data_keep_safe_fallback():
    x = torch.linspace(0, 1, 5, dtype=torch.float64)
    chart = _Spline(x, 1, x[:, None], torch.zeros(1, dtype=torch.float64))
    native = _compile_result(
        StructureLearningResult(
            x[:, None], x[:, None], (), "budget_exhausted", {"capacity": [{"weighted_cv_mse": 1.0}]}, [chart], (0, 1)
        )
    )
    assert refit_prediction(native)._compiled is native._compiled
    native.diagnostics["capacity"] = []
    assert refit_prediction(native) is native
    assert fit_kernel_predictor(x, x[:, None]) is None
    x = torch.linspace(0, 1, 8, dtype=torch.float64)
    assert fit_kernel_predictor(x, torch.zeros((8, 1), dtype=torch.float64)) is None
    assert fit_kernel_predictor(x, x[:, None].expand(-1, 2)) is None
    assert fit_kernel_predictor(x, x[:, None], noise=0.1) is None
    with pytest.raises(ValueError, match="noise"):
        fit_kernel_predictor(x, x[:, None], noise=True)
