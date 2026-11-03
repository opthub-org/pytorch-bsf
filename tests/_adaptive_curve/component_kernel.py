"""Independent conditional PRESS, support conversion and retained actual BSF."""

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

import mpmath as mp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from torch_bsf._adaptive_curve.component_kernel import (  # noqa: E402
    _contrasts,
    _covariance,
    _SmoothPosterior,
    fit_component_kernel,
)
from torch_bsf._adaptive_curve.kernel import _fixed_fit  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveGap  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def problem():
    x = torch.cat((torch.linspace(-1, -0.01, 12), torch.linspace(0.01, 1, 12))).double()
    y = torch.stack((torch.sin(2 * x), x.square(), torch.cos(x)), 1)
    y[12:] += torch.tensor([0.7, -0.3, 0.1], dtype=torch.float64)
    gap = CurveGap(float(x[11]), float(x[12]), 0.8, 0.02, 0)
    return x, y, gap


@given(eta=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
@settings(deadline=None)
def test_component_kernel_is_psd_and_is_unchanged_by_chart_renaming(eta):
    unit = np.linspace(0, 1, 13)
    labels = np.array([0] * 4 + [1] * 5 + [2] * 4, dtype=np.int64)
    covariance = _covariance(unit, unit, labels, labels, "matern32", 0.3, eta)
    renamed = np.array([17, -4, 8], dtype=np.int64)[labels]
    other = _covariance(unit, unit, renamed, renamed, "matern32", 0.3, eta)
    np.testing.assert_array_equal(covariance, other)
    tolerance = 128 * len(unit) * np.finfo(float).eps * max(1.0, float(np.linalg.norm(covariance, 2)))
    assert np.linalg.eigvalsh(covariance)[0] >= -tolerance


def test_conditional_press_matches_independent_deleted_row_augmented_systems():
    unit = np.linspace(0, 1, 14)
    labels = np.array([0] * 7 + [1] * 7, dtype=np.int64)
    covariance = _covariance(unit, unit, labels, labels, "matern52", 0.3, 0.2)
    trend = np.column_stack((np.ones(len(unit)), 2 * unit - 1))
    values = np.column_stack((np.sin(4 * unit), unit**2 + 0.8 * labels))
    diagonal = np.linspace(0.003, 0.009, len(unit))
    fitted = _fixed_fit(covariance, values, trend, diagonal)
    assert fitted is not None
    expected = []
    for i in range(len(unit)):
        use = np.arange(len(unit)) != i
        matrix = covariance[np.ix_(use, use)] + np.diag(diagonal[use])
        design = trend[use]
        augmented = np.block([[matrix, design], [design.T, np.zeros((2, 2))]])
        coefficients = np.linalg.solve(augmented, np.vstack((values[use], np.zeros((2, 2)))))
        prediction = np.r_[covariance[i, use], trend[i]] @ coefficients
        expected.append(values[i] - prediction)
    tolerance = float(np.finfo(float).eps * fitted.condition * 512 * np.linalg.norm(values))
    np.testing.assert_allclose(fitted.loo_residuals, np.asarray(expected), rtol=0, atol=tolerance)


def test_shared_smooth_plus_stable_intercepts_equals_full_covariance_kkt_prediction():
    unit = np.linspace(0, 1, 18)
    labels = np.array([0] * 9 + [1] * 9, dtype=np.int64)
    eta = 0.3
    covariance = _covariance(unit, unit, labels, labels, "rbf", 0.2, eta)
    diagonal = np.full(len(unit), 1e-6)
    trend = np.column_stack((np.ones(len(unit)), 2 * unit - 1))
    values = np.column_stack((np.sin(3 * unit) + labels, unit**2 - 0.3 * labels))
    fitted = _fixed_fit(covariance, values, trend, diagonal)
    assert fitted is not None
    factor = cho_factor(covariance + np.diag(diagonal), lower=True)
    residual = values - trend @ fitted.beta
    indicators = (labels[:, None] == np.arange(2)[None, :]).astype(float)
    offsets = cho_solve(factor, eta * indicators).T @ residual
    smooth = _SmoothPosterior(2 * unit - 1, unit, "rbf", 0.2, 1, factor, residual, fitted.beta, np.zeros(2), 1)
    at = np.linspace(0, 1, 37)
    assigned = (at > 0.5).astype(np.int64)
    actual = smooth.predict(2 * at - 1) + offsets[assigned]
    # An independently solved saddle system, rather than recombining alpha
    # from the implementation, checks the full covariance/trend semantics.
    matrix = covariance + np.diag(diagonal)
    augmented = np.block([[matrix, trend], [trend.T, np.zeros((2, 2))]])
    solution = np.linalg.solve(augmented, np.vstack((values, np.zeros((2, 2)))))
    cross = _covariance(at, unit, assigned, labels, "rbf", 0.2, eta)
    expected = np.column_stack((cross, np.ones(len(at)), 2 * at - 1)) @ solution
    tolerance = float(np.finfo(float).eps * fitted.condition * 128 * np.linalg.norm(values))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    with mp.workdps(80):
        reference = np.empty_like(actual)
        high_matrix = mp.matrix(augmented.tolist())
        evaluation = np.column_stack((cross, np.ones(len(at)), 2 * at - 1))
        for column in range(values.shape[1]):
            solution80 = mp.lu_solve(high_matrix, mp.matrix([*values[:, column], 0, 0]))
            for i, row in enumerate(evaluation):
                reference[i, column] = float(mp.fdot(row.tolist(), list(solution80)))
    np.testing.assert_allclose(actual, reference, rtol=0, atol=tolerance)


def test_gap_consistency_rejects_erasure_reversal_and_large_threshold():
    groups = [torch.tensor([0, 1]), torch.tensor([2, 3])]
    values = np.array([[0.0, 0], [0.0, 0], [1.0, 0], [1.0, 0]])
    assert _contrasts(values * 0.5, values, groups, [0.1])[0]["admissible"]
    for prediction in (np.zeros_like(values), -values, values * 0.01):
        assert not _contrasts(prediction, values, groups, [0.1])[0]["admissible"]
    x, y, gap = problem()
    impossible = CurveGap(gap.left, gap.right, gap.jump_estimate, 1e10, 0)
    assert fit_component_kernel(x, y, (impossible,)) is None


def test_connected_same_family_control_and_small_charts_are_explicit():
    x, y, gap = problem()
    connected = fit_component_kernel(x, y, ())
    assert connected is not None and len(connected.groups) == 1
    assert torch.isfinite(connected.predict(torch.tensor([[0.0]], dtype=torch.float64))).all()
    assert fit_component_kernel(x[:5], y[:5], ()) is None
    assert fit_component_kernel(x, y, (CurveGap(float(x[1]), float(x[2]), 1, 0, 0),)) is None
    assert fit_component_kernel(x, y, (gap,), noise=0.1) is None
    with pytest.raises(ValueError, match="one coordinate"):
        connected.predict(torch.zeros((3, 2), dtype=torch.float64))
