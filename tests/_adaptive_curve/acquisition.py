"""Acquisition bias is local, volume-weighted and observes training data only."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from torch_bsf._adaptive_curve.evidence import GapEvidence

pytest.importorskip("scipy")
from torch_bsf._adaptive_curve import acquisition as cv  # noqa: E402


def neutral(x):
    return tuple(
        GapEvidence(float(a), float(b), 0, 0, 0, 0.01, 0, "continuous_like", 0, None, None, None, float((a + b) / 2), "test")
        for a, b in zip(x[:-1], x[1:], strict=True)
    )


def test_exact_cubic_has_no_empirical_interpolation_bias():
    x = torch.linspace(-1, 1, 11, dtype=torch.float64)
    y = torch.stack((x**3 + x, x.square() - 2, 0.2 * x + 1), 1)
    score = cv.scores(x, y, neutral(x), 0.01)
    assert score.dtype == torch.float64 and score.shape == (10,)
    assert (score == 0).all()


def test_each_loo_target_response_is_excluded_from_its_interpolation(monkeypatch):
    original, training_rows = cv.CubicSpline, []

    def checked(x, y, **kwargs):
        training_rows.append(np.asarray(y).copy())
        return original(x, y, **kwargs)

    monkeypatch.setattr(cv, "CubicSpline", checked)
    x = np.linspace(-1, 1, 7)
    y = np.stack((np.arange(7), np.arange(7) ** 2), 1).astype(float)
    cv._residuals(x, y, shape_preserving=False)
    assert len(training_rows) == 5
    for heldout, rows in enumerate(training_rows, 1):
        assert heldout not in rows[:, 0] and len(rows) < 7


def test_local_peak_gets_more_bias_mass_than_its_flat_tails():
    x = torch.linspace(-1, 1, 17, dtype=torch.float64)
    y = torch.exp(-((x - 0.1) / 0.2).square())[:, None]
    score = cv.scores(x, y, neutral(x), 0.01)
    i = int(score.argmax())
    assert abs(float((x[i] + x[i + 1]) / 2) - 0.1) < 0.4
    assert float(score.max()) > 100 * float(score[0])


def test_same_local_error_is_weighted_by_input_volume_not_sample_density():
    left = torch.linspace(0, 0.1, 7, dtype=torch.float64)
    right = torch.linspace(0.5, 1, 7, dtype=torch.float64)
    x = torch.cat((left, right))
    t = torch.linspace(0, 1, 7, dtype=torch.float64)
    y = torch.cat((t**4, t**4))[:, None]
    evidence = list(neutral(x))
    evidence[6] = replace(evidence[6], status="jump_candidate", excess=1)
    score = cv.scores(x, y, evidence, 0.5)
    torch.testing.assert_close(score[7:], score[:6] * 5, rtol=1e-8, atol=1e-14)
    assert score[6] == 0


def test_confirmed_gap_stops_neighbor_bias_leakage_and_unresolved_jump_keeps_own_score():
    x = torch.linspace(-1, 1, 14, dtype=torch.float64)
    left = x[:7] ** 3
    right = torch.tensor([3, 8, 2, -4, 9, 1, 20], dtype=torch.float64)
    y = torch.cat((left, right))[:, None]
    evidence = list(neutral(x))
    evidence[6] = replace(evidence[6], status="jump_candidate", excess=5)
    score = cv.scores(x, y, evidence, 0.2)
    assert (score[:6] == 0).all() and score[6] == 0
    assert score[7:].max() > 0
    unlocalized = list(evidence)
    unlocalized[6] = replace(evidence[6], status="unresolved")
    another = cv.scores(x, y, unlocalized, 0.01)
    assert another[6] > 0 and (another[:6] == 0).all()


@pytest.mark.parametrize("factor", [0.001, 1.0, 1000.0])
def test_output_scale_and_orthogonal_rotation_do_not_change_vector_ranking(factor):
    x = torch.linspace(-1, 1, 15, dtype=torch.float64)
    y = torch.stack((torch.sin(7 * x), torch.cos(3 * x), x**4), 1)
    q = torch.linalg.qr(torch.tensor([[1.0, 2, 3], [-3, 1, 2], [2, -1, 1]], dtype=torch.float64)).Q
    reference = cv.scores(x, y, neutral(x), 0.01)
    transformed = cv.scores(x, factor * y @ q, neutral(x), 0.01)
    torch.testing.assert_close(reference, transformed, rtol=1e-8, atol=1e-14)


def test_input_units_do_not_change_dimensionless_risk_and_inputs_are_unchanged():
    x = torch.linspace(-1, 1, 15, dtype=torch.float64)
    y = torch.stack((torch.sin(7 * x), torch.cos(3 * x)), 1)
    before = y.clone()
    a = cv.scores(x, y, neutral(x), 0.01)
    z = 10 * x + 50
    b = cv.scores(z[:, None], y, neutral(z), 0.1)
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-14)
    torch.testing.assert_close(before, y, rtol=0, atol=0)


def test_empty_and_small_designs_have_no_invented_curvature_evidence():
    for n in (0, 1, 2):
        x = torch.arange(n, dtype=torch.float64)
        score = cv.scores(x, torch.ones((n, 2), dtype=torch.float64), neutral(x), 0.1)
        assert score.shape == (max(0, n - 1),) and (score == 0).all()


@pytest.mark.parametrize("target", [0, -1, float("nan")])
def test_invalid_target_and_misaligned_evidence_are_rejected(target):
    x = torch.linspace(-1, 1, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="target"):
        cv.scores(x, x[:, None], neutral(x), target)
    with pytest.raises(ValueError, match="align"):
        cv.scores(x, x[:, None], neutral(x)[:-1], 0.1)
