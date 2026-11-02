"""Finite-scale continuation evidence, observable history and charged refinement."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torch_bsf._adaptive_curve.evidence import MultiscaleGapDetector
from torch_bsf._adaptive_curve.queries import QuerySession


def observed(detector, session):
    x, y, counts = session.observations()
    return detector.update(x, y, counts=counts, noise_variance=session.noise_variance() or 0.0)


def test_charged_nested_queries_are_required_and_repeated_updates_do_not_count():
    session = QuerySession(lambda x: (x > 0.123).double(), [[-1, 1]], budget=12)
    session.query([[-1], [-0.5], [0], [0.5], [1]])
    detector = MultiscaleGapDetector()
    first = max(observed(detector, session), key=lambda row: row.excess)
    assert first.status == "unresolved" and first.nested_scales == 0
    assert max(observed(detector, session), key=lambda row: row.excess).nested_scales == 0
    for level in (1, 2):
        row = max(observed(detector, session), key=lambda row: row.excess)
        session.query([[row.probe]])
        row = max(observed(detector, session), key=lambda row: row.excess)
        assert row.nested_scales == level
    assert row.status == "jump_candidate" and row.left < 0.123 < row.right
    assert row.contraction == pytest.approx(1) and row.contraction_exponent == pytest.approx(0)
    assert row.direction_agreement == pytest.approx(1)
    assert session.cost == 7 and session.proposed_rows == 7


def test_resolved_steep_continuous_transition_revokes_a_finite_scale_candidate():
    session = QuerySession(lambda x: torch.tanh((x - 0.123) / 0.001), [[-1, 1]], budget=19)
    session.query([[-1], [-0.5], [0], [0.5], [1]])
    detector = MultiscaleGapDetector()
    accepted = False
    for _ in range(14):
        rows = observed(detector, session)
        accepted |= any(row.status == "jump_candidate" for row in rows)
        # Continue auditing even when continuation begins to explain the change.
        row = max(rows, key=lambda item: item.raw_change)
        session.query([[row.probe]])
    assert accepted
    assert not any(row.status == "jump_candidate" for row in observed(detector, session))
    assert session.cost == len(session.observations()[0]) == 19


@pytest.mark.parametrize("center", [-0.31, 0.0, 0.47])
@pytest.mark.parametrize("amplitude", [0.001, 1.0, 1000.0])
def test_continuous_cusp_peak_keeps_connected_continuation(center, amplitude):
    offsets = torch.tensor([-0.1, -0.04, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.04, 0.1], dtype=torch.float64)
    x = offsets + center
    y = amplitude / (1 + (offsets / 0.01).abs()).pow(4)
    rows = MultiscaleGapDetector(min_refinements=0).update(x, torch.stack((y, -0.7 * y), 1))
    assert not any(row.status == "jump_candidate" for row in rows)


def test_one_sided_curvature_prevents_parent_peak_cancellation_from_looking_persistent():
    # A coarser bracket crosses a peak; its net endpoint contrast cancels.
    # These are observations from an ordinary continuous cusp, with no boundary labels.
    x = torch.tensor([0.203125, 0.21875, 0.22265625, 0.2265625, 0.234375, 0.2421875], dtype=torch.float64)
    y = torch.tensor([0.0281234, 0.2153281, 0.4697719, 1.2689499, 1.0551432, 0.6645380], dtype=torch.float64)[:, None]
    rows = MultiscaleGapDetector(min_refinements=0).update(x, y)
    row = rows[2]
    first_order = (y[2] - y[1]).norm().item()
    assert row.raw_change > 3 * first_order + row.threshold
    assert row.local_continuation_bound > first_order
    assert row.status == "continuous_like"


def test_signed_curvature_projection_must_not_discard_oscillatory_continuation():
    # Removing curvature solely because its projection opposes the observed
    # chord is unsafe without monotonicity assumptions between sample sites.
    rows = MultiscaleGapDetector(min_refinements=0).update([-2, -1, 0, 1, 2, 3], [[-10], [-0.5], [0], [4], [4.5], [14]])
    assert rows[2].local_continuation_bound > rows[2].raw_change
    assert rows[2].status == "continuous_like"


@pytest.mark.parametrize("factor", [0.001, 1.0, 1000.0])
def test_vector_jumps_and_opposing_jumps_survive_scaling_and_rotation(factor):
    x = torch.tensor([-1, -0.5, 0, 0.05, 0.09, 0.11, 0.15, 0.17, 0.19, 0.21, 0.25, 0.3, 0.5, 1], dtype=torch.float64)
    level = ((x > 0.1).double() - (x > 0.2).double()) * factor
    y = torch.stack((level, -2 * level, 0.5 * level), 1)
    rotation = torch.linalg.qr(torch.tensor([[1.0, 2, 3], [-3, 1, 2], [2, -1, 1]], dtype=torch.float64)).Q
    for values in (y, y @ rotation):
        rows = MultiscaleGapDetector(min_refinements=0).update(x, values)
        candidates = [row for row in rows if row.status == "jump_candidate"]
        assert len(candidates) == 2
        assert candidates[0].left < 0.1 < candidates[0].right
        assert candidates[1].left < 0.2 < candidates[1].right


def test_noise_floor_and_replicate_counts_are_not_a_confidence_certificate():
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    y = (x > 0.1).double()[:, None]
    detector = MultiscaleGapDetector(min_refinements=0)
    rows = detector.update(x, y, counts=torch.ones(5), noise_variance=0.1)
    assert not any(row.status == "jump_candidate" for row in rows)
    assert any(row.reason == "noise_limited" for row in rows)
    rows = detector.update(x, y, counts=torch.full((5,), 1000), noise_variance=0.1)
    assert sum(row.status == "jump_candidate" for row in rows) == 1


def test_mean_updates_recompute_ancestral_contrasts_and_streams_do_not_mix():
    detector = MultiscaleGapDetector()
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    y = (x > 0.123).double()[:, None]
    detector.update(x, y)
    for point in (0.25, 0.125):
        x = torch.cat((x, x.new_tensor([point]))).sort().values
        y = (x > 0.123).double()[:, None]
        detector.update(x, y)
    rows = detector.update(x, y)
    assert any(row.status == "jump_candidate" for row in rows)
    # Increased counts permit updated means; every ancestor uses the new values.
    assert not any(row.status == "jump_candidate" for row in detector.update(x, x[:, None], counts=torch.full((7,), 2)))
    with pytest.raises(ValueError, match="without a count increase"):
        detector.update(x, y, counts=torch.full((7,), 2))
    with pytest.raises(ValueError, match="append-only"):
        detector.update(x[:-1], x[:-1, None], counts=torch.full((6,), 2))
    detector.reset()
    assert detector.update(torch.tensor([0.0]), torch.tensor([[1.0]])) == ()


def test_hidden_opposing_pulse_is_honestly_undetectable_without_an_interior_sample():
    x = torch.linspace(-1, 1, 9, dtype=torch.float64)
    y = ((x > 0.1) & (x < 0.11)).double()[:, None]
    assert not any(row.status == "jump_candidate" for row in MultiscaleGapDetector(min_refinements=0).update(x, y))


def test_adjacent_float_boundary_has_no_fabricated_probe():
    a, b = 1.0, math.nextafter(1.0, math.inf)
    rows = MultiscaleGapDetector(min_refinements=0).update([a, b], [[0], [1]])
    assert rows[0].probe is None and rows[0].status == "unresolved"


@given(
    st.lists(st.integers(-100, 100), min_size=5, max_size=18, unique=True), st.sampled_from([1e-9, 0.001, 1.0, 1000.0, 1e9])
)
@settings(max_examples=40, deadline=None)
def test_irregular_affine_vector_data_never_creates_a_cut(integer_points, scale):
    x = torch.tensor(sorted(integer_points), dtype=torch.float64) / 16
    y = torch.stack((2 * x + 3, -x + 0.1, 0.5 * x - 1), 1) * scale
    rows = MultiscaleGapDetector(min_refinements=0).update(x, y)
    assert len(rows) == len(x) - 1 and not any(row.status == "jump_candidate" for row in rows)


@pytest.mark.parametrize(
    "x,y,counts",
    [
        ([0, 0], [[0], [1]], None),
        ([1, 0], [[0], [1]], None),
        ([[0, 1]], [[0]], None),
        ([0, 1], [0, 1], None),
        ([0, 1], [[0], [1]], [1, 0]),
        ([0, 1], [[0], [1]], [1, 1.5]),
    ],
)
def test_invalid_observation_protocol_is_rejected(x, y, counts):
    with pytest.raises(ValueError):
        MultiscaleGapDetector().update(x, y, counts=counts)
