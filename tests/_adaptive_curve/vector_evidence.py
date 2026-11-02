"""Cross-fitted directions must not turn generic vector roughness into gaps."""

from dataclasses import asdict

import pytest
import torch

from torch_bsf._adaptive_curve.evidence import MultiscaleGapDetector
from torch_bsf._adaptive_curve.queries import QuerySession
from torch_bsf._adaptive_curve.vector_evidence import CrossFittedVectorGapDetector, _outer_models


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rough(x, *, jump):
    frequency = torch.arange(7, 38, dtype=torch.float64)
    nuisance = 0.12 * torch.sin(x[:, None] * frequency + frequency.square())
    signal = 0.2 * x + (x > 0.137).double() * jump
    return torch.cat((signal[:, None], nuisance), 1)


def test_outer_direction_and_selection_never_see_either_candidate_endpoint():
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)
    y = rough(x, jump=1)
    initial = _outer_models(x, y, 17)
    modified = y.clone()
    modified[17:19] += 1000 * torch.randn((2, 32), generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    after = _outer_models(x, modified, 17)
    assert len(initial) == len(after) == 3
    for first, second in zip(initial, after, strict=True):
        assert first.degree == second.degree and first.score == second.score
        torch.testing.assert_close(first.direction, second.direction, rtol=0, atol=0)
        torch.testing.assert_close(first.mean_direction, second.mean_direction, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["affine", "cusp", "jump", "oscillation"])
def test_scalar_path_matches_every_frozen_field_exactly_over_actual_insertions(kind):
    def oracle(points):
        x = points[:, 0]
        y = (
            x
            if kind == "affine"
            else x.abs()
            if kind == "cusp"
            else (x > 0.137).double()
            if kind == "jump"
            else torch.sin(17 * x)
        )
        return y[:, None]

    session = QuerySession(oracle, [[-1, 1]], budget=30)
    session.query(torch.linspace(-1, 1, 16, dtype=torch.float64)[:, None])
    base, new = MultiscaleGapDetector(), CrossFittedVectorGapDetector()
    for _ in range(12):
        x, y, counts = session.observations()
        a, b = base.update(x, y, counts=counts), new.update(x, y, counts=counts)
        assert [asdict(row) for row in a] == [asdict(row) for row in b]
        i = int((x[1:, 0] - x[:-1, 0]).argmax())
        session.query((x[i : i + 1] + x[i + 1 : i + 2]) / 2)
    assert new.last_projection_diagnostics == []


@pytest.mark.parametrize("seed", [1, 19, 127])
def test_continuous_high_dimensional_orthogonal_roughness_has_no_provisional_cuts(seed):
    x = torch.linspace(-1, 1, 65, dtype=torch.float64)
    frequency = torch.arange(7, 38, dtype=torch.float64)
    phase = torch.rand(31, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 6
    y = torch.cat((x[:, None], x.square()[:, None], 0.12 * torch.sin(x[:, None] * frequency + phase)), 1)
    rows = CrossFittedVectorGapDetector(min_refinements=0).update(x, y)
    assert not any(row.status == "jump_candidate" for row in rows)


@pytest.mark.parametrize("seed", [1, 19, 127])
def test_iid_high_dimensional_values_do_not_gain_self_selected_directions(seed):
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)
    y = torch.randn((33, 64), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    rows = CrossFittedVectorGapDetector(min_refinements=0).update(x, y)
    assert not any(row.status == "jump_candidate" for row in rows)


def test_coarse_vector_jump_requires_nested_evidence_from_additional_charged_samples():
    session = QuerySession(lambda x: rough(x[:, 0], jump=1), [[-1, 1]], budget=69)
    session.query(torch.linspace(-1, 1, 65, dtype=torch.float64)[:, None])
    detector = CrossFittedVectorGapDetector()
    x, y, counts = session.observations()
    rows = detector.update(x, y, counts=counts)
    assert not any(row.status == "jump_candidate" for row in rows)
    for _ in range(4):
        target = max(rows, key=lambda row: row.raw_change)
        session.query([[target.probe]])
        x, y, counts = session.observations()
        rows = detector.update(x, y, counts=counts)
    assert any(row.status == "jump_candidate" and row.left < 0.137 < row.right for row in rows)
    assert session.cost == len(x) == 69


def test_new_evidence_still_requires_charged_nested_refinements_and_resets():
    session = QuerySession(lambda x: rough(x[:, 0], jump=1), [[-1, 1]], budget=32)
    session.query(torch.linspace(-1, 1, 16, dtype=torch.float64)[:, None])
    detector = CrossFittedVectorGapDetector()
    for _ in range(8):
        x, y, counts = session.observations()
        rows = detector.update(x, y, counts=counts)
        target = max(rows, key=lambda row: row.excess)
        if target.probe is None:
            break
        session.query([[target.probe]])
    x, y, counts = session.observations()
    rows = detector.update(x, y, counts=counts)
    assert any(row.status == "jump_candidate" and row.left <= 0.137 <= row.right for row in rows)
    assert session.cost == len(x) == 24
    detector.reset()
    assert detector.last_projection_diagnostics == []


def test_clean_scalar_lift_retains_original_continuation_decisions():
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)
    signal = 1 / (1 + (x / 0.05).abs()).pow(4)
    y = signal[:, None] * torch.arange(1, 9, dtype=torch.float64)[None]
    base = MultiscaleGapDetector(min_refinements=0).update(x, y)
    new = CrossFittedVectorGapDetector(min_refinements=0).update(x, y)
    assert [asdict(row) for row in base] == [asdict(row) for row in new]


def test_vector_rotation_and_response_units_preserve_proposals():
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)
    y = rough(x, jump=1)
    q = torch.linalg.qr(torch.randn((32, 32), generator=torch.Generator().manual_seed(3), dtype=torch.float64)).Q
    first = CrossFittedVectorGapDetector(min_refinements=0).update(x, y)
    second = CrossFittedVectorGapDetector(min_refinements=0).update(x, 0.001 * y @ q)
    assert [row.status for row in first] == [row.status for row in second]
    torch.testing.assert_close(
        torch.tensor([row.excess for row in first]) * 0.001, torch.tensor([row.excess for row in second]), rtol=1e-5, atol=1e-8
    )


def test_outer_cache_matches_uncached_fits_and_invalidates_only_local_context():
    x = torch.linspace(-1, 1, 65, dtype=torch.float64)
    cached, uncached = CrossFittedVectorGapDetector(), CrossFittedVectorGapDetector(cache_outer=False)
    before = 0
    for step in range(4):
        y = rough(x, jump=1)
        first, second = cached.update(x, y), uncached.update(x, y)
        assert [asdict(row) for row in first] == [asdict(row) for row in second]
        previous = cached.outer_model_fits
        x = torch.sort(torch.cat((x, x.new_tensor([0.14 + step * 0.0001])))).values
        if step:
            assert cached.outer_model_fits - before <= 18
        before = previous
    assert cached.outer_model_cache_hits > 0
    assert len(cached._model_cache) == len(x) - 2
    cached.reset()
    assert cached.outer_model_cache_hits == cached.outer_model_fits == 0
    assert not cached._model_cache


def test_invalid_cache_flag_is_rejected():
    with pytest.raises(ValueError, match="cache_outer"):
        CrossFittedVectorGapDetector(cache_outer=1)  # type: ignore[arg-type]


def test_replicate_mean_and_noise_changes_recompute_evidence_without_stale_outer_fits():
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)
    y = rough(x, jump=1)
    counts = torch.ones(33, dtype=torch.long)
    cached, uncached = CrossFittedVectorGapDetector(), CrossFittedVectorGapDetector(cache_outer=False)
    cached.update(x, y, counts=counts)
    uncached.update(x, y, counts=counts)
    y[15] += 0.15
    counts[15] += 1
    first = cached.update(x, y, counts=counts, noise_variance=0.02)
    second = uncached.update(x, y, counts=counts, noise_variance=0.02)
    assert [asdict(row) for row in first] == [asdict(row) for row in second]
    fits = cached.outer_model_fits
    first = cached.update(x, y, counts=counts, noise_variance=2)
    second = uncached.update(x, y, counts=counts, noise_variance=2)
    assert cached.outer_model_fits == fits
    assert [asdict(row) for row in first] == [asdict(row) for row in second]
    assert all(
        row.noise_floor >= 3 * 2**0.5 * (1 / int(counts[i]) + 1 / int(counts[i + 1])) ** 0.5 for i, row in enumerate(first)
    )
