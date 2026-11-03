"""Cross-fitted directions must not turn generic vector roughness into gaps."""

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
import torch

from torch_bsf._adaptive_curve.evidence import MultiscaleGapDetector
from torch_bsf._adaptive_curve.queries import QuerySession
from torch_bsf._adaptive_curve.vector_evidence import (
    CrossFittedVectorGapDetector,
    _Outer,
    _outer_models,
    _projected_statistics,
    _Side,
)


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


def _projected_fixture(degree):
    first = torch.tensor([[0.5, 0.25], [0.75, -0.25], [0.125, 0.5]], dtype=torch.float64)[: degree + 1]
    second = torch.tensor([[1.0, 0.5], [0.25, 0.75], [-0.25, 0.125]], dtype=torch.float64)[: degree + 1]
    residual = torch.tensor([[0.125, -0.25], [-0.25, 0.125], [0.5, 0.125]], dtype=torch.float64)
    inverse = torch.tensor([[0.5, 0.25, -0.125], [0.25, -0.5, 0.125], [0.125, 0.5, -0.25]], dtype=torch.float64)[: degree + 1]
    return _Outer(
        _Side(first, residual, inverse, 0.125, 2),
        _Side(second, -residual, inverse, 0.125, 2),
        1.0,
        degree,
        3,
        3,
        torch.ones(2, dtype=torch.float64),
    )


def _projected_closed_form_and_roundoff(degree):
    # An independent rational oracle: u=(3/5,4/5) gives residual projections
    # (-1/8,-1/20,2/5), energy P=57/320, and raw energy T=27/64 per side.
    # Both Taylor maxima are 3/16. For t=+/-1/8 the forecast leverages are
    # (1381,1349)/4096 or (90453,87893)/262144, respectively.
    f = Fraction
    p, total = f(57, 320), f(27, 64)
    leverage_sum = f(1365, 2048) if degree == 1 else f(89173, 131072)
    variance = p * (2 + leverage_sum) / 2
    unit = f(1, 2**53)
    gamma2, gamma3 = 2 * unit / (1 - 2 * unit), 3 * unit / (1 - 3 * unit)

    # Including conversion of the rational direction, a two-term dot has
    # absolute error <= gamma3 * sum(abs(a_i*u_i)). The residual dot condition
    # numbers are 11/5, 5, and 1; using their absolute errors avoids treating
    # cancellation as a well-conditioned relative calculation.
    projections = (f(1, 8), f(1, 20), f(2, 5))
    dot_errors = tuple(gamma3 * a for a in (f(11, 40), f(1, 4), f(2, 5)))
    projected_error = sum(
        2 * value * error + error**2 + gamma3 * (value + error) ** 2
        for value, error in zip(projections, dot_errors, strict=True)
    )
    # Each square plus the three-term reduction has depth at most three.
    # All raw energies, forecast rows, leverages, and 1+leverage are dyadics
    # with fewer than 53 significant bits, so their arithmetic is exact.
    # P/2 is exact; two variance products and their sum add gamma2.
    variance_error = (p + projected_error) * (2 + leverage_sum) / 2 * (1 + gamma2) - variance

    # Multiplication by width=1/4 is exact. The two scalar additions in each
    # Taylor bound contribute gamma2 times the absolute-term envelope.
    bound_errors = []
    for slope, slope_abs, curvature, curvature_abs in (
        (f(1, 4), f(13, 20), f(19, 40), f(19, 40)),
        (f(3, 4), f(3, 4), f(1, 20), f(1, 4)),
    ):
        curvature = curvature if degree == 2 else f(0)
        slope_error = gamma3 * slope_abs
        curvature_error = gamma3 * curvature_abs if degree == 2 else f(0)
        input_error = slope_error / 4 + curvature_error / 8
        envelope = slope / 4 + curvature / 8
        bound_errors.append(input_error + gamma2 * (envelope + input_error))

    # Both projected energies are added before division by the exact 2*T.
    numerator_error = 2 * projected_error + unit * 2 * (p + projected_error)
    ratio_error = numerator_error / (2 * total) + unit * (p / total + numerator_error / (2 * total))
    orthogonal_error = ratio_error + unit * (f(26, 45) + ratio_error)
    with localcontext() as context:
        context.prec = 80

        def decimal(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        root = decimal(variance).sqrt()
        lower, upper = decimal(variance - variance_error).sqrt(), decimal(variance + variance_error).sqrt()
        # sqrt's one-ULP allowance (2u relative), followed by multiplication by
        # 3 (u relative), also permits platform libm rounding differences.
        floor_error = 3 * decimal(variance_error) / (root + lower)
        floor_error += 3 * upper * decimal((1 + 2 * unit) * (1 + unit) - 1)
        expected = (decimal(f(3, 16)), 3 * root, decimal(f(26, 45)))
        errors = (decimal(max(bound_errors)), floor_error, decimal(orthogonal_error))
        return expected, errors


@pytest.mark.parametrize("degree", [1, 2])
def test_ordinary_projected_statistics_match_closed_form_with_float64_roundoff(degree):
    model = _projected_fixture(degree)
    actual = _projected_statistics(model, torch.tensor([0.6, 0.8], dtype=torch.float64), 0.25)
    expected, errors = _projected_closed_form_and_roundoff(degree)
    for name, observed, reference, error in zip(
        ("Taylor bound", "forecast floor", "orthogonal fraction"), actual, expected, errors, strict=True
    ):
        with localcontext() as context:
            context.prec = 80
            difference = abs(Decimal.from_float(observed) - reference)
        assert difference <= error, (
            f"degree={degree}, {name}: actual={observed!r} ({observed.hex()}), "
            f"exact={reference}, absolute_error={difference}, roundoff_bound={error}"
        )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("scale", [1e-170, 1e170, 2.0**-600, 2.0**600])
def test_projected_statistics_are_finite_when_physical_input_squares_underflow_or_overflow(degree, scale):
    model, direction = _projected_fixture(degree), torch.tensor([0.6, 0.8], dtype=torch.float64)
    expected = _projected_statistics(model, direction, 0.25)
    actual = _projected_statistics(replace(model, radius=scale), direction, 0.25 * scale)
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)


def test_tiny_interval_ratio_is_not_squared_before_multiplying_its_curvature_coefficient():
    model = _projected_fixture(2)
    for side in (model.left, model.right):
        side.coefficients.zero_()
        side.coefficients[2, 0] = 1e300
        side.residual.zero_()
    bound, floor, orthogonal = _projected_statistics(model, torch.tensor([1.0, 0.0], dtype=torch.float64), 1e-200)
    assert bound == pytest.approx(2e-100, rel=2e-15, abs=0)
    assert floor == orthogonal == 0


@pytest.mark.parametrize(
    "radius,width,curvature,message",
    [(0.0, 1.0, 0.0, "positive"), (1e170, 5e-324, 0.0, "ratio"), (1e170, 1.5e170, 1e308, "bound")],
)
def test_unrepresentable_projected_diagnostics_fail_explicitly(radius, width, curvature, message):
    model = replace(_projected_fixture(2), radius=radius)
    for side in (model.left, model.right):
        side.coefficients.zero_()
        side.coefficients[2, 0] = curvature
    with pytest.raises(ArithmeticError, match=message):
        _projected_statistics(model, torch.tensor([1.0, 0.0], dtype=torch.float64), width)


@pytest.mark.parametrize("width", [1e-170, 1e170])
def test_public_vector_fit_completes_all_charged_rows_on_extreme_input_widths(width):
    pytest.importorskip("scipy")
    from torch_bsf import fit_adaptive_curve

    charged = []

    def oracle(points):
        charged.extend(points[:, 0].tolist())
        unit = points / width
        return torch.cat((unit.square(), unit**3), 1)

    result = fit_adaptive_curve(oracle, (0.0, width), budget=17)
    assert result.cost == len(charged) == 17
    assert not result.gaps
    unit = torch.linspace(0, 1, 33, dtype=torch.float64)[:, None]
    torch.testing.assert_close(result.predict(unit * width), torch.cat((unit.square(), unit**3), 1), rtol=0, atol=3e-14)
