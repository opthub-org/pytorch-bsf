import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torch_bsf._adaptive_curve.queries import QuerySession


def test_deterministic_grid_duplicates_cost_one_evaluation_and_preserve_inputs():
    calls = []

    def oracle(x):
        calls.extend(x[:, 0].tolist())
        return x.square()

    session = QuerySession(oracle, [[0.0, 1.0]], budget=3, resolution=0.1)
    points = torch.tensor([[0.11], [0.14], [0.91]], dtype=torch.float64)
    before = points.clone()
    output = session.query(points)
    assert session.cost == 2 and len(calls) == 2 and session.proposed_rows == 3
    assert output[0].item() == output[1].item()
    assert torch.equal(points, before)
    session.query([[0.1]])
    assert session.cost == 2
    _, _, counts = session.observations()
    assert counts.tolist() == [1, 1]
    assert session.noise_variance() is None


def test_stochastic_repeats_consume_budget_and_measure_noise():
    counter = 0

    def oracle(x):
        nonlocal counter
        output = torch.arange(counter, counter + len(x), dtype=torch.float64)[:, None]
        counter += len(x)
        return output

    session = QuerySession(oracle, [[0.0, 1.0]], budget=3, deterministic=False)
    session.query([[0.5], [0.5], [0.5]])
    x, means, counts = session.observations()
    assert session.cost == 3 and counts.tolist() == [3]
    assert x.item() == 0.5 and means.item() == 1
    assert session.noise_variance() == 1
    with pytest.raises(RuntimeError, match="budget"):
        session.query([[0.5]])


def test_invalid_requests_do_not_use_oracle_budget():
    session = QuerySession(lambda x: x, [[0.0, 1.0]], budget=2)
    with pytest.raises(ValueError, match="bounds"):
        session.query([[2.0]])
    assert session.cost == 0
    with pytest.raises(RuntimeError, match="budget"):
        session.query([[0.0], [0.5], [1.0]])
    assert session.cost == 0


def test_invalid_evaluator_responses_still_spend_requested_evaluations():
    session = QuerySession(lambda x: [[float("nan")]] * len(x), [[0.0, 1.0]], budget=2)
    with pytest.raises(ValueError, match="responses"):
        session.query([[0.0], [1.0]])
    assert session.cost == 2
    with pytest.raises(RuntimeError, match="budget"):
        session.query([[0.5]])


@pytest.mark.parametrize("upper, step", [(1.7, 0.1), (0.35, 0.01), (0.009, 0.001)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_rounded_grid_endpoint_keys_remain_inside_bounds(upper, step, sign):
    bounds = (0.0, upper) if sign == 1 else (-upper, 0.0)
    received = []

    def oracle(x):
        assert ((x >= bounds[0]) & (x <= bounds[1])).all()
        received.append(x.clone())
        return x

    session = QuerySession(oracle, [bounds], budget=2, resolution=step)
    canonical = session.canonical([[bounds[0]], [bounds[1]]])
    assert torch.equal(session.canonical(canonical), canonical)
    assert torch.equal(session.query([[bounds[0]], [bounds[1]]]), canonical)
    assert len(received) == 1 and session.cost == 2
    assert ((canonical >= bounds[0]) & (canonical <= bounds[1])).all()


def test_no_represented_grid_key_inside_narrow_bounds_uses_no_response():
    called = []

    def oracle(x):
        called.append(x)
        return x

    session = QuerySession(oracle, [[math.nextafter(1.7, -math.inf), 1.7]], budget=1, resolution=0.1)
    with pytest.raises(ValueError, match="grid.*bounds"):
        session.query([[1.7]])
    assert session.cost == 0 and not called


def test_extreme_grid_ratios_remain_finite_and_idempotent():
    session = QuerySession(lambda x: x, [[0.0, 1.0]], budget=3, resolution=1e-320)
    points = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    actual = session.query(points)
    assert torch.equal(actual, points)
    assert torch.equal(session.canonical(actual), actual)
    assert session.cost == 3


@pytest.mark.parametrize("bounds, step", [((-1.0, 1.0), 0.1), ((-1.8, 1.8), 1e-9), ((0.0, 3600.0), 1e-6)])
def test_ordinary_grid_reduction_remains_bitwise_identical(bounds, step):
    session = QuerySession(lambda x: x, [bounds], budget=1, resolution=step)
    points = torch.linspace(bounds[0], bounds[1], 1001, dtype=torch.float64)[:, None]
    minimum, maximum = math.ceil(bounds[0] / step), math.floor(bounds[1] / step)
    previous = torch.round(points / step).clamp(minimum, maximum) * step
    assert torch.equal(session.canonical(points).view(torch.int64), previous.view(torch.int64))


@given(
    step=st.sampled_from([0.001, 0.01, 0.1, 0.2, 0.3]),
    key=st.integers(1, 1000),
    fraction=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    sign=st.sampled_from([-1, 1]),
)
def test_canonical_grid_is_bounded_and_idempotent(step, key, fraction, sign):
    edge = math.nextafter(key * step, 0.0)
    bounds = (0.0, edge) if sign == 1 else (-edge, 0.0)
    session = QuerySession(lambda x: x, [bounds], budget=2, resolution=step)
    points = torch.tensor([[sign * fraction * edge], [sign * edge]], dtype=torch.float64)
    canonical = session.canonical(points)
    assert ((canonical >= bounds[0]) & (canonical <= bounds[1])).all()
    assert torch.equal(session.canonical(canonical), canonical)


def _assert_stable_grid_requery(bounds, step, point, deterministic):
    received = []

    def oracle(x):
        received.append(x.clone())
        return x

    session = QuerySession(oracle, [bounds], budget=2, resolution=step, deterministic=deterministic)
    canonical = session.canonical([[point]])
    assert torch.isfinite(canonical).all()
    assert ((canonical >= bounds[0]) & (canonical <= bounds[1])).all()
    repeated = canonical
    for _ in range(3):
        repeated = session.canonical(repeated)
        assert torch.equal(repeated, canonical)
    first = session.query([[point]])
    observed = session.observations()[0]
    assert torch.equal(observed, canonical)
    assert torch.equal(session.query(observed), first)
    expected = 1 if deterministic else 2
    assert len(received) == session.cost == expected
    assert torch.equal(received[0], received[-1])
    assert session.observations()[2].tolist() == [expected]


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("exponent", [-500, -100, 0, 100, 500])
@pytest.mark.parametrize("sign", [-1, 1])
def test_large_grid_keys_keep_observed_locations_stable(deterministic, exponent, sign):
    lower, upper = (math.ldexp(value, exponent) for value in (1.2520630658639984, 1.2520630658639995))
    bounds = (lower, upper) if sign == 1 else (-upper, -lower)
    step = math.ldexp(2.844936542058989e-16, exponent)
    _assert_stable_grid_requery(bounds, step, bounds[1], deterministic)


@given(
    base=st.floats(1, 2, allow_nan=False, allow_infinity=False),
    exponent=st.integers(-500, 500),
    width=st.integers(8, 32),
    spacing=st.floats(0.125, 4, allow_nan=False, allow_infinity=False),
    fraction=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    sign=st.sampled_from([-1, 1]),
    deterministic=st.booleans(),
)
def test_narrow_grid_requery_preserves_cache_or_replicate_location(
    base, exponent, width, spacing, fraction, sign, deterministic
):
    lower = math.ldexp(base, exponent)
    upper = lower + width * math.ulp(lower)
    bounds = (lower, upper) if sign == 1 else (-upper, -lower)
    step = spacing * math.ulp(lower)
    point = bounds[0] + fraction * (bounds[1] - bounds[0])
    _assert_stable_grid_requery(bounds, step, point, deterministic)


def test_constant_extreme_repeats_have_finite_means_and_zero_noise():
    expected = torch.tensor([[1e308, -1e308]], dtype=torch.float64)
    session = QuerySession(lambda x: expected.expand(len(x), -1), [[0.0, 1.0]], budget=3, deterministic=False)
    queried = session.query([[0.5], [0.5], [0.5]])
    assert torch.equal(queried, expected.expand(3, -1))
    _, mean, counts = session.observations()
    assert torch.equal(mean, expected) and counts.tolist() == [3]
    assert session.noise_variance() == 0


def test_ordinary_repeat_means_and_pooled_noise_remain_bitwise_identical():
    rows = torch.tensor([[0.1, -0.3], [1.7, 0.7], [0.4, -0.2]], dtype=torch.float64)
    session = QuerySession(lambda _: rows, [[0.0, 1.0]], budget=3, deterministic=False)
    actual = session.query([[0.5], [0.5], [0.5]])
    previous = rows.mean(0)
    assert torch.equal(actual.view(torch.int64), previous.expand(3, -1).contiguous().view(torch.int64))
    assert torch.equal(session.observations()[1].view(torch.int64), previous[None].view(torch.int64))
    assert session.noise_variance() == float((rows - previous).square().sum()) / 2


def test_pooled_noise_can_be_finite_when_squared_deviation_sum_overflows():
    rows = torch.tensor([[-1e154], [0.0], [1e154]], dtype=torch.float64)
    session = QuerySession(lambda _: rows, [[0.0, 1.0]], budget=3, deterministic=False)
    session.query([[0.5], [0.5], [0.5]])
    # The numerator is 2 * 1e308, but dividing by its two degrees of freedom
    # yields a representable variance. The symmetric mean is exactly zero.
    assert session.noise_variance() == 1e154 * 1e154


def test_unrepresentable_pooled_noise_fails_explicitly():
    rows = torch.tensor([[-1e308], [1e308]], dtype=torch.float64)
    session = QuerySession(lambda _: rows, [[0.0, 1.0]], budget=2, deterministic=False)
    session.query([[0.5], [0.5]])
    with pytest.raises(ArithmeticError, match="noise variance.*finite float64"):
        session.noise_variance()
