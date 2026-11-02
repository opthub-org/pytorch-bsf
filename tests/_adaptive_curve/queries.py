import pytest
import torch

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
