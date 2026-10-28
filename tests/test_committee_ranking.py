"""Stable committee acquisition over the finite prediction range."""

from fractions import Fraction
from unittest.mock import patch

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import extreme_scalings
from torch_bsf import BezierModel, BezierSpace, Ensemble, Points
from torch_bsf.active_learning import query_by_committee


def _query_order(predictions):
    count = predictions.shape[1]
    space = BezierSpace.simplex(count - 1, degree=1)
    model = BezierModel(space, torch.zeros(count, predictions.shape[2], dtype=predictions.dtype))
    ensemble = Ensemble([model])
    candidates = Points(torch.eye(count, dtype=torch.float64, requires_grad=True))
    with patch.object(ensemble, "members", return_value=predictions):
        result = query_by_committee(ensemble, candidates, n=count)
    assert result.parameters.requires_grad
    return result.parameters.argmax(dim=1).tolist()


def _exact_order(predictions):
    scores = []
    members = predictions.tolist()
    for candidate in range(predictions.shape[1]):
        score = Fraction(0)
        for output in range(predictions.shape[2]):
            values = [Fraction(member[candidate][output]) for member in members]
            mean = sum(values, Fraction(0)) / len(values)
            score += sum(((value - mean) * (value - mean) for value in values), Fraction(0)) / len(values)
        scores.append(score)
    return sorted(range(len(scores)), key=lambda index: -scores[index])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1.0, 1e20, 1e-30])
def test_issue_478_common_output_scale(dtype, scale):
    predictions = torch.tensor([[[-1.0], [-2.0]], [[1.0], [2.0]]], dtype=dtype) * scale
    assert _query_order(predictions) == [1, 0]


@pytest.mark.parametrize("scale", [torch.finfo(torch.float64).max / 2, 2.0**-1074])
def test_float64_boundary_disagreement(scale):
    predictions = torch.tensor([[[-1.0], [-2.0]], [[1.0], [2.0]]], dtype=torch.float64) * scale
    assert _query_order(predictions) == [1, 0]


def test_huge_agreeing_outputs_do_not_hide_tiny_disagreement():
    predictions = torch.tensor(
        [[[1e308, 0.0], [1e308, -1e-300], [0.0, -2e-300]], [[1e308, 0.0], [1e308, 1e-300], [0.0, 2e-300]]], dtype=torch.float64
    )
    assert _query_order(predictions) == [2, 1, 0]


def test_representable_disagreement_above_large_common_offset():
    offset = 2.0**1000
    step = 2.0**948
    predictions = torch.tensor(
        [[[offset], [offset]], [[offset + step], [offset + 2 * step]], [[offset + step], [offset + 4 * step]]],
        dtype=torch.float64,
    )
    assert _query_order(predictions) == _exact_order(predictions) == [1, 0]


def test_outputs_keep_relative_variance_weights_and_ties():
    predictions = torch.tensor(
        [[[-4.0, 0.0], [-1.0, -2.0], [-4.0, 0.0]], [[4.0, 0.0], [1.0, 2.0], [4.0, 0.0]]],
        dtype=torch.float64,
    )
    assert _query_order(predictions) == [0, 2, 1]
    assert _query_order(predictions[:1]) == [0, 1, 2]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_predictions_raise(value):
    with pytest.raises(ValueError, match="predictions must be finite"):
        _query_order(torch.tensor([[[0.0], [value]]]))


@given(
    st.lists(st.integers(-8, 8), min_size=12, max_size=12),
    extreme_scalings(),
)
def test_common_power_of_two_scaling_preserves_exact_variance_order(values, scale):
    # Small dyadic values make the two-member variance and output sums exact.
    # Scaling by a power of two is exact throughout this representable range.
    predictions = torch.tensor(values, dtype=torch.float64).reshape(2, 3, 2)
    expected = _exact_order(predictions)
    assert _query_order(predictions * scale) == expected


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_public_query_preserves_candidate_device_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    space = BezierSpace.simplex(1, degree=1)
    values = torch.tensor([[1e20], [2e20]], dtype=torch.float32, device=device)
    ensemble = Ensemble([BezierModel(space, values), BezierModel(space, -values)])
    parameters = torch.eye(2, device=device, requires_grad=True)
    result = query_by_committee(ensemble, Points(parameters), n=2)
    assert result.parameters.device == parameters.device
    assert torch.equal(result.parameters, parameters.flip(0))
    result.parameters[0].sum().backward()
    assert parameters.grad is not None
    assert torch.equal(parameters.grad, torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device))


@pytest.mark.parametrize("exponent", [-30, -500, -1074])
def test_subdominant_output_breaks_apparent_variance_ties(exponent):
    tiny = 2.0**exponent
    predictions = torch.tensor(
        [[[-1.0, 0.0], [-1.0, -tiny], [-1.0, -2 * tiny]], [[1.0, 0.0], [1.0, tiny], [1.0, 2 * tiny]]],
        dtype=torch.float64,
    )
    assert _query_order(predictions) == _exact_order(predictions) == [2, 1, 0]


@given(st.integers(-1074, -27), st.integers(2, 7))
def test_close_scores_match_exact_variance_across_member_counts(exponent, members):
    tiny = 2.0**exponent
    # Common leading variance hides the second output in ordinary float64
    # sums. Multiple members exercise centering and non-dyadic 1/K variance.
    predictions = torch.tensor(
        [[[float(member), tiny * candidate] for candidate in [0, 2, 1]] for member in range(members)],
        dtype=torch.float64,
    )
    predictions[0, :, 1] *= -1
    assert _query_order(predictions) == _exact_order(predictions) == [1, 2, 0]


def test_exact_ties_keep_candidate_order_after_approximate_roundoff():
    predictions = torch.tensor(
        [
            [[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    assert _query_order(predictions) == _exact_order(predictions) == [0, 1, 2]


def test_separated_and_zero_scores_do_not_need_exact_comparisons():
    predictions = torch.tensor([[[-1.0], [-2.0], [0.0], [0.0]], [[1.0], [2.0], [0.0], [0.0]]], dtype=torch.float64)
    with patch("torch_bsf.active_learning._exact_disagreement", side_effect=AssertionError("unexpected exact fallback")):
        assert _query_order(predictions) == [1, 0, 2, 3]
