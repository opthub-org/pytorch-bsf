import importlib.util
import math
import subprocess
import sys

import pytest
import torch

from torch_bsf import AdaptiveCurveResult, Points, fit_adaptive_curve


def test_default_bracket_limit_is_positive_for_subnormal_width():
    pytest.importorskip("scipy")
    from torch_bsf._adaptive_curve.engine import _default_bracket_limit

    assert _default_bracket_limit(math.ulp(0.0)) == math.ulp(0.0)
    assert _default_bracket_limit(1.0) == 2 / 256


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_package_and_facade_import_without_scipy_and_fit_explains_extra():
    code = """
import importlib.abc
import sys
class MissingSciPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ModuleNotFoundError('deliberately absent', name=fullname)
sys.meta_path.insert(0, MissingSciPy())
import torch_bsf
from torch_bsf.adaptive_curve import AdaptiveCurveResult, CurveGap, CurveInterval
assert 'scipy' not in sys.modules
try:
    torch_bsf.fit_adaptive_curve(lambda x: x, (0., 1.), budget=4)
except ImportError as error:
    assert 'pytorch-bsf[adaptive]' in str(error)
else:
    raise AssertionError('a fit requires the optional dependency')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


requires_scipy = pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="optional adaptive extra")


@requires_scipy
def test_public_fit_retains_actual_model_routing_cost_and_no_prediction_queries():
    received = []

    def oracle(x):
        assert x.dtype == torch.float64 and x.device.type == "cpu" and x.shape[1] == 1
        received.append(x.clone())
        return torch.cat((x, x.square()), 1)

    rng_before = torch.get_rng_state().clone()
    result = fit_adaptive_curve(oracle, (0.0, 1.0), budget=24)
    assert isinstance(result, AdaptiveCurveResult)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert result.cost == sum(map(len, received)) == int(result.counts.sum()) == 24
    assert result.proposed_rows >= result.cost
    assert result.status == "budget_exhausted" and not result.gaps
    assert result.domain.betti_numbers() == [1, 0]
    x = torch.tensor([[(cell.left + cell.right) / 2] for cell in result.intervals], dtype=torch.float64)
    points = Points(torch.full((len(x), 2), 0.5, dtype=torch.float64), [cell.cell_id for cell in result.intervals])
    expected = result.model(points)
    # Both paths use the same polynomial coefficients. Routing and evaluating
    # d+1 Bernstein terms incur at most O(d) unit-roundoff error.
    tolerance = 64 * (result.degree + 1) * torch.finfo(torch.float64).eps * float(expected.detach().abs().max())
    torch.testing.assert_close(result.predict(x), expected, rtol=0, atol=tolerance)
    assert result.predict(torch.empty((0, 1))).shape == (0, 2)
    assert sum(map(len, received)) == 24
    with pytest.raises(ValueError, match="bounds"):
        result.predict([[-0.1]])


@requires_scipy
def test_declared_grid_exhaustion_and_original_bound_endpoint_extension():
    result = fit_adaptive_curve(lambda x: x.square(), (0.03, 0.98), budget=20, resolution=0.25)
    assert result.cost == 3 and result.status == "input_resolution"
    assert torch.equal(result.inputs[:, 0], torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    assert torch.equal(result.predict([[0.03], [0.98]]), result.predict([[0.25], [0.75]]))
    assert result.diagnostics["topology_certified"] is False


@requires_scipy
def test_supplied_initial_design_is_charged_and_existing_anchors_are_cached():
    anchors = torch.linspace(0, 1, 9, dtype=torch.float64)[:, None]
    initial = torch.cat((anchors, (anchors[:-2] + anchors[1:-1]) / 2))
    batches = []

    def oracle(x):
        batches.append(x.clone())
        return x.square()

    result = fit_adaptive_curve(oracle, (0.0, 1.0), budget=16, initial_inputs=initial)
    assert len(batches) == 1 and len(batches[0]) == result.cost == 16
    assert torch.equal(result.inputs[:, 0], torch.sort(initial[:, 0]).values)


@requires_scipy
@pytest.mark.parametrize("vector", [False, True])
def test_seed_controls_only_private_vector_jitter(vector):
    def oracle(x):
        return torch.cat((x, x.square()), 1) if vector else x.square()

    before = torch.get_rng_state().clone()
    first = fit_adaptive_curve(oracle, (0.0, 1.0), budget=24, seed=0)
    other = fit_adaptive_curve(oracle, (0.0, 1.0), budget=24, seed=7)
    assert torch.equal(torch.get_rng_state(), before)
    assert first.cost == other.cost == 24
    assert torch.equal(first.inputs, other.inputs) is not vector


@requires_scipy
def test_insufficient_budget_still_returns_an_available_constant_bsf():
    result = fit_adaptive_curve(lambda x: x + 2, (0.0, 1.0), budget=1)
    assert result.cost == 1 and result.status == "insufficient_samples"
    assert result.model.n_outputs == 1 and result.domain.betti_numbers()[0] == 1
    assert torch.equal(result.predict([[0.0], [1.0]]), torch.full((2, 1), 2.0, dtype=torch.float64))


@requires_scipy
def test_stochastic_pilots_and_repeats_are_charged_to_one_actual_row_budget():
    generator = torch.Generator().manual_seed(723)
    rows = 0

    def oracle(x):
        nonlocal rows
        rows += len(x)
        return x + 0.01 * torch.randn(x.shape, generator=generator, dtype=torch.float64)

    result = fit_adaptive_curve(oracle, (0.0, 1.0), budget=32, deterministic=False)
    assert rows == result.cost == int(result.counts.sum()) == 32
    assert len(result.inputs) < rows and int(result.counts.max()) > 1
    assert result.diagnostics["noise_variance"] > 0
    assert result.diagnostics["noise_confidence_supported"] is False


@requires_scipy
def test_actual_jump_has_separate_bsf_components_and_an_explicit_nan_bracket():
    result = fit_adaptive_curve(lambda x: x + (x >= 0.413).double(), (0.0, 1.0), budget=64)
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.left <= 0.413 <= gap.right
    assert result.domain.betti_numbers()[0] == 2
    points = [[gap.left], [(gap.left + gap.right) / 2], [gap.right]]
    prediction = result.predict(points)
    assert torch.isfinite(prediction[[0, 2]]).all() and torch.isnan(prediction[1]).all()
    assert result.diagnostics["actual_bsf"]["finite_masks_equal"]


@requires_scipy
@pytest.mark.parametrize("response", ["exception", "invalid_shape"])
def test_failed_callback_keeps_actual_charged_rows_on_exception_note(response):
    rows = 0

    def oracle(x):
        nonlocal rows
        rows += len(x)
        if response == "exception":
            raise RuntimeError("expected callback failure")
        return x[:, 0]

    with pytest.raises((RuntimeError, ValueError)) as caught:
        fit_adaptive_curve(oracle, (0.0, 1.0), budget=4)
    assert rows == 4
    assert "requested 4 response rows" in " ".join(caught.value.__notes__)


@requires_scipy
@pytest.mark.parametrize(
    "options",
    [
        {"bounds": (0.0, 0.0)},
        {"bounds": (0.0, float("inf"))},
        {"budget": True},
        {"deterministic": 1},
        {"resolution": False},
        {"location_tolerance": 0.0},
        {"seed": -1},
    ],
)
def test_invalid_options_do_not_obtain_any_response(options):
    def forbidden(_):
        raise AssertionError("argument validation must precede any response")

    arguments = {"bounds": (0.0, 1.0), "budget": 4, **options}
    with pytest.raises(ValueError):
        fit_adaptive_curve(forbidden, **arguments)


@requires_scipy
@pytest.mark.parametrize("seed", [2**64, 2**128])
def test_oversized_seed_is_rejected_before_charging_initial_inputs(seed):
    called = []

    def oracle(x):
        called.append(x.clone())
        return x

    with pytest.raises(ValueError, match="seed.*2\\*\\*64"):
        fit_adaptive_curve(oracle, (0.0, 1.0), budget=4, initial_inputs=[[0.5]], seed=seed)
    assert not called


@requires_scipy
def test_largest_supported_seed_is_accepted():
    result = fit_adaptive_curve(lambda x: x, (0.0, 1.0), budget=4, seed=2**64 - 1)
    assert result.cost == 4


@requires_scipy
@pytest.mark.parametrize("initial_inputs", [None, [[1.7]]])
def test_rounded_grid_endpoint_never_reaches_callback_outside_bounds(initial_inputs):
    received = []

    def oracle(x):
        assert ((x >= 0.0) & (x <= 1.7)).all()
        received.append(x.clone())
        return x

    result = fit_adaptive_curve(oracle, (0.0, 1.7), budget=4, resolution=0.1, initial_inputs=initial_inputs)
    assert result.cost == sum(map(len, received)) == 4
    assert ((result.inputs >= 0.0) & (result.inputs <= 1.7)).all()
    assert torch.isfinite(result.predict([[0.0], [1.7]])).all()


@requires_scipy
def test_smallest_positive_interval_uses_a_representable_default_tolerance():
    width = math.ulp(0.0)
    default = fit_adaptive_curve(lambda x: torch.ones_like(x), (0.0, width), budget=2)
    explicit = fit_adaptive_curve(lambda x: torch.ones_like(x), (0.0, width), budget=2, location_tolerance=width)
    assert default.cost == explicit.cost == 2
    torch.testing.assert_close(default.inputs, explicit.inputs, rtol=0, atol=0)
    torch.testing.assert_close(default.model.coefficients, explicit.model.coefficients, rtol=0, atol=0)
    assert torch.equal(default.predict([[0.0], [width]]), torch.ones((2, 1), dtype=torch.float64))


@requires_scipy
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.filterwarnings("error:overflow encountered in divide:RuntimeWarning")
def test_near_duplicate_initial_rows_do_not_overflow_finite_continuation(constant):
    received = []

    def oracle(x):
        received.append(x.clone())
        return torch.ones_like(x) if constant else x

    result = fit_adaptive_curve(oracle, (0.0, 1.0), budget=10, initial_inputs=[[0.0], [1e-320]])
    assert result.cost == sum(map(len, received)) == 10
    assert not result.gaps
    at = torch.linspace(0, 1, 33, dtype=torch.float64)[:, None]
    expected = torch.ones_like(at) if constant else at
    torch.testing.assert_close(result.predict(at), expected, rtol=0, atol=1e-14)


@requires_scipy
def test_empty_initial_design_is_rejected_before_any_response_is_charged():
    received = []
    with pytest.raises(ValueError, match="empty query"):
        fit_adaptive_curve(lambda x: received.append(x), (0.0, 1.0), budget=9, initial_inputs=torch.empty((0, 1)))
    assert not received
