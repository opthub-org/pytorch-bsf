import math

import pytest
import torch
import torch.nn as nn
from hypothesis import given

from tests.strategies import U32, nearby_acquisition_points
from torch_bsf import Points
from torch_bsf._kernels._active_learning import suggest_next_points
from torch_bsf._kernels.api import BezierSimploid
from torch_bsf.active_learning import farthest_points


def _dense_farthest(candidates, observed, n):
    """Independent scalar reference: no torch distance kernels or matrix products."""
    coordinates = candidates.parameters.tolist()
    observed_coordinates = observed.parameters.tolist()
    candidate_ids = candidates.cell_ids.tolist()
    observed_ids = observed.cell_ids.tolist()
    distances = [
        min(
            (
                math.dist(point, other)
                for other, other_id in zip(observed_coordinates, observed_ids, strict=True)
                if cid == other_id
            ),
            default=math.inf,
        )
        for point, cid in zip(coordinates, candidate_ids, strict=True)
    ]
    selected = []
    for _ in range(n):
        index = max(range(len(distances)), key=distances.__getitem__)
        selected.append(index)
        for row, cid in enumerate(candidate_ids):
            if cid == candidate_ids[index]:
                distances[row] = min(distances[row], math.dist(coordinates[row], coordinates[index]))
        for row in selected:
            distances[row] = -1
    return candidates.take(torch.tensor(selected))


@pytest.mark.parametrize("chunk_size", [1, 3, 32, 1024])
@pytest.mark.parametrize("observations", [0, 5, 37])
@pytest.mark.parametrize("cells", [1, 3])
def test_chunked_farthest_matches_dense_order(chunk_size, observations, cells):
    generator = torch.Generator().manual_seed(71)
    candidates = Points(torch.rand(41, 3, dtype=torch.float64, generator=generator), torch.arange(41) % cells)
    observed = Points(
        torch.rand(observations, 3, dtype=torch.float64, generator=generator), torch.arange(observations) % min(cells, 2)
    )
    expected = _dense_farthest(candidates, observed, len(candidates))
    actual = farthest_points(candidates, observed, len(candidates), chunk_size=chunk_size)
    torch.testing.assert_close(actual.parameters, expected.parameters, rtol=0, atol=0)
    torch.testing.assert_close(actual.cell_ids, expected.cell_ids, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [1, 7, 25, 26, 1024])
@pytest.mark.parametrize("n_candidates,n_observed", [(25, 1), (26, 1), (2, 26), (26, 26)])
def test_farthest_preserves_nearby_simplex_distances(chunk_size, n_candidates, n_observed):
    t = torch.linspace(0.5, 0.5 + 2.5e-9, n_candidates, dtype=torch.float64)
    # Include both sides of PyTorch's 25-row algorithm threshold on each axis.
    candidates = Points(torch.stack([t, 1 - t], dim=1))
    observed = Points(torch.full((n_observed, 2), 0.5, dtype=torch.float64))
    result = farthest_points(candidates, observed, min(5, n_candidates), chunk_size=chunk_size)
    expected = _dense_farthest(candidates, observed, min(5, n_candidates))
    assert torch.equal(result.parameters, expected.parameters)
    assert torch.equal(result.parameters[0], candidates.parameters[-1])


@given(nearby_acquisition_points())
def test_farthest_nearby_points_match_reference(case):
    parameters, observations, chunk_size = case
    candidates, observed = Points(parameters), Points(observations)
    expected = _dense_farthest(candidates, observed, 5)
    actual = farthest_points(candidates, observed, 5, chunk_size=chunk_size)
    assert torch.equal(actual.parameters, expected.parameters)


def test_chunked_farthest_ties_duplicates_and_autograd():
    parameters = torch.tensor([[0.5], [0.5], [0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    candidates = Points(parameters)
    observed = Points([[0.0], [1.0]])
    result = farthest_points(candidates, observed, 4, chunk_size=1)
    torch.testing.assert_close(result.parameters, parameters[[0, 1, 2, 3]], rtol=0, atol=0)
    result.parameters.sum().backward()
    torch.testing.assert_close(parameters.grad, torch.ones_like(parameters), rtol=0, atol=0)


def test_farthest_distance_blocks_are_bounded(monkeypatch):
    original = torch.cdist
    shapes = []

    def bounded_cdist(x, y, **kwargs):
        shapes.append((len(x), len(y)))
        assert len(x) <= 7 and len(y) <= 7
        return original(x, y, **kwargs)

    monkeypatch.setattr(torch, "cdist", bounded_cdist)
    farthest_points(Points(torch.rand(41, 2)), Points(torch.rand(31, 2)), 4, chunk_size=7)
    assert len(shapes) == 30


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_farthest_rejects_invalid_chunk_size(invalid):
    with pytest.raises((ValueError, TypeError), match="chunk_size"):
        farthest_points(Points([[0.0]]), Points([[1.0]]), chunk_size=invalid)


def _make_models(n_params: int, n_values: int, degree: int, k: int = 2):
    """Return a list of k random Bézier simplices."""
    return [BezierSimploid.randn((n_params - 1,), degree, n_values) for _ in range(k)]


class _SimpleLinearModel(nn.Module):
    """A minimal nn.Module that maps (batch, n_params) -> (batch, n_values)."""

    # Attributes some tests set to emulate other frameworks; unset by default.
    n_params: int
    device: torch.device | str

    def __init__(self, n_params: int, n_values: int):
        super().__init__()
        self.linear = nn.Linear(n_params, n_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.linear(x)
        return y


class _BufferOnlyModel(nn.Module):
    """A minimal nn.Module with a registered buffer but no learnable parameters."""

    bias: torch.Tensor

    def __init__(self, n_params: int):
        super().__init__()
        self.register_buffer("bias", torch.zeros(n_params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class _EmptyModel(nn.Module):
    """A minimal nn.Module with no parameters and no buffers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestSuggestNextPointsQBC:
    def test_output_shape(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=3, n_candidates=50)
        assert result.shape == (3, 3)

    def test_output_is_on_simplex(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=5, n_candidates=100)
        # All components non-negative
        assert (result >= 0).all()
        # Rows sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(5), atol=1e-5)

    def test_single_suggestion(self):
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=1, n_candidates=50)
        assert result.shape == (1, 3)

    def test_single_model(self):
        # With one model, QBC variance is zero everywhere; should still return a valid point
        model = BezierSimploid.randn((2,), 2, 2)
        result = suggest_next_points([model], n_suggestions=1, n_candidates=50)
        assert result.shape == (1, 3)
        assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            suggest_next_points([])

    def test_mismatched_n_params_raises(self):
        m1 = BezierSimploid.randn((2,), 2, 2)
        m2 = BezierSimploid.randn((1,), 2, 2)
        with pytest.raises(ValueError, match="simploid_type"):
            suggest_next_points([m1, m2])

    def test_unknown_method_raises(self):
        models = _make_models(3, 2, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            suggest_next_points(models, method="unknown")  # type: ignore[arg-type]


class TestSuggestNextPointsDensity:
    def test_output_shape(self):
        models = _make_models(3, 2, 2)
        existing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = suggest_next_points(models, n_suggestions=2, n_candidates=50, method="density", params=existing)
        assert result.shape == (2, 3)

    def test_output_is_on_simplex(self):
        models = _make_models(3, 2, 2)
        existing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = suggest_next_points(models, n_suggestions=3, n_candidates=100, method="density", params=existing)
        assert (result >= 0).all()
        assert torch.allclose(result.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_density_requires_params(self):
        models = _make_models(3, 2, 2)
        with pytest.raises(ValueError, match="params"):
            suggest_next_points(models, method="density", params=None)


class TestSuggestNextPointsInteroperability:
    """Tests for improved interoperability: nn.ModuleList and generic nn.Module."""

    def test_accepts_module_list(self):
        """nn.ModuleList (from EnsembleLightningModule.models) should work directly."""
        bezier_list = _make_models(3, 2, 2)
        module_list = nn.ModuleList(bezier_list)
        result = suggest_next_points(module_list, n_suggestions=2, n_candidates=50)
        assert result.shape == (2, 3)
        assert (result >= 0).all()
        assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_generic_module_with_explicit_n_params(self):
        """Generic nn.Module without n_params attribute works when n_params is explicit."""
        models = [_SimpleLinearModel(3, 2) for _ in range(2)]
        result = suggest_next_points(models, n_suggestions=2, n_candidates=50, n_params=3)
        assert result.shape == (2, 3)

    def test_generic_module_missing_n_params_raises(self):
        """Generic nn.Module without n_params attribute raises when n_params not given."""
        models = [_SimpleLinearModel(3, 2)]
        with pytest.raises(ValueError, match="n_params"):
            suggest_next_points(models)

    def test_explicit_n_params_accepted(self):
        """Explicitly provided n_params is accepted and produces correct output shape."""
        models = _make_models(3, 2, 2)
        result = suggest_next_points(models, n_suggestions=1, n_candidates=50, n_params=3)
        assert result.shape == (1, 3)

    def test_explicit_n_params_mismatch_raises(self):
        """Explicit n_params must match models' n_params when models expose n_params."""
        models = _make_models(3, 2, 2, k=1)
        with pytest.raises(ValueError, match="n_params"):
            suggest_next_points(models, n_suggestions=1, n_candidates=50, n_params=4)

    def test_generic_module_n_params_mismatch_raises(self):
        """Without simploid_type, n_params is checked against the n_params attributes of the models."""
        models = [_SimpleLinearModel(3, 2), _SimpleLinearModel(4, 2)]
        for model, n in zip(models, (3, 4), strict=True):
            model.n_params = n
        with pytest.raises(ValueError, match="explicit n_params=4 but models\\[0\\].n_params=3"):
            suggest_next_points(models, n_params=4)
        with pytest.raises(ValueError, match="models\\[1\\].n_params=4 does not match"):
            suggest_next_points(models)

    def test_buffer_only_device_inference(self):
        """_infer_device falls back to buffer device for modules with no parameters."""
        models = [_BufferOnlyModel(3) for _ in range(2)]
        result = suggest_next_points(models, n_suggestions=1, n_candidates=50, n_params=3)
        assert result.shape == (1, 3)

    def test_empty_module_cpu_fallback(self):
        """_infer_device returns CPU for modules with no parameters and no buffers."""
        models = [_EmptyModel() for _ in range(2)]
        result = suggest_next_points(models, n_suggestions=1, n_candidates=50, n_params=3)
        assert result.shape == (1, 3)
        assert result.device.type == "cpu"

    def test_device_mismatch_raises(self):
        """Device mismatch across models raises ValueError."""
        m1 = _SimpleLinearModel(3, 2)
        m2 = _SimpleLinearModel(3, 2)
        # Override .device on m2 to simulate a different device
        m2.device = torch.device("meta")
        with pytest.raises(ValueError, match="same device"):
            suggest_next_points([m1, m2], n_suggestions=1, n_candidates=50, n_params=3)

    def test_string_cpu_device_normalization(self):
        """String-valued .device='cpu' is normalized and does not cause mismatch."""
        m1 = _SimpleLinearModel(3, 2)
        m2 = _SimpleLinearModel(3, 2)
        # Simulate a framework setting .device as a string instead of torch.device
        m2.device = "cpu"
        result = suggest_next_points([m1, m2], n_suggestions=1, n_candidates=50, n_params=3)
        assert result.shape == (1, 3)
        assert result.device.type == "cpu"


def _factor_sum_tolerance(width: int) -> float:
    """Bound on ``|sum - 1|`` for a factor of ``width`` float32 coordinates drawn by ``sample_simploid``.

    The sampler divides ``width`` positive numbers by their sum, which rounds ``width - 1`` times (an error of at most
    ``(width - 1) u`` relative to the sum), and rounds each quotient once (at most ``u`` in all, as the quotients sum
    to one); summing the quotients in the test rounds ``width - 1`` more times. With the float32 unit roundoff
    ``u = 2**-24``, the error is at most ``(2 * width - 1) u`` to first order; doubled.
    """
    return 2 * (2 * width - 1) * U32


class TestSuggestNextPointsSimploid:
    def test_candidates_lie_on_the_simploid(self):
        """The type of the models is used to sample the candidates."""
        from torch_bsf._kernels import sample_simploid

        models = [BezierSimploid.randn((2, 1), 2, 2) for _ in range(3)]
        existing = sample_simploid((2, 1), 5, seed=0)
        for method in ("qbc", "density"):
            result = suggest_next_points(models, n_suggestions=4, n_candidates=50, method=method, params=existing)
            assert result.shape == (4, 5)
            assert (result >= 0).all()
            torch.testing.assert_close(result[:, :3].sum(dim=1), torch.ones(4), rtol=0, atol=_factor_sum_tolerance(3))
            torch.testing.assert_close(result[:, 3:].sum(dim=1), torch.ones(4), rtol=0, atol=_factor_sum_tolerance(2))

    def test_explicit_type_for_plain_modules(self):
        models = [_SimpleLinearModel(4, 2) for _ in range(2)]
        result = suggest_next_points(models, n_suggestions=2, n_candidates=20, simploid_type=(1, 1))
        torch.testing.assert_close(result[:, :2].sum(dim=1), torch.ones(2), rtol=0, atol=_factor_sum_tolerance(2))
        torch.testing.assert_close(result[:, 2:].sum(dim=1), torch.ones(2), rtol=0, atol=_factor_sum_tolerance(2))

    def test_mismatched_types_raise(self):
        models = [BezierSimploid.randn((1, 1), 1, 2), BezierSimploid.randn((3,), 1, 2)]
        with pytest.raises(ValueError, match="simploid_type"):
            suggest_next_points(models)
        with pytest.raises(ValueError, match="simploid_type"):
            suggest_next_points(models[:1], simploid_type=(3,))


@pytest.mark.parametrize("simploid_type", [(2,), (1, 1), None])
def test_seed_makes_suggestions_reproducible(simploid_type):
    models: list[nn.Module]
    if simploid_type is None:
        models = [_SimpleLinearModel(3, 2) for _ in range(2)]
    else:
        models = [BezierSimploid.randn(simploid_type, 2, 2) for _ in range(2)]
    a = suggest_next_points(models, n_suggestions=3, n_candidates=50, n_params=3 if simploid_type is None else None, seed=1)
    b = suggest_next_points(models, n_suggestions=3, n_candidates=50, n_params=3 if simploid_type is None else None, seed=1)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
