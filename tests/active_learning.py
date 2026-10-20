import pytest
import torch
import torch.nn as nn

from torch_bsf import BezierSimploid
from torch_bsf.active_learning import suggest_next_points


def _make_models(n_params: int, n_values: int, degree: int, k: int = 2):
    """Return a list of k random Bézier simplices."""
    return [BezierSimploid.randn((n_params - 1,), degree, n_values) for _ in range(k)]


class _SimpleLinearModel(nn.Module):
    """A minimal nn.Module that maps (batch, n_params) -> (batch, n_values)."""

    def __init__(self, n_params: int, n_values: int):
        super().__init__()
        self.linear = nn.Linear(n_params, n_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _BufferOnlyModel(nn.Module):
    """A minimal nn.Module with a registered buffer but no learnable parameters."""

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
            suggest_next_points(models, method="unknown")


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
        for model, n in zip(models, (3, 4)):
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


class TestSuggestNextPointsSimploid:
    def test_candidates_lie_on_the_simploid(self):
        """The type of the models is used to sample the candidates."""
        from torch_bsf.simploidal import sample_simploid

        models = [BezierSimploid.randn((2, 1), 2, 2) for _ in range(3)]
        existing = sample_simploid((2, 1), 5, seed=0)
        for method in ("qbc", "density"):
            result = suggest_next_points(models, n_suggestions=4, n_candidates=50, method=method, params=existing)
            assert result.shape == (4, 5)
            assert (result >= 0).all()
            torch.testing.assert_close(result[:, :3].sum(dim=1), torch.ones(4))
            torch.testing.assert_close(result[:, 3:].sum(dim=1), torch.ones(4))

    def test_explicit_type_for_plain_modules(self):
        models = [_SimpleLinearModel(4, 2) for _ in range(2)]
        result = suggest_next_points(models, n_suggestions=2, n_candidates=20, simploid_type=(1, 1))
        torch.testing.assert_close(result[:, :2].sum(dim=1), torch.ones(2))
        torch.testing.assert_close(result[:, 2:].sum(dim=1), torch.ones(2))

    def test_mismatched_types_raise(self):
        models = [BezierSimploid.randn((1, 1), 1, 2), BezierSimploid.randn((3,), 1, 2)]
        with pytest.raises(ValueError, match="simploid_type"):
            suggest_next_points(models)
        with pytest.raises(ValueError, match="simploid_type"):
            suggest_next_points(models[:1], simploid_type=(3,))


@pytest.mark.parametrize("simploid_type", [(2,), (1, 1), None])
def test_seed_makes_suggestions_reproducible(simploid_type):
    if simploid_type is None:
        models = [_SimpleLinearModel(3, 2) for _ in range(2)]
    else:
        models = [BezierSimploid.randn(simploid_type, 2, 2) for _ in range(2)]
    a = suggest_next_points(models, n_suggestions=3, n_candidates=50, n_params=3 if simploid_type is None else None, seed=1)
    b = suggest_next_points(models, n_suggestions=3, n_candidates=50, n_params=3 if simploid_type is None else None, seed=1)
    torch.testing.assert_close(a, b)
