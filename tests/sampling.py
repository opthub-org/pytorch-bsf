import warnings

import pytest
import torch

from torch_bsf.sampling import simplex_grid, simplex_random, simplex_sobol


# ---------------------------------------------------------------------------
# simplex_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_params, degree",
    [
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 2),
        (4, 2),
    ],
)
def test_simplex_grid_shape(n_params, degree):
    result = simplex_grid(n_params, degree)
    assert result.ndim == 2
    assert result.shape[1] == n_params
    assert result.shape[0] >= 1


@pytest.mark.parametrize(
    "n_params, degree",
    [
        (2, 1),
        (2, 3),
        (3, 2),
    ],
)
def test_simplex_grid_sums_to_one(n_params, degree):
    result = simplex_grid(n_params, degree)
    assert torch.allclose(result.sum(dim=1), torch.ones(result.shape[0]), atol=1e-6)


def test_simplex_grid_n_params_zero():
    result = simplex_grid(0, 5)
    assert result.shape == (1, 0)


def test_simplex_grid_degree_zero():
    result = simplex_grid(3, 0)
    assert result.shape == (1, 3)
    assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-6)


def test_simplex_grid_n_params_one():
    result = simplex_grid(1, 5)
    assert result.shape == (1, 1)
    assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-6)


def test_simplex_grid_invalid_n_params():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_grid(-1, 3)


# ---------------------------------------------------------------------------
# simplex_random
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_params, n_samples",
    [
        (2, 5),
        (3, 10),
        (4, 1),
    ],
)
def test_simplex_random_shape(n_params, n_samples):
    result = simplex_random(n_params, n_samples)
    assert result.shape == (n_samples, n_params)


@pytest.mark.parametrize(
    "n_params, n_samples",
    [
        (2, 5),
        (3, 10),
    ],
)
def test_simplex_random_sums_to_one(n_params, n_samples):
    result = simplex_random(n_params, n_samples)
    assert torch.allclose(result.sum(dim=1), torch.ones(n_samples), atol=1e-6)


def test_simplex_random_zero_samples():
    result = simplex_random(3, 0)
    assert result.shape == (0, 3)


def test_simplex_random_non_negative():
    result = simplex_random(3, 100)
    assert (result >= 0).all()


def test_simplex_random_invalid_n_params():
    with pytest.raises(ValueError, match="positive"):
        simplex_random(0, 5)


def test_simplex_random_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_random(3, -1)


def test_simplex_random_seed_reproducibility():
    """Same seed produces identical results."""
    r1 = simplex_random(3, 50, seed=42)
    r2 = simplex_random(3, 50, seed=42)
    assert torch.equal(r1, r2)


def test_simplex_random_different_seeds_differ():
    """Different seeds produce different results (with very high probability)."""
    r1 = simplex_random(3, 50, seed=0)
    r2 = simplex_random(3, 50, seed=1)
    assert not torch.equal(r1, r2)


def test_simplex_random_seed_does_not_mutate_global_state():
    """Using seed= should not change the global numpy random state."""
    import numpy as np

    rng_state_before = np.random.get_state()
    simplex_random(3, 20, seed=99)
    rng_state_after = np.random.get_state()
    # Compare both the 'key' array and the position index in the MT state
    assert (rng_state_before[1] == rng_state_after[1]).all()
    assert rng_state_before[2] == rng_state_after[2]


# ---------------------------------------------------------------------------
# simplex_sobol
# ---------------------------------------------------------------------------

try:
    from scipy.stats import qmc  # noqa: F401
except ImportError:
    _has_scipy = False
else:
    _has_scipy = True

_scipy_skip = pytest.mark.skipif(
    not _has_scipy,
    reason="scipy.stats.qmc is required for simplex_sobol",
)


@_scipy_skip
@pytest.mark.parametrize(
    "n_params, n_samples",
    [
        (2, 4),
        (3, 8),
    ],
)
def test_simplex_sobol_shape(n_params, n_samples):
    result = simplex_sobol(n_params, n_samples)
    assert result.shape == (n_samples, n_params)


@_scipy_skip
@pytest.mark.parametrize(
    "n_params, n_samples",
    [
        (2, 4),
        (3, 8),
    ],
)
def test_simplex_sobol_sums_to_one(n_params, n_samples):
    result = simplex_sobol(n_params, n_samples)
    assert torch.allclose(result.sum(dim=1), torch.ones(n_samples), atol=1e-5)


@_scipy_skip
def test_simplex_sobol_zero_samples():
    result = simplex_sobol(2, 0)
    assert result.shape == (0, 2)


@_scipy_skip
def test_simplex_sobol_non_negative():
    result = simplex_sobol(3, 16)
    assert (result >= 0).all()


@_scipy_skip
def test_simplex_sobol_invalid_n_params():
    with pytest.raises(ValueError, match="at least 2"):
        simplex_sobol(1, 5)


@_scipy_skip
def test_simplex_sobol_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_sobol(2, -1)


@_scipy_skip
def test_simplex_sobol_power_of_two_no_warning():
    """Power-of-2 sample sizes should produce no Sobol-related UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simplex_sobol(3, 128)
    user_warning_messages = [
        str(x.message) for x in w if issubclass(x.category, UserWarning)
    ]
    # Ensure our custom "not a power of 2" warning is not emitted.
    assert not any(
        "not a power of 2" in msg for msg in user_warning_messages
    ), f"Unexpected 'not a power of 2' UserWarning(s): {user_warning_messages}"
    # Ensure the known SciPy Sobol warning is not emitted.
    assert not any(
        "balance properties of Sobol" in msg for msg in user_warning_messages
    ), (
        "Unexpected SciPy Sobol UserWarning(s): "
        f"{user_warning_messages}"
    )


@_scipy_skip
@pytest.mark.parametrize("n_samples", [3, 5, 100, 200])
def test_simplex_sobol_non_power_of_two_warns(n_samples):
    """Non-power-of-2 sample sizes should emit exactly one UserWarning."""
    with pytest.warns(UserWarning, match="not a power of 2") as w:
        result = simplex_sobol(3, n_samples)
    assert len(w) == 1, f"Expected exactly 1 UserWarning, got {len(w)}: {w}"
    # Result is still valid despite the warning.
    assert result.shape == (n_samples, 3)
    assert torch.allclose(result.sum(dim=1), torch.ones(n_samples), atol=1e-5)


@_scipy_skip
def test_simplex_sobol_seed_reproducibility():
    """Same seed produces identical Sobol samples."""
    r1 = simplex_sobol(3, 64, seed=0)
    r2 = simplex_sobol(3, 64, seed=0)
    assert torch.equal(r1, r2)


@_scipy_skip
def test_simplex_sobol_different_seeds_differ():
    """Different seeds produce different samples (with very high probability)."""
    r1 = simplex_sobol(3, 64, seed=0)
    r2 = simplex_sobol(3, 64, seed=1)
    assert not torch.equal(r1, r2)


def test_simplex_sobol_scipy_import_error(monkeypatch):
    """simplex_sobol should raise ImportError when scipy is not available."""
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.stats", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.stats.qmc", raising=False)

    real_import = builtins.__import__

    def _import_without_scipy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("No module named 'scipy'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_scipy)

    # Re-import the function from a fresh copy of the module to bypass the cached import.
    import importlib
    import torch_bsf.sampling as sampling_mod
    importlib.reload(sampling_mod)

    with pytest.raises(ImportError, match="scipy"):
        sampling_mod.simplex_sobol(3, 4)
