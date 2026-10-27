import warnings
from math import comb

import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import seeds
from torch_bsf._kernels._sampling import simplex_grid, simplex_random, simplex_sobol

# ---------------------------------------------------------------------------
# simplex_grid
# ---------------------------------------------------------------------------


@example(n_params=0, degree=0)
@example(n_params=1, degree=5)
@example(n_params=3, degree=0)
@given(n_params=st.integers(0, 5), degree=st.integers(0, 6))
def test_simplex_grid_shape_and_coordinates(n_params, degree):
    result = simplex_grid(n_params, degree)
    count = 1 if n_params == 0 or degree == 0 else comb(degree + n_params - 1, n_params - 1)
    assert result.shape == (count, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    if n_params:
        assert len(torch.unique(result, dim=0)) == count
    if n_params:
        # Float32 division and n_params-term summation: at most (n_params + 1) unit roundoffs.
        bound = (n_params + 1) * torch.finfo(result.dtype).eps / 2
        assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    if degree > 0:
        # Each coordinate is an integer lattice coordinate divided by degree.
        integer_coordinates = result * degree
        assert ((integer_coordinates - integer_coordinates.round()).abs() <= degree * torch.finfo(result.dtype).eps).all()


def test_simplex_grid_invalid_n_params():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_grid(-1, 3)


# ---------------------------------------------------------------------------
# simplex_random
# ---------------------------------------------------------------------------


@example(n_params=1, n_samples=0, seed=0)
@example(n_params=1, n_samples=1, seed=0)
@given(n_params=st.integers(1, 8), n_samples=st.integers(0, 64), seed=seeds())
def test_simplex_random_shape_and_coordinates(n_params, n_samples, seed):
    result = simplex_random(n_params, n_samples, seed=seed)
    assert result.shape == (n_samples, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    # Dirichlet normalization is float64, then one float32 cast and a row reduction.
    bound = (n_params + 2) * torch.finfo(result.dtype).eps / 2
    assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    assert torch.equal(result, simplex_random(n_params, n_samples, seed=seed))


def test_simplex_random_invalid_n_params():
    with pytest.raises(ValueError, match="positive"):
        simplex_random(0, 5)


def test_simplex_random_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_random(3, -1)


def test_simplex_random_different_seeds_differ():
    """Different seeds produce different results (with very high probability)."""
    r1 = simplex_random(3, 50, seed=0)
    r2 = simplex_random(3, 50, seed=1)
    assert not torch.equal(r1, r2)


@given(seed=seeds())
def test_simplex_random_seed_does_not_mutate_global_state(seed):
    """Using seed= should not change the global numpy random state."""
    import numpy as np

    rng_state_before = np.random.get_state()
    simplex_random(3, 20, seed=seed)
    rng_state_after = np.random.get_state()
    assert isinstance(rng_state_before, tuple) and isinstance(rng_state_after, tuple)
    # Compare both the 'key' array and the position index in the MT state
    assert (rng_state_before[1] == rng_state_after[1]).all()
    assert rng_state_before[2:] == rng_state_after[2:]


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
@example(n_params=2, exponent=0, seed=0)
@given(n_params=st.integers(2, 8), exponent=st.integers(0, 6), seed=seeds())
def test_simplex_sobol_shape_and_coordinates(n_params, exponent, seed):
    n_samples = 2**exponent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = simplex_sobol(n_params, n_samples, seed=seed)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert result.shape == (n_samples, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    # Sorted differences are computed in float64, then cast and summed in float32.
    bound = (n_params + 2) * torch.finfo(result.dtype).eps / 2
    assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    assert torch.equal(result, simplex_sobol(n_params, n_samples, seed=seed))


@_scipy_skip
@given(n_params=st.integers(2, 8))
def test_simplex_sobol_zero_samples(n_params):
    assert simplex_sobol(n_params, 0).shape == (0, n_params)


@_scipy_skip
def test_simplex_sobol_invalid_n_params():
    with pytest.raises(ValueError, match="at least 2"):
        simplex_sobol(1, 5)


@_scipy_skip
def test_simplex_sobol_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_sobol(2, -1)


@_scipy_skip
@given(n_samples=st.integers(3, 256).filter(lambda n: n & (n - 1) != 0))
def test_simplex_sobol_non_power_of_two_warns(n_samples):
    """Non-power-of-2 sample sizes should emit exactly one UserWarning."""
    with pytest.warns(UserWarning, match="not a power of 2") as w:
        result = simplex_sobol(3, n_samples, seed=0)
    assert len(w) == 1, f"Expected exactly 1 UserWarning, got {len(w)}: {w}"
    # Result is still valid despite the warning.
    assert result.shape == (n_samples, 3)
    assert ((result.sum(dim=1) - 1).abs() <= 5 * torch.finfo(result.dtype).eps / 2).all()


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

    import torch_bsf._kernels._sampling as sampling_mod

    importlib.reload(sampling_mod)

    with pytest.raises(ImportError, match="scipy"):
        sampling_mod.simplex_sobol(3, 4)
