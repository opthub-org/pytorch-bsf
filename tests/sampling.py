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


# ---------------------------------------------------------------------------
# simplex_sobol
# ---------------------------------------------------------------------------

try:
    import scipy  # noqa: F401
except Exception:
    _has_scipy = False
else:
    _has_scipy = True

_scipy_skip = pytest.mark.skipif(
    not _has_scipy,
    reason="scipy is required for simplex_sobol",
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
