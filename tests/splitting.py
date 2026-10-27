"""Tests for torch_bsf._kernels._splitting."""

from fractions import Fraction
from typing import Literal

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import (
    FLOAT64_TINY,
    U,
    degrees,
    extreme_scalings,
    factor_edges,
    log_uniform,
    matrices,
    scalings,
    simploid_points,
    simploid_types,
    split_fractions,
)
from torch_bsf import BezierModel, BezierSpace, Points
from torch_bsf._kernels._splitting import longest_edge_criterion, max_error_criterion, reparametrize, split, split_by_criterion
from torch_bsf._kernels.api import BezierSimploid
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.splitting import EdgeSplit, ParameterMap, subdivide, subdivide_space

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(n_params: int, degree: int, n_values: int) -> BezierSimploid:
    """A random Bézier simplex with ``n_params`` vertices (a vertex, of type ``()``, if ``n_params == 1``)."""
    return BezierSimploid.rand((n_params - 1,) if n_params > 1 else (), degree, n_values=n_values)


def _identity_curve(n_values: int = 1) -> BezierSimploid:
    """Degree-1 Bézier curve B(t) = t_1  (for each value dimension)."""
    return BezierSimploid(
        (1,),
        1,
        {
            (1, 0): [0.0] * n_values,
            (0, 1): [1.0] * n_values,
        },
    )


def _identity_curve_degree2(n_values: int = 1) -> BezierSimploid:
    """Degree-2 Bézier curve B(t) = t_1  (identity, n_params=2)."""
    return BezierSimploid(
        (1,),
        2,
        {
            (2, 0): [0.0] * n_values,
            (1, 1): [0.5] * n_values,
            (0, 2): [1.0] * n_values,
        },
    )


# ---------------------------------------------------------------------------
# split(): validation
# ---------------------------------------------------------------------------


def test_split_wrong_n_params():
    bs = _rand(1, 3, 2)
    with pytest.raises(ValueError, match="factor"):
        split(bs, 0, 1)


def test_split_bad_edge_indices():
    bs = _rand(3, 2, 2)
    with pytest.raises(ValueError, match="i="):
        split(bs, 0, 0)
    with pytest.raises(ValueError, match="i="):
        split(bs, -1, 1)
    with pytest.raises(ValueError, match="i="):
        split(bs, 0, 5)


def test_split_bad_s():
    bs = _rand(2, 2, 2)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=0.0)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=1.0)
    with pytest.raises(ValueError, match="s="):
        split(bs, 0, 1, s=1.5)


# ---------------------------------------------------------------------------
# split(): structural properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("n_values", [1, 3])
def test_split_preserves_shape(n_params, degree, n_values):
    """Sub-simplices have the same n_params, degree, and n_values."""
    bs = _rand(n_params, degree, n_values)
    bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
    assert bs_A.n_params == bs.n_params
    assert bs_A.degree == bs.degree
    assert bs_A.n_values == bs.n_values
    assert bs_B.n_params == bs.n_params
    assert bs_B.degree == bs.degree
    assert bs_B.n_values == bs.n_values


# ---------------------------------------------------------------------------
# split(): correctness for degree-1 Bézier curve (identity)
# ---------------------------------------------------------------------------


def test_split_degree1_control_points_a():
    """bs_A control points are correct for the identity degree-1 curve."""
    bs = _identity_curve()
    bs_A, _ = split(bs, i=0, j=1, s=0.5)
    assert torch.allclose(bs_A[(1, 0)], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(bs_A[(0, 1)], torch.tensor([0.5]), atol=1e-6)


def test_split_degree1_control_points_b():
    """bs_B control points are correct for the identity degree-1 curve."""
    _, bs_B = split(_identity_curve(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_B[(1, 0)], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(bs_B[(0, 1)], torch.tensor([1.0]), atol=1e-6)


def test_split_degree2_control_points_a():
    """bs_A control points are correct for the identity degree-2 curve."""
    bs_A, _ = split(_identity_curve_degree2(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_A[(2, 0)], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(bs_A[(1, 1)], torch.tensor([0.25]), atol=1e-6)
    assert torch.allclose(bs_A[(0, 2)], torch.tensor([0.5]), atol=1e-6)


def test_split_degree2_control_points_b():
    """bs_B control points are correct for the identity degree-2 curve."""
    _, bs_B = split(_identity_curve_degree2(), i=0, j=1, s=0.5)
    assert torch.allclose(bs_B[(2, 0)], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(bs_B[(1, 1)], torch.tensor([0.75]), atol=1e-6)
    assert torch.allclose(bs_B[(0, 2)], torch.tensor([1.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# split(): continuity at the split point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_split_continuity_at_new_vertex(n_params, degree, s):
    """Both sub-simplices evaluate to the same value at the new vertex."""
    bs = _rand(n_params, degree, 3)
    i, j = 0, 1
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    # New vertex in A: e_j  (all weight at j in A's local coords)
    # New vertex in B: e_i  (all weight at i in B's local coords)
    t_new_A = torch.zeros(1, n_params)
    t_new_A[0, j] = 1.0
    t_new_B = torch.zeros(1, n_params)
    t_new_B[0, i] = 1.0

    with torch.no_grad():
        val_A = bs_A(t_new_A)
        val_B = bs_B(t_new_B)

    assert torch.allclose(val_A, val_B, atol=1e-5), f"Split point mismatch: {val_A} != {val_B}"


# ---------------------------------------------------------------------------
# split(): reproduces original on each sub-domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("s", [0.3, 0.5, 0.7])
def test_split_reproduces_original_curve(degree, s):
    """For a Bézier curve, sub-simplices reproduce the original on their domain."""
    bs = _rand(2, degree, 3)
    i, j = 0, 1
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    # Sample 20 points in sub-domain A  (t_j / (t_i + t_j) <= s → t_j <= s * t_i/(1-s)... simplified: s * t_i >= (1-s) * t_j)
    alpha_vals = torch.linspace(0.0, s, 20)  # t_j = alpha, t_i = 1 - alpha
    params_A = torch.stack([1.0 - alpha_vals, alpha_vals], dim=1)  # t_j/(t_i+t_j) = alpha <= s

    u_A, _ = reparametrize(params_A, i, j, s, "A")
    with torch.no_grad():
        orig_vals = bs(params_A)
        split_vals = bs_A(u_A)

    assert torch.allclose(orig_vals, split_vals, atol=1e-5), (
        f"Sub-simplex A does not reproduce original: max diff = {(orig_vals - split_vals).abs().max().item()}"
    )

    # Sample 20 points in sub-domain B  (t_j >= s)
    alpha_vals = torch.linspace(s, 1.0, 20)
    params_B = torch.stack([1.0 - alpha_vals, alpha_vals], dim=1)

    u_B, _ = reparametrize(params_B, i, j, s, "B")
    with torch.no_grad():
        orig_vals = bs(params_B)
        split_vals = bs_B(u_B)

    assert torch.allclose(orig_vals, split_vals, atol=1e-5), (
        f"Sub-simplex B does not reproduce original: max diff = {(orig_vals - split_vals).abs().max().item()}"
    )


@pytest.mark.parametrize("n_params", [3, 4])
@pytest.mark.parametrize("degree", [1, 2])
def test_split_reproduces_original_higher_dim(n_params, degree):
    """Sub-simplices reproduce the original for higher-dimensional parameter spaces."""
    bs = _rand(n_params, degree, 2)
    i, j, s = 0, 1, 0.5
    bs_A, bs_B = split(bs, i=i, j=j, s=s)

    from torch_bsf._kernels._sampling import simplex_random

    torch.manual_seed(42)
    params_all = simplex_random(n_params, n_samples=50)

    ti = params_all[:, i]
    tj = params_all[:, j]
    denom = ti + tj

    # Sub-domain A: t_j / (t_i + t_j) <= s  (including t_i+t_j=0)
    mask_A = (denom == 0) | (tj <= s * denom)
    if mask_A.any():
        u_A, _ = reparametrize(params_all[mask_A], i, j, s, "A")
        with torch.no_grad():
            orig = bs(params_all[mask_A])
            approx = bs_A(u_A)
        assert torch.allclose(orig, approx, atol=1e-5)

    # Sub-domain B: t_j / (t_i + t_j) >= s
    mask_B = (denom == 0) | (tj >= s * denom)
    if mask_B.any():
        u_B, _ = reparametrize(params_all[mask_B], i, j, s, "B")
        with torch.no_grad():
            orig = bs(params_all[mask_B])
            approx = bs_B(u_B)
        assert torch.allclose(orig, approx, atol=1e-5)


# ---------------------------------------------------------------------------
# reparametrize()
# ---------------------------------------------------------------------------


def test_reparametrize_t_not_2d():
    """reparametrize raises ValueError when t is not a 2-D tensor."""
    t_1d = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError, match="2-D"):
        reparametrize(t_1d, i=0, j=1, s=0.5, part="A")
    t_3d = torch.tensor([[[0.5, 0.5]]])
    with pytest.raises(ValueError, match="2-D"):
        reparametrize(t_3d, i=0, j=1, s=0.5, part="A")


def test_reparametrize_t_too_few_columns():
    """reparametrize raises ValueError when t has fewer than 2 columns."""
    t = torch.tensor([[0.5]])
    with pytest.raises(ValueError, match="n_params >= 2"):
        reparametrize(t, i=0, j=1, s=0.5, part="A")


def test_reparametrize_t_not_floating():
    """reparametrize raises ValueError when t is not a floating-point tensor."""
    t_int = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="floating-point"):
        reparametrize(t_int, i=0, j=1, s=0.5, part="A")
    t_bool = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="floating-point"):
        reparametrize(t_bool, i=0, j=1, s=0.5, part="A")


def test_reparametrize_bad_s():
    """reparametrize raises ValueError when s is not in (0, 1)."""
    t = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=0.0, part="A")
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=1.0, part="A")
    with pytest.raises(ValueError, match="s="):
        reparametrize(t, i=0, j=1, s=-0.1, part="B")


def test_reparametrize_bad_edge_indices():
    """reparametrize raises ValueError for invalid edge indices."""
    t = torch.tensor([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=0, j=0, s=0.5, part="A")
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=-1, j=1, s=0.5, part="A")
    with pytest.raises(ValueError, match="i="):
        reparametrize(t, i=0, j=5, s=0.5, part="A")


def test_reparametrize_bad_part():
    """reparametrize raises ValueError for an invalid part argument."""
    t = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="part"):
        reparametrize(t, i=0, j=1, s=0.5, part="C")  # type: ignore[arg-type]


def test_reparametrize_a_domain():
    """Points in sub-domain A get a valid mask and remapped coordinates."""
    # t = (0.8, 0.2) → t_j/(t_i+t_j) = 0.2 ≤ 0.5  → belongs to A
    t = torch.tensor([[0.8, 0.2]])
    u, mask = reparametrize(t, i=0, j=1, s=0.5, part="A")
    assert mask[0].item()
    assert torch.allclose(u[0, 1], torch.tensor(0.4), atol=1e-6)  # t_j/s = 0.4
    assert torch.allclose(u[0, 0], torch.tensor(0.6), atol=1e-6)  # 0.8 - 0.5*0.4


def test_reparametrize_b_domain():
    """Points in sub-domain B get a valid mask and remapped coordinates."""
    # t = (0.3, 0.7) → t_j/(t_i+t_j) = 0.7 ≥ 0.5  → belongs to B
    t = torch.tensor([[0.3, 0.7]])
    u, mask = reparametrize(t, i=0, j=1, s=0.5, part="B")
    assert mask[0].item()
    assert torch.allclose(u[0, 0], torch.tensor(0.6), atol=1e-6)  # t_i/(1-s) = 0.6
    assert torch.allclose(u[0, 1], torch.tensor(0.4), atol=1e-6)  # 0.7 - 0.5*0.6


def test_reparametrize_zero_face_belongs_to_both():
    """A point on the opposite face (t_i = t_j = 0) belongs to both sub-domains."""
    t = torch.tensor([[0.0, 0.0, 1.0]])
    _, mask_A = reparametrize(t, i=0, j=1, s=0.5, part="A")
    _, mask_B = reparametrize(t, i=0, j=1, s=0.5, part="B")
    assert mask_A[0].item()
    assert mask_B[0].item()


def test_reparametrize_coordinates_sum_to_one():
    """Re-parameterised coordinates sum to 1 (up to floating-point error)."""
    torch.manual_seed(7)
    from torch_bsf._kernels._sampling import simplex_random

    t = simplex_random(n_params=4, n_samples=50)
    for sub in ("A", "B"):
        u, mask = reparametrize(t, i=0, j=1, s=0.5, part=sub)
        if mask.any():
            row_sums = u[mask].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# longest_edge_criterion()
# ---------------------------------------------------------------------------


def test_longest_edge_criterion_wrong_n_params():
    bs = _rand(1, 1, 2)
    with pytest.raises(ValueError, match="type is"):
        longest_edge_criterion(bs)


def test_longest_edge_criterion_returns_valid_edge():
    bs = _rand(4, 2, 3)
    k, i, j, s = longest_edge_criterion(bs)
    assert k == 0
    assert 0 <= i < j < bs.n_params
    assert 0.0 < s < 1.0


def test_longest_edge_criterion_identity_curve():
    """For the identity curve (edge length = 1), picks edge (0, 1)."""
    bs = _identity_curve(n_values=1)
    k, i, j, s = longest_edge_criterion(bs)
    assert k == 0 and i == 0 and j == 1
    assert s == 0.5


def test_longest_edge_criterion_custom_s():
    bs = _rand(3, 1, 2)
    _, _, _, s = longest_edge_criterion(bs, s=0.3)
    assert s == 0.3


# ---------------------------------------------------------------------------
# max_error_criterion()
# ---------------------------------------------------------------------------


def test_max_error_criterion_returns_valid_edge():
    torch.manual_seed(0)
    bs = _rand(2, 1, 1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    values = torch.tensor([[0.0], [0.5], [1.0]])
    criterion = max_error_criterion(params, values, grid_size=5)
    k, i, j, s = criterion(bs)
    assert k == 0
    assert 0 <= i < bs.n_params
    assert 0 <= j < bs.n_params
    assert i != j
    assert 0.0 < s < 1.0


def test_max_error_criterion_bad_grid_size():
    params = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="grid_size must be >= 1"):
        max_error_criterion(params, values, grid_size=0)


def test_max_error_criterion_wrong_n_params():
    bs = _rand(1, 1, 2)
    params = torch.tensor([[1.0]])
    values = torch.tensor([[0.0]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match="type is"):
        criterion(bs)


def test_max_error_criterion_params_not_2d():
    bs = _rand(2, 1, 1)
    params = torch.tensor([1.0, 0.0, 0.5])  # 1-D
    values = torch.tensor([[0.0], [1.0], [0.5]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`params` must be a 2D tensor"):
        criterion(bs)


def test_max_error_criterion_values_not_2d():
    bs = _rand(2, 1, 1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    values = torch.tensor([0.0, 0.5, 1.0])  # 1-D
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`values` must be a 2D tensor"):
        criterion(bs)


def test_max_error_criterion_mismatched_n():
    bs = _rand(2, 1, 1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5]])  # N=2
    values = torch.tensor([[0.0], [0.5], [1.0]])  # N=3
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match="same number of samples"):
        criterion(bs)


def test_max_error_criterion_wrong_params_shape():
    bs = _rand(2, 1, 1)
    params = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2]])  # wrong n_params=3
    values = torch.tensor([[0.0], [0.5]])
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`params` must have shape"):
        criterion(bs)


def test_max_error_criterion_wrong_values_shape():
    bs = _rand(2, 1, 1)
    params = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    values = torch.tensor([[0.0, 1.0], [0.5, 0.6]])  # wrong n_values=2
    criterion = max_error_criterion(params, values, grid_size=2)
    with pytest.raises(ValueError, match=r"`values` must have shape"):
        criterion(bs)


def test_max_error_criterion_piece_without_data():
    # All data lie at vertex 0, so one of the two pieces holds no data for every s.
    # That piece contributes nothing to the error; the error is 0.25 for every s,
    # so the first candidate s = 0.25 wins (an empty piece yielding NaN would keep the default s = 0.5).
    bs = _identity_curve()
    params = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[0.5]])
    criterion = max_error_criterion(params, values, grid_size=3)
    assert criterion(bs) == (0, 0, 1, 0.25)


# ---------------------------------------------------------------------------
# split_by_criterion()
# ---------------------------------------------------------------------------


def test_split_by_criterion_longest_edge():
    bs = _rand(3, 2, 2)
    bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
    assert bs_A.n_params == bs.n_params
    assert bs_B.n_params == bs.n_params


def test_split_by_criterion_max_error():
    torch.manual_seed(1)
    from torch_bsf._kernels._sampling import simplex_random

    n_params, n_values = 3, 2
    params = simplex_random(n_params, n_samples=20)
    bs = _rand(n_params, 2, n_values)
    values = bs(params).detach()
    criterion = max_error_criterion(params, values, grid_size=3)
    bs_A, bs_B = split_by_criterion(bs, criterion)
    assert bs_A.degree == bs.degree
    assert bs_B.degree == bs.degree


# ---------------------------------------------------------------------------
# Bézier simploids (several factors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "simploid_type, degree, factor, i, j",
    [
        ((1, 1), (2, 3), 0, 0, 1),
        ((1, 1), (2, 3), 1, 1, 0),
        ((2, 1), 2, 0, 0, 2),
        ((2, 1), (1, 3), 1, 0, 1),
        ((1, 1, 2), (2, 1, 2), 2, 1, 2),
    ],
)
@pytest.mark.parametrize("s", [0.3, 0.5])
def test_split_simploid_pieces_reparametrize_the_original(simploid_type, degree, factor, i, j, s):
    """Each piece of a split simploid reproduces the original on its part of the domain."""
    from torch_bsf._kernels import sample_simploid

    model = BezierSimploid.rand(simploid_type, degree, n_values=2).double()
    model_A, model_B = split(model, i, j, s, factor=factor)
    for piece in (model_A, model_B):
        assert piece.simploid_type == model.simploid_type
        assert piece.degree == model.degree
        assert piece.n_values == model.n_values
    t = sample_simploid(simploid_type, 200, seed=0).double()
    pieces: list[tuple[Literal["A", "B"], BezierSimploid]] = [("A", model_A), ("B", model_B)]
    for part, piece in pieces:
        u, mask = reparametrize(t, i, j, s, part, simploid_type=simploid_type, factor=factor)
        assert mask.any()
        with torch.no_grad():
            torch.testing.assert_close(piece(u[mask]), model(t[mask]))


def test_split_simploid_rejects_bad_factor_and_edge():
    model = BezierSimploid.rand((2, 1), 2, n_values=1)
    with pytest.raises(ValueError, match="factor"):
        split(model, 0, 1, factor=2)
    with pytest.raises(ValueError, match="i="):
        split(model, 0, 2, factor=1)  # the second factor is a segment


def test_reparametrize_simploid_leaves_other_factors_unchanged():
    t = torch.tensor([[0.8, 0.2, 0.0, 0.3, 0.7]])
    u, mask = reparametrize(t, 0, 1, 0.5, "B", simploid_type=(2, 1), factor=1)
    assert mask.item()
    torch.testing.assert_close(u[:, :3], t[:, :3])
    torch.testing.assert_close(u[:, 3:], torch.tensor([[0.6, 0.4]]))
    with pytest.raises(ValueError, match="columns for type"):
        reparametrize(t[:, :4], 0, 1, 0.5, "A", simploid_type=(2, 1))


def test_longest_edge_criterion_picks_the_longest_factor():
    """A bilinear patch that is long along the second factor is split along it."""
    patch = BezierSimploid(
        (1, 1),
        1,
        {
            ((1, 0), (1, 0)): [0.0],
            ((1, 0), (0, 1)): [5.0],
            ((0, 1), (1, 0)): [1.0],
            ((0, 1), (0, 1)): [6.0],
        },
    )
    assert longest_edge_criterion(patch) == (1, 0, 1, 0.5)


def test_criteria_split_simploids():
    from torch_bsf._kernels import sample_simploid

    model = BezierSimploid.rand((2, 1), 2, n_values=2)
    params = sample_simploid((2, 1), 30, seed=0)
    values = torch.rand(30, 2)
    for criterion in (longest_edge_criterion, max_error_criterion(params, values, grid_size=3)):
        k, i, j, s = criterion(model)
        assert 0 <= k < 2 and 0 <= i < j <= model.simploid_type[k] and 0.0 < s < 1.0
        model_A, model_B = split_by_criterion(model, criterion)
        assert model_A.simploid_type == model_B.simploid_type == (2, 1)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_public_api_accessible_from_top_level():
    """Only the unified subdivision entry point is public."""
    from torch_bsf.splitting import EdgeSplit, LongestEdge, subdivide

    assert callable(subdivide)
    assert EdgeSplit(0, 1).fraction == LongestEdge().fraction


# ---------------------------------------------------------------------------
# Conforming subdivision of a space (subdivide_space)
# ---------------------------------------------------------------------------


def _uniform_points(space, n: int = 8):
    """``n`` random barycentric points (exact per-factor sums) on every cell of ``space``, vertices included."""
    from torch_bsf._kernels.combinatorics import sample_simploid
    from torch_bsf.core import Points

    data = []
    for k, cell in enumerate(space.cells):
        t = sample_simploid(cell.dimensions, n, seed=k).double()
        offset = 0
        for a in cell.dimensions:
            t[:, offset : offset + a + 1] /= t[:, offset : offset + a + 1].sum(1, keepdim=True)
            offset += a + 1
        data.append((k, t))
    return Points.from_cells(space, data)


def _glued():
    from torch_bsf.geometry import SimploidalSet

    S = SimploidalSet()
    t, p, c = S.add_simplex(3), S.add_simploid((2, 1)), S.add_simploid((1, 1, 1))
    S.identify(t.face(0, 0), p.face(1, 0))
    S.identify(p.face(0, 0), c.face(0, 0))
    return S


def _domains():
    from torch_bsf.geometry import SimploidalSet

    def single(simploid_type):
        S = SimploidalSet()
        S.add_simploid(simploid_type)
        return S

    def collapsed_square():
        S = SimploidalSet()
        e = S.add_simploid((1, 1)).face(0, 0)
        S.degenerate(e, e.face(0, 0), 0, -1)
        return S

    def pyramid():
        S = SimploidalSet()
        S.cone([S.add_simploid((1, 1))])
        return S

    def double_cone():
        S = SimploidalSet()
        cone, _ = S.cone([S.add_simplex(2)])
        S.cone([cone])
        return S

    def prism_with_collapsed_triangle():
        # A triangle (v0, v1, v2) = (w0, w1, w1): two edges of the triangular factor map to the same edge.
        S = SimploidalSet()
        tri = S.add_simploid((2, 1)).face(1, 0)
        S.identify(tri, S.add_simplex(1).degeneracy(0, 1))
        return S

    return {
        "triangle": single((2,)),
        "tetrahedron": single((3,)),
        "prism": single((2, 1)),
        "cube": single((1, 1, 1)),
        "glued": _glued(),
        "collapsed_square": collapsed_square(),
        "pyramid": pyramid(),
        "double_cone": double_cone(),
        "prism_with_collapsed_triangle": prism_with_collapsed_triangle(),
    }


@pytest.mark.parametrize("name", list(_domains()))
@pytest.mark.parametrize("fraction", [0.5, 0.3])
def test_subdivide_space_is_exact_on_every_cell_and_edge(name, fraction):
    """Every factor edge of every cell, in both directions, refines the space without changing the function."""
    from itertools import permutations

    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    torch.manual_seed(0)
    space = BezierSpace.on_set(_domains()[name], degree=2)
    model = BezierModel(space, torch.randn(space.n_controls, 2, dtype=torch.float64))
    points = _uniform_points(space)
    expected = model(points)
    for cid, cell in enumerate(space.cells):
        for factor, a in enumerate(cell.dimensions):
            for i, j in permutations(range(a + 1), 2):
                result = subdivide_space(model, cell=cid, policy=EdgeSplit(i, j, fraction, factor))
                assert result.split == EdgeSplit(i, j, fraction, factor)
                assert len(result.parents) == result.model.space.n_cells
                assert len(result.model.space.main_cells) > len(space.main_cells)
                torch.testing.assert_close(result.model(result.transfer(points)), expected, rtol=0, atol=1e-10)


def test_subdivide_space_propagates_through_shared_faces():
    """Splitting the tetrahedron edge shared with the prism also splits the prism, but not the cube."""
    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    S = _glued()
    tetra, prism, cube = S.main_simploids()
    space = BezierSpace.on_set(S, degree=2)
    model = BezierModel(space, torch.zeros(space.n_controls, 1, dtype=torch.float64))
    # Face (0, 0) of the tetrahedron is the bottom triangle of the prism; its edge (0, 1) is the tetrahedron's (1, 2).
    result = subdivide_space(model, cell=space.cell_id(tetra), policy=EdgeSplit(1, 2))
    parents = [result.parents[k] for k in result.model.space.main_cells]
    assert sorted(parents) == sorted([space.cell_id(tetra)] * 2 + [space.cell_id(prism)] * 2 + [space.cell_id(cube)])
    # The segment factor of the prism is parallel to an edge of the cube.
    result = subdivide_space(model, cell=space.cell_id(prism), policy=EdgeSplit(0, 1, factor=1))
    parents = [result.parents[k] for k in result.model.space.main_cells]
    assert sorted(parents) == sorted([space.cell_id(tetra)] + [space.cell_id(prism)] * 2 + [space.cell_id(cube)] * 2)


def test_subdivide_space_keeps_mixed_degrees_and_repeats():
    """Subdivision is closed: the result can be subdivided again, and every cell keeps its degrees."""
    import random

    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.splitting import EdgeSplit, LongestEdge, subdivide_space

    S = _glued()
    tetra, prism, cube = S.main_simploids()
    space = BezierSpace.on_set(S, degree={tetra: 3, prism: (3, 1), cube: (2, 3, 1)})
    torch.manual_seed(1)
    model = BezierModel(space, torch.randn(space.n_controls, 1, dtype=torch.float64))
    points = _uniform_points(space)
    expected = model(points)
    rng = random.Random(0)
    current, current_points = model, points
    for _ in range(4):
        cid = rng.choice([k for k, c in enumerate(current.space.cells) if c.dimensions])
        cell = current.space.cells[cid]
        factor = rng.randrange(len(cell.dimensions))
        i, j = rng.sample(range(cell.dimensions[factor] + 1), 2)
        result = subdivide_space(current, cell=cid, policy=EdgeSplit(i, j, rng.uniform(0.1, 0.9), factor))
        for k, parent in enumerate(result.parents):
            child, original = result.model.space.cells[k], current.space.cells[parent]
            if child.dimensions == original.dimensions:
                assert child.degree == original.degree
        current_points = result.transfer(current_points)
        current = result.model
        torch.testing.assert_close(current(current_points), expected, rtol=0, atol=1e-9)
    result = subdivide_space(current, cell=current.space.main_cells[0], policy=LongestEdge())
    torch.testing.assert_close(result.model(result.transfer(current_points)), expected, rtol=0, atol=1e-9)


def test_subdivide_space_on_a_single_cell_matches_subdivide():
    """On a one-cell space the two pieces are the children of subdivide."""
    from torch_bsf.core import BezierModel, BezierSpace, Samples
    from torch_bsf.splitting import EdgeSplit, ResidualSplit, subdivide, subdivide_space

    space = BezierSpace.simploid((2, 1), degree=(2, 3))
    torch.manual_seed(2)
    model = BezierModel(space, torch.randn(space.n_controls, 3, dtype=torch.float64))
    policy = EdgeSplit(2, 0, 0.25, 0)
    children = subdivide(model, policy=policy).models
    result = subdivide_space(model, policy=policy)
    assert result.split == policy
    pieces = [result.model.space.cells[k] for k in result.model.space.main_cells]
    nets = {tuple(map(tuple, result.model.coefficients[list(c.controls)].tolist())) for c in pieces}
    assert nets == {tuple(map(tuple, child.coefficients.tolist())) for child in children}
    samples = Samples(_uniform_points(space).parameters, torch.randn(8, 3, dtype=torch.float64))
    assert subdivide_space(model, policy=ResidualSplit(samples)).split.fraction > 0


def test_subdivide_space_rejects_bad_edges_and_points():
    from torch_bsf.core import BezierModel, BezierSpace, Points
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    model = BezierModel(BezierSpace.simplex(2, degree=2), torch.zeros(6, 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="factor"):
        subdivide_space(model, policy=EdgeSplit(0, 1, factor=1))
    with pytest.raises(ValueError, match="Edge indices"):
        subdivide_space(model, policy=EdgeSplit(0, 0))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        subdivide_space(model, policy=EdgeSplit(0, 1, 1.0))
    with pytest.raises(TypeError, match="policy"):
        subdivide_space(model, policy="longest")  # type: ignore[arg-type]
    result = subdivide_space(model, policy=EdgeSplit(0, 1))
    with pytest.raises(ValueError, match="barycentric"):
        result.transfer(Points(torch.tensor([[1.5, -0.5, 0.0]], dtype=torch.float64)))


def test_transfer_accepts_points_within_validation_drift():
    """Points that validation accepts (within 1e-6 of the simplex) are transferred, even onto tiny pieces."""
    from torch_bsf.core import BezierModel, BezierSpace, Points
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    model = BezierModel(BezierSpace.simplex(1, degree=3), torch.tensor([[0.0], [1.0], [-2.0], [3.0]], dtype=torch.float64))
    points = Points(torch.tensor([[-1e-7, 1.0000001], [1.0000001, -1e-7], [0.5, 0.5]], dtype=torch.float64))
    for fraction in (0.5, 1e-12):
        result = subdivide_space(model, policy=EdgeSplit(0, 1, fraction))
        moved = result.transfer(points)
        clean = Points(points.parameters.clamp_min(0) / points.parameters.clamp_min(0).sum(1, keepdim=True))
        torch.testing.assert_close(result.model(moved), model(clean), rtol=0, atol=1e-12)


def test_subdivide_space_on_simplices_bisects_exactly_the_simplices_of_the_edge():
    """On simplices, the simplices containing the edge are bisected (as by interpolation.refine) and share rows."""
    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.geometry import SimploidalSet
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    # A fan of three triangles around the edge (0, 1) of each, plus a triangle glued to the first elsewhere.
    S = SimploidalSet()
    fan = [S.add_simplex(2) for _ in range(3)]
    other = S.add_simplex(2)
    for t in fan[1:]:
        S.identify(fan[0].face(0, 2), t.face(0, 2))
    S.identify(fan[0].face(0, 0), other.face(0, 0))
    space = BezierSpace.on_set(S, degree=2)
    model = BezierModel(space, torch.zeros(space.n_controls, 1, dtype=torch.float64))
    result = subdivide_space(model, cell=space.cell_id(fan[1]), policy=EdgeSplit(0, 1, 0.4))
    refined = result.model.space
    parents = sorted(result.parents[k] for k in refined.main_cells)
    assert parents == sorted([space.cell_id(t) for t in fan] * 2 + [space.cell_id(other)])
    # 6 + 1 vertices and 9 + 1 (split) + 3 (new interior) edges: every shared quadratic row appears once.
    assert refined.n_controls == 7 + 13


@pytest.mark.parametrize("fraction", [1e-300, 1e-12, 1 - 1e-12])
def test_subdivide_space_with_extreme_fractions(fraction):
    """Reversed edges are split at ``fraction`` exactly, even where ``1 - fraction`` rounds to one."""
    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    space = BezierSpace.on_set(_glued(), degree=2)
    torch.manual_seed(3)
    model = BezierModel(space, torch.randn(space.n_controls, 1, dtype=torch.float64))
    points = _uniform_points(space)
    expected = model(points)
    for i, j in [(0, 1), (1, 0)]:
        result = subdivide_space(model, cell=space.main_cells[0], policy=EdgeSplit(i, j, fraction))
        torch.testing.assert_close(result.model(result.transfer(points)), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_subdivide_space_with_extreme_coefficients(scale):
    """Subdivision is linear in the coefficients, so it neither underflows nor overflows at the float64 limits."""
    from torch_bsf.core import BezierModel, BezierSpace
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    space = BezierSpace.on_set(_glued(), degree=3)
    torch.manual_seed(4)
    model = BezierModel(space, torch.rand(space.n_controls, 1, dtype=torch.float64) * scale)
    points = _uniform_points(space)
    result = subdivide_space(model, cell=space.main_cells[1], policy=EdgeSplit(0, 2, 0.3))
    torch.testing.assert_close(result.model(result.transfer(points)), model(points), rtol=1e-13, atol=0)


_ACCELERATORS = [d for d, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]


@pytest.mark.skipif(not _ACCELERATORS, reason="no accelerator available")
def test_transfer_keeps_points_on_their_device():
    from torch_bsf.core import BezierModel, BezierSpace, Points
    from torch_bsf.splitting import EdgeSplit, subdivide_space

    model = BezierModel(BezierSpace.simplex(2, degree=2), torch.randn(6, 1, dtype=torch.float64))
    result = subdivide_space(model, policy=EdgeSplit(0, 1))
    device = _ACCELERATORS[0]
    points = Points(torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32, device=device), torch.tensor([0], device=device))
    moved = result.transfer(points)
    assert moved.parameters.device == points.parameters.device and moved.cell_ids.device == points.cell_ids.device
    cpu = Points(moved.parameters.cpu(), moved.cell_ids.cpu())
    original = Points(points.parameters.cpu(), points.cell_ids.cpu())
    torch.testing.assert_close(result.model(cpu.to()), model(original.to()), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Properties (Hypothesis)
# ---------------------------------------------------------------------------


def _split_tolerance(simploid_type: tuple[int, ...], degree: tuple[int, ...], factor: int, magnitude: float) -> float:
    r"""Bound on ``|child(u) - parent(to_parent(u))|`` for float64 coefficients of magnitude at most ``magnitude``.

    With the unit roundoff :math:`u`, :math:`N` controls, total degree :math:`D = \sum_k d_k`, :math:`w` barycentric
    coordinates and the degree :math:`n` of the split factor:

    * each de Casteljau step is a convex combination ``s a + (1 - s) b``, which errs by at most :math:`3u` times the
      largest control, and convex combinations do not amplify earlier errors: :math:`3nu` after :math:`n` steps;
    * evaluating a model sums :math:`N` products of a control with a basis function, each basis function rounded
      at most :math:`D + 2w` times: :math:`(N + D + 2w)u` relative to the largest control, once for each side;
    * ``to_parent`` moves the parent coordinates by at most :math:`3u`, and the partial derivatives of a
      Bézier simploid with respect to its coordinates are at most :math:`D` times the largest control: :math:`3Du`.

    The sum is doubled, as the bounds of each rounding are not tight.
    """
    n_controls = len(_indices(simploid_type, degree))
    width = sum(a + 1 for a in simploid_type)
    total = sum(degree)
    return 2 * (3 * degree[factor] + 2 * (n_controls + total + 2 * width) + 3 * total) * U * magnitude


@given(data=st.data())
def test_children_reproduce_the_parent_through_to_parent(data):
    """For any edge and split ratio, each child equals the parent at the parent point of every child point."""
    simploid_type = data.draw(simploid_types())
    degree = data.draw(degrees(simploid_type))
    factor, i, j = data.draw(factor_edges(simploid_type))
    s = data.draw(split_fractions(), label="s")
    space = BezierSpace.simploid(simploid_type, degree=degree)
    coefficients = data.draw(matrices(space.n_controls, 2, scale=scalings() | extreme_scalings()), label="coefficients")
    model = BezierModel(space, coefficients)
    result = subdivide(model, policy=EdgeSplit(i, j, s, factor))
    u = data.draw(simploid_points(simploid_type, 6), label="u")
    tol = _split_tolerance(simploid_type, degree, factor, float(coefficients.abs().max()))
    with torch.no_grad():
        for child, mapping in zip(result.models, result.maps, strict=True):
            assert child.coefficients.dtype == torch.float64
            difference = (child(u) - model(mapping.to_parent(u))).abs().max()
            assert difference <= tol, (mapping.part, float(difference), tol)


@given(data=st.data())
def test_from_parent_inverts_to_parent(data):
    """``from_parent`` recovers child points from their parent points, except within rounding of the cut."""
    simploid_type = data.draw(simploid_types())
    factor, i, j = data.draw(factor_edges(simploid_type))
    s = data.draw(split_fractions(), label="s")
    u = data.draw(simploid_points(simploid_type, 6), label="u")
    offset = sum(a + 1 for a in simploid_type[:factor])
    parts: tuple[tuple[Literal["A", "B"], int], ...] = (("A", offset + i), ("B", offset + j))
    for part, cut in parts:
        mapping = ParameterMap(0, simploid_type, EdgeSplit(i, j, s, factor), part)
        back, mask = mapping.from_parent(mapping.to_parent(u))
        # The child coordinates are at most one; each is recovered by a few roundings of terms below one in
        # magnitude, e.g. u_i = t_i - (1 - s) / s * t_j with t_j = s u_j for part A: 16 u bounds them.
        torch.testing.assert_close(back.parameters, u[mask], atol=16 * U, rtol=0)
        # A point is lost only if it lies on the cut face (u_i = 0 for A, u_j = 0 for B) up to rounding.
        assert (u[~mask, cut] <= 4 * U).all()


@given(data=st.data())
def test_children_cover_the_parent_and_to_parent_inverts_from_parent(data):
    """Every parent point lies in a child, and mapping it there and back returns it."""
    simploid_type = data.draw(simploid_types())
    factor, i, j = data.draw(factor_edges(simploid_type))
    s = data.draw(split_fractions(), label="s")
    t = Points(data.draw(simploid_points(simploid_type, 6), label="t"))
    masks = []
    for part in ("A", "B"):
        mapping = ParameterMap(0, simploid_type, EdgeSplit(i, j, s, factor), part)
        child, mask = mapping.from_parent(t)
        # The child coordinates are valid barycentric coordinates (to_parent validates them) and map back to within
        # a few roundings of numbers at most one; see test_from_parent_inverts_to_parent.
        torch.testing.assert_close(mapping.to_parent(child).parameters, t.parameters[mask], atol=16 * U, rtol=0)
        masks.append(mask)
    assert (masks[0] | masks[1]).all()


@given(
    data=st.data(),
    bad=st.sampled_from([float("nan"), float("inf"), -float("inf")]),
)
def test_to_parent_rejects_non_finite_points(data, bad):
    simploid_type = data.draw(simploid_types())
    factor, i, j = data.draw(factor_edges(simploid_type))
    u = data.draw(simploid_points(simploid_type, 3))
    u[data.draw(st.integers(0, 2)), data.draw(st.integers(0, u.shape[1] - 1))] = bad
    with pytest.raises(ValueError, match="finite"):
        ParameterMap(0, simploid_type, EdgeSplit(i, j, 0.5, factor), "A").to_parent(u)


@pytest.mark.parametrize("part", ["A", "B"])
def test_reparametrize_point_next_to_the_cut_of_a_thin_piece(part: Literal["A", "B"]):
    """Regression test for a counterexample of the properties above: t_i = 1e-12, t_j ~ 1 and s = 1 - 1e-12.

    The point lies within rounding of the cut. Testing t_j <= s (t_i + t_j) lost t_i and put the point in piece B,
    where it got u_j = -2.2e-5, so that ``to_parent`` rejected it.
    """
    t = torch.tensor([[1.0, 0.0, 1e-12]], dtype=torch.float64)
    t = t / t.sum(dim=1, keepdim=True)
    u, mask = reparametrize(t, 2, 0, 0.999999999999, part)
    assert (u[mask] >= -4 * U).all()


# Edge coordinates down to the smallest subnormal float64, and zero.
_edge_coordinates = st.one_of(st.just(0.0), log_uniform(5e-324, 1.0))


@given(t_i=_edge_coordinates, t_j=_edge_coordinates, s=split_fractions(FLOAT64_TINY) | st.sampled_from([1 - U, 5e-324]))
def test_reparametrize_puts_each_point_in_the_piece_of_its_exact_ratio(t_i: float, t_j: float, s: float):
    """A point is in piece A iff t_j / (t_i + t_j) <= s, and in B iff it is >= s, decided on exact rationals.

    The only rounding is that of 1 - s, a relative error of at most U, so ratios within U max(s, 1 - s) of s may
    go either way; every other point must be in exactly the piece of its exact ratio, however small its coordinates.
    """
    t = torch.tensor([[t_i, t_j, max(0.0, 1.0 - t_i - t_j)]], dtype=torch.float64)
    _, in_a = reparametrize(t, 0, 1, s, "A")
    _, in_b = reparametrize(t, 0, 1, s, "B")
    assert bool(in_a) or bool(in_b)
    if t_i + t_j == 0:
        assert bool(in_a) and bool(in_b)
        return
    ratio, cut = Fraction(t_j) / (Fraction(t_i) + Fraction(t_j)), Fraction(s)
    if abs(ratio - cut) > Fraction(U) * max(cut, 1 - cut):
        assert bool(in_a) == (ratio < cut)
        assert bool(in_b) == (ratio > cut)


def test_reparametrize_point_with_an_underflowing_product():
    """Regression test: for t = [0, DBL_MIN, 1] and s = nextafter(1, 0), (1 - s) t_j and s t_i both underflowed
    to 0, so the point, whose ratio t_j / (t_i + t_j) is 1, was also put in piece A.
    """
    t = torch.tensor([[0.0, FLOAT64_TINY, 1.0]], dtype=torch.float64)
    _, in_a = reparametrize(t, 0, 1, 1 - U, "A")
    _, in_b = reparametrize(t, 0, 1, 1 - U, "B")
    assert not bool(in_a)
    assert bool(in_b)


@given(data=st.data())
def test_subdivide_space_keeps_the_function_at_any_fraction(data):
    """Refining a whole space at any fraction, including within 1e-12 of an end, keeps the value at every point.

    The tolerance is that of :func:`_split_tolerance` for the largest cell type of the domain: the refined
    controls come from the same de Casteljau steps, and ``transfer`` maps a point to the coordinates of its
    piece with the same few roundings as ``to_parent``.
    """
    name = data.draw(st.sampled_from(sorted(_domains())), label="domain")
    space = BezierSpace.on_set(_domains()[name], degree=2)
    coefficients = data.draw(matrices(space.n_controls, 2, scale=scalings() | extreme_scalings()), label="coefficients")
    model = BezierModel(space, coefficients)
    cid = data.draw(st.integers(0, space.n_cells - 1), label="cell")
    dimensions = space.cells[cid].dimensions
    if not dimensions:
        return
    factor, i, j = data.draw(factor_edges(dimensions), label="edge")
    s = data.draw(split_fractions(), label="s")
    result = subdivide_space(model, cell=cid, policy=EdgeSplit(i, j, s, factor))
    cells = [k for k, cell in enumerate(space.cells) if cell.dimensions]
    pairs = [(k, data.draw(simploid_points(space.cells[k].dimensions, 3), label=f"points of cell {k}")) for k in cells]
    points = Points.from_cells(space, pairs)
    magnitude = coefficients.abs().max().item()
    tolerance = max(
        _split_tolerance(cell.dimensions, (2,) * len(cell.dimensions), 0, magnitude) for cell in space.cells if cell.dimensions
    )
    torch.testing.assert_close(result.model(result.transfer(points)), model(points), rtol=0, atol=tolerance)
