"""Tests for torch_bsf.interpolation (shape-preserving cubic interpolation on simplicial complexes)."""

import itertools
import math
from collections import Counter
from fractions import Fraction

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import (
    FLOAT64_MAX,
    FLOAT64_TINY,
    U,
    axis_scalings,
    extreme_scalings,
    kuhn_meshes,
    log_uniform,
    matrices,
    points_in_cells,
    scalings,
    signed,
    simplex_points,
    translations,
)
from torch_bsf import BezierModel, BezierSpace, Points
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.geometry import SimploidalSet
from torch_bsf.interpolation import SimplicialMesh, estimate_gradients, hermite_interpolant, refinement_indicator
from torch_bsf.splitting import subdivide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kuhn_mesh(dim: int, n: int, jitter: float = 0.0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The Kuhn triangulation of an ``n``-per-axis grid of the unit cube, with jittered interior vertices."""
    grid = list(itertools.product(range(n + 1), repeat=dim))
    number = {p: k for k, p in enumerate(grid)}
    x = torch.tensor(grid, dtype=torch.float64) / n
    interior = ((x > 0) & (x < 1)).all(dim=1)
    gen = torch.Generator().manual_seed(seed)
    x[interior] += jitter / n * (torch.rand(int(interior.sum()), dim, generator=gen, dtype=torch.float64) - 0.5)
    simplices = []
    for corner in itertools.product(range(n), repeat=dim):
        for perm in itertools.permutations(range(dim)):
            p = list(corner)
            simplex = [number[tuple(p)]]
            for axis in perm:
                p[axis] += 1
                simplex.append(number[tuple(p)])
            simplices.append(simplex)
    # Shuffle the vertex order within each simplex to exercise the sorting of shared faces.
    rows = [s[::-1] if k % 2 else s for k, s in enumerate(simplices)]
    return x, torch.tensor(rows)


def _random_points_in_cells(mesh: SimplicialMesh, per_cell: int, seed: int = 0) -> Points:
    gen = torch.Generator().manual_seed(seed)
    n = mesh.dimension + 1
    lam = -torch.log(torch.rand(mesh.n_simplices * per_cell, n, generator=gen, dtype=torch.float64))
    lam = lam / lam.sum(dim=1, keepdim=True)
    return Points(lam, mesh.cell_ids.repeat_interleave(per_cell))


def _global_positions(mesh: SimplicialMesh, points: Points) -> torch.Tensor:
    simplex = {c: k for k, c in enumerate(mesh.cell_ids.tolist())}
    rows = torch.tensor([simplex[c] for c in points.cell_ids.tolist()])
    return torch.einsum("nk,nkd->nd", points.parameters, mesh.vertices[mesh.simplices[rows]])


def _quadratic(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([1 + x.sum(1) + x.square().sum(1) + x[:, 0] * x[:, -1], x[:, 0] - 2 * x[:, -1] + 3 * x[:, 0] ** 2], 1)


def _linear(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([2 - x.sum(1), 3 * x[:, 0]], 1)


def _kinked(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([(x - 0.4).abs().sum(1), torch.where(x[:, 0] > 0.5, 1.0, 0.0).double() + x[:, -1]], 1)


MESHES = [(1, 8), (2, 4), (3, 3)]


# ---------------------------------------------------------------------------
# Reproduction and interpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, n", MESHES)
@pytest.mark.parametrize("data", [_linear, _quadratic])
def test_reproduces_polynomials_of_degree_two(dim, n, data):
    """Linear and quadratic data are reproduced to machine precision without clamping."""
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    model = hermite_interpolant(mesh, data(x), clamp=False)
    points = _random_points_in_cells(mesh, 5)
    torch.testing.assert_close(model(points), data(_global_positions(mesh, points)), atol=1e-11, rtol=0)


@pytest.mark.parametrize("dim, n", MESHES)
def test_estimated_gradients_of_quadratic_are_exact(dim, n):
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    g = estimate_gradients(mesh, x.square().sum(1))
    torch.testing.assert_close(g[:, :, 0], 2 * x, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_linear_fallback_reproduces_linear_data(dim):
    """A single simplex has too few points for the quadratic fit; the one-ring linear fit is exact for linear data."""
    x = torch.cat([torch.zeros(1, dim), torch.eye(dim) + 0.1]).double()
    mesh = SimplicialMesh(x, [list(range(dim + 1))])
    model = hermite_interpolant(mesh, _linear(x), clamp=False)
    points = _random_points_in_cells(mesh, 20)
    torch.testing.assert_close(model(points), _linear(_global_positions(mesh, points)), atol=1e-12, rtol=0)


def test_given_gradients_are_used():
    """With exact gradients, a single triangle (too few neighbours to estimate them) reproduces a quadratic."""
    x = torch.tensor([[0.0, 0.0], [1.0, 0.2], [0.3, 0.9]], dtype=torch.float64)
    mesh = SimplicialMesh(x, [[2, 0, 1]])
    f = (x[:, 0] ** 2 - x[:, 0] * x[:, 1] + 2 * x[:, 1])[:, None]
    g = torch.stack([2 * x[:, 0] - x[:, 1], -x[:, 0] + 2], dim=1)[:, :, None]
    model = hermite_interpolant(mesh, f, gradients=g, clamp=False)
    points = _random_points_in_cells(mesh, 20)
    p = _global_positions(mesh, points)
    torch.testing.assert_close(model(points)[:, 0], p[:, 0] ** 2 - p[:, 0] * p[:, 1] + 2 * p[:, 1], atol=1e-12, rtol=0)


@pytest.mark.parametrize("dim, n", MESHES)
@pytest.mark.parametrize("clamp", [False, True])
def test_interpolates_vertex_data(dim, n, clamp):
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    f = _kinked(x)
    model = hermite_interpolant(mesh, f, clamp=clamp)
    torch.testing.assert_close(model(mesh.locate(x)), f, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Continuity: shared faces share coefficient rows
# ---------------------------------------------------------------------------


def _control_keys(mesh: SimplicialMesh) -> dict[tuple[tuple[int, int], ...], set[int]]:
    """Map each control, named by its global vertices and multiplicities, to the rows it resolves to."""
    keys: dict[tuple[tuple[int, int], ...], set[int]] = {}
    for simplex, cell in zip(mesh.simplices.tolist(), mesh.cell_ids.tolist(), strict=True):
        for (alpha,) in _indices((mesh.dimension,), (3,)):
            key = tuple(sorted((v, a) for v, a in zip(simplex, alpha, strict=True) if a))
            keys.setdefault(key, set()).add(mesh.space.control_index(alpha, cell=cell))
    return keys


@pytest.mark.parametrize("dim, n", MESHES)
def test_shared_faces_share_coefficient_rows(dim, n):
    """Every control on a shared face resolves to one row from all incident simplices, and only there."""
    x, simplices = _kuhn_mesh(dim, n)
    mesh = SimplicialMesh(x, simplices)
    keys = _control_keys(mesh)
    assert all(len(rows) == 1 for rows in keys.values())
    assert len({next(iter(rows)) for rows in keys.values()}) == len(keys) == mesh.space.n_controls
    # Some controls are actually shared (the test is not vacuous).
    counts = Counter(tuple(sorted(f)) for s in mesh.simplices.tolist() for f in itertools.combinations(s, dim))
    assert max(counts.values()) == 2


def test_rows_match_explicit_identification():
    """The mesh space equals gluing triangles with ``identify`` on faces of matching vertex order."""
    x, simplices = _kuhn_mesh(2, 2)
    mesh = SimplicialMesh(x, simplices)
    domain = SimploidalSet()
    sorted_simplices = mesh.simplices.tolist()
    handles = [domain.add_simplex(2) for _ in sorted_simplices]
    for a, b in itertools.combinations(range(len(sorted_simplices)), 2):
        shared = set(sorted_simplices[a]) & set(sorted_simplices[b])
        if len(shared) == 2:
            # face(0, j) removes the j-th vertex; with ascending vertex numbers both faces list `shared` in order.
            ja = next(j for j, v in enumerate(sorted_simplices[a]) if v not in shared)
            jb = next(j for j, v in enumerate(sorted_simplices[b]) if v not in shared)
            domain.identify(handles[a].face(0, ja), handles[b].face(0, jb))
    space = BezierSpace.on_set(domain, degree=3)
    assert space.n_controls == mesh.space.n_controls
    rows: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    for simplex, handle, cell in zip(sorted_simplices, handles, mesh.cell_ids.tolist(), strict=True):
        for (alpha,) in _indices((2,), (3,)):
            key = tuple(sorted((v, a) for v, a in zip(simplex, alpha, strict=True) if a))
            pair = (space.control_index(alpha, cell=handle), mesh.space.control_index(alpha, cell=cell))
            assert rows.setdefault(key, pair) == pair


@pytest.mark.parametrize("dim, n", MESHES)
@pytest.mark.parametrize("clamp", [False, True])
def test_predictions_agree_across_shared_faces(dim, n, clamp):
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    model = hermite_interpolant(mesh, _kinked(x), clamp=clamp)
    gen = torch.Generator().manual_seed(1)
    rows = mesh.simplices.tolist()
    checked = 0
    for a, b in itertools.combinations(range(len(rows)), 2):
        shared = sorted(set(rows[a]) & set(rows[b]))
        if len(shared) != dim:
            continue
        w = torch.rand(4, dim, generator=gen, dtype=torch.float64)
        p = (w / w.sum(1, keepdim=True)) @ mesh.vertices[shared]
        lam = mesh.barycentric(p)
        left = model(Points(lam[:, a].clamp_min(0), mesh.cell_ids[a].repeat(4)))
        right = model(Points(lam[:, b].clamp_min(0), mesh.cell_ids[b].repeat(4)))
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, n", MESHES)
@pytest.mark.parametrize("seed", [0, 1])
def test_clamped_prediction_stays_in_cell_data_range(dim, n, seed):
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3, seed=seed)
    mesh = SimplicialMesh(x, simplices)
    gen = torch.Generator().manual_seed(seed)
    f = torch.cat([_kinked(x), torch.randn(len(x), 1, generator=gen, dtype=torch.float64)], dim=1)
    model = hermite_interpolant(mesh, f, clamp=True)
    per_cell = 30
    points = _random_points_in_cells(mesh, per_cell, seed)
    y = model(points).detach().reshape(mesh.n_simplices, per_cell, -1)
    corner = f[mesh.simplices]  # (m, d+1, M)
    assert (y >= corner.amin(dim=1, keepdim=True) - 1e-12).all()
    assert (y <= corner.amax(dim=1, keepdim=True) + 1e-12).all()


def test_unclamped_interpolant_can_overshoot():
    """Clamping is what bounds the prediction (the range test above is not vacuous)."""
    x = torch.arange(8, dtype=torch.float64)[:, None]
    f = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    mesh = SimplicialMesh(x, [[k, k + 1] for k in range(7)])
    t = torch.linspace(0, 7, 701, dtype=torch.float64)[:, None]
    loose = hermite_interpolant(mesh, f, clamp=False)(mesh.locate(t))
    tight = hermite_interpolant(mesh, f, clamp=True)(mesh.locate(t))
    assert loose.min() < -0.01 and loose.max() > 1.01
    assert tight.min() >= -1e-12 and tight.max() <= 1 + 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_monotone_data_give_monotone_interpolant_in_one_dimension(seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.rand(12, generator=gen, dtype=torch.float64) + 0.05, 0)[:, None]
    steps = torch.rand(12, 2, generator=gen, dtype=torch.float64) * (torch.rand(12, 2, generator=gen) > 0.4)
    f = torch.stack([torch.cumsum(steps[:, 0], 0), -torch.cumsum(steps[:, 1], 0)], dim=1)
    order = torch.randperm(11, generator=gen)
    mesh = SimplicialMesh(x, torch.stack([order, order + 1], dim=1))
    t = torch.linspace(float(x[0]), float(x[-1]), 2001, dtype=torch.float64)[:, None]
    y = hermite_interpolant(mesh, f, clamp=True)(mesh.locate(t)).detach()
    assert (y[:, 0].diff() >= -1e-12).all()
    assert (y[:, 1].diff() <= 1e-12).all()


# ---------------------------------------------------------------------------
# Point location
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, n", MESHES)
def test_locate_interior_points(dim, n):
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    expected = _random_points_in_cells(mesh, 3)
    found = mesh.locate(_global_positions(mesh, expected))
    assert torch.equal(found.cell_ids, expected.cell_ids)
    torch.testing.assert_close(found.parameters, expected.parameters, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dim, n", MESHES)
def test_locate_boundary_points(dim, n):
    """Vertices, points on shared faces and on the hull get a containing cell and consistent coordinates."""
    x, simplices = _kuhn_mesh(dim, n, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    gen = torch.Generator().manual_seed(0)
    faces = [mesh.vertices[list(f)] for s in mesh.simplices.tolist() for f in itertools.combinations(s, dim)]
    w = torch.rand(len(faces), dim, generator=gen, dtype=torch.float64)
    on_faces = torch.einsum("fk,fkd->fd", w / w.sum(1, keepdim=True), torch.stack(faces))
    hull = torch.rand(10, dim, generator=gen, dtype=torch.float64)
    hull[:, 0] = 1.0
    p = torch.cat([x, on_faces, hull])
    found = mesh.locate(p)
    assert (found.parameters >= 0).all()
    torch.testing.assert_close(_global_positions(mesh, found), p, atol=1e-12, rtol=0)
    mesh.space.validate(found)


def test_locate_tolerance_and_outside_points():
    mesh = SimplicialMesh([[0.0], [1.0], [2.0]], [[0, 1], [1, 2]])
    found = mesh.locate(torch.tensor([[-1e-10], [2.0 + 1e-10]]))
    torch.testing.assert_close(_global_positions(mesh, found), torch.tensor([[0.0], [2.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="outside"):
        mesh.locate(torch.tensor([[2.5]]))
    with pytest.raises(ValueError, match="outside"):
        mesh.locate(torch.tensor([[2.0 + 1e-6]]), tol=1e-9)
    small = mesh.locate(torch.linspace(0, 2, 7)[:, None], batch_size=2)
    assert len(small) == 7


# ---------------------------------------------------------------------------
# Compatibility with the rest of the API
# ---------------------------------------------------------------------------


def test_model_is_a_bezier_model_that_can_be_subdivided():
    x, simplices = _kuhn_mesh(2, 3, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    model = hermite_interpolant(mesh, _kinked(x))
    assert isinstance(model, BezierModel)
    assert model.space is mesh.space and model.n_outputs == 2
    cell = int(mesh.cell_ids[4])
    result = subdivide(model, cell=cell)
    lam = _random_points_in_cells(mesh, 6).parameters[:6]
    for child, mapping in zip(result.models, result.maps, strict=True):
        torch.testing.assert_close(child(Points(lam)), model(mapping.to_parent(Points(lam))), atol=1e-12, rtol=0)


def test_model_is_differentiable_in_points():
    x, simplices = _kuhn_mesh(2, 2)
    mesh = SimplicialMesh(x, simplices)
    model = hermite_interpolant(mesh, _quadratic(x), clamp=False)
    points = mesh.locate(torch.tensor([[0.3, 0.6]], dtype=torch.float64))
    t = points.parameters.clone().requires_grad_(True)
    model(Points(t, points.cell_ids)).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()


def test_mesh_properties():
    x, simplices = _kuhn_mesh(3, 2)
    mesh = SimplicialMesh(x, simplices)
    assert (mesh.dimension, mesh.n_vertices, mesh.n_simplices) == (3, 27, 48)
    torch.testing.assert_close(mesh.volumes().sum(), torch.tensor(1.0, dtype=torch.float64))
    assert (mesh.simplices.diff(dim=1) > 0).all()
    assert len(set(mesh.cell_ids.tolist())) == mesh.n_simplices
    assert set(mesh.cell_ids.tolist()) == set(mesh.space.main_cells)
    assert "n_simplices=48" in repr(mesh)
    assert mesh.neighbors()[0] == [1, 3, 4, 9, 10, 12, 13]


def test_refinement_indicator():
    x, simplices = _kuhn_mesh(2, 4, jitter=0.3)
    mesh = SimplicialMesh(x, simplices)
    score = refinement_indicator(mesh, _kinked(x))
    assert score.shape == (mesh.n_simplices,) and (score >= 0).all() and score.max() > 0
    torch.testing.assert_close(refinement_indicator(mesh, _linear(x)), torch.zeros(mesh.n_simplices, dtype=torch.float64))
    q = _quadratic(x)
    unclamped = refinement_indicator(mesh, q, clamp=False, gradients=estimate_gradients(mesh, q))
    torch.testing.assert_close(unclamped, refinement_indicator(mesh, q, clamp=False))


def test_delaunay_mesh():
    """A Delaunay triangulation from the optional SciPy dependency can be passed directly."""
    spatial = pytest.importorskip("scipy.spatial")
    gen = torch.Generator().manual_seed(0)
    x = torch.cat([torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), torch.rand(40, 2, generator=gen)]).double()
    mesh = SimplicialMesh(x, spatial.Delaunay(x.numpy()).simplices)
    model = hermite_interpolant(mesh, _quadratic(x), clamp=False)
    p = torch.rand(100, 2, generator=gen, dtype=torch.float64)
    torch.testing.assert_close(model(mesh.locate(p)), _quadratic(p), atol=1e-11, rtol=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vertices, simplices, message",
    [
        ([0.0, 1.0], [[0, 1]], "vertices"),
        ([[0.0], [float("nan")]], [[0, 1]], "vertices"),
        ([[0.0], [1.0]], [[0.0, 1.0]], "integer"),
        ([[0.0], [1.0]], [[0, 1, 1]], "shape"),
        ([[0.0], [1.0]], [[0, 2]], "does not exist"),
        ([[0.0], [1.0]], [[0, 0]], "repeats"),
        ([[0.0], [1.0]], [[0, 1], [1, 0]], "distinct"),
        ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[0, 1, 2]], "degenerate"),
    ],
)
def test_invalid_mesh(vertices, simplices, message):
    with pytest.raises(ValueError, match=message):
        SimplicialMesh(vertices, simplices)


def test_invalid_values_and_gradients():
    mesh = SimplicialMesh([[0.0], [1.0], [2.0]], [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="values"):
        hermite_interpolant(mesh, torch.zeros(2))
    with pytest.raises(ValueError, match="values"):
        hermite_interpolant(mesh, torch.tensor([0.0, float("inf"), 1.0]))
    with pytest.raises(ValueError, match="gradients"):
        hermite_interpolant(mesh, torch.zeros(3), gradients=torch.zeros(3, 1))
    with pytest.raises(ValueError, match="max_rings"):
        estimate_gradients(mesh, torch.zeros(3), max_rings=1)
    with pytest.raises(ValueError, match="points"):
        mesh.locate(torch.zeros(3, 2))


def test_list_inputs_are_read_as_float64():
    """Regression test for #431: Python floats must not be rounded to float32 on the way in."""
    vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.2]]
    mesh = SimplicialMesh(vertices, [[0, 1, 2]])
    assert mesh.vertices[2, 1].item() == 0.2
    points = [[0.5, 0.1]]
    exact = SimplicialMesh(torch.tensor(vertices, dtype=torch.float64), [[0, 1, 2]])
    expected = exact.locate(torch.tensor(points, dtype=torch.float64))
    assert torch.equal(mesh.locate(points).parameters, expected.parameters)
    assert torch.equal(mesh.barycentric(points), exact.barycentric(torch.tensor(points, dtype=torch.float64)))
    values = [0.1, 0.2, 0.3]
    gradients = [[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.7]]]
    model = hermite_interpolant(mesh, values, gradients=gradients, clamp=False)
    reference = hermite_interpolant(
        exact,
        torch.tensor(values, dtype=torch.float64),
        gradients=torch.tensor(gradients, dtype=torch.float64),
        clamp=False,
    )
    assert torch.equal(model.coefficients, reference.coefficients)


def test_complex_inputs_are_rejected():
    with pytest.raises(ValueError, match="vertices"):
        SimplicialMesh([[0.0], [1j]], [[0, 1]])
    mesh = SimplicialMesh([[0.0], [1.0]], [[0, 1]])
    with pytest.raises(ValueError, match="points"):
        mesh.locate([[0.5j]])
    with pytest.raises(ValueError, match="values"):
        hermite_interpolant(mesh, [0.0, 1j])
    with pytest.raises(ValueError, match="gradients"):
        hermite_interpolant(mesh, [0.0, 1.0], gradients=[[[0.0]], [[1j]]])


# ---------------------------------------------------------------------------
# Properties (Hypothesis)
# ---------------------------------------------------------------------------


def _barycentric_tolerance(mesh: SimplicialMesh, offset: float) -> float:
    r"""Bound on the change of barycentric coordinates when every coordinate is rounded at magnitude ``offset + 2``.

    Rounding moves each vertex and point coordinate by at most :math:`u(|t| + 2)` (the unit cube is translated by
    :math:`t`), so the right-hand side :math:`x - x_0` and the matrix of edges :math:`A` of every simplex move by at
    most :math:`2u(|t| + 2)` per entry. The coordinates :math:`A^{-1}(x - x_0)` (with :math:`\|x - x_0\| \le` the
    diameter, at most :math:`\sqrt d`) then move by at most :math:`\|A^{-1}\|_2 (d + 1)^2 \cdot 2u(|t| + 2)` in
    each component, and the remaining coordinate by :math:`d` times that. :math:`\|A^{-1}\|_2 = 1 / \sigma_{\min}(A)`
    is the condition of the location problem. The bound is doubled for the rounding inside the solve.
    """
    d = mesh.dimension
    edges = mesh.vertices[mesh.simplices[:, 1:]] - mesh.vertices[mesh.simplices[:, :1]]
    sigma = torch.linalg.svdvals(edges).min()
    return float(4 * d * (d + 1) ** 2 * U * (offset + 2) / sigma)


def _check_same_location(mesh: SimplicialMesh, other: SimplicialMesh, p: torch.Tensor, q: torch.Tensor, tol: float):
    """``other`` locates ``q`` as ``mesh`` locates ``p``, up to ``tol``, wherever that is decidable at ``tol``."""
    lam, moved = mesh.barycentric(p), other.barycentric(q)
    torch.testing.assert_close(moved, lam, atol=tol, rtol=0)
    margin = lam.min(dim=2).values.max(dim=1).values
    clear = margin > 2 * tol  # the cell is decided by more than the rounding error
    found, refound = mesh.locate(p[clear]), other.locate(q[clear])
    assert torch.equal(found.cell_ids, refound.cell_ids)
    torch.testing.assert_close(refound.parameters, found.parameters, atol=4 * tol, rtol=0)


@given(data=st.data())
def test_mesh_is_translation_invariant(data):
    """Translating vertices and points by up to 1e12 changes neither validity nor location beyond rounding."""
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    _, _, p = data.draw(points_in_cells(x, simplices, 8), label="points")
    t = data.draw(translations(x.shape[1]), label="t")
    mesh, moved = SimplicialMesh(x, simplices), SimplicialMesh(x + t, simplices)
    _check_same_location(mesh, moved, p, p + t, _barycentric_tolerance(mesh, float(t.abs().max())))


@given(data=st.data())
def test_mesh_is_exactly_translation_invariant_for_exact_translations(data):
    """When the translated coordinates are exact, so is everything computed from them.

    Vertices are multiples of 2**-10 in [0, 1], points multiples of 2**-20, and translations multiples of 2**-10 up
    to 2**32, so every translated coordinate fits in 53 bits; differences of coordinates are then exact.
    """
    x, simplices = data.draw(kuhn_meshes(quantum=2.0**-10), label="mesh")
    _, _, p = data.draw(points_in_cells(x, simplices, 8), label="points")
    p = torch.round(p * 2.0**20) / 2.0**20
    t = torch.round(data.draw(translations(x.shape[1], max_magnitude=2.0**32), label="t") * 2.0**10) / 2.0**10
    mesh, moved = SimplicialMesh(x, simplices), SimplicialMesh(x + t, simplices)
    assert torch.equal((x + t) - t, x) and torch.equal((p + t) - t, p)
    assert torch.equal(moved.barycentric(p + t), mesh.barycentric(p))


@given(data=st.data())
def test_mesh_is_scale_invariant(data):
    """Scaling each axis by its own factor from 1e-6 to 1e6 changes neither validity nor location beyond rounding."""
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    scale = data.draw(axis_scalings(x.shape[1], scalings()), label="scale")
    _, _, p = data.draw(points_in_cells(x, simplices, 8), label="points")
    mesh, scaled = SimplicialMesh(x, simplices), SimplicialMesh(x * scale, simplices)
    # Scaling rounds each coordinate once more, relative to its magnitude, i.e. by at most 2u in the unscaled units
    # of its axis: offset 0 bounds that as well.
    _check_same_location(mesh, scaled, p, p * scale, 2 * _barycentric_tolerance(mesh, 0.0))


def _times_power_of_two(x: float, exponent: int) -> float:
    """``x * 2**exponent``, correctly rounded, or infinity if it overflows."""
    try:
        return math.ldexp(x, exponent)
    except OverflowError:
        return math.inf


@given(data=st.data())
def test_mesh_is_exactly_invariant_under_power_of_two_axis_scalings(data):
    """Scaling each axis by its own power of two up to 2**±1000 (1e±301) scales every result exactly.

    Barycentric coordinates, cells and interpolants do not change, gradients are divided by the scales, and volumes
    and refinement scores are multiplied by their product, rounded once (or inf when it overflows). Axes then differ
    in scale by up to 2**2000, which overflows any determinant, norm or product of coordinates computed directly.
    Points lie on a grid of 2**-30, so that their scaled coordinates are exact as well.
    """
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    scale = data.draw(axis_scalings(x.shape[1]), label="scale")
    _, _, p = data.draw(points_in_cells(x, simplices, 8), label="points")
    p = torch.round(p * 2.0**30) / 2.0**30
    f = data.draw(matrices(len(x), 2), label="f")
    mesh, scaled = SimplicialMesh(x, simplices), SimplicialMesh(x * scale, simplices)
    assert torch.equal(scaled.barycentric(p * scale), mesh.barycentric(p))
    found, refound = mesh.locate(p), scaled.locate(p * scale)
    assert torch.equal(refound.cell_ids, found.cell_ids) and torch.equal(refound.parameters, found.parameters)
    # Both fits are the same up to the final power of two, applied with one rounding. A gradient that is subnormal
    # in one of the meshes loses bits there, so it is compared after scaling the other one onto it.
    g, g_scaled = estimate_gradients(mesh, f), estimate_gradients(scaled, f)
    s = scale[None, :, None]
    assert ((g_scaled * s == g) | (g / s == g_scaled)).all()
    # The edge controls f_i + g . (x_j - x_i) / 3 multiply each gradient component by an offset scaled inversely, which
    # is exact except for that subnormal ulp of g / scale times an offset of at most 2**1000: 2**-74 in all.
    torch.testing.assert_close(
        hermite_interpolant(scaled, f, gradients=g_scaled).coefficients,
        hermite_interpolant(mesh, f).coefficients,
        rtol=2 * U,
        atol=2.0**-70,
    )
    exponent = int(torch.log2(scale).sum())
    for actual, volume in zip(scaled.volumes().tolist(), mesh.volumes().tolist(), strict=True):
        expected = _times_power_of_two(volume, exponent)
        if expected == math.inf or expected >= FLOAT64_TINY:
            assert actual == expected
        else:  # a subnormal result, rounded twice
            assert abs(actual - expected) <= 2.0**-1074
    # A score is a correction, which may differ by the 2**-70 of the controls above, times the volume.
    expected_scores = refinement_indicator(mesh, f).tolist()
    volumes = scaled.volumes().tolist()
    for actual, score, volume in zip(refinement_indicator(scaled, f).tolist(), expected_scores, volumes, strict=True):
        expected = _times_power_of_two(score, exponent)
        assert actual == expected or abs(actual - expected) <= 2 * U * abs(expected) + 2.0**-70 * volume + 2.0**-1073


@given(data=st.data(), scale=extreme_scalings())
def test_mesh_accepts_coordinates_near_the_float64_limits(data, scale):
    """A mesh scaled by a power of two up to 2**±1000 (1e±301) is valid and locates points as the unscaled one."""
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    _, _, p = data.draw(points_in_cells(x, simplices, 8), label="points")
    mesh, scaled = SimplicialMesh(x, simplices), SimplicialMesh(x * scale, simplices)
    _check_same_location(mesh, scaled, p, p * scale, _barycentric_tolerance(mesh, 0.0))
    torch.testing.assert_close(estimate_gradients(scaled, x.sum(dim=1)) * scale, estimate_gradients(mesh, x.sum(dim=1)))


@given(data=st.data())
def test_locate_raises_or_returns_barycentric_coordinates(data):
    """For any finite point, even one whose offset from the mesh overflows, locate raises or returns valid coordinates."""
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    scale = data.draw(extreme_scalings() | scalings(), label="scale")
    mesh = SimplicialMesh(x * scale, simplices)
    coordinate = st.just(0.0) | signed(log_uniform(FLOAT64_TINY, FLOAT64_MAX))
    q = torch.tensor(data.draw(st.lists(coordinate, min_size=mesh.dimension, max_size=mesh.dimension)), dtype=torch.float64)
    try:
        found = mesh.locate(q[None])
    except ValueError as error:
        assert "outside" in str(error)
        return
    t = found.parameters
    assert torch.isfinite(t).all() and (t >= 0).all()
    torch.testing.assert_close(t.sum(dim=1), torch.ones(1, dtype=torch.float64))


def _range_tolerance(mesh: SimplicialMesh, magnitude: torch.Tensor) -> torch.Tensor:
    r"""Bound on how far a clamped prediction may leave the range of its cell data of magnitude ``magnitude``.

    The controls of a cell lie in that range exactly (they are clipped), and the prediction is a convex combination
    of them computed with the error bound of ``_evaluation_tolerance`` in ``tests/test_core.py``:
    :math:`2(N + D + 3w)u` times the largest control, with :math:`N` controls of degree :math:`D = 3` in :math:`w` coordinates.
    """
    w = mesh.dimension + 1
    n = len(_indices((mesh.dimension,), (3,)))
    return 2 * (n + 3 + 3 * w) * U * magnitude + 2 * n * FLOAT64_TINY


@given(data=st.data(), scale=scalings() | extreme_scalings(), offset=st.just(0.0) | signed(log_uniform(1e-6, 1e12)))
def test_clamped_interpolant_preserves_the_range_of_cell_data(data, scale, offset):
    """At any point of a cell, the clamped prediction lies between the smallest and largest data of the cell.

    The data are scaled by up to 2**±1000 and offset by up to 1e12, so that a cell may have nearly equal data.
    """
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    mesh = SimplicialMesh(x, simplices)
    f = data.draw(matrices(mesh.n_vertices, 2), label="f") * scale + offset
    rows, lam, _ = data.draw(points_in_cells(x, simplices, 8), label="points")
    model = hermite_interpolant(mesh, f, clamp=True)
    with torch.no_grad():
        y = model(Points(lam, mesh.cell_ids[rows]))
    corner = f[mesh.simplices[rows]]  # (n, d + 1, M)
    lo, hi = corner.amin(dim=1), corner.amax(dim=1)
    tol = _range_tolerance(mesh, torch.maximum(lo.abs(), hi.abs()))
    assert (y >= lo - tol).all() and (y <= hi + tol).all()


@given(data=st.data(), scale=scalings() | extreme_scalings(), clamp=st.booleans())
def test_interpolant_is_continuous_across_shared_faces(data, scale, clamp):
    """At any point of a face shared by two cells, the predictions of both cells agree up to rounding."""
    x, simplices = data.draw(kuhn_meshes(dimensions=(2, 3)), label="mesh")
    mesh = SimplicialMesh(x, simplices)
    model = hermite_interpolant(mesh, data.draw(matrices(mesh.n_vertices, 2), label="f") * scale, clamp=clamp)
    rows = mesh.simplices.tolist()
    pairs = [
        (a, b) for a, b in itertools.combinations(range(len(rows)), 2) if len(set(rows[a]) & set(rows[b])) == mesh.dimension
    ]
    a, b = data.draw(st.sampled_from(pairs), label="cells")
    shared = sorted(set(rows[a]) & set(rows[b]))
    w = data.draw(simplex_points(len(shared), 4), label="w")

    def on_cell(k: int) -> Points:
        lam = torch.zeros(len(w), mesh.dimension + 1, dtype=torch.float64)
        lam[:, [rows[k].index(v) for v in shared]] = w
        return Points(lam, mesh.cell_ids[k].repeat(len(w)))

    with torch.no_grad():
        left, right = model(on_cell(a)), model(on_cell(b))
    # The controls on the face are shared rows, the other controls get zero weight, and each side errs by at most
    # the bound of _range_tolerance relative to the largest coefficient.
    tol = 2 * _range_tolerance(mesh, model.coefficients.detach().abs().max())
    assert (left - right).abs().max() <= tol


@given(data=st.data(), bad=st.sampled_from([float("nan"), float("inf"), -float("inf")]))
def test_non_finite_mesh_inputs_are_rejected(data, bad):
    """A single NaN or infinity in vertices, points, values or gradients raises ValueError."""
    x, simplices = data.draw(kuhn_meshes(), label="mesh")
    mesh = SimplicialMesh(x, simplices)
    _, _, p = data.draw(points_in_cells(x, simplices, 3), label="points")
    f = data.draw(matrices(mesh.n_vertices, 2), label="f")
    g = torch.zeros(mesh.n_vertices, mesh.dimension, 2, dtype=torch.float64)

    def broken(t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        t.view(-1)[data.draw(st.integers(0, t.numel() - 1))] = bad
        return t

    with pytest.raises(ValueError, match="vertices"):
        SimplicialMesh(broken(x), simplices)
    with pytest.raises(ValueError, match="points"):
        mesh.locate(broken(p))
    with pytest.raises(ValueError, match="values"):
        hermite_interpolant(mesh, broken(f))
    with pytest.raises(ValueError, match="gradients"):
        hermite_interpolant(mesh, f, gradients=broken(g))


# Regression tests for the counterexamples found by the properties above.


@pytest.mark.parametrize("scale", [2.0**-1000, 2.0**-600, 2.0**600, 2.0**1000])
def test_mesh_far_from_unit_scale_is_not_degenerate(scale):
    """The determinant of the edges overflowed (or underflowed) to inf (or 0), so the triangle was called degenerate."""
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    mesh = SimplicialMesh(x * scale, [[0, 1, 2]])
    p = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    assert mesh.locate(p * scale).parameters.tolist() == [[0.25, 0.25, 0.5]]


def test_estimated_gradients_at_large_coordinates():
    """Squared distances overflowed beyond coordinates of 1e154, so that every weight and the gradient became zero."""
    x = torch.linspace(0, 1, 6, dtype=torch.float64)[:, None]
    scale = 2.0**1000
    mesh = SimplicialMesh(x * scale, [[k, k + 1] for k in range(5)])
    g = estimate_gradients(mesh, x.square())
    torch.testing.assert_close(g[:, 0, 0] * scale, 2 * x[:, 0])


def test_locate_rejects_a_point_whose_coordinates_overflow():
    """The barycentric coordinates of this point overflow to -inf and inf, and their sum to NaN.

    The NaN passed the outside test (NaN < -tol is false), and locate returned NaN coordinates.
    """
    mesh = SimplicialMesh([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="outside"):
        mesh.locate([[0.0, FLOAT64_MAX]])


def test_mesh_rejects_vertices_whose_differences_overflow():
    with pytest.raises(ValueError, match="overflow"):
        SimplicialMesh([[-FLOAT64_MAX], [FLOAT64_MAX]], [[0, 1]])


def _hadamard_ratio_squared(rows: list[list[Fraction]]) -> Fraction:
    """``(det / product of row lengths)**2`` of an upper triangular matrix, exactly."""
    det = math.prod((rows[k][k] for k in range(len(rows))), start=Fraction(1))
    return det**2 / math.prod((sum((v**2 for v in row), start=Fraction(0)) for row in rows), start=Fraction(1))


@given(data=st.data())
def test_simplex_with_edges_and_axes_of_extreme_scales(data: st.DataObject):
    """Edges diag(2**a) T diag(2**b), T upper triangular with integer entries, against exact rational arithmetic.

    Edges and axes then differ in scale by up to 2**2000, as in the triangle [0, 0], [1e308, 1e308],
    [1e-300, -1e-300]. A simplex is degenerate when its volume is at most 1e-12 times the product of its edge
    lengths both as given (Euclidean) and with each axis divided by the power of two 2**floor(log2(extent))
    below twice its extent over the simplex. Ratios within a factor of two of 1e-12 may go either way, as the
    determinant is rounded. Otherwise the volume must have a relative error of at most 2 (d**2 + 1) U: the
    equilibrated edges stay triangular, so LU factorization perturbs only their diagonal, each entry by at most
    gamma_d ~ d U, and dividing by d! rounds once (plus the subnormal spacing 2**-1074 for a subnormal volume).
    """
    d = data.draw(st.integers(1, 3), label="d")
    t = [[0] * d for _ in range(d)]
    for k in range(d):
        t[k][k] = data.draw(st.integers(1, 2**10), label="diagonal")
        for j in range(k + 1, d):
            t[k][j] = data.draw(st.integers(-(2**10), 2**10), label="off-diagonal")
    a = data.draw(st.lists(st.integers(-1000, 1000), min_size=d, max_size=d), label="edge exponents")
    b = data.draw(st.lists(st.integers(-1000, 1000), min_size=d, max_size=d), label="axis exponents")
    exact = [[Fraction(t[k][j]) * Fraction(2) ** (a[k] + b[j]) for j in range(d)] for k in range(d)]
    # Only simplices whose edges are float64 values are drawn, so that T gives the exact simplex.
    if any(v != 0 and not (FLOAT64_TINY <= abs(v) <= FLOAT64_MAX) for row in exact for v in row):
        return
    vertices = torch.tensor([[0.0] * d] + [[float(v) for v in row] for row in exact], dtype=torch.float64)
    axis = [Fraction(2) ** math.frexp(float(max(abs(row[j]) for row in exact)))[1] for j in range(d)]
    normalized = [[v / axis[j] for j, v in enumerate(row)] for row in exact]
    ratio = max(_hadamard_ratio_squared(exact), _hadamard_ratio_squared(normalized))
    if ratio <= Fraction(1, 2 * 10**12) ** 2:
        with pytest.raises(ValueError, match="degenerate"):
            SimplicialMesh(vertices, [list(range(d + 1))])
        return
    if ratio < Fraction(2, 10**12) ** 2:
        return
    mesh = SimplicialMesh(vertices, [list(range(d + 1))])
    volume = abs(math.prod((exact[k][k] for k in range(d)), start=Fraction(1))) / math.factorial(d)
    actual = mesh.volumes().item()
    if volume > Fraction(FLOAT64_MAX):
        assert actual >= FLOAT64_MAX * (1 - 2 * (d**2 + 1) * U)
    else:
        assert abs(Fraction(actual) - volume) <= 2 * (d**2 + 1) * Fraction(U) * volume + Fraction(2) ** -1074
    # Linear data that varies by about one along every axis of the simplex, i.e. with gradient 1 / axis. Least
    # squares is accurate relative to the largest variation, so a gradient component whose variation is far
    # smaller could be lost in rounding, and its value relative to a tiny axis could then overflow.
    f = torch.tensor([0.0] + [float(sum(normalized[k])) for k in range(d)], dtype=torch.float64)
    assert torch.isfinite(estimate_gradients(mesh, f)).all()
    assert not refinement_indicator(mesh, f).isnan().any()


def test_mesh_with_edges_of_very_different_lengths():
    """Regression test (Codex review of #440): for the triangle [0, 0], [1e308, 1e308], [1e-300, -1e-300] (volume
    1e8), dividing the edges by the largest entry of each axis first made the short edge 0, and the constructor
    said "degenerate". The gradient of linear data also lost the short edge and was 0 at the vertices it joins.
    """
    mesh = SimplicialMesh([[0.0, 0.0], [1e308, 1e308], [1e-300, -1e-300]], [[0, 1, 2]])
    torch.testing.assert_close(mesh.volumes(), torch.tensor([1e8], dtype=torch.float64), rtol=4 * U, atol=0)
    torch.testing.assert_close(
        mesh.locate([[1e-301, 0.0]]).parameters, torch.tensor([[0.95, 0.0, 0.05]], dtype=torch.float64), rtol=0, atol=8 * U
    )
    # f = x - y. At vertex 1, the offsets to vertices 0 and 2 round to the same float64 vector, so its gradient
    # cannot be determined; the other two vertices see both edges.
    g = estimate_gradients(mesh, [0.0, 0.0, 2e-300])
    torch.testing.assert_close(g[[0, 2], :, 0], torch.tensor([[1.0, -1.0]] * 2, dtype=torch.float64), rtol=8 * U, atol=0)


def test_volume_of_an_anisotropic_simplex():
    """Regression test (Codex review of #440): edges diag(1e200, 1e-100) gave a volume of inf instead of 5e99.

    The largest entry of the edges, 1e200, was squared before multiplying by the determinant of the scaled edges.
    """
    mesh = SimplicialMesh([[0.0, 0.0], [1e200, 0.0], [0.0, 1e-100]], [[0, 1, 2]])
    torch.testing.assert_close(mesh.volumes(), torch.tensor([5e99], dtype=torch.float64), rtol=4 * U, atol=0)
    assert torch.isfinite(refinement_indicator(mesh, [0.0, 1.0, 4.0])).all()
    assert mesh.locate([[2.5e199, 2.5e-101]]).parameters.tolist() == [[0.5, 0.25, 0.25]]


def test_gradients_of_axes_of_very_different_scales():
    """With axes of scales 1e200 and 1e-200, Euclidean weights differed by more than 1e308 and underflowed."""
    x, simplices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64), [[0, 1, 2], [1, 2, 3]]
    scale = torch.tensor([1e200, 1e-200], dtype=torch.float64)
    f = x[:, :1] - 2 * x[:, 1:]  # linear in the unscaled coordinates
    g = estimate_gradients(SimplicialMesh(x * scale, simplices), f)
    torch.testing.assert_close(g[:, :, 0] * scale, torch.tensor([[1.0, -2.0]] * 4, dtype=torch.float64))


def test_refinement_score_without_correction_on_a_huge_simplex_is_zero():
    """A volume beyond the float64 range is inf, and 0 * inf gave a NaN score to a simplex with linear data."""
    mesh = SimplicialMesh([[0.0, 0.0], [2.0**1000, 0.0], [0.0, 2.0**25]], [[0, 1, 2]])
    assert mesh.volumes().item() == math.inf
    assert refinement_indicator(mesh, [1.0, 1.0, 1.0]).tolist() == [0.0]


def test_refinement_score_of_a_tiny_correction_on_a_huge_thin_simplex():
    """Regression test (Codex review of #440): a subnormal correction times the small volume mantissa of a thin
    simplex underflowed to 0 before the volume exponent scaled it up, so a curved cell got a zero score.

    The triangle [0, 0], [1, 1], [1, 1 + 2**-30] with its x axis scaled by 2**1000, and data and gradients scaled
    by 2**-1060 (all exact), must score that of the unscaled triangle times 2**(1000 - 1060). The correction is
    subnormal, about 2**-1095, so it keeps only about 12 bits: relative tolerance 2**-10.
    """
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0 + 2.0**-30]], dtype=torch.float64)
    g = torch.tensor([[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [1.0]]], dtype=torch.float64)
    expected = refinement_indicator(SimplicialMesh(x, [[0, 1, 2]]), [0.0, 0.0, 1.0], gradients=g).item()
    scale = torch.tensor([2.0**1000, 1.0], dtype=torch.float64)
    huge = SimplicialMesh(x * scale, [[0, 1, 2]])
    actual = refinement_indicator(huge, [0.0, 0.0, 2.0**-1060], gradients=g * 2.0**-1060).item()
    assert expected > 0
    assert math.isclose(actual, math.ldexp(expected, -60), rel_tol=2.0**-10)


def test_refinement_score_of_a_simplex_whose_volume_overflows():
    """The score is the correction times the volume, which may be finite although the volume itself is not."""
    big = SimplicialMesh([[0.0, 0.0], [2.0**1000, 0.0], [0.0, 2.0**25]], [[0, 1, 2]])
    unit = SimplicialMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    assert big.volumes().item() == math.inf
    expected = refinement_indicator(unit, [0.0, 0.0, 1.0], gradients=[[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [3.0]]])
    actual = refinement_indicator(big, [0.0, 0.0, 1.0], gradients=[[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [3.0 * 2.0**-25]]])
    assert 0 < expected.item() < 0.5 and actual.tolist() == [math.ldexp(expected.item(), 1025)]
