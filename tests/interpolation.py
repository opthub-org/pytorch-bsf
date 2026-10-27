"""Tests for torch_bsf.interpolation (shape-preserving cubic interpolation on simplicial complexes)."""

import itertools
from collections import Counter

import pytest
import torch

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
