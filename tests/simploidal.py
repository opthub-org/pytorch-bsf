from math import comb, factorial, prod
from pathlib import Path

import pytest
import torch

import torch_bsf
from torch_bsf import fit
from torch_bsf.simploidal import (
    BezierSimploid,
    Simploid,
    SimploidalBezierSpace,
    SimploidalSet,
    bernstein_basis,
    fit_simploidal_set,
    sample_simploid,
    simploid_indices,
    simploid_n_params,
)

TYPES = [(), (1,), (2,), (3,), (1, 1), (2, 1), (1, 2), (1, 1, 1), (2, 2)]
TRAINER_KWARGS = dict(max_epochs=5, enable_progress_bar=False, logger=False, enable_checkpointing=False)


def _face_param(simploid_type: tuple[int, ...], u: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Embeds parameters of face (i, j) into the parameters of the simploid."""
    blocks, offset, k = [], 0, 0
    for p, a in enumerate(simploid_type):
        if p == i:
            if a == 1:
                blocks.append(torch.tensor([[0.0, 1.0] if j == 0 else [1.0, 0.0]]).expand(len(u), 2))
                continue
            block = u[:, offset : offset + a]
            blocks.append(torch.cat([block[:, :j], torch.zeros(len(u), 1), block[:, j:]], dim=1))
            offset += a
        else:
            blocks.append(u[:, offset : offset + a + 1])
            offset += a + 1
        k += 1
    return torch.cat(blocks, dim=1)


def _all_faces(simploid_type):
    return [(i, j) for i, a in enumerate(simploid_type) for j in range(a + 1)]


def _face_type(simploid_type, i):
    a = simploid_type[i]
    return simploid_type[:i] + ((a - 1,) if a > 1 else ()) + simploid_type[i + 1 :]


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simploid_type", TYPES)
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_simploid_indices_count(simploid_type, degree):
    indices = simploid_indices(simploid_type, degree)
    assert len(indices) == prod(comb(a + degree, a) for a in simploid_type)
    assert len(set(indices)) == len(indices)
    assert all(sum(m) == degree and len(m) == a + 1 for alpha in indices for m, a in zip(alpha, simploid_type))


def test_simploid_indices_invalid():
    with pytest.raises(ValueError):
        simploid_indices((0, 1), 2)
    with pytest.raises(ValueError):
        simploid_indices((1, 1), (1,))
    with pytest.raises(ValueError):
        simploid_indices((1,), -1)


@pytest.mark.parametrize("simploid_type", TYPES)
def test_bernstein_basis_partition_of_unity(simploid_type):
    t = sample_simploid(simploid_type, 20, seed=0)
    basis = bernstein_basis(simploid_type, 3, t)
    assert basis.shape == (20, len(simploid_indices(simploid_type, 3)))
    assert torch.allclose(basis.sum(dim=1), torch.ones(20))
    with pytest.raises(ValueError):
        bernstein_basis(simploid_type, 3, torch.zeros(2, simploid_n_params(simploid_type) + 1))


def test_sample_simploid():
    t = sample_simploid((2, 1), 10, seed=3)
    assert torch.equal(t, sample_simploid((2, 1), 10, seed=3))
    assert (t >= 0).all()
    assert torch.allclose(t[:, :3].sum(dim=1), torch.ones(10))
    assert torch.allclose(t[:, 3:].sum(dim=1), torch.ones(10))
    assert sample_simploid((), 4).shape == (4, 0)
    with pytest.raises(ValueError):
        sample_simploid((1,), -1)


# ---------------------------------------------------------------------------
# BezierSimploid
# ---------------------------------------------------------------------------


def _bezier_simplex(t: torch.Tensor, points: dict[tuple[int, ...], torch.Tensor], degree: int) -> torch.Tensor:
    """The Bézier simplex of the given control points, by the textbook formula."""
    out = 0
    for alpha, point in points.items():
        coeff = factorial(degree) / prod(factorial(a) for a in alpha)
        out = out + coeff * prod((t[:, i : i + 1] ** a for i, a in enumerate(alpha)), start=torch.ones(len(t), 1)) * point
    return out


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("degree", [0, 1, 2, 4])
def test_bezier_simploid_of_one_factor_is_bezier_simplex(dim, degree):
    bsd = BezierSimploid.randn((dim,), degree, n_values=3)
    t = sample_simploid((dim,), 15, seed=1)
    expected = _bezier_simplex(t, {alpha[0]: bsd[alpha].detach() for alpha in bsd.indices()}, degree)
    assert torch.allclose(bsd(t), expected, atol=1e-5)


def test_bezier_simploid_vertex_is_constant():
    bsd = BezierSimploid((), 3, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(bsd(torch.zeros(4, 0)), torch.tensor([[1.0, 2.0]] * 4))


def test_bezier_simploid_mixed_degrees():
    bsd = BezierSimploid.rand((2, 1), (1, 3), n_values=2)
    assert bsd.degree == (1, 3)
    assert len(bsd.indices()) == 3 * 4
    assert bsd(sample_simploid((2, 1), 5, seed=0)).shape == (5, 2)


def test_bezier_simploid_init_errors():
    with pytest.raises(TypeError):
        BezierSimploid((1,), 2)
    with pytest.raises(ValueError):
        BezierSimploid((1,), 1, {((1, 0),): [0.0]})
    with pytest.raises(ValueError):
        BezierSimploid((1,), 1, torch.zeros(3, 1))


def test_models_are_plain_modules():
    import lightning.pytorch as L

    S = SimploidalSet()
    S.add_simplex(2)
    for model in (BezierSimploid.zeros((2, 1), 2, n_values=3), SimploidalBezierSpace(S, 2, n_values=1)):
        assert isinstance(model, torch.nn.Module)
        assert not isinstance(model, L.LightningModule)


def test_bezier_simploid_state_dict_rebuild():
    bsd = BezierSimploid.randn((2, 1), 2, n_values=3)
    assert list(bsd.state_dict()) == ["control_points"]  # the cached basis is not persistent
    rebuilt = BezierSimploid(bsd.simploid_type, bsd.degree, n_values=bsd.n_values)
    rebuilt.load_state_dict(bsd.state_dict())
    t = sample_simploid((2, 1), 5, seed=0)
    assert torch.equal(rebuilt(t), bsd(t))


def test_bezier_simploid_forward_uses_cached_basis(monkeypatch):
    """forward rebuilds neither the multi-indices nor the multinomial coefficients."""
    import torch_bsf.simploidal.combinatorics as comb_module

    bsd = BezierSimploid.randn((2, 1), 3, n_values=2)
    t = sample_simploid((2, 1), 5, seed=0)
    expected = bernstein_basis((2, 1), 3, t) @ bsd.control_points

    def fail(*args, **kwargs):
        raise AssertionError("the basis tables must not be rebuilt")

    for name in ("_simplex_indices", "_multinomial", "_factor_tables", "_factor_tensors"):
        monkeypatch.setattr(comb_module, name, fail)
    assert torch.allclose(bsd(t), expected)


def test_bezier_simploid_cached_basis_follows_dtype():
    bsd = BezierSimploid.randn((2, 1), 2, n_values=2).double()
    t = sample_simploid((2, 1), 5, seed=0).double()
    assert bsd.design_matrix(t).dtype == torch.float64
    assert bsd(t).dtype == torch.float64
    assert torch.allclose(bsd(t), bernstein_basis((2, 1), 2, t) @ bsd.control_points)


def test_bezier_simploid_design_matrix_errors():
    bsd = BezierSimploid.randn((2, 1), 2, n_values=2)
    with pytest.raises(ValueError):
        bsd.design_matrix(torch.zeros(3, 4))
    assert bsd.design_matrix(torch.ones(1, 5, dtype=torch.long)).dtype == torch.get_default_dtype()


def test_space_design_matrix_matches_forward():
    S, mains = _glued_set()
    space = _random_space(S, 2, n_values=2)
    width = max(simploid_n_params(s.simploid_type) for s in mains)
    ts, ids = [], []
    for k, s in enumerate(mains):
        t = sample_simploid(s.simploid_type, 4, seed=k)
        ts.append(torch.nn.functional.pad(t, (0, width - t.shape[1])))
        ids.append(torch.full((4,), space.simploid_id(s)))
    t, sid = torch.cat(ts), torch.cat(ids)
    design = space.design_matrix(t, sid)
    assert design.shape == (len(t), space.n_proper_control_points)
    assert torch.allclose(design @ space.proper_control_points, space(t, sid), atol=1e-6)


def test_solvers_share_one_code_path(monkeypatch):
    """fit_simploid and fit_simploidal_set hand every model to the same solver function."""
    import torch_bsf.simploidal.fitting as fitting

    seen = []
    original = fitting._fit

    def spy(model, *args, **kwargs):
        seen.append(type(model))
        return original(model, *args, **kwargs)

    monkeypatch.setattr(fitting, "_fit", spy)
    t = sample_simploid((2,), 20, seed=0)
    fit(t, t[:, :1], 2, simploid_type=(2,))
    S = SimploidalSet()
    tri = S.add_simplex(2)
    fit_simploidal_set(S, [(tri, t, t[:, :1])], 2)
    assert seen == [BezierSimploid, SimploidalBezierSpace]


TRAINER_CONVERGENCE_KWARGS = dict(max_epochs=10, enable_progress_bar=False, logger=False, enable_checkpointing=False)


def test_fit_simploid_trainer_converges_to_lstsq():
    n = len(simploid_indices((2, 1), 2))
    # A fixed truth: L-BFGS in float32 stalls at an error of ~5e-4, so an unlucky random truth may exceed atol.
    truth = BezierSimploid((2, 1), 2, torch.randn(n, 2, generator=torch.Generator().manual_seed(0)))
    t = sample_simploid((2, 1), 200, seed=0)
    x = truth(t).detach() + 0.01 * torch.randn(200, 2, generator=torch.Generator().manual_seed(0))
    exact = fit(t, x, 2, simploid_type=(2, 1))
    trained = fit(t, x, 2, simploid_type=(2, 1), solver="trainer", seed=0, **TRAINER_CONVERGENCE_KWARGS)
    assert torch.allclose(trained.control_points, exact.control_points, atol=1e-3)


def test_fit_simploidal_set_trainer_converges_to_lstsq():
    S = SimploidalSet()
    tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    S.identify(tri.face(0, 2), sq.face(1, 1))
    data = []
    for k, s in enumerate([tri, sq]):
        t = sample_simploid(s.simploid_type, 100, seed=k)
        noise = 0.01 * torch.randn(100, 2, generator=torch.Generator().manual_seed(k))
        data.append((s, t, torch.sin(3 * t[:, :2]) + noise))
    exact = fit_simploidal_set(S, data, 2)
    trained = fit_simploidal_set(S, data, 2, solver="trainer", seed=0, **TRAINER_CONVERGENCE_KWARGS)
    assert torch.allclose(trained.proper_control_points, exact.proper_control_points, atol=1e-3)


@pytest.mark.parametrize("simploid_type", [t for t in TYPES if t])
def test_bezier_simploid_face_property(simploid_type):
    """Property 2-1: the restriction of a Bézier simploid to a face is the Bézier simploid of the face."""
    bsd = BezierSimploid.randn(simploid_type, 3, n_values=2)
    for i, j in _all_faces(simploid_type):
        face = bsd.face(i, j)
        assert face.simploid_type == _face_type(simploid_type, i)
        u = sample_simploid(face.simploid_type, 10, seed=i * 7 + j)
        assert torch.allclose(bsd(_face_param(simploid_type, u, i, j)), face(u), atol=1e-5)
    with pytest.raises(IndexError):
        bsd.face(len(simploid_type), 0)


@pytest.mark.parametrize("simploid_type", [t for t in TYPES if t])
def test_bezier_simploid_degenerate_property(simploid_type):
    """Property 2-2: a degenerate Bézier simploid has the same shape as the original."""
    bsd = BezierSimploid.randn(simploid_type, 3, n_values=2)
    for i, a in enumerate(simploid_type):
        offset = sum(b + 1 for b in simploid_type[:i])
        for j in range(a + 1):
            deg = bsd.degenerate(i, j)
            u = sample_simploid(deg.simploid_type, 10, seed=j)
            block = u[:, offset : offset + a + 2]
            merged = torch.cat([block[:, :j], block[:, j : j + 2].sum(dim=1, keepdim=True), block[:, j + 2 :]], dim=1)
            v = torch.cat([u[:, :offset], merged, u[:, offset + a + 2 :]], dim=1)
            assert torch.allclose(deg(u), bsd(v), atol=1e-5)
    for i in range(len(simploid_type) + 1):
        deg = bsd.degenerate(i, -1)
        u = sample_simploid(deg.simploid_type, 10, seed=i)
        offset = sum(b + 1 for b in simploid_type[:i])
        v = torch.cat([u[:, :offset], u[:, offset + 2 :]], dim=1)
        assert torch.allclose(deg(u), bsd(v), atol=1e-5)


def test_bezier_simploid_degenerate_errors():
    with pytest.raises(ValueError):
        BezierSimploid.randn((1, 1), (1, 2), n_values=1).degenerate(0, -1)
    assert BezierSimploid.randn((1, 1), (1, 2), n_values=1).degenerate(0, -1, degree=2).degree == (2, 1, 2)
    with pytest.raises(IndexError):
        BezierSimploid.randn((1,), 2, n_values=1).degenerate(0, 2)
    with pytest.raises(IndexError):
        BezierSimploid.randn((1,), 2, n_values=1).degenerate(2, -1)


def test_bezier_simploid_product():
    b1 = BezierSimploid.randn((2,), 2, n_values=3)
    b2 = BezierSimploid.randn((1, 1), (1, 3), n_values=3)
    p = b1.product(b2)
    assert p.simploid_type == (2, 1, 1)
    assert p.degree == (2, 1, 3)
    u, v = sample_simploid((2,), 8, seed=0), sample_simploid((1, 1), 8, seed=1)
    assert torch.allclose(p(torch.cat([u, v], dim=1)), b1(u) + b2(v), atol=1e-5)
    with pytest.raises(ValueError):
        b1.product(BezierSimploid.randn((1,), 1, n_values=2))


# ---------------------------------------------------------------------------
# SimploidalSet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_add_simplex(dim):
    S = SimploidalSet()
    top = S.add_simplex(dim)
    S.check()
    assert top.simploid_type == ((dim,) if dim else ())
    for k in range(dim + 1):
        assert len(S.simploids(k)) == comb(dim + 1, k + 1)
    assert len(top.vertices()) == dim + 1
    assert S.main_simploids() == [top]
    with pytest.raises(ValueError):
        S.add_simplex(-1)


@pytest.mark.parametrize(
    "simploid_type, counts",
    [
        ((1, 1), [4, 4, 1]),
        ((2, 1), [6, 9, 5, 1]),
        ((1, 2), [6, 9, 5, 1]),
        ((1, 1, 1), [8, 12, 6, 1]),
        ((2, 2), [9, 18, 15, 6, 1]),
    ],
)
def test_add_simploid(simploid_type, counts):
    S = SimploidalSet()
    top = S.add_simploid(simploid_type)
    S.check()
    assert top.simploid_type == simploid_type
    assert [len(S.simploids(k)) for k in range(len(counts))] == counts
    for (i, _), f in top.faces():
        assert f.simploid_type == _face_type(simploid_type, i)
        assert not f.is_degenerate


def test_product_of_sets():
    A, B = SimploidalSet(), SimploidalSet()
    tri, edge = A.add_simplex(2), B.add_simplex(1)
    v = A.add_vertex()
    P, pairs = A.product(B)
    P.check()
    assert pairs[(tri, edge)].simploid_type == (2, 1)
    assert pairs[(v, edge)].simploid_type == (1,)
    assert len(P) == len(A) * len(B)


def test_product_with_degenerate_faces():
    """The product handles degenerate faces and merges s^k_{-1} x id with id x s^0_{-1}."""
    A = SimploidalSet()
    square = A.add_simploid((1, 1))
    _, cones = A.cone([square])
    B = SimploidalSet()
    edge = B.add_simplex(1)
    P, pairs = A.product(B)
    P.check()
    prism = pairs[(cones[square], edge)]
    assert prism.simploid_type == (1, 1, 1, 1)
    apex_face = prism.face(2, 0)
    assert apex_face.is_degenerate
    vertex = cones[square].face(2, 0).base
    assert apex_face.base is pairs[(vertex, edge)]
    # (sigma1 s^k_{-1}) x sigma2 and sigma1 x (sigma2 s^0_{-1}) are the same simploid.
    for (s1, s2), s in pairs.items():
        assert s.degeneracy(s1.length, -1).simploid_type == s1.simploid_type + (1,) + s2.simploid_type


def test_degenerate_simploids_are_canonical():
    S = SimploidalSet()
    edge = S.add_simplex(1)
    v = edge.face(0, 0)
    # Simplicial identity: s0 s0 = s1 s0 (as operators applied left to right).
    assert edge.degeneracy(0, 0).degeneracy(0, 0) is edge.degeneracy(0, 0).degeneracy(0, 1)
    assert edge.degeneracy(0, 0) is not edge.degeneracy(0, 1)
    # All degeneracies of a vertex with the same type are equal.
    assert v.degeneracy(0, -1).degeneracy(0, 0) is v.degeneracy(0, -1).degeneracy(0, 1)
    # Faces of degenerate simploids (relation (IV)).
    d = edge.degeneracy(0, 1)  # (v0, v1, v1)
    assert d.face(0, 1) is edge and d.face(0, 2) is edge
    assert d.face(0, 0) is v.degeneracy(0, -1)
    e = edge.degeneracy(1, -1)  # edge x degenerate edge
    assert e.simploid_type == (1, 1)
    assert e.face(1, 0) is edge and e.face(1, 1) is edge
    assert e.face(0, 0) is v.degeneracy(0, -1)
    assert e.is_degenerate and e.base is edge
    with pytest.raises(IndexError):
        edge.degeneracy(3, -1)
    with pytest.raises(IndexError):
        edge.face(0, 2)


def test_degenerate_simploids_satisfy_commutation():
    from torch_bsf.simploidal.sets import _check_commutation

    S = SimploidalSet()
    prism = S.add_simploid((2, 1))
    for i, a in enumerate(prism.simploid_type):
        for j in range(a + 1):
            _check_commutation(prism.degeneracy(i, j))
            _check_commutation(prism.degeneracy(i, j).degeneracy(0, 0))
    for i in range(3):
        _check_commutation(prism.degeneracy(i, -1))


def test_identify_glues_faces():
    S = SimploidalSet()
    tetra = S.add_simplex(3)
    prism = S.add_simploid((2, 1))
    cube = S.add_simploid((1, 1, 1))
    tri = S.identify(tetra.face(0, 0), prism.face(1, 0))
    sq = S.identify(prism.face(0, 0), cube.face(0, 0))
    S.check()
    assert tetra.face(0, 0) is tri and prism.face(1, 0) is tri
    assert prism.face(0, 0) is sq and cube.face(0, 0) is sq
    assert len(S.simploids(0)) == 4 + 6 + 8 - 3 - 4
    assert S.main_simploids() == [tetra, prism, cube]
    assert tri in S and sq in S
    assert S.dim == 3


def test_identify_self_loop():
    S = SimploidalSet()
    edge = S.add_simplex(1)
    v = S.identify(edge.face(0, 0), edge.face(0, 1))
    S.check()
    assert edge.face(0, 0) is v and edge.face(0, 1) is v
    assert len(S) == 2


def test_identify_errors():
    S = SimploidalSet()
    tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    with pytest.raises(ValueError):
        S.identify(tri, sq)
    other = SimploidalSet().add_simplex(2)
    with pytest.raises(ValueError):
        S.identify(tri, other)


def test_identify_degeneracies_collapses_edge():
    """(v0, v0, v1) ~ (v0, v1, v1) forces v0 ~ v1 and the edge to degenerate into a vertex."""
    S = SimploidalSet()
    tri = S.add_simplex(2)
    edge = tri.face(0, 0)
    d = S.identify(edge.degeneracy(0, 0), edge.degeneracy(0, 1))
    S.check()
    assert d.is_degenerate and d.base.is_vertex
    assert tri.face(0, 0).is_degenerate
    assert len(tri.vertices()) == 2


def test_degenerate_square_edge():
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    edge = square.face(0, 0)
    d = S.degenerate(edge, edge.face(0, 0), 0, -1)
    S.check()
    assert d.is_degenerate
    assert square.face(0, 0) is d
    assert len(S.simploids(0)) == 3
    assert len(S.simploids(1)) == 3


@pytest.mark.parametrize("simploid_type", [(1,), (2,), (1, 1), (2, 1)])
def test_cone(simploid_type):
    S = SimploidalSet()
    top = S.add_simploid(simploid_type)
    n = len(S)
    v, cones = S.cone()
    S.check()
    assert len(cones) == n
    assert len(S) == 2 * n + 1
    c = cones[top]
    assert c.simploid_type == simploid_type + (1,)
    assert c.face(len(simploid_type), 1) is top
    apex = c.face(len(simploid_type), 0)
    assert apex.is_degenerate and apex.base is v
    assert set(map(id, c.vertices())) == set(map(id, top.vertices() + [v]))


def test_cone_of_subset():
    S = SimploidalSet()
    tri = S.add_simplex(2)
    sq = S.add_simploid((1, 1))
    v, cones = S.cone([tri.face(0, 0)])
    S.check()
    assert len(cones) == 3  # the edge and its two vertices
    assert sq not in cones
    with pytest.raises(ValueError):
        S.cone([SimploidalSet().add_simplex(1)])


def test_simploid_repr():
    S = SimploidalSet()
    e = S.add_simplex(1)
    assert "type=(1,)" in repr(e)
    assert "degenerate" in repr(e.degeneracy(0, 0))
    assert "SimploidalSet" in repr(S)
    assert isinstance(e, Simploid)
    assert 1 not in S


# ---------------------------------------------------------------------------
# Homology
# ---------------------------------------------------------------------------


def _circle() -> SimploidalSet:
    S = SimploidalSet()
    edge = S.add_simplex(1)
    S.identify(edge.face(0, 0), edge.face(0, 1))
    return S


def _torus() -> SimploidalSet:
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    S.identify(square.face(0, 0), square.face(0, 1))
    S.identify(square.face(1, 0), square.face(1, 1))
    return S


def _figure_1c() -> SimploidalSet:
    S = SimploidalSet()
    tetra = S.add_simplex(3)
    prism = S.add_simploid((2, 1))
    cube = S.add_simploid((1, 1, 1))
    S.identify(tetra.face(0, 0), prism.face(1, 0))
    S.identify(prism.face(0, 0), cube.face(0, 0))
    return S


def _coned_torus() -> SimploidalSet:
    S = _torus()
    S.cone()
    return S


def _pyramid() -> SimploidalSet:
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    S.cone([square])
    return S


def _degenerate_square() -> SimploidalSet:
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    edge = square.face(0, 0)
    S.degenerate(edge, edge.face(0, 0), 0, -1)
    return S


def _degenerate_sphere() -> SimploidalSet:
    """A square whose two edges face(1, *) are collapsed into vertices and whose two other edges are identified."""
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    for j in range(2):
        edge = square.face(1, j)
        S.degenerate(edge, edge.face(0, 0), 0, -1)
    S.identify(square.face(0, 0), square.face(0, 1))
    return S


def _two_triangles(pairs) -> SimploidalSet:
    """Two triangles whose edges ``(triangle, j)`` are identified in pairs."""
    S = SimploidalSet()
    triangles = [S.add_simplex(2), S.add_simplex(2)]
    for (t, j), (u, k) in pairs:
        S.identify(triangles[t].face(0, j), triangles[u].face(0, k))
    return S


def _product(A: SimploidalSet, B: SimploidalSet) -> SimploidalSet:
    return A.product(B)[0]


def _single(simploid_type) -> SimploidalSet:
    S = SimploidalSet()
    S.add_simploid(simploid_type)
    return S


def _two_edges() -> SimploidalSet:
    S = SimploidalSet()
    S.add_simplex(1)
    S.add_simplex(1)
    return S


# (name, builder, Betti numbers over Q, Betti numbers over Z/2)
HOMOLOGY_CASES = [
    ("empty", SimploidalSet, [], []),
    ("vertex", lambda: _single(()), [1], [1]),
    ("two edges", _two_edges, [2, 0], [2, 0]),
    ("edge", lambda: _single((1,)), [1, 0], [1, 0]),
    ("tetrahedron", lambda: _single((3,)), [1, 0, 0, 0], [1, 0, 0, 0]),
    ("prism", lambda: _single((2, 1)), [1, 0, 0, 0], [1, 0, 0, 0]),
    ("cube", lambda: _single((1, 1, 1)), [1, 0, 0, 0], [1, 0, 0, 0]),
    ("simploid (2, 2)", lambda: _single((2, 2)), [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]),
    ("circle", _circle, [1, 1], [1, 1]),
    ("torus", _torus, [1, 2, 1], [1, 2, 1]),
    ("product torus", lambda: _product(_circle(), _circle()), [1, 2, 1], [1, 2, 1]),
    ("solid torus", lambda: _product(_circle(), _single((2,))), [1, 1, 0, 0], [1, 1, 0, 0]),
    ("3-torus", lambda: _product(_torus(), _circle()), [1, 3, 3, 1], [1, 3, 3, 1]),
    ("figure 1(c)", _figure_1c, [1, 0, 0, 0], [1, 0, 0, 0]),
    ("pyramid", _pyramid, [1, 0, 0, 0], [1, 0, 0, 0]),
    ("cone over torus", _coned_torus, [1, 0, 0, 0], [1, 0, 0, 0]),
    ("degenerate square", _degenerate_square, [1, 0, 0], [1, 0, 0]),
    ("degenerate sphere", _degenerate_sphere, [1, 0, 1], [1, 0, 1]),
    ("sphere", lambda: _two_triangles([((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2))]), [1, 0, 1], [1, 0, 1]),
    ("projective plane", lambda: _two_triangles([((0, 0), (1, 0)), ((0, 1), (1, 2)), ((0, 2), (1, 1))]), [1, 0, 0], [1, 1, 1]),
    ("Klein bottle", lambda: _two_triangles([((0, 0), (1, 2)), ((0, 1), (1, 0)), ((0, 2), (1, 1))]), [1, 1, 0], [1, 2, 1]),
]


@pytest.mark.parametrize("name, build, betti_q, betti_z2", HOMOLOGY_CASES, ids=[c[0] for c in HOMOLOGY_CASES])
def test_boundary_of_boundary_is_zero(name, build, betti_q, betti_z2):
    S = build()
    S.check()
    for k in range(S.dim + 2):
        d = S.boundary_matrix(k)
        assert d.dtype == torch.long
        assert d.shape == (len(S.simploids(k - 1)), len(S.simploids(k)))
        if k >= 1:
            assert (S.boundary_matrix(k - 1) @ d == 0).all()


@pytest.mark.parametrize("name, build, betti_q, betti_z2", HOMOLOGY_CASES, ids=[c[0] for c in HOMOLOGY_CASES])
def test_betti_numbers(name, build, betti_q, betti_z2):
    S = build()
    assert S.betti_numbers() == betti_q
    assert S.betti_numbers("Q") == betti_q
    assert S.betti_numbers("Z2") == betti_z2
    # The Euler characteristic does not depend on the coefficients.
    chi = sum((-1) ** k * len(S.simploids(k)) for k in range(S.dim + 1))
    assert sum((-1) ** k * b for k, b in enumerate(betti_q)) == chi


def test_boundary_matrix_signs():
    S = SimploidalSet()
    tri = S.add_simplex(2)
    d = S.boundary_matrix(2)
    rows = S.simploids(1)
    assert d[:, 0].tolist() == [{tri.face(0, j): (-1) ** j for j in range(3)}[e] for e in rows]
    # Koszul sign: faces of the second factor of a prism get the sign (-1)^(a_1 + j).
    S = SimploidalSet()
    prism = S.add_simploid((2, 1))
    d = S.boundary_matrix(3)
    rows = S.simploids(2)
    expected = {prism.face(0, j): (-1) ** j for j in range(3)}
    expected.update({prism.face(1, j): (-1) ** (2 + j) for j in range(2)})
    assert d[:, 0].tolist() == [expected[f] for f in rows]


def test_boundary_matrix_drops_degenerate_faces():
    S = _pyramid()
    pyramid = S.simploids(3)[0]
    apex_face = pyramid.face(2, 0)
    assert apex_face.is_degenerate
    d = S.boundary_matrix(3)
    assert int(d.abs().sum()) == 4 + 1  # four side faces and the base; the apex face is dropped


def test_homology_errors():
    S = _torus()
    with pytest.raises(ValueError):
        S.boundary_matrix(-1)
    with pytest.raises(ValueError):
        S.betti_numbers("Z")  # type: ignore[arg-type]
    edge = S.simploids(1)[0]
    edge._faces[(0, 0)] = Simploid(())  # a face outside the set
    with pytest.raises(ValueError):
        S.boundary_matrix(1)


def test_rank_is_exact():
    from torch_bsf.simploidal.sets import _rank

    m = torch.tensor([[2, 4, 6], [1, 2, 3], [0, 0, 0], [3, 1, 2]])
    assert _rank(m, None) == 2
    assert _rank(m, 2) == 2
    assert _rank(torch.tensor([[2, 0], [0, 2]]), 2) == 0
    assert _rank(torch.zeros(0, 3, dtype=torch.long), None) == 0


# ---------------------------------------------------------------------------
# SimploidalBezierSpace
# ---------------------------------------------------------------------------


def _glued_set() -> tuple[SimploidalSet, list[Simploid]]:
    """A tetrahedron, a prism, a cube and a pyramid (cone of a cube face), glued together."""
    S = SimploidalSet()
    tetra = S.add_simplex(3)
    prism = S.add_simploid((2, 1))
    cube = S.add_simploid((1, 1, 1))
    S.identify(tetra.face(0, 0), prism.face(1, 0))
    S.identify(prism.face(0, 0), cube.face(0, 0))
    _, cones = S.cone([cube.face(0, 1)])
    pyramid = cones[cube.face(0, 1)]
    S.check()
    return S, [tetra, prism, cube, pyramid]


def _random_space(S: SimploidalSet, degree: int, n_values: int = 2, seed: int = 0) -> SimploidalBezierSpace:
    space = SimploidalBezierSpace(S, degree, n_values)
    with torch.no_grad():
        space.proper_control_points.copy_(
            torch.randn(space.n_proper_control_points, n_values, generator=torch.Generator().manual_seed(seed))
        )
    return space


def _expected_proper_count(s: Simploid, degree: int) -> int:
    return 1 if s.is_vertex else prod(comb(degree - 1, a) for a in s.simploid_type)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_space_stores_each_control_point_once(degree):
    S, mains = _glued_set()
    space = _random_space(S, degree)
    assert space.n_proper_control_points == sum(_expected_proper_count(s, degree) for s in S)
    used = set()
    for s in S.main_simploids():
        rows = space.control_point_rows(s)
        assert len(rows) == len(simploid_indices(s.simploid_type, degree))
        used.update(rows)
    assert used == set(range(space.n_proper_control_points))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_space_face_and_degeneracy_consistency(degree):
    """Property 2 holds between every simploid of the space and its faces and degeneracies."""
    S, _ = _glued_set()
    space = _random_space(S, degree)
    for s in S:
        cp = space.bezier_simploid(s)
        for (i, j), f in s.faces():
            assert torch.equal(cp.face(i, j).control_points, space.control_points(f).detach())
        for i, a in enumerate(s.simploid_type):
            for j in range(a + 1):
                assert torch.equal(cp.degenerate(i, j).control_points, space.control_points(s.degeneracy(i, j)).detach())
        for i in range(s.length + 1):
            assert torch.equal(
                cp.degenerate(i, -1, degree=degree).control_points, space.control_points(s.degeneracy(i, -1)).detach()
            )


def test_space_continuity_across_shared_faces():
    S, (tetra, prism, cube, pyramid) = _glued_set()
    space = _random_space(S, 3)
    u = sample_simploid((2,), 10, seed=0)
    on_tetra = torch.cat([torch.zeros(10, 1), u], dim=1)  # face(0, 0) of the tetrahedron
    on_prism = torch.cat([u, torch.tensor([[0.0, 1.0]]).expand(10, 2)], dim=1)  # face(1, 0) of the prism
    assert torch.allclose(space.evaluate(tetra, on_tetra), space.evaluate(prism, on_prism), atol=1e-5)
    # The apex face of the pyramid collapses to the new vertex.
    apex = pyramid.face(2, 0)
    w = sample_simploid((1, 1), 10, seed=1)
    values = space.evaluate(pyramid, torch.cat([w, torch.tensor([[0.0, 1.0]]).expand(10, 2)], dim=1))
    assert torch.allclose(values, space.control_points(apex.base).expand(10, -1), atol=1e-5)


def test_space_forward_mixed_batch():
    S, mains = _glued_set()
    space = _random_space(S, 2)
    width = max(simploid_n_params(s.simploid_type) for s in mains)
    ts, ids, expected = [], [], []
    for s in mains:
        t = sample_simploid(s.simploid_type, 4, seed=space.simploid_id(s))
        expected.append(space.evaluate(s, t))
        ts.append(torch.nn.functional.pad(t, (0, width - t.shape[1])))
        ids.append(torch.full((4,), space.simploid_id(s)))
    perm = torch.randperm(4 * len(mains), generator=torch.Generator().manual_seed(0))
    out = space(torch.cat(ts)[perm], torch.cat(ids)[perm])
    assert torch.allclose(out, torch.cat(expected)[perm], atol=1e-6)


def test_space_errors():
    S, _ = _glued_set()
    with pytest.raises(ValueError):
        SimploidalBezierSpace(S, 0, 1)
    with pytest.raises(ValueError):
        SimploidalBezierSpace(S, 2, 1, torch.zeros(1, 1))
    space = SimploidalBezierSpace(S, 2, 1)
    foreign = SimploidalSet().add_simplex(1)
    with pytest.raises(ValueError):
        space.simploid_id(foreign)
    with pytest.raises(ValueError):
        space.control_point_rows(foreign)
    assert "n_simploids" in space.extra_repr()


# ---------------------------------------------------------------------------
# SimploidalBezierSpace with per-factor degrees
# ---------------------------------------------------------------------------


def _mixed_degrees(mains: list[Simploid]) -> dict[Simploid, tuple[int, ...]]:
    """Consistent per-factor degrees of the main simploids of :func:`_glued_set`."""
    tetra, prism, cube, pyramid = mains
    # cube factors 1, 2 = prism factors 0, 1; pyramid factors 0, 1 = cube factors 1, 2; tetra = prism factor 0.
    return {prism: (2, 3), cube: (4, 2, 3), pyramid: (2, 3, 1)}


def _random_mixed_space(S: SimploidalSet, degree, n_values: int = 2, seed: int = 0, **kwargs) -> SimploidalBezierSpace:
    space = SimploidalBezierSpace(S, degree, n_values, **kwargs)
    with torch.no_grad():
        space.proper_control_points.copy_(
            torch.randn(space.n_proper_control_points, n_values, generator=torch.Generator().manual_seed(seed))
        )
    return space


def _all_simploids_and_faces(S: SimploidalSet) -> list[Simploid]:
    out = list(S)
    for s in S:
        out.extend(f for _, f in s.faces())
        out.extend(s.degeneracy(i, j) for i, a in enumerate(s.simploid_type) for j in range(-1, a + 1))
        out.append(s.degeneracy(s.length, -1))
    return out


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_space_uniform_mapping_matches_integer_degree(degree):
    """A uniform degree given as a mapping or a default builds the same space as an integer degree."""
    S, mains = _glued_set()
    reference = SimploidalBezierSpace(S, degree, 1)
    for space in [
        SimploidalBezierSpace(S, {s: degree for s in mains}, 1),
        SimploidalBezierSpace(S, {}, 1, default_degree=degree),
    ]:
        assert space.degree == degree
        assert space.n_proper_control_points == reference.n_proper_control_points
        assert space.extra_repr() == reference.extra_repr()
        for s in _all_simploids_and_faces(S):
            assert space.simploid_degree(s) == (degree,) * s.length
            assert space.control_point_rows(s) == reference.control_point_rows(s)


def test_space_mixed_degrees_assignment():
    S, mains = _glued_set()
    tetra, prism, cube, pyramid = mains
    space = SimploidalBezierSpace(S, _mixed_degrees(mains), 1)
    assert space.simploid_degree(tetra) == (2,)
    assert space.simploid_degree(prism.face(1, 0)) == (2,)  # triangle
    assert space.simploid_degree(prism.face(0, 0)) == (2, 3)  # square shared with the cube
    assert space.simploid_degree(cube.face(0, 1)) == (2, 3)  # base of the pyramid
    # Degrees propagate from the prism to the cube and the pyramid; the rest comes from default_degree.
    space = SimploidalBezierSpace(S, {prism: (2, 3)}, 1, default_degree=5)
    assert space.simploid_degree(cube) == (5, 2, 3)
    assert space.simploid_degree(pyramid) == (2, 3, 5)
    assert space.degree == 5 and "max_degree=5" in space.extra_repr()
    # Inserted edge factors of a degenerate simploid get the maximum degree.
    assert space.simploid_degree(prism.degeneracy(1, -1)) == (2, 5, 3)
    assert space.simploid_degree(prism.degeneracy(0, 1)) == (2, 3)
    assert space.simploid_degree(pyramid.face(2, 0)) == (5, 5)


def test_space_mixed_degrees_stores_each_control_point_once():
    S, mains = _glued_set()
    space = _random_mixed_space(S, _mixed_degrees(mains))
    expected = sum(
        1 if s.is_vertex else prod(comb(d - 1, a) for a, d in zip(s.simploid_type, space.simploid_degree(s))) for s in S
    )
    assert space.n_proper_control_points == expected
    used = set()
    for s in S.main_simploids():
        rows = space.control_point_rows(s)
        assert len(rows) == len(simploid_indices(s.simploid_type, space.simploid_degree(s)))
        used.update(rows)
    assert used == set(range(space.n_proper_control_points))


def test_space_mixed_degrees_proper_count_by_hand():
    """A prism of degrees (1, 3) glued to a cube of degrees (2, 1, 3) along a square face."""
    S = SimploidalSet()
    prism, cube = S.add_simploid((2, 1)), S.add_simploid((1, 1, 1))
    S.identify(prism.face(0, 0), cube.face(0, 0))
    space = SimploidalBezierSpace(S, {prism: (1, 3), cube: (2, 1, 3)}, 1)
    # 10 vertices; 3 + 2 vertical edges of degree 3 (2 points each) and 4 cube edges of degree 2 (1 point each);
    # 2 cube squares of degrees (2, 3) (2 points each); no proper point in the other cells.
    assert space.n_proper_control_points == 10 + 5 * 2 + 4 * 1 + 2 * 2


@pytest.mark.parametrize("seed", [0, 1])
def test_space_mixed_degrees_face_and_degeneracy_consistency(seed):
    """Property 2 holds with per-factor degrees, including degenerate faces of different degrees."""
    S, mains = _glued_set()
    space = _random_mixed_space(S, _mixed_degrees(mains), seed=seed)
    for s in S:
        cp = space.bezier_simploid(s)
        assert cp.degree == space.simploid_degree(s)
        for (i, j), f in s.faces():
            u = sample_simploid(f.simploid_type, 5, seed=seed)
            assert torch.allclose(cp.face(i, j)(u), space.evaluate(f, u), atol=1e-5)
            if not f.is_degenerate:
                assert torch.equal(cp.face(i, j).control_points, space.control_points(f).detach())
        for i, a in enumerate(s.simploid_type):
            for j in range(a + 1):
                assert torch.equal(cp.degenerate(i, j).control_points, space.control_points(s.degeneracy(i, j)).detach())
        for i in range(s.length + 1):
            deg = cp.degenerate(i, -1, degree=space.degree)
            assert torch.equal(deg.control_points, space.control_points(s.degeneracy(i, -1)).detach())


def test_space_mixed_degrees_continuity_across_shared_faces():
    S = SimploidalSet()
    prism, cube = S.add_simploid((2, 1)), S.add_simploid((1, 1, 1))
    S.identify(prism.face(0, 0), cube.face(0, 0))
    space = _random_mixed_space(S, {prism: (1, 3), cube: (2, 1, 3)}, n_values=3)
    w, v = sample_simploid((1,), 20, seed=0), sample_simploid((1,), 20, seed=1)
    on_prism = torch.cat([torch.zeros(20, 1), w, v], dim=1)  # face(0, 0) of the prism
    on_cube = torch.cat([torch.tensor([[0.0, 1.0]]).expand(20, 2), w, v], dim=1)  # face(0, 0) of the cube
    assert torch.allclose(space.evaluate(prism, on_prism), space.evaluate(cube, on_cube), atol=1e-5)
    # The pieces are not trivially polynomials of the lowest degree: the height direction is cubic.
    t = sample_simploid((2, 1), 20, seed=2)
    linear = fit_simploidal_set(S, [(prism, t, space.evaluate(prism, t).detach())], 1)
    assert not torch.allclose(linear.evaluate(prism, t), space.evaluate(prism, t), atol=1e-3)


def test_space_mixed_degrees_errors():
    S, mains = _glued_set()
    tetra, prism, cube, pyramid = mains
    with pytest.raises(ValueError, match="Inconsistent"):
        SimploidalBezierSpace(S, {prism: (2, 3), cube: (4, 3, 3)}, 1)
    with pytest.raises(ValueError, match="Inconsistent"):
        SimploidalBezierSpace(S, {prism: (2, 3), tetra: 3}, 1, default_degree=1)
    with pytest.raises(ValueError, match="not determined"):
        SimploidalBezierSpace(S, {prism: (2, 3)}, 1)
    with pytest.raises(ValueError, match="not a non-degenerate simploid"):
        SimploidalBezierSpace(S, {prism.degeneracy(0, 0): 2}, 1, default_degree=1)
    with pytest.raises(ValueError, match="not a non-degenerate simploid"):
        SimploidalBezierSpace(S, {SimploidalSet().add_simplex(1): 2}, 1, default_degree=1)
    with pytest.raises(ValueError, match="components"):
        SimploidalBezierSpace(S, {prism: (2, 3, 4)}, 1, default_degree=1)
    with pytest.raises(ValueError, match="at least 1"):
        SimploidalBezierSpace(S, {prism: (0, 3)}, 1, default_degree=1)
    with pytest.raises(ValueError, match="at least 1"):
        SimploidalBezierSpace(S, {prism: (2, 3)}, 1, default_degree=0)
    with pytest.raises(ValueError, match="Inconsistent"):
        fit_simploidal_set(S, [(prism, sample_simploid((2, 1), 5), torch.zeros(5, 1))], {prism: 2, cube: 3})


def test_space_mixed_degrees_through_degenerate_face():
    """A degenerate face constrains the degree of the factor of its base it comes from."""
    S = SimploidalSet()
    prism = S.add_simploid((2, 1))
    edge = S.add_simplex(1)
    S.identify(prism.face(1, 0), edge.degeneracy(0, 0))  # collapse a triangle of the prism onto the edge
    S.check()
    assert prism.face(1, 0).is_degenerate
    edge = prism.face(1, 0).base
    space = _random_mixed_space(S, {prism: (2, 3)}, n_values=2)
    assert space.simploid_degree(edge) == (2,)
    assert space.simploid_degree(prism.face(1, 0)) == (2,)
    u = sample_simploid((2,), 10, seed=0)
    on_prism = torch.cat([u, torch.tensor([[0.0, 1.0]]).expand(10, 2)], dim=1)
    on_edge = torch.stack([u[:, 0] + u[:, 1], u[:, 2]], dim=1)  # vertex 0 of the edge is repeated
    assert torch.allclose(space.evaluate(prism, on_prism), space.evaluate(edge, on_edge), atol=1e-5)
    with pytest.raises(ValueError, match="Inconsistent"):
        SimploidalBezierSpace(S, {prism: (2, 3), edge: 3}, 1)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simploid_type, degree", [((1,), 3), ((2,), 2), ((1, 1), (2, 1)), ((2, 1), 2), ((1, 1, 1), 1)])
def test_fit_simploid_recovers_model(simploid_type, degree):
    truth = BezierSimploid.randn(simploid_type, degree, n_values=3)
    t = sample_simploid(simploid_type, 200, seed=0)
    fitted = fit(t, truth(t).detach(), degree, simploid_type=simploid_type)
    assert torch.allclose(fitted.control_points, truth.control_points, atol=1e-3)


def test_fit_simploid_ridge_and_errors():
    t = sample_simploid((2,), 3, seed=0)
    x = t[:, :1]
    fitted = fit(t, x, 3, simploid_type=(2,), ridge=1e-3)  # under-determined: 10 control points, 3 samples
    assert fitted.control_points.shape == (10, 1)
    assert torch.isfinite(fit(t, x, 3, simploid_type=(2,)).control_points).all()
    with pytest.raises(ValueError):
        fit(t, x, 3, simploid_type=(2,), solver="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit(t, x, 3, simploid_type=(2,), solver="trainer", ridge=1.0)
    with pytest.raises(ValueError):
        fit(t, x, 3, simploid_type=(1,))
    with pytest.raises(ValueError):
        fit(t, x, 3, simploid_type=(2,), ridge=-1.0)


def test_fit_simploid_trainer():
    truth = BezierSimploid.randn((2, 1), 1, n_values=2)
    t = sample_simploid((2, 1), 100, seed=0)
    x = truth(t).detach()
    fitted = fit(t, x, 1, simploid_type=(2, 1), solver="trainer", seed=0, **TRAINER_KWARGS)
    assert torch.nn.functional.mse_loss(fitted(t), x) < 1e-3


def _sample_space(space: SimploidalBezierSpace, simploids, n: int):
    data = []
    for k, s in enumerate(simploids):
        t = sample_simploid(s.simploid_type, n, seed=k)
        data.append((s, t, space.evaluate(s, t).detach()))
    return data


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fit_simploidal_set_recovers_space(degree):
    S, mains = _glued_set()
    truth = _random_space(S, degree, n_values=3)
    fitted = fit_simploidal_set(S, _sample_space(truth, mains, 150), degree)
    assert torch.allclose(fitted.proper_control_points, truth.proper_control_points, atol=1e-3)


@pytest.mark.parametrize("solver", ["lstsq", "trainer"])
def test_fit_simploidal_set_recovers_mixed_degree_space(solver):
    S, mains = _glued_set()
    degree = _mixed_degrees(mains)
    truth = _random_mixed_space(S, degree, n_values=3)
    data = _sample_space(truth, mains, 200)
    if solver == "lstsq":
        fitted = fit_simploidal_set(S, data, degree)
        assert torch.allclose(fitted.proper_control_points, truth.proper_control_points, atol=1e-3)
    else:
        fitted = fit_simploidal_set(S, data, degree, solver="trainer", seed=0, **TRAINER_KWARGS)
        assert fitted.n_proper_control_points == truth.n_proper_control_points
    for s in S:
        assert fitted.simploid_degree(s) == truth.simploid_degree(s)
    # Degrees propagated from a partial mapping with a default give the same space.
    partial = fit_simploidal_set(S, data, {mains[1]: (2, 3), mains[2]: (4, 2, 3)}, default_degree=1)
    assert torch.allclose(partial.proper_control_points, truth.proper_control_points, atol=1e-3)


def test_fit_simploidal_set_is_continuous():
    """Pieces fitted to unrelated data still agree on their shared face."""
    S = SimploidalSet()
    t1, t2 = S.add_simplex(2), S.add_simplex(2)
    S.identify(t1.face(0, 0), t2.face(0, 0))
    p1, p2 = sample_simploid((2,), 50, seed=0), sample_simploid((2,), 50, seed=1)
    space = fit_simploidal_set(S, [(t1, p1, p1[:, :1] ** 2), (t2, p2, -p2[:, 1:2])], degree=3)
    u = torch.cat([torch.zeros(5, 1), sample_simploid((1,), 5, seed=2)], dim=1)
    assert torch.allclose(space.evaluate(t1, u), space.evaluate(t2, u), atol=1e-6)


def test_fit_simploidal_set_data_on_lower_dimensional_simploid():
    S = SimploidalSet()
    sq = S.add_simploid((1, 1))
    edge = sq.face(0, 0)
    t_edge = sample_simploid((1,), 20, seed=0)
    t_sq = sample_simploid((1, 1), 40, seed=1)
    data = [(sq, t_sq, t_sq[:, 2:3]), (edge, t_edge, t_edge[:, :1]), (edge, t_edge, t_edge[:, :1])]
    space = fit_simploidal_set(S, data, degree=2, ridge=1e-8)
    assert torch.allclose(space.evaluate(edge, t_edge), t_edge[:, :1], atol=1e-3)


def test_fit_simploidal_set_trainer():
    S, mains = _glued_set()
    truth = _random_space(S, 1, n_values=2)
    data = _sample_space(truth, mains, 60)
    space = fit_simploidal_set(S, data, 1, solver="trainer", seed=0, **TRAINER_KWARGS)
    mse = sum(torch.nn.functional.mse_loss(space.evaluate(s, t), x) for s, t, x in data) / len(data)
    assert mse < 1e-3


def test_fit_simploidal_set_errors():
    S, mains = _glued_set()
    t = sample_simploid((3,), 5, seed=0)
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [], 2)
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [(mains[0], t, t[:, :1])], 2, solver="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [(mains[0], t, t[:, :1])], 2, solver="trainer", ridge=1.0)
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [(mains[0], t, t[:, :1]), (mains[0], t, t[:, :2])], 2)
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [(mains[0].degeneracy(0, 0), t, t[:, :1])], 2)
    with pytest.raises(ValueError):
        fit_simploidal_set(S, [(mains[1], t, t[:, :1])], 2)


def test_space_accepts_integer_control_points():
    S = SimploidalSet()
    S.add_simplex(1)
    space = SimploidalBezierSpace(S, 1, 2, torch.ones(2, 2, dtype=torch.long))
    assert space.proper_control_points.is_floating_point()
    assert torch.equal(space.proper_control_points, torch.ones(2, 2))


@pytest.mark.parametrize("solver", ["lstsq", "trainer"])
def test_fit_preserves_float64(solver):
    """Fitting float64 data yields a float64 model (and the trainer does not hit a dtype mismatch)."""
    kwargs = dict(TRAINER_KWARGS, max_epochs=1, accelerator="cpu") if solver == "trainer" else {}
    t = sample_simploid((2,), 30, seed=0).double()
    x = t[:, :1] * t[:, 1:2]
    bs = fit(t, x, 2, simploid_type=(2,), solver=solver, **kwargs)
    assert bs.control_points.dtype == torch.float64
    assert bs(t).dtype == torch.float64
    S = SimploidalSet()
    tri = S.add_simplex(2)
    space = fit_simploidal_set(S, [(tri, t, x)], 2, solver=solver, **kwargs)
    assert space.proper_control_points.dtype == torch.float64
    if solver == "lstsq":
        assert torch.allclose(space.evaluate(tri, t), x, atol=1e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_fit_lstsq_accepts_mps_inputs():
    """MPS does not support float64, so the exact solver must move the inputs to the CPU before promoting them."""
    t = sample_simploid((2,), 30, seed=0)
    x = t[:, :1] * t[:, 1:2]
    for normalize in ["none", "std"]:
        bs = torch_bsf.fit(t.to("mps"), x.to("mps"), 2, normalize=normalize)
        assert torch.allclose(bs(t), x, atol=1e-5)
    S = SimploidalSet()
    tri = S.add_simplex(2)
    space = fit_simploidal_set(S, [(tri, t.to("mps"), x.to("mps"))], 2)
    assert torch.allclose(space.evaluate(tri, t), x, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_fit_lstsq_accepts_mps_init():
    """The exact solver builds the design matrix on the CPU even if init is on another device."""
    t = sample_simploid((2,), 30, seed=0)
    x = t[:, :1] * t[:, 1:2]
    init = BezierSimploid.zeros((2,), 2, n_values=1).to("mps")
    for normalize in ["none", "std"]:
        bs = torch_bsf.fit(t, x, 2, init=init, freeze=[(2, 0, 0)], normalize=normalize)
        assert bs.control_points.device.type == "mps"
        assert torch.allclose(bs.cpu()(t), x, atol=1e-5)


def test_public_exports():
    for name in [
        "BezierSimploid",
        "Simploid",
        "SimploidalSet",
        "SimploidalBezierSpace",
        "fit",
        "fit_kfold",
        "fit_simploidal_set",
    ]:
        assert hasattr(torch_bsf, name)


# ---------------------------------------------------------------------------
# Additional coverage: chained degeneracies, consistency check, Lightning steps
# ---------------------------------------------------------------------------


def test_chained_degeneration():
    """Degenerating the base of a degenerate face composes the degeneracy maps."""
    S = SimploidalSet()
    cube = S.add_simploid((1, 1, 1))
    square = cube.face(2, 0)
    edge = square.face(1, 0)
    flat = S.degenerate(square, edge, 1, -1)
    assert cube.face(2, 0) is flat and flat.base.simploid_type == (1,)
    edge = flat.base  # the two opposite edges of the square were merged; `edge` may be stale
    point = S.degenerate(edge, edge.face(0, 0), 0, -1)
    S.check()
    assert point.base.is_vertex
    face = cube.face(2, 0)
    assert face.is_degenerate and face.base is point.base
    assert face.simploid_type == (1, 1)
    space = _random_space(S, 2)
    t = sample_simploid((1, 1), 5, seed=0)
    assert torch.allclose(space.evaluate(face, t), space.control_points(point.base).expand(5, -1))


def test_cone_of_simploid_with_degenerate_face():
    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    edge = square.face(0, 0)
    S.degenerate(edge, edge.face(0, 0), 0, -1)
    v, cones = S.cone([square])
    S.check()
    c = cones[square]
    assert c.face(0, 0).is_degenerate
    assert c.face(0, 0).base is cones[square.face(0, 0).base]


def test_check_detects_inconsistencies():
    S = SimploidalSet()
    tri = S.add_simplex(2)
    edges = dict(tri._faces)
    tri._faces[(0, 0)] = SimploidalSet().add_simplex(1)
    with pytest.raises(ValueError, match="does not belong"):
        S.check()
    tri._faces[(0, 0)] = edges[(0, 0)].face(0, 0)
    with pytest.raises(ValueError, match="wrong type"):
        S.check()
    tri._faces[(0, 0)] = edges[(0, 2)]
    with pytest.raises(ValueError, match="!="):
        S.check()

    S = SimploidalSet()
    square = S.add_simploid((1, 1))
    square._faces[(1, 0)] = square._faces[(1, 1)]
    with pytest.raises(ValueError, match="!="):
        S.check()


def test_lightning_validation_and_test_steps():
    import lightning.pytorch as L
    from torch.utils.data import DataLoader, TensorDataset

    from torch_bsf.simploidal.fitting import _LightningWrapper

    trainer = L.Trainer(accelerator="cpu", enable_progress_bar=False, logger=False, enable_model_summary=False)
    t = sample_simploid((2, 1), 10, seed=0)
    bsd = BezierSimploid.randn((2, 1), 1, n_values=1)
    dl = DataLoader(TensorDataset(t, bsd(t).detach()))
    assert trainer.validate(_LightningWrapper(bsd), dl, verbose=False)[0]["val_mse"] < 1e-10
    assert trainer.test(_LightningWrapper(bsd), dl, verbose=False)[0]["test_mse"] < 1e-10

    S = SimploidalSet()
    tri = S.add_simplex(2)
    space = _random_space(S, 1, n_values=1)
    p = sample_simploid((2,), 10, seed=0)
    ids = torch.full((10,), space.simploid_id(tri))
    dl = DataLoader(TensorDataset(p, ids, space.evaluate(tri, p).detach()))
    assert trainer.validate(_LightningWrapper(space), dl, verbose=False)[0]["val_mae"] < 1e-5
    assert trainer.test(_LightningWrapper(space), dl, verbose=False)[0]["test_mae"] < 1e-5


def test_bezier_simploid_constructors_and_properties():
    zeros = BezierSimploid.zeros((2, 1), 2, n_values=3)
    assert torch.equal(zeros.control_points, torch.zeros(18, 3))
    assert zeros.n_params == 5
    assert "simploid_type=(2, 1)" in repr(zeros)
    integer = BezierSimploid((1,), 1, torch.tensor([[1], [2]]))
    assert integer.control_points.is_floating_point()
    mapping = BezierSimploid((1,), 1, {((1, 0),): [0.0, 1.0], ((0, 1),): [2.0, 3.0]})
    assert torch.equal(mapping[((0, 1),)].detach(), torch.tensor([2.0, 3.0]))


def test_fit_with_integer_values_and_shape_errors():
    S = SimploidalSet()
    edge = S.add_simplex(1)
    t = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    space = fit_simploidal_set(S, [(edge, t, torch.tensor([[0], [1], [2]]))], 1)
    assert space.proper_control_points.dtype == torch.get_default_dtype()
    assert torch.allclose(space.evaluate(edge, t), torch.tensor([[0.0], [1.0], [2.0]]), atol=1e-5)
    with pytest.raises(ValueError, match="values must have shape"):
        fit(t, torch.zeros(2, 1), 1, simploid_type=(1,))


# ---------------------------------------------------------------------------
# Saving and loading Bézier simploids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".yml", ".yaml", ".pt", ".CSV"])
@pytest.mark.parametrize("simploid_type, degree", [((2, 1), 2), ((1, 2), (3, 1)), ((3,), 2), ((1, 1, 1), 1), ((), 0)])
def test_save_load_roundtrip(tmp_path, suffix, simploid_type, degree):
    """A saved Bézier simploid should load with the same type, degree, control points and predictions."""
    torch.manual_seed(0)
    bs = BezierSimploid.randn(simploid_type, degree, 3)
    path = tmp_path / f"model{suffix}"
    torch_bsf.simploidal.save(path, bs)
    loaded = torch_bsf.simploidal.load(path)
    assert loaded.simploid_type == bs.simploid_type
    assert loaded.degree == bs.degree
    assert loaded.indices() == bs.indices()
    assert torch.allclose(loaded.control_points, bs.control_points)
    t = sample_simploid(simploid_type, 20, seed=1)
    assert torch.allclose(loaded(t), bs(t), atol=1e-6)


def test_save_load_fitted_model_and_float64(tmp_path):
    """A fitted model should predict the same after a round trip; .pt keeps the dtype."""
    t = sample_simploid((2, 1), 50, seed=0).double()
    bs = fit(t, t[:, :3] * t[:, 3:4], degree=(1, 2), simploid_type=(2, 1))
    for suffix in (".csv", ".pt"):
        loaded = torch_bsf.simploidal.load(_saved(tmp_path, bs, suffix))
        assert torch.allclose(loaded(t).double(), bs(t), atol=1e-6)
    assert torch_bsf.simploidal.load(tmp_path / "model.pt").control_points.dtype == torch.float64


def _saved(tmp_path, bs, suffix):
    path = tmp_path / f"model{suffix}"
    torch_bsf.simploidal.save(path, bs)
    return path


def test_save_formats(tmp_path):
    """The text formats use the index tuple as the key, in the order of simploid_indices."""
    import json

    import yaml

    bs = BezierSimploid((1, 1), 1, torch.arange(8.0).reshape(4, 2))
    keys = [str(index) for index in simploid_indices((1, 1), 1)]
    assert keys[0] == "((1, 0), (1, 0))"
    assert _saved(tmp_path, bs, ".tsv").read_text().splitlines()[1] == "((1, 0), (0, 1))\t2.0\t3.0"
    assert json.loads(_saved(tmp_path, bs, ".json").read_text()) == {k: [2.0 * i, 2.0 * i + 1] for i, k in enumerate(keys)}
    assert list(yaml.safe_load(_saved(tmp_path, bs, ".yml").read_text())) == keys
    assert set(torch.load(_saved(tmp_path, bs, ".pt"), weights_only=True)) == {"simploid_type", "degree", "control_points"}


@pytest.mark.parametrize(
    "suffix, content, match",
    [
        (".csv", "", "No control points found"),
        (".csv", '"(1, 0)",1.0\n"((1, 0), (0, 1))",2.0\n', "does not match the simploid type"),
        (".csv", '"1",1.0\n', "Invalid control point index '1'"),
        (".csv", '"((1, 0), (1, 0))",1.0\n"((1, 0), (0, 1))",2.0\n', "Control points are missing"),
        (".csv", '"((1, 0),)",1.0\n"((0, 1),)",2.0\n"((1, 0), )",3.0\n', "Duplicate control point index"),
        (".csv", '"((1, 0),)",1.0\n"((0, 0, 1),)",2.0\n', "does not match the simploid type \\(1,\\) and degree \\(1,\\)"),
        (".csv", '"((1, 0),)",1.0\n"((0, 2),)",2.0\n', "does not match the simploid type"),
        (".csv", '"((1, 0),)",1.0\n"((0, 1),)",2.0,3.0\n', "has 2 values, expected 1"),
        (".csv", '"((1, -1),)",1.0\n', "Invalid control point index"),
        (".csv", '"((1, 0.5),)",1.0\n', "Invalid control point index"),
        (".csv", '"((1, 0)",1.0\n', "Invalid control point index"),
        (".csv", '"((0,),)",1.0\n', "positive"),
        (".json", "[1, 2]", "must contain a mapping"),
        (".json", "{}", "No control points found"),
        (".yml", "", "must contain a mapping"),
        (".yml", "'((1, 0),)': [1.0]\n'((0, 1),)': [2.0]\n'((2, 0),)': [3.0]\n", "does not match"),
        (".txt", "", "Unknown file type"),
    ],
)
def test_load_errors(tmp_path, suffix, content, match):
    """Invalid or incomplete control point files should raise a clear ValueError."""
    path = tmp_path / f"model{suffix}"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        torch_bsf.simploidal.load(path)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".yml"])
def test_load_flat_keys(suffix):
    """Files with flat keys such as "(2, 0)" hold a Bézier simplex."""
    loaded = torch_bsf.simploidal.load(Path(__file__).parent / "data" / f"bezier_simplex{suffix}")
    assert (loaded.simploid_type, loaded.degree, loaded.n_values) == ((1,), (2,), 3)
    assert loaded[(1, 1)].tolist() == pytest.approx([0.5890220403671265, -0.5538082122802734, 2.7124223709106445])


@pytest.mark.parametrize("suffix", [".csv", ".json", ".yml"])
def test_save_one_factor_writes_flat_keys_and_loads_nested_keys(tmp_path, suffix):
    """A Bézier simplex is saved with flat keys, and the same file with nested keys loads the same."""
    bs = BezierSimploid.randn((2,), 2, n_values=2)
    path = _saved(tmp_path, bs, suffix)
    text = path.read_text()
    assert "(2, 0, 0)" in text and "((2, 0, 0),)" not in text
    nested = tmp_path / f"nested{suffix}"
    nested.write_text(
        "".join(line.replace("(", "((", 1).replace(")", "),)", 1) if "(" in line else line for line in text.splitlines(True))
    )
    assert "((2, 0, 0),)" in nested.read_text()
    for loaded in (torch_bsf.simploidal.load(path), torch_bsf.simploidal.load(nested)):
        assert (loaded.simploid_type, loaded.degree) == ((2,), (2,))
        assert torch.equal(loaded.control_points, bs.control_points.detach())


@pytest.mark.parametrize("suffix", [".csv", ".json", ".yml"])
def test_load_one_coordinate_flat_key_as_vertex(tmp_path, suffix):
    """A flat key with one coordinate, written by earlier versions for one parameter, loads as a vertex."""
    path = tmp_path / f"legacy{suffix}"
    path.write_text({".csv": '"(3,)",1.0,2.0\n', ".json": '{"(3,)": [1.0, 2.0]}', ".yml": '"(3,)": [1.0, 2.0]\n'}[suffix])
    loaded = torch_bsf.simploidal.load(path)
    assert (loaded.simploid_type, loaded.degree) == ((), ())
    assert loaded(torch.empty(4, 0)).tolist() == [[1.0, 2.0]] * 4


def test_load_pt_requires_a_dictionary(tmp_path):
    """.pt files are loaded with weights_only=True and must hold the dictionary written by save."""
    bs = BezierSimploid.randn((2, 1), 1, n_values=2)
    data = torch.load(_saved(tmp_path, bs, ".pt"), weights_only=True)
    assert data["simploid_type"] == [2, 1] and data["degree"] == [1, 1]
    assert torch.equal(data["control_points"], bs.control_points.detach())
    torch.save({"control_points": torch.zeros(1, 1)}, tmp_path / "dict.pt")
    with pytest.raises(ValueError, match="must contain a dictionary"):
        torch_bsf.simploidal.load(tmp_path / "dict.pt")
    torch.save(torch.nn.Linear(1, 1), tmp_path / "module.pt")  # a pickled module, e.g. of an old version
    with pytest.raises(ValueError, match="must contain a dictionary"):
        torch_bsf.simploidal.load(tmp_path / "module.pt")


def test_save_unknown_file_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown file type"):
        torch_bsf.simploidal.save(tmp_path / "model.txt", BezierSimploid.zeros((1,), 1, 1))


def test_bezier_simploid_pickles_under_public_path():
    import pickle

    bsd = BezierSimploid.randn((2, 1), 2, n_values=2)
    assert type(bsd).__module__ == "torch_bsf.simploidal"
    restored = pickle.loads(pickle.dumps(bsd))
    t = sample_simploid((2, 1), 5, seed=0)
    assert torch.equal(restored(t), bsd(t))


# ---------------------------------------------------------------------------
# Smoothness penalty and frozen control points
# ---------------------------------------------------------------------------


def _laplacian(adjacency: torch.Tensor, n: int) -> torch.Tensor:
    lap = torch.zeros(n, n, dtype=torch.float64)
    for p, q in adjacency.tolist():
        lap[p, p] += 1
        lap[q, q] += 1
        lap[p, q] -= 1
        lap[q, p] -= 1
    return lap


def _noisy(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    return x + 0.05 * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("n_params", [2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_adjacency_of_one_factor(n_params, degree):
    """Adjacent control points of a Bézier simplex differ by moving one unit between two components."""
    simploid = BezierSimploid.zeros((n_params - 1,), degree, n_values=1)
    rows = [alpha[0] for alpha in simploid.indices()]
    expected = {
        (a, b)
        for a in rows
        for b in rows
        if a < b and sorted(x - y for x, y in zip(a, b)) == [-1] + [0] * (n_params - 2) + [1]
    }
    actual = {tuple(sorted((rows[p], rows[q]))) for p, q in simploid.adjacency().tolist()}
    assert actual == expected
    assert len(simploid.adjacency()) == len(expected)


@pytest.mark.parametrize("simploid_type,degree", [((1, 1), 1), ((1, 1), (2, 3)), ((2, 1), 2), ((1, 1, 1), 1)])
def test_adjacency_of_several_factors(simploid_type, degree):
    """Adjacent control points differ in one factor by moving one unit between two components."""
    bs = BezierSimploid.zeros(simploid_type, degree, n_values=1)
    indices = bs.indices()

    def adjacent(alpha, beta):
        diffs = [k for k in range(len(alpha)) if alpha[k] != beta[k]]
        if len(diffs) != 1:
            return False
        d = [u - v for u, v in zip(alpha[diffs[0]], beta[diffs[0]])]
        return sorted(d) == [-1] + [0] * (len(d) - 2) + [1]

    expected = [(p, q) for p in range(len(indices)) for q in range(p + 1, len(indices)) if adjacent(indices[p], indices[q])]
    assert bs.adjacency().tolist() == [list(pq) for pq in expected]
    if simploid_type == (1, 1) and degree == 1:  # the four edges of a bilinear patch
        assert bs.adjacency().tolist() == [[0, 1], [0, 2], [1, 3], [2, 3]]


def test_space_adjacency_counts_shared_faces_once():
    S = SimploidalSet()
    t1, t2 = S.add_simplex(2), S.add_simplex(2)
    S.identify(t1.face(0, 0), t2.face(0, 0))
    space = SimploidalBezierSpace(S, degree=2, n_values=1)
    pairs = space.adjacency()
    assert len({tuple(p) for p in pairs.tolist()}) == len(pairs)
    assert bool((pairs[:, 0] < pairs[:, 1]).all())
    one = BezierSimploid.zeros((2,), 2, n_values=1).adjacency()
    shared = BezierSimploid.zeros((1,), 2, n_values=1).adjacency()
    assert len(pairs) == 2 * len(one) - len(shared)
    for s in (t1, t2):
        rows = space.control_point_rows(s)
        local = {tuple(sorted((rows[p], rows[q]))) for p, q in one.tolist()}
        assert local <= {tuple(p) for p in pairs.tolist()}


def test_fit_simploid_smoothness_is_exact():
    t = sample_simploid((2, 1), 60, seed=0)
    x = _noisy(torch.sin(3 * t[:, :2]))
    w = 0.01
    fitted = fit(t, x, 2, simploid_type=(2, 1), smoothness_weight=w)
    a = bernstein_basis((2, 1), 2, t).double()
    lap = _laplacian(fitted.adjacency(), a.shape[1])
    expected = torch.linalg.solve(a.T @ a + x.numel() * w * lap, a.T @ x.double())
    assert torch.allclose(fitted.control_points.double(), expected, atol=1e-5)
    # The penalty does smooth the control net.
    rough = fit(t, x, 2, simploid_type=(2, 1))
    i, j = fitted.adjacency().T
    assert (fitted.control_points[i] - fitted.control_points[j]).pow(2).sum() < (
        rough.control_points[i] - rough.control_points[j]
    ).pow(2).sum()


def test_fit_simploid_smoothness_trainer_matches_lstsq():
    t = sample_simploid((2, 1), 100, seed=0)
    x = _noisy(torch.sin(3 * t[:, :2]))
    exact = fit(t, x, 2, simploid_type=(2, 1), smoothness_weight=0.01)
    trained = fit(
        t, x, 2, simploid_type=(2, 1), smoothness_weight=0.01, solver="trainer", seed=0, **TRAINER_CONVERGENCE_KWARGS
    )
    assert torch.allclose(trained.control_points, exact.control_points, atol=1e-3)


@pytest.mark.parametrize("solver", ["lstsq", "trainer"])
@pytest.mark.parametrize("smoothness_weight", [0.0, 0.01])
def test_fit_simploid_freeze_keeps_init(solver, smoothness_weight):
    t = sample_simploid((2, 1), 60, seed=0)
    x = _noisy(torch.sin(3 * t[:, :2]))
    init = BezierSimploid.randn((2, 1), 2, n_values=2)
    freeze = [((2, 0, 0), (2, 0)), ((0, 1, 1), (1, 1))]
    kwargs = dict(seed=0, **TRAINER_KWARGS) if solver == "trainer" else {}
    fitted = fit(
        t,
        x,
        2,
        simploid_type=(2, 1),
        init=init,
        freeze=freeze,
        smoothness_weight=smoothness_weight,
        solver=solver,
        **kwargs,
    )
    for index in freeze:
        assert torch.equal(fitted[index], init[index])
    assert not torch.equal(fitted.control_points, init.control_points)
    # The frozen rows are no longer masked once fitting is done.
    fitted(t).sum().backward()
    assert fitted.control_points.grad[fitted.row(freeze[0])].abs().sum() > 0


def test_fit_simploid_freeze_is_exact():
    """lstsq with frozen rows solves the reduced system for the free rows."""
    t = sample_simploid((2,), 40, seed=0)
    x = _noisy(t[:, :1] ** 2)
    init = BezierSimploid.randn((2,), 3, n_values=1)
    freeze = [((3, 0, 0),), ((0, 3, 0),), ((0, 0, 3),)]
    w = 0.1
    fitted = fit(t, x, 3, simploid_type=(2,), init=init, freeze=freeze, smoothness_weight=w)
    a = bernstein_basis((2,), 3, t).double()
    lap = _laplacian(fitted.adjacency(), a.shape[1])
    frozen = torch.tensor([init.row(i) for i in freeze])
    free = torch.ones(a.shape[1], dtype=torch.bool)
    free[frozen] = False
    p0 = init.control_points.detach().double()
    c = x.numel() * w
    lhs = a[:, free].T @ a[:, free] + c * lap[free][:, free]
    rhs = a[:, free].T @ (x.double() - a[:, ~free] @ p0[~free]) - c * lap[free][:, ~free] @ p0[~free]
    expected = torch.linalg.solve(lhs, rhs)
    assert torch.allclose(fitted.control_points.double()[free], expected, atol=1e-5)


def test_fit_simploid_init_without_freeze():
    t = sample_simploid((1,), 30, seed=0)
    x = t[:, :1]
    init = {((1, 0),): [5.0], ((0, 1),): [5.0]}
    assert torch.allclose(fit(t, x, 1, simploid_type=(1,), init=init).control_points, torch.tensor([[1.0], [0.0]]), atol=1e-5)
    trained = fit(t, x, 1, simploid_type=(1,), init=torch.full((2, 1), 5.0), solver="trainer", max_epochs=0, logger=False)
    assert torch.equal(trained.control_points, torch.full((2, 1), 5.0))  # no epoch: the trainer starts from init


def test_fit_simploid_all_rows_frozen():
    t = sample_simploid((1,), 10, seed=0)
    init = BezierSimploid.randn((1,), 1, n_values=1)
    fitted = fit(t, t[:, :1], 1, simploid_type=(1,), init=init, freeze=init.indices())
    assert torch.equal(fitted.control_points, init.control_points)


def test_fit_simploid_smoothness_and_freeze_errors():
    t = sample_simploid((2,), 10, seed=0)
    x = t[:, :1]
    with pytest.raises(ValueError, match="init is required"):
        fit(t, x, 2, simploid_type=(2,), freeze=[((2, 0, 0),)])
    with pytest.raises(ValueError, match="smoothness_weight"):
        fit(t, x, 2, simploid_type=(2,), smoothness_weight=-1.0)
    init = BezierSimploid.zeros((2,), 2, n_values=1)
    with pytest.raises(ValueError, match="not a control point index"):
        fit(t, x, 2, simploid_type=(2,), init=init, freeze=[((3, 0, 0),)])
    prism = BezierSimploid.zeros((2, 1), 2, n_values=1)
    with pytest.raises(ValueError, match="not a control point index"):
        fit(sample_simploid((2, 1), 10), x, 2, simploid_type=(2, 1), init=prism, freeze=[(2, 0, 0)])
    with pytest.raises(ValueError, match="ridge must be non-negative"):  # even if no row is free
        fit(t, x, 2, simploid_type=(2,), init=init, freeze=simploid_indices((2,), 2), ridge=-1.0)
    with pytest.raises(ValueError, match="degree"):
        fit(t, x, 3, simploid_type=(2,), init=init)
    with pytest.raises(ValueError, match="values"):
        fit(t, x, 2, simploid_type=(2,), init=BezierSimploid.zeros((2,), 2, n_values=2))


def _two_triangles_with_data():
    S = SimploidalSet()
    t1, t2 = S.add_simplex(2), S.add_simplex(2)
    S.identify(t1.face(0, 0), t2.face(0, 0))
    data = []
    for k, s in enumerate((t1, t2)):
        p = sample_simploid((2,), 50, seed=k)
        data.append((s, p, _noisy(torch.sin(3 * p[:, :2]), seed=k)))
    return S, (t1, t2), data


def _stacked(space, data):
    width = max(p.shape[1] for _, p, _ in data)
    t = torch.cat([torch.nn.functional.pad(p, (0, width - p.shape[1])) for _, p, _ in data])
    ids = torch.cat([torch.full((len(p),), space.simploid_id(s)) for s, p, _ in data])
    return t, ids, torch.cat([x for _, _, x in data])


def test_fit_simploidal_set_smoothness_is_exact():
    S, _, data = _two_triangles_with_data()
    w = 0.01
    space = fit_simploidal_set(S, data, 2, smoothness_weight=w)
    t, ids, x = _stacked(space, data)
    a = space.design_matrix(t.double(), ids)
    lap = _laplacian(space.adjacency(), a.shape[1])
    expected = torch.linalg.solve(a.T @ a + x.numel() * w * lap, a.T @ x.double())
    assert torch.allclose(space.proper_control_points.double(), expected, atol=1e-5)


def test_fit_simploidal_set_smoothness_trainer_matches_lstsq():
    S, _, data = _two_triangles_with_data()
    exact = fit_simploidal_set(S, data, 2, smoothness_weight=0.01)
    trained = fit_simploidal_set(S, data, 2, smoothness_weight=0.01, solver="trainer", seed=0, **TRAINER_CONVERGENCE_KWARGS)
    assert torch.allclose(trained.proper_control_points, exact.proper_control_points, atol=1e-3)


@pytest.mark.parametrize("solver", ["lstsq", "trainer"])
def test_fit_simploidal_set_freeze_keeps_init(solver):
    S, (t1, t2), data = _two_triangles_with_data()
    init = _random_space(S, 2, n_values=2)
    shared = t1.face(0, 0)
    freeze = [(t1, ((2, 0, 0),)), (shared, ((1, 1),)), (t2, ((1, 1, 0),))]
    kwargs = dict(seed=0, **TRAINER_KWARGS) if solver == "trainer" else {}
    space = fit_simploidal_set(S, data, 2, init=init, freeze=freeze, smoothness_weight=0.01, solver=solver, **kwargs)
    for simploid, index in freeze:
        row = space.row(simploid, index)
        assert torch.equal(space.proper_control_points[row], init.proper_control_points[row])
    assert not torch.equal(space.proper_control_points, init.proper_control_points)
    # A tensor of proper control points is accepted as init too.
    again = fit_simploidal_set(S, data, 2, init=init.proper_control_points.detach(), freeze=freeze, smoothness_weight=0.01)
    if solver == "lstsq":
        assert torch.allclose(again.proper_control_points, space.proper_control_points)


def test_fit_simploidal_set_freeze_keeps_float64_init():
    """A float64 init is not rounded to the default dtype before the space is converted to float64."""
    S, (t1, _), data = _two_triangles_with_data()
    data = [(s, t.double(), x.double()) for s, t, x in data]
    init = _random_space(S, 2, n_values=2).double()
    with torch.no_grad():
        init.proper_control_points.add_(1e-10)  # not representable in float32
    freeze = [(t1, ((2, 0, 0),))]
    space = fit_simploidal_set(S, data, 2, init=init, freeze=freeze)
    row = space.row(*freeze[0])
    assert torch.equal(space.proper_control_points[row], init.proper_control_points[row])


def test_fit_simploidal_set_smoothness_and_freeze_errors():
    S, (t1, _), data = _two_triangles_with_data()
    with pytest.raises(ValueError, match="init is required"):
        fit_simploidal_set(S, data, 2, freeze=[(t1, ((2, 0, 0),))])
    with pytest.raises(ValueError, match="smoothness_weight"):
        fit_simploidal_set(S, data, 2, smoothness_weight=-1.0)
    init = _random_space(S, 2, n_values=2)
    with pytest.raises(ValueError, match="not a control point index"):
        fit_simploidal_set(S, data, 2, init=init, freeze=[(t1, ((3, 0, 0),))])
    with pytest.raises(ValueError, match="shape"):
        fit_simploidal_set(S, data, 2, init=torch.zeros(3, 2))


# ---------------------------------------------------------------------------
# The unified fit and fit_kfold
# ---------------------------------------------------------------------------


def test_fit_infers_a_simplex():
    from torch_bsf.sampling import simplex_grid

    t = simplex_grid(3, 4)
    x = t**2
    bs = torch_bsf.fit(t, x, degree=2)
    assert isinstance(bs, BezierSimploid)
    assert (bs.simploid_type, bs.degree) == ((2,), (2,))
    assert torch.allclose(bs(t), x, atol=1e-5)


def test_fit_needs_a_type_for_fewer_than_two_columns():
    with pytest.raises(ValueError, match="simploid_type"):
        torch_bsf.fit(torch.ones(3, 1), torch.zeros(3, 1), degree=1)
    constant = torch_bsf.fit(torch.ones(3, 0), torch.tensor([[1.0], [2.0], [3.0]]), degree=1, simploid_type=())
    assert constant.control_points.tolist() == [[2.0]]


def test_fit_takes_type_and_degree_from_init():
    t = sample_simploid((2, 1), 40, seed=0)
    init = BezierSimploid.zeros((2, 1), (2, 1), n_values=1)
    bs = torch_bsf.fit(t, t[:, :1] * t[:, 3:4], init=init)
    assert (bs.simploid_type, bs.degree) == ((2, 1), (2, 1))
    with pytest.raises(ValueError, match="degree is required"):
        torch_bsf.fit(t, t[:, :1], init=torch.zeros(9, 1), simploid_type=(2, 1))
    with pytest.raises(ValueError, match="init has type"):
        torch_bsf.fit(t, t[:, :1], degree=3, init=init)


@pytest.mark.parametrize("normalize", ["none", "max", "std", "quantile"])
@pytest.mark.parametrize("smoothness_weight", [0.0, 0.01])
def test_fit_normalize_predicts_in_original_scale(normalize, smoothness_weight):
    t = sample_simploid((2, 1), 60, seed=0).double()
    x = torch.cat([1000 * t[:, :1] * t[:, 3:4] + 50, -0.01 * t[:, 1:2]], dim=1)
    bs = torch_bsf.fit(t, x, 2, simploid_type=(2, 1), normalize=normalize)
    assert torch.allclose(bs(t), x, atol=1e-8 * 1000)
    if smoothness_weight:
        # The normalization weights the columns of the objective: the model of the normalized values,
        # mapped back, is the fitted model.
        from torch_bsf.preprocessing import make_scaler

        scaler = make_scaler(normalize)
        scaled = torch_bsf.fit(t, scaler.fit_transform(x), 2, simploid_type=(2, 1), smoothness_weight=smoothness_weight)
        bs = torch_bsf.fit(t, x, 2, simploid_type=(2, 1), normalize=normalize, smoothness_weight=smoothness_weight)
        assert torch.allclose(bs.control_points, scaler.inverse_transform(scaled.control_points.detach()), atol=1e-9)


def test_fit_normalize_keeps_frozen_rows_exactly():
    t = sample_simploid((2,), 30, seed=0)
    x = 1e3 * t[:, :2] + 7.0
    init = BezierSimploid.randn((2,), 2, n_values=2)
    fitted = torch_bsf.fit(t, x, init=init, freeze=[(2, 0, 0)], normalize="std", smoothness_weight=0.1)
    assert torch.equal(fitted[(2, 0, 0)], init[(2, 0, 0)])


def test_fit_rejects_trainer_options_with_lstsq():
    t = sample_simploid((1,), 5, seed=0)
    with pytest.raises(ValueError, match="max_epochs is supported only by the 'trainer' solver"):
        torch_bsf.fit(t, t, 1, max_epochs=3)
    with pytest.raises(ValueError, match="batch_size, logger are supported only"):
        torch_bsf.fit(t, t, 1, batch_size=2, logger=False)
    with pytest.raises(ValueError, match="normalize"):
        torch_bsf.fit(t, t, 1, normalize="unknown")  # type: ignore[arg-type]


def test_bezier_simplex_accepts_flat_indices():
    bs = BezierSimploid((2,), 1, {(1, 0, 0): [1.0], (0, 1, 0): [2.0], ((0, 0, 1),): [3.0]})
    assert bs[(0, 1, 0)].item() == 2.0
    assert bs.row((0, 0, 1)) == bs.row(((0, 0, 1),)) == 2
    with pytest.raises(ValueError, match="not a control point index"):
        BezierSimploid.zeros((1, 1), 1, n_values=1).row((1, 0))


def test_fit_kfold_fits_each_model_to_the_other_folds():
    t = sample_simploid((2, 1), 23, seed=0)
    x = t[:, :1] * t[:, 3:4] + 0.1 * torch.randn(23, 1, generator=torch.Generator().manual_seed(0))
    models = torch_bsf.fit_kfold(t, x, 1, n_folds=4, shuffle=False, simploid_type=(2, 1))
    assert len(models) == 4
    for model, test in zip(models, torch.tensor_split(torch.arange(23), 4)):
        train = torch.ones(23, dtype=torch.bool)
        train[test] = False
        expected = torch_bsf.fit(t[train], x[train], 1, simploid_type=(2, 1))
        assert torch.allclose(model.control_points, expected.control_points)


def test_fit_kfold_freezes_in_every_fold_given_an_iterator():
    from torch_bsf.sampling import simplex_grid

    t = simplex_grid(3, 3)
    x = t[:, :1]
    init = BezierSimploid.zeros((2,), 1, n_values=1)
    vertex = (1, 0, 0)
    models = torch_bsf.fit_kfold(t, x, 1, n_folds=3, seed=0, init=init, freeze=iter([vertex]))
    for model in models:
        assert model[vertex].item() == 0.0


def test_fit_kfold_rejects_mismatched_sample_counts():
    t = sample_simploid((2,), 6, seed=0)
    with pytest.raises(ValueError, match="params has 6 samples, but values has 7 samples"):
        torch_bsf.fit_kfold(t, torch.zeros(7, 1), 1, n_folds=3)


def test_fit_rejects_empty_data():
    with pytest.raises(ValueError, match="At least one sample"):
        torch_bsf.fit(torch.zeros(0, 3), torch.zeros(0, 1), 1)


def test_fit_std_normalization_with_a_single_sample():
    t = torch.tensor([[0.2, 0.3, 0.5]])
    bs = torch_bsf.fit(t, torch.tensor([[1.5]]), 0, normalize="std")
    assert torch.allclose(bs(t), torch.tensor([[1.5]]))


def test_fit_kfold_shuffles_reproducibly_and_caps_folds():
    from torch_bsf.sampling import simplex_grid

    t = simplex_grid(2, 2)
    x = t[:, :1]
    first = torch_bsf.fit_kfold(t, x, 1, n_folds=5, seed=0)
    second = torch_bsf.fit_kfold(t, x, 1, n_folds=5, seed=0)
    assert len(first) == 3  # capped at the number of samples
    for a, b in zip(first, second):
        assert torch.equal(a.control_points, b.control_points)
    with pytest.raises(ValueError, match="n_folds"):
        torch_bsf.fit_kfold(t, x, 1, n_folds=1)
    with pytest.raises(ValueError, match="At least 2 samples"):
        torch_bsf.fit_kfold(t[:1], x[:1], 1)
