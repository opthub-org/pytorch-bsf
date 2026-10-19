from math import comb, prod

import pytest
import torch

import torch_bsf
from torch_bsf.bezier_simplex import BezierSimplex
from torch_bsf.simploidal import (
    BezierSimploid,
    Simploid,
    SimploidalBezierSpace,
    SimploidalSet,
    bernstein_basis,
    fit_simploid,
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


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_bezier_simploid_matches_bezier_simplex(dim, degree):
    bsd = BezierSimploid.randn((dim,), degree, n_values=3)
    bs = BezierSimplex({alpha[0]: bsd[alpha].tolist() for alpha in bsd.indices()})
    t = sample_simploid((dim,), 15, seed=1)
    assert torch.allclose(bsd(t), bs(t), atol=1e-5)


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


def test_bezier_simploid_hparams_rebuild():
    bsd = BezierSimploid.randn((2, 1), 2, n_values=3)
    rebuilt = BezierSimploid(**bsd.hparams)
    assert rebuilt.control_points.shape == bsd.control_points.shape
    rebuilt.load_state_dict(bsd.state_dict())
    t = sample_simploid((2, 1), 5, seed=0)
    assert torch.equal(rebuilt(t), bsd(t))


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
    from torch_bsf.simploidal import _check_commutation

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
    fitted = fit_simploid(t, truth(t).detach(), simploid_type, degree)
    assert torch.allclose(fitted.control_points, truth.control_points, atol=1e-3)


def test_fit_simploid_ridge_and_errors():
    t = sample_simploid((2,), 3, seed=0)
    x = t[:, :1]
    fitted = fit_simploid(t, x, (2,), 3, ridge=1e-3)  # under-determined: 10 control points, 3 samples
    assert fitted.control_points.shape == (10, 1)
    assert torch.isfinite(fit_simploid(t, x, (2,), 3).control_points).all()
    with pytest.raises(ValueError):
        fit_simploid(t, x, (2,), 3, solver="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit_simploid(t, x, (2,), 3, solver="trainer", ridge=1.0)
    with pytest.raises(ValueError):
        fit_simploid(t, x, (1,), 3)
    with pytest.raises(ValueError):
        fit_simploid(t, x, (2,), 3, ridge=-1.0)


def test_fit_simploid_trainer():
    truth = BezierSimploid.randn((2, 1), 1, n_values=2)
    t = sample_simploid((2, 1), 100, seed=0)
    x = truth(t).detach()
    fitted = fit_simploid(t, x, (2, 1), 1, solver="trainer", seed=0, **TRAINER_KWARGS)
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
    bs = fit_simploid(t, x, (2,), 2, solver=solver, **kwargs)
    assert bs.control_points.dtype == torch.float64
    assert bs(t).dtype == torch.float64
    S = SimploidalSet()
    tri = S.add_simplex(2)
    space = fit_simploidal_set(S, [(tri, t, x)], 2, solver=solver, **kwargs)
    assert space.proper_control_points.dtype == torch.float64
    if solver == "lstsq":
        assert torch.allclose(space.evaluate(tri, t), x, atol=1e-12)


def test_public_exports():
    for name in ["BezierSimploid", "Simploid", "SimploidalSet", "SimploidalBezierSpace", "fit_simploid", "fit_simploidal_set"]:
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

    trainer = L.Trainer(accelerator="cpu", enable_progress_bar=False, logger=False, enable_model_summary=False)
    t = sample_simploid((2, 1), 10, seed=0)
    bsd = BezierSimploid.randn((2, 1), 1, n_values=1)
    dl = DataLoader(TensorDataset(t, bsd(t).detach()))
    assert trainer.validate(bsd, dl, verbose=False)[0]["val_mse"] < 1e-10
    assert trainer.test(bsd, dl, verbose=False)[0]["test_mse"] < 1e-10

    S = SimploidalSet()
    tri = S.add_simplex(2)
    space = _random_space(S, 1, n_values=1)
    p = sample_simploid((2,), 10, seed=0)
    ids = torch.full((10,), space.simploid_id(tri))
    dl = DataLoader(TensorDataset(ids, p, space.evaluate(tri, p).detach()))
    assert trainer.validate(space, dl, verbose=False)[0]["val_mae"] < 1e-5
    assert trainer.test(space, dl, verbose=False)[0]["test_mae"] < 1e-5


def test_least_squares_module_requires_predict_batch():
    from torch_bsf.simploidal import _LeastSquaresModule

    with pytest.raises(NotImplementedError):
        _LeastSquaresModule()._predict_batch(None)


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
        fit_simploid(t, torch.zeros(2, 1), (1,), 1)
