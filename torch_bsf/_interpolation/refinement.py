"""Lattice meshes and conforming bisection of :mod:`torch_bsf.interpolation`."""

import operator
from collections.abc import Sequence
from itertools import accumulate, combinations, combinations_with_replacement
from typing import Literal

import torch

from torch_bsf._interpolation.exact import _length_key
from torch_bsf._interpolation.mesh import SimplicialMesh, _Tagged
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._validation import _choice


def lattice_mesh(dim: int, k: int) -> SimplicialMesh:
    r"""The Kuhn (Freudenthal) triangulation of the degree-``k`` lattice of the standard ``dim``-simplex.

    The vertices are the lattice points :math:`a / k` with :math:`a \in \mathbb{N}^{d + 1}` and
    :math:`|a| = k`, in the order of :func:`~torch_bsf.sampling.grid` with ``resolution=k``. Their
    coordinates are the first ``dim`` barycentric coordinates (the last one is one minus their sum).
    The ``k ** dim`` simplices have equal volume, have lattice points as vertices and meet face to face.
    They are not congruent for ``dim >= 3``, because the map from the staircase coordinates below to
    these coordinates is not an isometry.
    In the staircase coordinates :math:`z_j = a_0 + \dots + a_j` (:math:`j < d`) the lattice is
    :math:`\{0 \le z_0 \le \dots \le z_{d-1} \le k\}`, and the simplices are the Kuhn simplices
    :math:`(c, c + e_{\pi(1)}, c + e_{\pi(1)} + e_{\pi(2)}, \dots)` of unit cubes whose vertices all lie in it.
    The mesh records these paths, so that :func:`refine` with ``rule="newest_vertex"`` can bisect each
    simplex across the diagonal of its cube.

    Lattice points are co-spherical, so their Delaunay triangulation is not unique; qhull returns flat
    simplices, and removing them leaves a mesh that is not conforming. Use this mesh instead.

    Parameters
    ----------
    dim
        Dimension of the simplex, at least one.
    k
        Lattice degree (number of intervals per edge), at least one.

    Returns
    -------
    SimplicialMesh
        A conforming mesh with ``comb(k + dim, dim)`` vertices and ``k ** dim`` simplices.

    Examples
    --------
    >>> from torch_bsf.interpolation import lattice_mesh
    >>> mesh = lattice_mesh(2, 3)
    >>> mesh.n_vertices, mesh.n_simplices, mesh.unmatched_faces
    (10, 9, 0)
    >>> float(mesh.volumes().sum())
    0.5
    """
    for name, value in (("dim", dim), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    lattice = [alpha for (alpha,) in _indices((dim,), (k,))]
    number = {tuple(accumulate(a[:dim])): n for n, a in enumerate(lattice)}
    simplices = []
    newest_vertex: dict[tuple[int, ...], _Tagged] = {}
    # A Kuhn simplex stays in the lattice exactly when its corner is non-decreasing with entries below k and,
    # wherever two consecutive entries are equal, the later axis is stepped first; every such simplex is
    # generated once, in the lexicographic order of (corner, axis order).
    for corner in combinations_with_replacement(range(k), dim):
        for order in _orders(corner):
            z = list(corner)
            path = [number[tuple(z)]]
            for axis in order:
                z[axis] += 1
                path.append(number[tuple(z)])
            simplices.append(path)
            newest_vertex[tuple(sorted(path))] = _Tagged(tuple(path), dim, 0)
    vertices = torch.tensor(lattice, dtype=torch.float64)[:, :dim] / k
    mesh = SimplicialMesh(vertices, simplices)
    # Newest-vertex bisection starts from the Kuhn paths with tag d: the refinement edge is the cube diagonal.
    mesh._newest_vertex = newest_vertex
    return mesh


def _orders(corner: tuple[int, ...]) -> list[list[int]]:
    """The axis orders, in lexicographic order, in which axis ``j`` comes after ``j + 1`` if they tie in ``corner``."""
    dim = len(corner)
    orders: list[list[int]] = []
    order: list[int] = []
    done = [False] * (dim + 1)
    done[dim] = True

    def extend() -> None:
        if len(order) == dim:
            orders.append(order.copy())
            return
        for axis in range(dim):
            if not done[axis] and (done[axis + 1] or corner[axis] != corner[axis + 1]):
                done[axis] = True
                order.append(axis)
                extend()
                order.pop()
                done[axis] = False

    extend()
    return orders


class _Bisection:
    """Mutable state of :func:`refine`: vertices, live simplices and the simplices around each edge.

    With ``newest_vertex`` (newest-vertex bisection), every live simplex also has its :class:`_Tagged` simplex.
    """

    def __init__(
        self,
        x: list[torch.Tensor],
        cells: list[tuple[int, ...]],
        newest_vertex: dict[tuple[int, ...], _Tagged] | None = None,
    ):
        self.x = x
        self.alive: set[tuple[int, ...]] = set()
        self.cells_of: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        self.newest_vertex = None if newest_vertex is None else dict(newest_vertex)
        for cell in cells:
            self.add(cell)

    def add(self, cell: tuple[int, ...]) -> None:
        self.alive.add(cell)
        for e in combinations(cell, 2):
            self.cells_of.setdefault(e, set()).add(cell)

    def remove(self, cell: tuple[int, ...]) -> None:
        self.alive.discard(cell)
        for e in combinations(cell, 2):
            self.cells_of[e].discard(cell)

    def edge(self, cell: tuple[int, ...]) -> tuple[int, int]:
        """The refinement edge: ``x0 xk`` of the ordered vertices with tag ``k``, or else the longest edge."""
        if self.newest_vertex is not None:
            tagged = self.newest_vertex[cell]
            a, b = tagged.vertices[0], tagged.vertices[tagged.tag]
            return (a, b) if a < b else (b, a)
        return self.longest(cell)

    def longest(self, cell: tuple[int, ...]) -> tuple[int, int]:
        """The longest edge; ties are broken by vertex numbers, the same way in every simplex."""
        return max(combinations(cell, 2), key=lambda e: (_length_key(self.x[e[0]], self.x[e[1]]), e))

    def bisect(self, e: tuple[int, int]) -> None:
        """Split every simplex containing edge ``e`` at its midpoint, which keeps the mesh conforming."""
        m = len(self.x)
        a, b = self.x[e[0]], self.x[e[1]]
        # The halved sum is correctly rounded (halving is exact for normal numbers and rounds only once for
        # subnormal ones); the sum of the halves is used only where the sum overflows.
        total = a + b
        self.x.append(torch.where(torch.isfinite(total), total / 2, a / 2 + b / 2))
        for cell in sorted(self.cells_of[e]):
            self.remove(cell)
            if self.newest_vertex is None:
                children = [tuple(sorted(m if w == v else w for w in cell)) for v in e]
            else:
                # Maubach: (x0, ..., x_{k-1}, z, x_{k+1}, ...) and (x1, ..., xk, z, x_{k+1}, ...), tag k - 1 or d.
                tagged = self.newest_vertex.pop(cell)
                ordered, tag = tagged.vertices, tagged.tag
                new_tag = tag - 1 if tag > 1 else len(cell) - 1
                halves = (
                    ordered[:tag] + (m,) + ordered[tag + 1 :],
                    ordered[1 : tag + 1] + (m,) + ordered[tag + 1 :],
                )
                children = []
                for half in halves:
                    child = tuple(sorted(half))
                    self.newest_vertex[child] = _Tagged(half, new_tag, tagged.generation + 1)
                    children.append(child)
            for child in children:
                self.add(child)
        del self.cells_of[e]

    def refine(self, target: tuple[int, ...]) -> None:
        stack = [target]
        while stack:
            cell = stack[-1]
            if cell not in self.alive:
                stack.pop()
                continue
            e = self.edge(cell)
            other = next((c for c in sorted(self.cells_of[e]) if self.edge(c) != e), None)
            if other is None:
                self.bisect(e)
            else:
                # Longest edge: the refinement edge of `other` is longer than e, so the recursion ends. Newest vertex:
                # on Kuhn meshes `other` is of an earlier generation (Maubach 1995; Stevenson 2008), so it ends too.
                if (
                    self.newest_vertex is not None
                    and self.newest_vertex[other].generation >= self.newest_vertex[cell].generation
                ):
                    raise RuntimeError("newest-vertex bisection lost its matching condition; please report this.")
                stack.append(other)


def refine(
    mesh: SimplicialMesh,
    simplices: Sequence[int],
    *,
    rule: Literal["longest_edge", "newest_vertex"] = "longest_edge",
) -> SimplicialMesh:
    r"""Bisect simplices at the midpoint of their refinement edge, keeping the mesh conforming.

    An edge is bisected in every simplex that contains it, which keeps the mesh conforming. Before an
    edge is bisected, every simplex containing it whose refinement edge is a different one is refined
    first, recursively, so that cells are always cut across their own refinement edge. Refining one
    simplex may therefore add several vertices. The refinement edge depends on ``rule``:

    * ``"longest_edge"`` (Rivara): the longest edge, measured in the vertex coordinates. Ties between
      edges of equal length are broken by their vertex numbers, consistently over the whole mesh.
    * ``"newest_vertex"`` (Maubach :cite:p:`maubach1995local`): every simplex is an ordered vertex tuple
      :math:`(x_0, \dots, x_d)` with a tag :math:`k \in \{1, \dots, d\}`, its refinement edge is
      :math:`x_0 x_k`, and the children :math:`(x_0, \dots, x_{k-1}, z, x_{k+1}, \dots, x_d)` and
      :math:`(x_1, \dots, x_k, z, x_{k+1}, \dots, x_d)` get the tag :math:`k - 1` (:math:`d` after 1). It
      starts from a mesh of :func:`lattice_mesh`, whose simplices are the Kuhn paths with tag :math:`d`,
      and the refined mesh carries the ordered simplices and tags for the next call. It cuts a Kuhn
      simplex across the diagonal of its cube, so ``dim`` rounds of refining every simplex of
      ``lattice_mesh(dim, k)`` give a triangulation of the degree-``2k`` lattice, and the simplices take
      only finitely many shapes up to similarity :cite:p:`maubach1995local`. Longest-edge bisection of a
      Kuhn mesh does neither: the longest edge of a Kuhn simplex is not the cube diagonal in the
      coordinates of :func:`lattice_mesh`, so rounds of refinement do not return to a lattice mesh and
      closure adds vertices that were not asked for.

    Parameters
    ----------
    mesh
        A conforming mesh. For ``rule="newest_vertex"``, a mesh returned by :func:`lattice_mesh` or by
        this function with ``rule="newest_vertex"``.
    simplices
        Integer indices into ``mesh.simplices`` of the simplices to bisect.
    rule
        ``"longest_edge"`` (default) or ``"newest_vertex"``.

    Returns
    -------
    SimplicialMesh
        The refined mesh. It keeps the vertices of ``mesh`` with their numbers and appends the new
        vertices (edge midpoints) in the order they were created. Its simplices are sorted.

    Raises
    ------
    ValueError
        If ``rule`` is not one of the strings ``"longest_edge"`` and ``"newest_vertex"``, if an index
        is not one of a simplex, or if ``rule="newest_vertex"`` is used on a mesh that carries no
        newest-vertex state (e.g. one built directly with :class:`SimplicialMesh`).

    Examples
    --------
    >>> from torch_bsf.interpolation import lattice_mesh, refine
    >>> coarse = lattice_mesh(2, 2)
    >>> fine = refine(coarse, [0])
    >>> fine.n_vertices - coarse.n_vertices, fine.n_simplices - coarse.n_simplices, fine.unmatched_faces
    (1, 2, 0)

    Two rounds of newest-vertex bisection of every triangle double the lattice degree:

    >>> mesh = lattice_mesh(2, 2)
    >>> for _ in range(2):
    ...     mesh = refine(mesh, range(mesh.n_simplices), rule="newest_vertex")
    >>> mesh.n_vertices, mesh.n_simplices, mesh.unmatched_faces
    (15, 16, 0)
    >>> sorted(set((4 * mesh.vertices).flatten().tolist()))
    [0.0, 1.0, 2.0, 3.0, 4.0]
    """
    _choice(
        rule,
        "rule",
        ("longest_edge", "newest_vertex"),
        message=f"rule must be 'longest_edge' or 'newest_vertex', not {rule!r}.",
    )
    newest_vertex = None
    if rule == "newest_vertex":
        newest_vertex = mesh._newest_vertex
        if newest_vertex is None:
            raise ValueError("rule='newest_vertex' needs a mesh from lattice_mesh or from refine(..., rule='newest_vertex').")
    cells = [tuple(c) for c in mesh.simplices.tolist()]
    try:
        # operator.index accepts integers of any kind (Python, NumPy, 0-d tensors) but not floats or strings.
        if any(isinstance(k, bool) or (isinstance(k, torch.Tensor) and k.dtype == torch.bool) for k in simplices):
            raise TypeError
        targets = [cells[operator.index(k)] for k in simplices]
    except (IndexError, TypeError):
        raise ValueError("simplices must be indices into mesh.simplices.") from None
    state = _Bisection(list(mesh.vertices), cells, newest_vertex)
    for target in targets:
        state.refine(target)
    refined = SimplicialMesh(torch.stack(state.x), sorted(state.alive))
    refined._newest_vertex = state.newest_vertex
    return refined
