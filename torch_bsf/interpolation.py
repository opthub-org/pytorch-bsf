r"""Shape-preserving cubic Bézier interpolation on simplicial complexes ("simplicial PCHIP").

A :class:`SimplicialMesh` glues user-supplied simplices of :math:`\mathbb{R}^d` into one
:class:`~torch_bsf.core.BezierSpace`, so that simplices sharing a face share the controls on it.
:func:`hermite_interpolant` fills that space with a cubic Bézier model that interpolates vertex data:

* a vertex control is the datum :math:`f_i`;
* the control on edge :math:`i \to j` next to vertex :math:`i` is :math:`f_i + g_i \cdot (x_j - x_i) / 3`,
  with a vertex gradient :math:`g_i`;
* the control at the centre of face :math:`(i, j, l)` is the sum of the six edge controls of the face
  divided by four, minus :math:`(f_i + f_j + f_l) / 6`, which reproduces quadratic data exactly.

Every control depends only on the data of the edge or face it lies on, so the interpolant is
continuous across shared faces. With ``clamp=True`` every edge (face) control is clipped into the
range of the data at the vertices of that edge (face). By the convex-hull property, the prediction
on each simplex then stays within the range of its vertex data, which bounds the sup error by the
modulus of continuity of the data at the mesh size. In one dimension this is the range limiter of
piecewise cubic Hermite interpolation (PCHIP) :cite:p:`fritsch1980monotone`.
"""

from collections.abc import Sequence
from itertools import combinations
from math import factorial
from typing import Any

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._kernels.sets import Simploid, SimploidalSet
from torch_bsf.core import BezierModel, BezierSpace, Points

__all__ = ["SimplicialMesh", "estimate_gradients", "hermite_interpolant", "refinement_indicator"]

_DEGREE = 3


def _simplicial_set(simplices: Sequence[Sequence[int]]) -> tuple[SimploidalSet, list[Simploid]]:
    """Glue simplices with ascending global vertex numbers into a simploidal set.

    This is the result of adding each simplex with :meth:`SimploidalSet.add_simplex` and identifying every
    pair of faces with the same vertex set, but it builds each shared face once instead of merging copies.
    Because the vertices of every simplex are in ascending order, ``face(0, j)`` of a simplex is the face
    without its ``j``-th smallest vertex, and the vertices of a shared face appear in the same order in
    every simplex containing it. Hence its controls resolve to the same coefficient rows from every side.
    """
    cells: dict[tuple[int, ...], Simploid] = {}
    tops = []
    for simplex in simplices:
        top = tuple(simplex)
        for k in range(len(top)):
            for face in combinations(top, k + 1):
                if face in cells:
                    continue
                s = Simploid((k,) if k else ())
                for j in range(k + 1 if k else 0):
                    s._faces[(0, j)] = cells[face[:j] + face[j + 1 :]]
                cells[face] = s
        tops.append(cells[top])
    domain = SimploidalSet()
    domain._add(list(cells.values()))
    return domain, tops


class SimplicialMesh:
    r"""A conforming simplicial complex in :math:`\mathbb{R}^d` compiled into a cubic :class:`~torch_bsf.core.BezierSpace`.

    Parameters
    ----------
    vertices
        Vertex coordinates of shape ``(n_vertices, d)``, ``d >= 1``.
    simplices
        Vertex numbers of shape ``(n_simplices, d + 1)``. The simplices must be non-degenerate and meet
        face to face (as in a Delaunay triangulation); the mesh is not generated here.

    Attributes
    ----------
    vertices
        Float64 copy of the vertex coordinates.
    simplices
        Each simplex with its vertex numbers sorted in ascending order. The barycentric coordinates of
        cell ``k`` (as returned by :meth:`locate`) refer to the vertices ``simplices[k]`` in this order.
    space
        The compiled degree-3 space, in which simplices sharing a face share the controls on it.
    cell_ids
        ``cell_ids[k]`` is the cell ID of simplex ``k`` in :attr:`space`.

    Examples
    --------
    Two triangles sharing the diagonal of the unit square:

    >>> import torch
    >>> from torch_bsf.interpolation import SimplicialMesh
    >>> mesh = SimplicialMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0, 1, 2], [3, 2, 1]])
    >>> mesh.simplices.tolist()
    [[0, 1, 2], [1, 2, 3]]
    >>> points = mesh.locate(torch.tensor([[0.25, 0.25], [0.75, 0.75]]))
    >>> points.cell_ids.tolist() == mesh.cell_ids.tolist()
    True
    >>> points.parameters[0].tolist()
    [0.5, 0.25, 0.25]
    """

    def __init__(self, vertices: Any, simplices: Any):
        x = torch.as_tensor(vertices)
        if x.ndim != 2 or x.shape[1] < 1 or not len(x) or x.is_complex() or not torch.isfinite(x).all():
            raise ValueError("vertices must be finite and real with shape (n_vertices, d), d >= 1.")
        x = x.to(dtype=torch.float64, device="cpu")
        s = torch.as_tensor(simplices)
        if s.is_floating_point() or s.is_complex() or s.dtype == torch.bool:
            raise ValueError("simplices must contain integer vertex numbers.")
        d = x.shape[1]
        if s.ndim != 2 or s.shape[1] != d + 1 or not len(s):
            raise ValueError(f"simplices must have shape (n_simplices, {d + 1}).")
        s = torch.sort(s.to(dtype=torch.long, device="cpu"), dim=1).values
        if (s < 0).any() or (s >= len(x)).any():
            raise ValueError("simplices refer to a vertex that does not exist.")
        if (s[:, 1:] == s[:, :-1]).any():
            raise ValueError("a simplex repeats a vertex.")
        if len(torch.unique(s, dim=0)) != len(s):
            raise ValueError("simplices must be distinct.")
        edges = x[s[:, 1:]] - x[s[:, :1]]  # (m, d, d)
        det = torch.linalg.det(edges)
        scale = edges.norm(dim=2).prod(dim=1)
        if (det.abs() <= 1e-12 * scale).any():
            raise ValueError("a simplex is degenerate (zero volume).")
        self.vertices = x
        self.simplices = s
        self._inverse = torch.linalg.inv(edges.transpose(1, 2))  # maps x - x_0 to barycentric coordinates 1..d
        self._volumes: torch.Tensor = det.abs() / factorial(d)
        domain, tops = _simplicial_set(s.tolist())
        self.space = BezierSpace.on_set(domain, degree=_DEGREE)
        self.cell_ids = torch.tensor([self.space.cell_id(t) for t in tops], dtype=torch.long)

    def __repr__(self) -> str:
        return f"SimplicialMesh(n_vertices={self.n_vertices}, n_simplices={self.n_simplices}, dimension={self.dimension})"

    @property
    def dimension(self) -> int:
        """Dimension ``d`` of the ambient space and of every simplex."""
        return int(self.vertices.shape[1])

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_simplices(self) -> int:
        """Number of simplices."""
        return len(self.simplices)

    def volumes(self) -> torch.Tensor:
        """Volumes of the simplices, of shape ``(n_simplices,)``."""
        return self._volumes.clone()

    def neighbors(self) -> list[list[int]]:
        """The vertices sharing an edge with each vertex (its one-ring), in ascending order."""
        ring: list[set[int]] = [set() for _ in range(self.n_vertices)]
        for simplex in self.simplices.tolist():
            for a, b in combinations(simplex, 2):
                ring[a].add(b)
                ring[b].add(a)
        return [sorted(r) for r in ring]

    def barycentric(self, x: Any) -> torch.Tensor:
        """Barycentric coordinates of points with respect to every simplex.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.

        Returns
        -------
        torch.Tensor
            Coordinates of shape ``(n_points, n_simplices, d + 1)``, in the vertex order of :attr:`simplices`.
            They sum to one and are all nonnegative exactly when the point lies in the simplex.
        """
        p = self._check_points(x)
        origin = self.vertices[self.simplices[:, 0]]  # (m, d)
        rest = torch.einsum("mij,nmj->nmi", self._inverse, p[:, None, :] - origin[None])
        return torch.cat([1 - rest.sum(dim=2, keepdim=True), rest], dim=2)

    def locate(self, x: Any, *, tol: float = 1e-8, batch_size: int = 1024) -> Points:
        """Find the simplex containing each point and its barycentric coordinates there.

        Every simplex is tested (brute force). A point on a shared face is assigned to one of the simplices
        containing it; the prediction does not depend on that choice because the interpolant is continuous.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.
        tol
            Points whose barycentric coordinates are at least ``-tol`` in some simplex are accepted. Their
            coordinates are clipped to be nonnegative and renormalized, which moves points just outside the
            mesh onto its boundary.
        batch_size
            Number of points tested at a time, which bounds the memory to ``batch_size * n_simplices * (d + 1)``.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in :attr:`space`, in the order of ``x``.

        Raises
        ------
        ValueError
            If a point lies outside the mesh by more than ``tol``.
        """
        p = self._check_points(x)
        ids, coordinates = [], []
        for chunk in torch.split(p, max(1, batch_size)):
            lam = self.barycentric(chunk)
            margin, best = lam.min(dim=2).values.max(dim=1)
            if (margin < -tol).any():
                raise ValueError("a point lies outside the mesh.")
            t = lam[torch.arange(len(chunk)), best].clamp_min(0)
            coordinates.append(t / t.sum(dim=1, keepdim=True))
            ids.append(self.cell_ids[best])
        return Points(torch.cat(coordinates), torch.cat(ids))

    def _check_points(self, x: Any) -> torch.Tensor:
        p = torch.as_tensor(x)
        if p.ndim != 2 or p.shape[1] != self.dimension or p.is_complex() or not torch.isfinite(p).all():
            raise ValueError(f"points must be finite and real with shape (n_points, {self.dimension}).")
        return p.to(dtype=torch.float64, device="cpu")

    def _check_values(self, values: Any) -> torch.Tensor:
        f = torch.as_tensor(values)
        if f.ndim == 1:
            f = f[:, None]
        if f.ndim != 2 or len(f) != self.n_vertices or f.shape[1] < 1 or f.is_complex() or not torch.isfinite(f).all():
            raise ValueError("values must be finite and real with shape (n_vertices, n_outputs).")
        return f.to(dtype=torch.float64, device="cpu")


def _gradient_at(v: int, x: torch.Tensor, f: torch.Tensor, ring: list[list[int]], max_rings: int) -> torch.Tensor:
    """Weighted least-squares gradient of the data at vertex ``v``."""
    d = x.shape[1]
    iu = torch.triu_indices(d, d)
    n_quadratic = d + iu.shape[1]
    near = set(ring[v])
    for _ in range(max_rings - 1):
        near |= {w for u in near for w in ring[u]}
        near.discard(v)
        if len(near) >= n_quadratic + 2:
            break
    if len(near) >= n_quadratic + 2:
        rows = torch.tensor(sorted(near), dtype=torch.long)
        offsets = x[rows] - x[v]
        design = torch.cat([offsets, offsets[:, iu[0]] * offsets[:, iu[1]]], dim=1)
    else:  # The one-ring of a vertex of a non-degenerate simplex spans R^d.
        rows = torch.tensor(ring[v], dtype=torch.long)
        offsets = x[rows] - x[v]
        design = offsets
    weight = offsets.square().sum(dim=1).clamp_min(1e-300).rsqrt()[:, None]  # square root of 1 / |x_w - x_v|^2
    solution: torch.Tensor = torch.linalg.lstsq(design * weight, (f[rows] - f[v]) * weight, driver="gelsd").solution
    return solution[:d]


def estimate_gradients(mesh: SimplicialMesh, values: Any, *, max_rings: int = 4) -> torch.Tensor:
    r"""Estimate the gradient of vertex data at every vertex from its mesh neighbours.

    At each vertex, a quadratic polynomial is fitted by least squares with weights
    :math:`1 / |x_w - x_v|^2` to the data on the smallest :math:`k`-ring neighbourhood
    (:math:`2 \le k \le` ``max_rings``) that contains at least two more points than the quadratic has
    unknowns. If no such neighbourhood exists, a linear polynomial is fitted on the one-ring instead.
    The quadratic fit reproduces the gradient of quadratic data exactly, and either fit reproduces that
    of linear data. Vertices that belong to no simplex get a zero gradient.

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``.
    max_rings
        The largest neighbourhood to try.

    Returns
    -------
    torch.Tensor
        Float64 gradients of shape ``(n_vertices, d, n_outputs)``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import SimplicialMesh, estimate_gradients
    >>> x = torch.linspace(0, 1, 6)[:, None]
    >>> mesh = SimplicialMesh(x, [[k, k + 1] for k in range(5)])
    >>> g = estimate_gradients(mesh, x.square())
    >>> torch.allclose(g[:, 0, 0], 2 * x[:, 0].double())
    True
    """
    if isinstance(max_rings, bool) or not isinstance(max_rings, int) or max_rings < 2:
        raise ValueError("max_rings must be an integer >= 2.")
    f = mesh._check_values(values)
    ring = mesh.neighbors()
    out = torch.zeros(mesh.n_vertices, mesh.dimension, f.shape[1], dtype=torch.float64)
    for v in torch.unique(mesh.simplices).tolist():
        out[v] = _gradient_at(v, mesh.vertices, f, ring, max_rings)
    return out


def _clip(c: torch.Tensor, corners: torch.Tensor, clamp: bool) -> torch.Tensor:
    """Clip control values ``c`` (m, M) into the range of ``corners`` (m, k, M) if ``clamp``."""
    return torch.clamp(c, corners.amin(dim=1), corners.amax(dim=1)) if clamp else c


def _control_values(x: torch.Tensor, f: torch.Tensor, g: torch.Tensor, alpha: tuple[int, ...], clamp: bool) -> torch.Tensor:
    """Values (m, M) of the control with multi-index ``alpha`` of every simplex.

    ``x`` (m, d+1, d), ``f`` (m, d+1, M) and ``g`` (m, d+1, d, M) are the positions, data and gradients
    at the vertices of each simplex.
    """

    def edge(i: int, j: int) -> torch.Tensor:
        c = f[:, i] + torch.einsum("md,mdk->mk", x[:, j] - x[:, i], g[:, i]) / 3
        return _clip(c, f[:, [i, j]], clamp)

    support = [i for i, a in enumerate(alpha) if a]
    if len(support) == 1:
        return f[:, support[0]]
    if len(support) == 2:
        i, j = support if alpha[support[0]] == 2 else support[::-1]
        return edge(i, j)
    i, j, k = support
    total = edge(i, j) + edge(j, i) + edge(i, k) + edge(k, i) + edge(j, k) + edge(k, j)
    c = total / 4 - (f[:, i] + f[:, j] + f[:, k]) / 6
    return _clip(c, f[:, [i, j, k]], clamp)


def hermite_interpolant(mesh: SimplicialMesh, values: Any, *, gradients: Any = None, clamp: bool = True) -> BezierModel:
    r"""Interpolate vertex data by a continuous piecewise cubic Bézier model on a simplicial mesh.

    The model reproduces the data at the vertices. Without clamping, it reproduces quadratic data
    exactly when the gradients do (for example, with enough neighbours for :func:`estimate_gradients`).
    With ``clamp=True``, on every simplex and for every output, the prediction lies between the smallest
    and largest data at the vertices of that simplex; in one dimension, monotone data then give a
    monotone interpolant.

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``.
    gradients
        Vertex gradients of shape ``(n_vertices, d, n_outputs)``. Estimated with
        :func:`estimate_gradients` if ``None``.
    clamp
        Whether to clip edge and face controls into the range of their vertex data (shape preservation).

    Returns
    -------
    BezierModel
        A float64 model on ``mesh.space``. Evaluate it at ``mesh.locate(x)``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import SimplicialMesh, hermite_interpolant
    >>> x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    >>> mesh = SimplicialMesh(x, [[0, 1], [1, 2], [2, 3]])
    >>> model = hermite_interpolant(mesh, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    >>> y = model(mesh.locate(torch.linspace(0, 3, 31)[:, None]))
    >>> bool((y.diff(dim=0) >= 0).all() and y.min() >= 0 and y.max() <= 1)
    True
    """
    f = mesh._check_values(values)
    if gradients is None:
        g = estimate_gradients(mesh, f)
    else:
        g = torch.as_tensor(gradients)
        if g.shape != (mesh.n_vertices, mesh.dimension, f.shape[1]) or g.is_complex() or not torch.isfinite(g).all():
            raise ValueError("gradients must be finite and real with shape (n_vertices, d, n_outputs).")
        g = g.to(dtype=torch.float64, device="cpu")
    s = mesh.simplices
    xs, fs, gs = mesh.vertices[s], f[s], g[s]
    controls = torch.tensor([mesh.space.cells[c].controls for c in mesh.cell_ids.tolist()], dtype=torch.long)
    coefficients = torch.zeros(mesh.space.n_controls, f.shape[1], dtype=torch.float64)
    for k, (alpha,) in enumerate(_indices((mesh.dimension,), (_DEGREE,))):
        # A shared control gets bitwise equal values from every simplex, so duplicate rows are harmless.
        coefficients[controls[:, k]] = _control_values(xs, fs, gs, alpha, bool(clamp))
    return BezierModel(mesh.space, coefficients)


def refinement_indicator(mesh: SimplicialMesh, values: Any, *, gradients: Any = None, clamp: bool = True) -> torch.Tensor:
    """Score simplices for refinement by the cubic correction to linear interpolation.

    The score of a simplex is its volume times the mean over outputs of the absolute difference between
    :func:`hermite_interpolant` and linear interpolation at its centroid. It is large where the data bend
    within a large simplex. Refine the highest-scoring simplices first, for example by evaluating the
    response at the midpoint of their longest edge and remeshing (``LongestEdge`` in
    :mod:`torch_bsf.splitting` selects that edge on a single cell).

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``.
    gradients
        Vertex gradients, as in :func:`hermite_interpolant`.
    clamp
        Whether the cubic interpolant is clamped, as in :func:`hermite_interpolant`.

    Returns
    -------
    torch.Tensor
        Nonnegative float64 scores of shape ``(n_simplices,)``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import SimplicialMesh, refinement_indicator
    >>> x = torch.arange(6.0)[:, None]
    >>> mesh = SimplicialMesh(x, [[k, k + 1] for k in range(5)])
    >>> [round(s, 6) for s in refinement_indicator(mesh, x.square()).tolist()]
    [0.25, 0.25, 0.25, 0.25, 0.25]
    """
    f = mesh._check_values(values)
    model = hermite_interpolant(mesh, f, gradients=gradients, clamp=clamp)
    n = mesh.dimension + 1
    centroid = Points(torch.full((mesh.n_simplices, n), 1 / n, dtype=torch.float64), mesh.cell_ids)
    with torch.no_grad():
        cubic: torch.Tensor = model(centroid)
    linear = f[mesh.simplices].mean(dim=1)
    return (cubic - linear).abs().mean(dim=1) * mesh.volumes()
