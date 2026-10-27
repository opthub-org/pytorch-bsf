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


def _times_power_of_two(x: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """``x * 2**exponent`` for integer ``exponent`` of any size, rounded once, even where ``2**exponent`` is not a float64.

    The power is applied in three factors of the same sign, each at most ``2**767``, so the intermediate products
    move monotonically towards the result and only the result can overflow or underflow. Beyond ``2**+-2300``,
    every nonzero float64 ``x`` gives ``inf`` or ``0`` anyway.
    """
    e = exponent.to(torch.float64).clamp(-2300, 2300)
    first = torch.trunc(e / 3)
    second = torch.trunc((e - first) / 2)
    two = torch.tensor(2.0, dtype=torch.float64)
    return x * torch.pow(two, first) * torch.pow(two, second) * torch.pow(two, e - first - second)


def _equilibrate(
    mantissa: torch.Tensor, exponent: torch.Tensor, axis_exponent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edges ``mantissa * 2**exponent`` of shape ``(m, d, d)`` as ``diag(2**edge) @ unit @ diag(2**axis)``.

    Each edge (row) of ``unit`` has its largest entry in ``[1, 2)``, after the axes (columns) are divided by
    ``2**axis_exponent``. The exponents are integers, so no entry underflows on the way and each entry of ``unit``
    is its edge entry times a power of two. A common power of two moves from the edges to the axes so that the edge
    exponents are centred on zero, because they multiply the barycentric coordinates in
    :meth:`SimplicialMesh.barycentric`.

    Returns ``(unit, edge_exponent, axis_exponent, ok, det)``: ``ok`` tells whether the volume of ``unit`` exceeds
    ``1e-12`` times the product of its edge lengths, and ``det`` is its determinant.
    """
    shift = exponent - axis_exponent[:, None, :]
    edge_exponent = shift.amax(dim=2) - 1  # (m, d)
    unit = torch.where(mantissa == 0, 0.0, _times_power_of_two(mantissa, shift - edge_exponent[:, :, None]))
    centre = torch.div(edge_exponent.amax(dim=1) + edge_exponent.amin(dim=1), 2, rounding_mode="floor")[:, None]
    # The determinant from the LU factors: torch.linalg.det was seen to be off by tens of ulps even for a triangular
    # matrix, whose LU factorization is exact.
    lu, pivots, _ = torch.linalg.lu_factor_ex(unit)
    swaps = (pivots != torch.arange(1, unit.shape[-1] + 1, dtype=pivots.dtype)).sum(dim=-1)
    det = torch.where(swaps % 2 == 1, -1.0, 1.0) * lu.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    ok = det.abs() > 1e-12 * unit.norm(dim=2).prod(dim=1)
    return unit, edge_exponent - centre, axis_exponent + centre, ok, det


def _as_real_tensor(x: Any) -> torch.Tensor:
    """Read ``x`` as a tensor, parsing non-tensor floating-point input in float64 rather than the default dtype."""
    t = torch.as_tensor(x)
    if not isinstance(x, torch.Tensor) and t.is_floating_point():
        t = torch.as_tensor(x, dtype=torch.float64)
    return t


class SimplicialMesh:
    r"""A conforming simplicial complex in :math:`\mathbb{R}^d` compiled into a cubic :class:`~torch_bsf.core.BezierSpace`.

    Parameters
    ----------
    vertices
        Vertex coordinates of shape ``(n_vertices, d)``, ``d >= 1``.
    simplices
        Vertex numbers of shape ``(n_simplices, d + 1)``. The simplices must be non-degenerate and meet
        face to face (as in a Delaunay triangulation); the mesh is not generated here. A simplex counts as
        degenerate when its volume is at most ``1e-12`` times the product of its edge lengths both in the given
        coordinates and with each axis divided by the extent of the simplex along it (rounded down to a power
        of two), so that neither the units of the axes nor the scale of the coordinates matters.

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
        x = _as_real_tensor(vertices)
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
        edges = x[s[:, 1:]] - x[s[:, :1]]  # (m, d, d): row k is the edge from vertex 0 to vertex k + 1
        if not torch.isfinite(edges).all():
            raise ValueError("the vertices of a simplex are too far apart: their differences overflow float64.")
        # The edges are E = diag(2**edge_exponent) @ unit @ diag(2**axis_exponent), with the largest entry of
        # every edge of `unit` in [1, 2). The determinant, the edge lengths and the inverse of E itself overflow or
        # underflow for coordinates beyond about 1e+-150, or for axes or edges of very different lengths; those of
        # `unit` do not. Two choices of axis exponents are tried: one per axis, from the extent of the simplex
        # along it, under which scaling an axis by a power of two changes only its exponent; and a common one,
        # which keeps the Euclidean shape. The first is used unless only the second gives a non-degenerate `unit`.
        mantissa, exponent = torch.frexp(edges)
        exponent = torch.where(mantissa == 0, -(2**20), exponent.to(torch.long))  # zeros never set a scale
        per_axis = _equilibrate(mantissa, exponent, exponent.amax(dim=1))
        common = _equilibrate(mantissa, exponent, torch.zeros_like(exponent[:, 0]))
        use_common = (~per_axis[3]) & common[3]
        if not (per_axis[3] | common[3]).all():
            raise ValueError("a simplex is degenerate (zero volume).")
        unit, self._edge_exponent, self._axis_exponent, _, det = (
            torch.where(use_common.view((-1,) + (1,) * (a.ndim - 1)), b, a) for a, b in zip(per_axis, common, strict=True)
        )
        self.vertices = x
        self.simplices = s
        # Maps (x - x_0) / 2**axis_exponent to barycentric coordinates 1..d times 2**edge_exponent.
        self._inverse = torch.linalg.inv(unit.transpose(1, 2))
        # Volumes are kept as mantissa * 2**exponent too, so that a product with a volume beyond the float64 range
        # (e.g. a refinement score) can still be finite.
        self._volume_mantissa = det.abs() / factorial(d)
        self._volume_exponent = self._edge_exponent.sum(dim=1) + self._axis_exponent.sum(dim=1)
        self._volumes: torch.Tensor = _times_power_of_two(self._volume_mantissa, self._volume_exponent)
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
        offset = _times_power_of_two(p[:, None, :] - origin[None], -self._axis_exponent[None])
        rest = _times_power_of_two(torch.einsum("mij,nmj->nmi", self._inverse, offset), -self._edge_exponent[None])
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
            # Coordinates that overflow (NaN or inf) belong to a simplex far from the point.
            margin, best = torch.nan_to_num(lam.min(dim=2).values, nan=-torch.inf).max(dim=1)
            if not (margin >= -tol).all():
                raise ValueError("a point lies outside the mesh.")
            t = lam[torch.arange(len(chunk)), best].clamp_min(0)
            coordinates.append(t / t.sum(dim=1, keepdim=True))
            ids.append(self.cell_ids[best])
        return Points(torch.cat(coordinates), torch.cat(ids))

    def _check_points(self, x: Any) -> torch.Tensor:
        p = _as_real_tensor(x)
        if p.ndim != 2 or p.shape[1] != self.dimension or p.is_complex() or not torch.isfinite(p).all():
            raise ValueError(f"points must be finite and real with shape (n_points, {self.dimension}).")
        return p.to(dtype=torch.float64, device="cpu")

    def _check_values(self, values: Any) -> torch.Tensor:
        f = _as_real_tensor(values)
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
    quadratic = len(near) >= n_quadratic + 2
    # The one-ring of a vertex of a non-degenerate simplex spans R^d, for the linear fallback.
    rows = torch.tensor(sorted(near) if quadratic else ring[v], dtype=torch.long)
    offsets = x[rows] - x[v]
    # The fit is solved in the coordinates q = H^{-1} offsets, H = diag(extent of each axis), with the rows
    # weighted by 1 / |q|: each weighted linear row is the unit vector q / |q| and its right-hand side the slope
    # (f_w - f_v) / |q|. They are computed from the mantissas and exponents of the offsets, so that nothing
    # overflows or underflows on the way however the axes and the neighbours differ in scale.
    mantissa, exponent = torch.frexp(offsets)
    extent_mantissa, extent_exponent = torch.frexp(offsets.abs().amax(dim=0))  # H_j = extent_mantissa_j * 2**...
    exponent = torch.where(mantissa == 0, -(2**20), exponent.to(torch.long) - extent_exponent.to(torch.long))
    row_exponent = exponent.amax(dim=1, keepdim=True)  # q_k = 2**row_exponent_k * p_k, at most 0
    p = torch.where(mantissa == 0, 0.0, _times_power_of_two(mantissa / extent_mantissa, exponent - row_exponent))
    length = p.norm(dim=1, keepdim=True)  # in [1/2, 2 sqrt(d)]
    unit = p / length
    if quadratic:
        q = _times_power_of_two(p, row_exponent)  # entries of q are at most 1 in magnitude
        design = torch.cat([unit, unit[:, iu[0]] * q[:, iu[1]]], dim=1)
    else:
        design = unit
    # Slopes as mantissa * 2**exponent, scaled by one power of two per output so that the largest is near one.
    slope_mantissa, slope_exponent = torch.frexp(f[rows] - f[v])
    slope_mantissa = slope_mantissa / length
    slope_exponent = torch.where(slope_mantissa == 0, -(2**20), slope_exponent.to(torch.long) - row_exponent)
    top = slope_exponent.amax(dim=0)  # (n_outputs,)
    top = torch.where(top < -(2**19), 0, top)  # data constant on the neighbourhood
    rhs = _times_power_of_two(slope_mantissa, slope_exponent - top)
    solution: torch.Tensor = torch.linalg.lstsq(design, rhs, driver="gelsd").solution
    # The linear coefficients are the gradient times H, divided by 2**top.
    shift = top[None, :] - extent_exponent.to(torch.long)[:, None]
    return _times_power_of_two(solution[:d] / extent_mantissa[:, None], shift)


def estimate_gradients(mesh: SimplicialMesh, values: Any, *, max_rings: int = 4) -> torch.Tensor:
    r"""Estimate the gradient of vertex data at every vertex from its mesh neighbours.

    At each vertex, a quadratic polynomial is fitted by least squares with weights
    :math:`1 / |H^{-1}(x_w - x_v)|^2` to the data on the smallest :math:`k`-ring neighbourhood
    (:math:`2 \le k \le` ``max_rings``) that contains at least two more points than the quadratic has
    unknowns. If no such neighbourhood exists, a linear polynomial is fitted on the one-ring instead.
    The quadratic fit reproduces the gradient of quadratic data exactly, and either fit reproduces that
    of linear data. Vertices that belong to no simplex get a zero gradient.

    :math:`H` is the diagonal matrix of the largest :math:`|x_w - x_v|` along each axis over the
    neighbourhood, so that the estimate does not depend on the units of the axes: scaling an axis by
    :math:`c` divides that component of the gradient by :math:`c`, even for axes of very different scales.

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
        g = _as_real_tensor(gradients)
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
    correction = (cubic - linear).abs().mean(dim=1)
    # Multiplied by the volume as mantissa * 2**exponent, with the correction split the same way: a volume beyond
    # the float64 range may still give a finite score, a tiny correction is not lost before the exponent scales it
    # up, and a simplex without correction scores zero rather than 0 * inf = NaN.
    correction_mantissa, correction_exponent = torch.frexp(correction)
    score = _times_power_of_two(
        correction_mantissa * mesh._volume_mantissa, correction_exponent.to(torch.long) + mesh._volume_exponent
    )
    return torch.where(correction > 0, score, 0.0)
