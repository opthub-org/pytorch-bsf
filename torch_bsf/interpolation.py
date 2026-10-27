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

import math
import operator
import warnings
from collections.abc import Sequence
from fractions import Fraction
from itertools import accumulate, combinations, combinations_with_replacement
from math import factorial
from typing import Any

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._kernels.sets import Simploid, SimploidalSet
from torch_bsf.core import BezierModel, BezierSpace, Points

__all__ = ["SimplicialMesh", "estimate_gradients", "hermite_interpolant", "lattice_mesh", "refine", "refinement_indicator"]

_DEGREE = 3
_ROUNDING = 1  # lengths differing by this many units of rounding (times d) are equal


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


def _check_simplices(simplices: Any, n_vertices: int, d: int) -> torch.Tensor:
    """Read ``simplices`` as distinct, sorted rows of ``d + 1`` distinct vertex numbers below ``n_vertices``."""
    s = torch.as_tensor(simplices)
    if s.is_floating_point() or s.is_complex() or s.dtype == torch.bool:
        raise ValueError("simplices must contain integer vertex numbers.")
    if s.ndim != 2 or s.shape[1] != d + 1 or not len(s):
        raise ValueError(f"simplices must have shape (n_simplices, {d + 1}).")
    s = torch.sort(s.to(dtype=torch.long, device="cpu"), dim=1).values
    if (s < 0).any() or (s >= n_vertices).any():
        raise ValueError("simplices refer to a vertex that does not exist.")
    if (s[:, 1:] == s[:, :-1]).any():
        raise ValueError("a simplex repeats a vertex.")
    if len(torch.unique(s, dim=0)) != len(s):
        raise ValueError("simplices must be distinct.")
    return s


def _overlapping_boxes(
    low: torch.Tensor, high: torch.Tensor, slack: torch.Tensor, own: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs of boxes ``[low, high]`` (n, d) of different owners that meet when grown by the larger slack of the two.

    The boxes are swept in the order of their lower bounds along the first axis, so that only boxes overlapping
    along it are compared, not all pairs; each pair is returned once.
    """
    if not len(low):
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    order = torch.argsort(low[:, 0])
    start = low[order, 0]
    # Every box that a box meets along the first axis starts before its end plus the largest slack.
    end = torch.searchsorted(start, (high[order, 0] + slack.max()).contiguous(), right=True)
    position = torch.arange(len(order))
    counts = (end - position - 1).clamp_min(0)
    pairs = []
    for chunk in torch.split(position, max(1, 2**16)):
        repeats = counts[chunk]
        i = torch.repeat_interleave(chunk, repeats)
        if not len(i):
            continue
        # The k-th later box of each run, for k = 1, 2, ...
        run_start = torch.repeat_interleave(torch.cumsum(repeats, dim=0) - repeats, repeats)
        j = i + 1 + torch.arange(len(i)) - run_start
        a, b = order[i], order[j]
        pad = torch.maximum(slack[a], slack[b])[:, None]
        meet = ((low[a] <= high[b] + pad) & (low[b] <= high[a] + pad)).all(dim=1) & (own[a] != own[b])
        pairs.append(torch.stack([a[meet], b[meet]]))
    if not pairs:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    first, second = torch.cat(pairs, dim=1)
    return first, second


def _difference_times_power_of_two(a: torch.Tensor, b: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """``(a - b) * 2**exponent``, also where ``a - b`` itself overflows, as for the opposite ends of a long face."""
    difference = a - b
    return torch.where(
        torch.isfinite(difference),
        _times_power_of_two(difference, exponent),
        _times_power_of_two(a / 2 - b / 2, exponent + 1),
    )


def _split_difference(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``a - b`` as ``mantissa * 2**exponent`` (:func:`torch.frexp`, long exponents), also where it overflows.

    Where the difference overflows, that of the halves is taken, which is exact: both ends then exceed ``2**970``
    in magnitude.
    """
    difference = a - b
    overflow = ~torch.isfinite(difference)
    mantissa, exponent = torch.frexp(torch.where(overflow, a / 2 - b / 2, difference))
    return mantissa, exponent.to(torch.long) + overflow


def _pair_exponents(f: torch.Tensor, g: torch.Tensor, axis_exponent: torch.Tensor) -> torch.Tensor:
    """Per pair of faces ``f`` and ``g`` (p, d, d) and axis, the power of two that scales the pair to order one.

    Each axis is divided by ``2**axis_exponent`` (from the whole mesh, so that the units of the axes do not matter),
    and then all axes of a pair by a common power of two, so that the largest extent of the pair is of order one.
    Both are applied together, so that a pair far smaller than the mesh does not underflow on the way.
    """
    both = torch.cat([f, g], dim=1)
    half = both.amax(dim=1) / 2 - both.amin(dim=1) / 2  # (p, d): half the extent, which may overflow itself
    relative = torch.where(half > 0, torch.frexp(half).exponent.to(torch.long) + 1 - axis_exponent, -(2**20))
    return axis_exponent + relative.amax(dim=1, keepdim=True)


def _local_pair_coordinates(
    f: torch.Tensor, g: torch.Tensor, axis_exponent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faces ``f`` and ``g`` (p, d, d) relative to the first vertex of ``f`` and scaled by :func:`_pair_exponents`
    (exactly), with the largest absolute coordinate of each pair in the same units (p,)."""
    e = -_pair_exponents(f, g, axis_exponent)
    magnitude = torch.maximum(f.abs().amax(dim=1), g.abs().amax(dim=1))  # (p, d)
    return (
        _difference_times_power_of_two(f, f[:, :1], e[:, None]),
        _difference_times_power_of_two(g, f[:, :1], e[:, None]),
        _times_power_of_two(magnitude, e).amax(dim=1),
    )


def _as_real_tensor(x: Any) -> torch.Tensor:
    """Read ``x`` as a tensor, parsing non-tensor floating-point input in float64 rather than the default dtype."""
    t = torch.as_tensor(x)
    if not isinstance(x, torch.Tensor) and t.is_floating_point():
        t = torch.as_tensor(x, dtype=torch.float64)
    return t


def _length_key(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    """A key ordering segments ``ab`` by Euclidean length, without overflow or underflow.

    The difference is taken of the halves where that of the points overflows, as for points near opposite ends of
    the float64 range.
    """
    v, shift = b - a, 0
    if not torch.isfinite(v).all():
        v, shift = b / 2 - a / 2, 1
    exponent = math.frexp(float(v.abs().max()))[1]
    mantissa, extra = math.frexp(math.hypot(*(math.ldexp(float(t), -exponent) for t in v.tolist())))
    return exponent + extra + shift, mantissa


class SimplicialMesh:
    r"""A conforming simplicial complex in :math:`\mathbb{R}^d` compiled into a cubic :class:`~torch_bsf.core.BezierSpace`.

    Parameters
    ----------
    vertices
        Vertex coordinates of shape ``(n_vertices, d)``, ``d >= 1``.
    simplices
        Vertex numbers of shape ``(n_simplices, d + 1)``. The simplices must be non-degenerate and meet
        face to face (conforming); the mesh is not generated here. A Delaunay triangulation of points in
        general position is conforming; one of lattice points with its flat simplices removed is not
        (use :func:`lattice_mesh` instead). A simplex counts as
        degenerate when its volume is at most ``1e-12`` times the product of its edge lengths both in the given
        coordinates and with each axis divided by the extent of the simplex along it (rounded down to a power
        of two), so that neither the units of the axes nor the scale of the coordinates matters.
    strict
        Whether a non-conforming mesh raises :class:`ValueError`. If ``False``, a warning is issued
        instead and :attr:`unmatched_faces` reports the defect.

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

    def __init__(self, vertices: Any, simplices: Any, *, strict: bool = True):
        x = _as_real_tensor(vertices)
        if x.ndim != 2 or x.shape[1] < 1 or not len(x) or x.is_complex() or not torch.isfinite(x).all():
            raise ValueError("vertices must be finite and real with shape (n_vertices, d), d >= 1.")
        x = x.to(dtype=torch.float64, device="cpu")
        d = x.shape[1]
        s = _check_simplices(simplices, len(x), d)
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
        self._unmatched = self._count_unmatched_faces()
        if self._unmatched:
            message = (
                f"the mesh is not conforming: {self._unmatched} faces are not shared face to face by two simplices "
                "(e.g. a Delaunay triangulation of lattice points with flat simplices removed; see lattice_mesh)."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
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

    @property
    def unmatched_faces(self) -> int:
        """Number of faces that break conformity (0 for a conforming mesh).

        A face of dimension ``d - 1`` is counted if it belongs to more than two simplices, or if it belongs
        to one simplex and overlaps, in a set of positive :math:`(d - 1)`-volume, another such face in the
        same hyperplane (a hanging vertex, or a flat polygon split differently on its two sides). The
        overlap is tested over the whole faces, however thin it is, and faces count as in the same
        hyperplane only if they are so up to rounding. Rounding here is ``d`` units of rounding (machine
        epsilon) of the larger diameter plus the largest absolute coordinate of the two faces, which is how
        faces meeting at a common vertex or edge touch after rounding. In 1-D, where faces are points given
        as input, only equal points overlap. There is no other tolerance. Faces of the
        same simplex are never compared. Faces on the
        meshed domain are not counted, even when another part of the mesh lies close to them or touches
        them at a lower-dimensional face. Overlapping simplices that share no face are not detected.
        A mesh without boundary faces (every face in two or more simplices, which only overlapping simplices
        allow) has all its faces counted.
        """
        return self._unmatched

    def _count_unmatched_faces(self) -> int:
        d = self.dimension
        owners: dict[tuple[int, ...], list[int]] = {}
        for k, simplex in enumerate(self.simplices.tolist()):
            for i in range(d + 1):
                owners.setdefault(tuple(simplex[:i] + simplex[i + 1 :]), []).append(k)
        crowded = sum(1 for o in owners.values() if len(o) > 2)
        single = [(face, o[0]) for face, o in owners.items() if len(o) == 1]
        if not single:  # a bounded domain has a boundary, so overlapping simplices close up the mesh
            return len(owners)
        faces = self.vertices[torch.tensor([face for face, _ in single])]  # (n, d, d)
        own = torch.tensor([k for _, k in single])
        # In a conforming mesh the faces of one simplex only are boundary faces, and no two of them overlap.
        # A face partly covered by such faces of other simplices (a hanging face) overlaps one of them in a
        # set of positive (d - 1)-volume within their common hyperplane. Candidates are pairs of faces of
        # different simplices whose bounding boxes meet up to rounding; the boxes are compared as given, which
        # cannot overflow or underflow. Rounding is d units of the larger diameter and of the larger coordinate
        # of the pair: coordinates far from the origin carry an error of their own unit of rounding.
        # In 1-D the faces are input coordinates themselves, and only equal ones coincide.
        unit = 0.0 if d == 1 else _ROUNDING * d * torch.finfo(torch.float64).eps
        low, high = faces.amin(dim=1), faces.amax(dim=1)  # (n, d)
        # The diameter is at most sqrt(d) times the largest side of the box, which bounds the slack from above.
        # Halves are subtracted, since the extent of a face may overflow although its vertices do not.
        slack = (2 * math.sqrt(d) * unit) * (high / 2 - low / 2).amax(dim=1) + unit * faces.abs().amax(dim=(1, 2))
        first, second = _overlapping_boxes(low, high, slack, own)
        overlap = torch.ones(len(first), dtype=torch.bool)  # in 1-D, the candidates coincide
        if d > 1 and len(first):
            axis_exponent = torch.frexp(self.vertices.abs().amax(dim=0)).exponent.to(torch.long)
            f, g, magnitude = _local_pair_coordinates(faces[first], faces[second], axis_exponent)
            exact = "donot_use_mm_for_euclid_dist"
            both = torch.cat([f, g], dim=1)
            diameter = torch.maximum(
                torch.cdist(f, f, compute_mode=exact).amax(dim=(1, 2)), torch.cdist(g, g, compute_mode=exact).amax(dim=(1, 2))
            )
            gap = unit * (diameter + magnitude)
            normal = torch.linalg.svd(f[:, 1:] - f[:, :1]).Vh[:, -1]  # (p, d)
            # The faces must lie in one hyperplane up to rounding.
            height = torch.einsum("pkd,pd->pk", both - f[:, :1], normal)
            overlap &= (height.abs() <= gap[:, None]).all(dim=1)
            f, g = f - f[:, :1], g - f[:, :1]
            # Faces meeting at a common vertex or edge touch up to rounding; any wider overlap, however thin,
            # counts. Two convex polytopes have interiors that do not overlap exactly when a facet normal of their
            # Minkowski difference separates them. In the hyperplane, such a normal is orthogonal to the
            # edges of a face of each simplex whose dimensions sum to d - 2.
            for a in range(1, d):
                for left in combinations(range(d), a):
                    for right in combinations(range(d), d - a):
                        spans = torch.cat(
                            [
                                normal[:, None],
                                f[:, left[1:]] - f[:, left[:1]],
                                g[:, right[1:]] - g[:, right[:1]],
                            ],
                            dim=1,
                        )  # (p, d - 1, d)
                        spans = spans / spans.norm(dim=2, keepdim=True)
                        # A unit normal of the spans (their generalized cross product), by cofactor expansion.
                        u = torch.stack(
                            [(-1) ** c * torch.linalg.det(spans[:, :, [k for k in range(d) if k != c]]) for c in range(d)],
                            dim=1,
                        )
                        # Any direction in the hyperplane may separate the faces; one lost to rounding (when the
                        # spans are dependent) is skipped.
                        u -= (u * normal).sum(dim=1, keepdim=True) * normal
                        length = u.norm(dim=1)
                        u = u / length.clamp_min(1e-300)[:, None]
                        pf, pg = (f @ u[:, :, None])[..., 0], (g @ u[:, :, None])[..., 0]
                        apart = (pf.amax(dim=1) <= pg.amin(dim=1) + gap) | (pg.amax(dim=1) <= pf.amin(dim=1) + gap)
                        overlap &= ~(apart & (length > unit))
        hanging = torch.zeros(len(single), dtype=torch.bool)
        hanging[first[overlap]] = True
        hanging[second[overlap]] = True
        return crowded + int(hanging.sum())

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
    vertices = torch.tensor(lattice, dtype=torch.float64)[:, :dim] / k
    return SimplicialMesh(vertices, simplices)


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
    """Mutable state of :func:`refine`: vertices, live simplices and the simplices around each edge."""

    def __init__(self, x: list[torch.Tensor], cells: list[tuple[int, ...]]):
        self.x = x
        self.alive: set[tuple[int, ...]] = set()
        self.cells_of: dict[tuple[int, int], set[tuple[int, ...]]] = {}
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
            for v in e:
                self.add(tuple(sorted(m if w == v else w for w in cell)))
        del self.cells_of[e]

    def refine(self, target: tuple[int, ...]) -> None:
        stack = [target]
        while stack:
            cell = stack[-1]
            if cell not in self.alive:
                stack.pop()
                continue
            e = self.longest(cell)
            other = next((c for c in sorted(self.cells_of[e]) if self.longest(c) != e), None)
            if other is None:
                self.bisect(e)
            else:  # its longest edge comes after e, so the recursion ends
                stack.append(other)


def refine(mesh: SimplicialMesh, simplices: Sequence[int]) -> SimplicialMesh:
    """Bisect simplices at the midpoint of their longest edge, keeping the mesh conforming.

    An edge is bisected in every simplex that contains it, which keeps the mesh conforming. Before an
    edge is bisected, every simplex containing it whose longest edge is a different one is refined
    first, recursively (the longest-edge bisection of Rivara), so that cells are always cut across their
    longest edge. Ties between edges of equal length are broken by their vertex numbers, consistently
    over the whole mesh. Refining one simplex may therefore add several vertices.

    Parameters
    ----------
    mesh
        A conforming mesh.
    simplices
        Integer indices into ``mesh.simplices`` of the simplices to bisect.

    Returns
    -------
    SimplicialMesh
        The refined mesh. It keeps the vertices of ``mesh`` with their numbers and appends the new
        vertices (edge midpoints) in the order they were created. Its simplices are sorted.

    Examples
    --------
    >>> from torch_bsf.interpolation import lattice_mesh, refine
    >>> coarse = lattice_mesh(2, 2)
    >>> fine = refine(coarse, [0])
    >>> fine.n_vertices - coarse.n_vertices, fine.n_simplices - coarse.n_simplices, fine.unmatched_faces
    (1, 2, 0)
    """
    cells = [tuple(c) for c in mesh.simplices.tolist()]
    try:
        # operator.index accepts integers of any kind (Python, NumPy, 0-d tensors) but not floats or strings.
        if any(isinstance(k, bool) or (isinstance(k, torch.Tensor) and k.dtype == torch.bool) for k in simplices):
            raise TypeError
        targets = [cells[operator.index(k)] for k in simplices]
    except (IndexError, TypeError):
        raise ValueError("simplices must be indices into mesh.simplices.") from None
    state = _Bisection(list(mesh.vertices), cells)
    for target in targets:
        state.refine(target)
    return SimplicialMesh(torch.stack(state.x), sorted(state.alive))


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
    # The fit is solved in the coordinates q = H^{-1} offsets, H = diag(extent of each axis), with the rows
    # weighted by 1 / |q|: each weighted linear row is the unit vector q / |q| and its right-hand side the slope
    # (f_w - f_v) / |q|. They are computed from the mantissas and exponents of the offsets, so that nothing
    # overflows or underflows on the way however the axes and the neighbours differ in scale, even where the
    # offsets themselves overflow, between vertices two rings apart near opposite ends of the float64 range.
    mantissa, exponent = _split_difference(x[rows], x[v])
    exponent = torch.where(mantissa == 0, -(2**20), exponent)  # zeros never set the extent
    extent_exponent = exponent.amax(dim=0)  # H_j = extent_mantissa_j * 2**extent_exponent_j
    extent_mantissa = torch.where(exponent == extent_exponent, mantissa.abs(), 0.0).amax(dim=0)
    exponent = torch.where(mantissa == 0, -(2**20), exponent - extent_exponent)
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
    slope_mantissa, slope_exponent = _split_difference(f[rows], f[v])
    slope_mantissa = slope_mantissa / length
    slope_exponent = torch.where(slope_mantissa == 0, -(2**20), slope_exponent - row_exponent)
    top = slope_exponent.amax(dim=0)  # (n_outputs,)
    top = torch.where(top < -(2**19), 0, top)  # data constant on the neighbourhood
    rhs = _times_power_of_two(slope_mantissa, slope_exponent - top)
    solution: torch.Tensor = torch.linalg.lstsq(design, rhs, driver="gelsd").solution
    # The linear coefficients are the gradient times H, divided by 2**top.
    shift = top[None, :] - extent_exponent[:, None]
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
        Components beyond the float64 range are returned as signed infinity, without a warning.
        Finite vertex data can produce such gradients, for example on very short edges.

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


def _rounded(value: Fraction) -> float:
    """``value`` rounded to the nearest float64, or an infinity of its sign beyond the float64 range."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _control_values(x: torch.Tensor, f: torch.Tensor, g: torch.Tensor, alpha: tuple[int, ...], clamp: bool) -> torch.Tensor:
    """Values (m, M) of the control with multi-index ``alpha`` of every simplex.

    ``x`` (m, d+1, d), ``f`` (m, d+1, M) and ``g`` (m, d+1, d, M) are the positions, data and gradients
    at the vertices of each simplex.
    """

    def edge(i: int, j: int) -> torch.Tensor:
        third = torch.einsum("md,mdk->mk", x[:, j] - x[:, i], g[:, i]) / 3
        c = f[:, i] + third
        # Only the differences from the first vertex of a simplex are known to be finite. Where the correction
        # overflows, in a difference, a product or their sum, the control is computed exactly and rounded once,
        # unless the gradient itself is not finite (an estimate may overflow).
        overflow = ~torch.isfinite(third) & torch.isfinite(g[:, i]).all(dim=1)
        for r, o in overflow.nonzero().tolist():
            terms = zip(x[r, j].tolist(), x[r, i].tolist(), g[r, i, :, o].tolist(), strict=True)
            correction = sum(((Fraction(b) - Fraction(a)) * Fraction(h) for b, a, h in terms), Fraction(0))
            c[r, o] = _rounded(Fraction(f[r, i, o].item()) + correction / 3)
        return _clip(c, f[:, [i, j]], clamp)

    support = [i for i, a in enumerate(alpha) if a]
    if len(support) == 1:
        return f[:, support[0]]
    if len(support) == 2:
        i, j = support if alpha[support[0]] == 2 else support[::-1]
        return edge(i, j)
    i, j, k = support
    edges = [edge(i, j), edge(j, i), edge(i, k), edge(k, i), edge(j, k), edge(k, j)]
    total = edges[0] + edges[1] + edges[2] + edges[3] + edges[4] + edges[5]
    c = total / 4 - (f[:, i] + f[:, j] + f[:, k]) / 6
    # The controls of a face spanning the float64 range may add up beyond it although the face control does not:
    # where a sum overflows, the control is computed exactly from finite edge controls and rounded once.
    stacked = torch.stack(edges)
    for r, o in (~torch.isfinite(c) & torch.isfinite(stacked).all(dim=0)).nonzero().tolist():
        edge_sum = sum(map(Fraction, stacked[:, r, o].tolist()), Fraction(0))
        corner_sum = sum(map(Fraction, f[r, [i, j, k], o].tolist()), Fraction(0))
        c[r, o] = _rounded(edge_sum / 4 - corner_sum / 6)
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

    Raises
    ------
    ValueError
        If the data or supplied gradients are invalid, or the interpolant's control values are
        non-finite because the data or estimated gradients cause float64 overflow. Clamping can
        clip infinite controls into the data range, but cannot repair NaN controls.

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
    if not torch.isfinite(coefficients).all():
        raise ValueError("the interpolant is not finite: the vertex data or gradients cause its controls to overflow float64.")
    return BezierModel(mesh.space, coefficients)


def refinement_indicator(mesh: SimplicialMesh, values: Any, *, gradients: Any = None, clamp: bool = True) -> torch.Tensor:
    """Score simplices for refinement by the cubic correction to linear interpolation.

    The score of a simplex is its volume times the mean over outputs of the absolute difference between
    :func:`hermite_interpolant` and linear interpolation at its centroid. It is large where the data bend
    within a large simplex. Refine the highest-scoring simplices first, for example
    with :func:`refine`, and evaluate the response at the new vertices.

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

    Raises
    ------
    ValueError
        If :func:`hermite_interpolant` cannot construct a finite interpolant, including when the
        vertex data or gradients cause its controls to overflow float64.

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
