"""The :class:`~torch_bsf.interpolation.SimplicialMesh` of :mod:`torch_bsf.interpolation`."""

import math
import warnings
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from math import factorial
from typing import Any, NamedTuple

import torch

from torch_bsf._interpolation.exact import (
    _as_real_tensor,
    _barycentric_expressions,
    _check_simplices,
    _equilibrate,
    _ExactExpressions,
    _Frexp,
    _local_pair_coordinates,
    _overlapping_boxes,
    _rational_inverse,
    _rounded_fraction,
    _times_power_of_two,
)
from torch_bsf._kernels.sets import Simploid, SimploidalSet
from torch_bsf.core import BezierSpace, Points

_DEGREE = 3
_ROUNDING = 1  # lengths differing by this many units of rounding (times d) are equal


# The default batch of SimplicialMesh.locate keeps each (batch, n_simplices, d + 1) float64 array within this size.
_LOCATE_ARRAY_BYTES = 2**24


class _Tagged(NamedTuple):
    """A simplex of newest-vertex bisection; its refinement edge is ``vertices[0] vertices[tag]``."""

    vertices: tuple[int, ...]  # in the order of bisection (a Kuhn path on a lattice mesh), not sorted
    tag: int  # in 1..d
    generation: int  # the number of bisections since the lattice mesh


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
        Float64 snapshot of the vertex coordinates, independent of the caller's storage.
    simplices
        Each simplex with its vertex numbers sorted in ascending order. The barycentric coordinates of
        cell ``k`` (as returned by :meth:`locate`) refer to the vertices ``simplices[k]`` in this order.
    space
        The compiled degree-3 space, in which simplices sharing a face share the controls on it.
    cell_ids
        ``cell_ids[k]`` is the cell ID of simplex ``k`` in :attr:`space`.

    Notes
    -----
    Construction copies vertex and simplex inputs, including CPU float64 tensors and NumPy arrays.
    Treat the public ``vertices`` and ``simplices`` tensors as read-only: direct in-place edits are
    unsupported because geometric caches and the compiled space are not rebuilt. Construct a new mesh
    to change its geometry or topology. The vertex copy preserves autograd connections to tensor inputs.

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
        x = x.to(device="cpu").to(dtype=torch.float64, copy=True)
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
        mantissa, exponent = _Frexp.apply(edges)
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
        # Keep discrete topology in Python too: older functorch versions cannot expose wrapped
        # tensor storage through tolist(), even for constant integer metadata.
        self._simplex_rows = [tuple(int(v) for v in row.unbind()) for row in s.unbind()]
        # Maps (x - x_0) / 2**axis_exponent to barycentric coordinates 1..d times 2**edge_exponent.
        self._inverse = torch.linalg.inv(unit.transpose(1, 2))
        # The final dot-product error alone misses rounding in the factorization of a nearly singular
        # unit matrix. Also account for amplification when coordinates of differently scaled edges are
        # recovered, even if a spurious inverse entry hides cancellation in the final dot product.
        condition = unit.abs().sum(dim=1).amax(dim=1) * self._inverse.abs().sum(dim=2).amax(dim=1)
        amplification = self._edge_exponent.amax(dim=1) - self._edge_exponent.amin(dim=1)
        self._inexact_inverse = (8 * d * torch.finfo(torch.float64).eps) * _times_power_of_two(
            condition, amplification
        ) > 1e-12
        self._exact_inverses: dict[int, tuple[list[Fraction], list[list[Fraction]]]] = {}
        self._exact_formulas: dict[tuple[int, bool], tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]] = {}
        # Volumes are kept as mantissa * 2**exponent too, so that a product with a volume beyond the float64 range
        # (e.g. a refinement score) can still be finite.
        self._volume_mantissa = det.abs() / factorial(d)
        self._volume_exponent = self._edge_exponent.sum(dim=1) + self._axis_exponent.sum(dim=1)
        self._volumes: torch.Tensor = _times_power_of_two(self._volume_mantissa, self._volume_exponent)
        # Newest-vertex bisection state (set by lattice_mesh and refine): sorted simplex -> its _Tagged simplex.
        self._newest_vertex: dict[tuple[int, ...], _Tagged] | None = None
        self._unmatched = self._count_unmatched_faces()
        if self._unmatched:
            message = (
                f"the mesh is not conforming: {self._unmatched} faces are not shared face to face by two simplices "
                "(e.g. a Delaunay triangulation of lattice points with flat simplices removed; see lattice_mesh)."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        domain, tops = _simplicial_set(self._simplex_rows)
        self.space = BezierSpace.on_set(domain, degree=_DEGREE)
        self.cell_ids = torch.tensor([self.space.cell_id(t) for t in tops], dtype=torch.long)

    def __repr__(self) -> str:
        return f"SimplicialMesh(n_vertices={self.n_vertices}, n_simplices={self.n_simplices}, dimension={self.dimension})"

    @property
    def dimension(self) -> int:
        """Dimension ``d`` of the ambient space and of every simplex.

        Returns
        -------
        int
            Common ambient and simplex dimension.
        """
        return int(self.vertices.shape[1])

    @property
    def n_vertices(self) -> int:
        """Number of vertices.

        Returns
        -------
        int
            Number of rows in the vertex array.
        """
        return len(self.vertices)

    @property
    def n_simplices(self) -> int:
        """Number of simplices.

        Returns
        -------
        int
            Number of rows in the simplex array.
        """
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

        Returns
        -------
        int
            Count of nonconforming faces according to the overlap test above.
        """
        return self._unmatched

    def _count_unmatched_faces(self) -> int:  # overridden by _SplitMesh, which relies on __init__ calling it once
        d = self.dimension
        owners: dict[tuple[int, ...], list[int]] = {}
        for k, simplex in enumerate(self._simplex_rows):
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
            signs = torch.tensor([(-1) ** c for c in range(d)], dtype=faces.dtype)
            for a in range(1, d):
                for left in combinations(range(d), a):
                    for right in combinations(range(d), d - a):
                        # Once every candidate has a separating axis, further determinants cannot
                        # change the result. This is common for conforming boundary faces.
                        if not overlap.any():
                            return crowded
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
                        cofactors = torch.stack([spans[:, :, [k for k in range(d) if k != c]] for c in range(d)], dim=1)
                        u = torch.linalg.det(cofactors) * signs
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
        """Return the volumes of the simplices.

        Returns
        -------
        torch.Tensor
            Copy of simplex volumes of shape ``(n_simplices,)``.
        """
        return self._volumes.clone()

    def neighbors(self) -> list[list[int]]:
        """The vertices sharing an edge with each vertex (its one-ring), in ascending order.

        Returns
        -------
        list of list of int
            One sorted list of adjacent vertex IDs per vertex.
        """
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

        Notes
        -----
        The usual path uses the equilibrated float64 inverse. If its condition estimate indicates excessive
        factorization error, an absolute-product estimate of roundoff
        exceeds ``1e-12 * (1 + abs(coordinate))``, or the solve overflows, the affected point/simplex pairs
        are evaluated with exact rational arithmetic on the input floats and rounded once to float64.
        This slower path caches one exact inverse per affected simplex. It recovers coordinates lost by
        the solve, but cannot recover information already lost when the input points were rounded.
        Differentiable affected pairs additionally use cached SymPy expressions. Their derivatives are
        evaluated exactly at every order before rounding, independently for each point. Symbolic work and
        retained autograd state grow with dimension and derivative order; this path is intentionally slower.
        Forward-mode queries with any affected pair use symbolic expressions for the whole query so that
        ``forward_ad`` and ``torch.func.jvp`` retain tangents without reading dual tensor storage in Python.
        """
        p = self._check_points(x)
        origin = self.vertices[self.simplices[:, 0]]  # (m, d)
        offset = _times_power_of_two(p[:, None, :] - origin[None], -self._axis_exponent[None])
        rest = _times_power_of_two(torch.einsum("mij,nmj->nmi", self._inverse, offset), -self._edge_exponent[None])
        coordinates = torch.cat([1 - rest.sum(dim=2, keepdim=True), rest], dim=2)
        products = _times_power_of_two(
            torch.einsum("mij,nmj->nmi", self._inverse.abs(), offset.abs()), -self._edge_exponent[None]
        )
        magnitude = torch.cat([products.sum(dim=2, keepdim=True), products], dim=2)
        error = (8 * self.dimension * torch.finfo(torch.float64).eps) * magnitude
        uncertain = (error > 1e-12 * (1 + coordinates.abs())).any(dim=2) | ~torch.isfinite(coordinates).all(dim=2)
        uncertain |= self._inexact_inverse[None]
        if not uncertain.any():
            return coordinates
        return self._exact_barycentric(p, coordinates, uncertain, offset)

    def _forward_barycentric(self, p: torch.Tensor) -> torch.Tensor:
        """Evaluate dual queries without exposing transformed tensor storage to Python."""
        # torch.func duals cannot expose storage via tolist/nonzero. Evaluate the full query using
        # transform-compatible exact expressions, retaining both point and vertex tangents.
        cells = []
        for cell in range(self.n_simplices):
            key = (cell, True)
            if key not in self._exact_formulas:
                self._exact_formulas[key] = _barycentric_expressions(self.dimension, [], [], True)
            expressions, symbols = self._exact_formulas[key]
            vertices = self.vertices[self.simplices[cell]]
            cells.append(
                torch.stack(
                    [
                        _ExactExpressions.apply(expressions, symbols, (self.dimension + 1,), point, vertices)
                        for point in p.unbind()
                    ]
                )
            )
        return torch.stack(cells, dim=1)

    def _exact_barycentric(
        self, p: torch.Tensor, coordinates: torch.Tensor, uncertain: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Correct cancellation-prone pairs, retaining the derivative of the same affine map."""
        if (
            torch.autograd.forward_ad.unpack_dual(p).tangent is not None
            or torch.autograd.forward_ad.unpack_dual(self.vertices).tangent is not None
        ):
            return self._forward_barycentric(p)
        result = coordinates.detach().clone()
        if coordinates.requires_grad:
            # Rebuild the ordinary derivative path with finite offsets: even a zero upstream gradient
            # through an overflowed dot product can otherwise contaminate vertex gradients with NaN.
            safe_offset = torch.where(uncertain[:, :, None], 0.0, offset)
            rest = _times_power_of_two(torch.einsum("mij,nmj->nmi", self._inverse, safe_offset), -self._edge_exponent[None])
            result = torch.cat([1 - rest.sum(dim=2, keepdim=True), rest], dim=2)
        for cell_tensor in uncertain.any(dim=0).nonzero().flatten().unbind():
            cell = int(cell_tensor)
            vertices = self.vertices[self.simplices[cell]]
            if cell not in self._exact_inverses:
                self._exact_inverses[cell] = _rational_inverse(vertices)
            origin, inverse = self._exact_inverses[cell]
            indices = uncertain[:, cell].nonzero().flatten()
            if coordinates.requires_grad:
                key = (cell, vertices.requires_grad)
                if key not in self._exact_formulas:
                    self._exact_formulas[key] = _barycentric_expressions(
                        self.dimension, origin, inverse, vertices.requires_grad
                    )
                expressions, symbols = self._exact_formulas[key]
                rows = []
                for index_tensor in indices.unbind():
                    index = int(index_tensor)
                    arguments = (p[index], vertices) if vertices.requires_grad else (p[index],)
                    rows.append(_ExactExpressions.apply(expressions, symbols, (self.dimension + 1,), *arguments))
                values = torch.stack(rows)
            else:
                exact_rows = []
                for point in p[indices].tolist():
                    exact_offset = [Fraction(v) - o for v, o in zip(point, origin, strict=True)]
                    exact_rest = [sum((a * b for a, b in zip(row, exact_offset, strict=True)), Fraction(0)) for row in inverse]
                    exact_rows.append([_rounded_fraction(v) for v in [1 - sum(exact_rest, Fraction(0)), *exact_rest]])
                values = torch.tensor(exact_rows, dtype=torch.float64)
            result[indices, cell] = values
        return result

    def locate(self, x: Any, *, tol: float = 1e-8, batch_size: int | None = None) -> Points:
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
            Number of points tested at a time, bounding temporary float64 storage by
            ``O(batch_size * n_simplices * (d + 1))``. By default it is chosen from the number of simplices so
            that each temporary array of shape ``(batch_size, n_simplices, d + 1)`` takes at most 16 MiB
            (about a dozen such arrays are alive at once), and at least one point is tested at a time. Exact
            arithmetic for cancellation-prone pairs additionally caches ``O(d**2)`` rational coefficients per
            affected simplex, whose integer sizes depend on the input exponents. Those pairs are processed
            sequentially without an array of rational coordinates. Differentiable queries additionally retain
            symbolic expression caches and autograd state.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in :attr:`space`, in the order of ``x``.

        Raises
        ------
        ValueError
            If ``tol`` is negative or not finite, or a point lies outside the mesh by more than ``tol``.
        """
        if not math.isfinite(float(tol)) or tol < 0:
            raise ValueError(f"tol must be finite and nonnegative, but {tol} is given.")
        p = self._check_points(x)
        if batch_size is None:
            batch_size = _LOCATE_ARRAY_BYTES // (8 * max(1, self.n_simplices) * (self.dimension + 1))
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
        return p.to(device="cpu").to(dtype=torch.float64)

    def _check_values(self, values: Any) -> torch.Tensor:
        f = _as_real_tensor(values)
        if f.ndim == 1:
            f = f[:, None]
        if f.ndim != 2 or len(f) != self.n_vertices or f.shape[1] < 1 or f.is_complex() or not torch.isfinite(f).all():
            raise ValueError("values must be finite and real with shape (n_vertices, n_outputs).")
        return f.to(device="cpu").to(dtype=torch.float64)
