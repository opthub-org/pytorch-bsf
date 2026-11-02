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
modulus of continuity of the data at the mesh size. In one dimension the edge limiter restricts each
endpoint slope, separately on each incident interval, to the interval between zero and three times
the secant slope. The normalized slopes lie in the square :math:`[0, 3]^2` of the Fritsch–Carlson
monotonicity region :cite:p:`fritsch1980monotone`. The interpolant is C0, but need not be C1, even
on a chain of intervals. It agrees with PCHIP when supplied with its vertex slopes (the
Fritsch–Butland weighted harmonic means at interior vertices :cite:p:`fritsch1984method`, with
PCHIP's one-sided endpoint rule); the default :func:`estimate_gradients` uses different slopes.
"""

import math
import operator
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate, combinations, combinations_with_replacement
from math import factorial
from typing import Any, Literal, NamedTuple

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._kernels.sets import Simploid, SimploidalSet
from torch_bsf._validation import _choice
from torch_bsf.core import BezierModel, BezierSpace, Points

__all__ = [
    "C0",
    "C1",
    "CloughTocher",
    "MacroInterpolant",
    "NoSplit",
    "SimplicialMesh",
    "estimate_gradients",
    "hermite_interpolant",
    "lattice_mesh",
    "macro_interpolant",
    "refine",
    "refinement_indicator",
]

# PyTorch 2.3 imports Dynamo on the first custom-autograd transform. Its nested-tensor
# initialization uses random meta operations, which fail if that import occurs inside vmap.
# Initialize it before transforms; PyTorch 2.4 replaced the random module-level dummy.
if torch.__version__.startswith("2.3."):
    import torch._dynamo

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


class _Frexp(torch.autograd.Function):
    """Differentiate frexp without materializing an infinite scale factor for subnormal inputs."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mantissa, exponent = torch.frexp(value)
        return mantissa, exponent

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, exponent = output
        ctx.save_for_backward(exponent)
        ctx.save_for_forward(exponent)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor, exponent_gradient: None) -> torch.Tensor:
        (exponent,) = ctx.saved_tensors
        return _times_power_of_two(gradient, -exponent)

    @staticmethod
    def jvp(ctx: Any, tangent: torch.Tensor) -> tuple[torch.Tensor, None]:
        (exponent,) = ctx.saved_tensors
        return _times_power_of_two(tangent, -exponent), None


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
    """Return faces ``f`` and ``g`` (p, d, d) in exact local units, with their largest coordinates (p,).

    The faces are taken relative to the first vertex of ``f`` and scaled by :func:`_pair_exponents` (exactly);
    the largest absolute coordinate of each pair is returned in the same units.
    """
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


def _rational_inverse(vertices: torch.Tensor) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Origin and inverse edge matrix of input float64 vertices, without rounding their differences.

    Used lazily for point/simplex pairs whose floating-point solve suffers large cancellation. An exact
    inverse is shared by all such points; no inverse about a distant, numerically singular anchor is needed.
    """
    points = [[Fraction(float(v.detach())) for v in row.unbind()] for row in vertices.unbind()]
    origin, d = points[0], vertices.shape[1]
    rows = [[points[j + 1][i] - origin[i] for j in range(d)] + [Fraction(i == j) for j in range(d)] for i in range(d)]
    for column in range(d):
        pivot = max(range(column, d), key=lambda i: abs(rows[i][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        value = rows[column][column]
        if value == 0:
            raise ValueError("a simplex is degenerate (zero volume).")
        rows[column] = [v / value for v in rows[column]]
        for i in range(d):
            if i != column:
                value = rows[i][column]
                rows[i] = [a - value * b for a, b in zip(rows[i], rows[column], strict=True)]
    return origin, [row[d:] for row in rows]


def _rounded_fraction(value: Fraction) -> float:
    """Round a rational to float64, including coordinates of far points that overflow."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


class _ExactExpressions(torch.autograd.Function):
    """Evaluate and differentiate rational expressions before rounding, at every derivative order.

    Each invocation describes one point, so non-finite cotangents of another point cannot contaminate it.
    SymPy is loaded only for differentiable queries that require the exact fallback.
    """

    @staticmethod
    def forward(
        expressions: tuple[Any, ...],
        symbols: tuple[tuple[Any, ...], ...],
        shape: tuple[int, ...],
        *arguments: torch.Tensor,
    ) -> torch.Tensor:
        import sympy as sp

        bindings = {}
        for group, argument in zip(symbols, arguments, strict=True):
            for symbol, value in zip(group, argument.reshape(-1).tolist(), strict=True):
                if math.isfinite(value):
                    bindings[symbol] = sp.Rational(value)
                elif math.isnan(value):
                    bindings[symbol] = sp.nan
                else:
                    bindings[symbol] = sp.oo if value > 0 else -sp.oo
        numbers = [expression.xreplace(bindings) for expression in expressions]
        values = [_rounded_fraction(Fraction(int(v.p), int(v.q))) if v.is_Rational else float(v) for v in numbers]
        return torch.tensor(values, dtype=torch.float64).reshape(shape)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        ctx.expressions, ctx.symbols, ctx.shape, *arguments = inputs
        ctx.save_for_backward(*arguments)
        ctx.save_for_forward(*arguments)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Any, ...],
        expressions: tuple[Any, ...],
        symbols: tuple[tuple[Any, ...], ...],
        shape: tuple[int, ...],
        *arguments: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # SymPy reads concrete scalar bindings in forward. Iterate this transform's batch outside it,
        # keeping independent exact evaluations rather than exposing batched tensor storage to Python.
        rows = []
        for index in range(info.batch_size):
            values = tuple(
                argument if axis is None else argument.select(axis, index)
                for argument, axis in zip(arguments, in_dims[3:], strict=True)
            )
            rows.append(_ExactExpressions.apply(expressions, symbols, shape, *values))
        return torch.stack(rows), 0

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        import sympy as sp

        arguments = ctx.saved_tensors
        cotangents = tuple(sp.Dummy(real=True) for _ in ctx.expressions)
        symbols = (*ctx.symbols, cotangents)
        result: list[torch.Tensor | None] = []
        for index, argument in enumerate(arguments):
            if not ctx.needs_input_grad[index + 3]:
                result.append(None)
                continue
            derivatives = tuple(
                sp.Add(*(g * sp.diff(expression, variable) for g, expression in zip(cotangents, ctx.expressions, strict=True)))
                for variable in ctx.symbols[index]
            )
            result.append(_ExactExpressions.apply(derivatives, symbols, tuple(argument.shape), *arguments, gradient))
        return None, None, None, *result

    @staticmethod
    def jvp(
        ctx: Any, expressions_tangent: None, symbols_tangent: None, shape_tangent: None, *tangents: torch.Tensor
    ) -> torch.Tensor:
        import sympy as sp

        arguments = ctx.saved_tensors
        directions = tuple(tuple(sp.Dummy(real=True) for _ in group) for group in ctx.symbols)
        derivatives = tuple(
            sp.Add(
                *(
                    sp.diff(expression, variable) * direction
                    for group, tangent_group in zip(ctx.symbols, directions, strict=True)
                    for variable, direction in zip(group, tangent_group, strict=True)
                )
            )
            for expression in ctx.expressions
        )
        result: torch.Tensor = _ExactExpressions.apply(
            derivatives, (*ctx.symbols, *directions), ctx.shape, *arguments, *tangents
        )
        return result


def _barycentric_expressions(
    dimension: int, origin: list[Fraction], inverse: list[list[Fraction]], vertex_gradients: bool
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Compile a cell's exact affine formula; symbolic vertex coordinates retain inverse derivatives."""
    import sympy as sp

    point = tuple(sp.Dummy(real=True) for _ in range(dimension))
    symbols: tuple[tuple[Any, ...], ...]
    if vertex_gradients:
        vertices = tuple(sp.Dummy(real=True) for _ in range((dimension + 1) * dimension))
        edges = sp.Matrix(dimension, dimension, lambda i, j: vertices[(j + 1) * dimension + i] - vertices[i])
        offset = sp.Matrix([p - v for p, v in zip(point, vertices[:dimension], strict=True)])
        rest = edges.inv() * offset
        symbols = (point, vertices)
    else:
        matrix = sp.Matrix([[sp.Rational(v.numerator, v.denominator) for v in row] for row in inverse])
        offset = sp.Matrix([p - sp.Rational(v.numerator, v.denominator) for p, v in zip(point, origin, strict=True)])
        rest = matrix * offset
        symbols = (point,)
    return (1 - sum(rest), *rest), symbols


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


def _neighbourhood(v: int, ring: list[list[int]], max_rings: int, n_quadratic: int) -> tuple[list[int], bool]:
    """The neighbours of vertex ``v`` used by its gradient fit, and whether they suffice for the quadratic fit."""
    near = set(ring[v])
    for _ in range(max_rings - 1):
        near |= {w for u in near for w in ring[u]}
        near.discard(v)
        if len(near) >= n_quadratic + 2:
            break
    quadratic = len(near) >= n_quadratic + 2
    # The one-ring of a vertex of a non-degenerate simplex spans R^d, for the linear fallback.
    return (sorted(near) if quadratic else ring[v]), quadratic


def _gradients_at(v: torch.Tensor, rows: torch.Tensor, x: torch.Tensor, f: torch.Tensor, quadratic: bool) -> torch.Tensor:
    """Weighted least-squares gradients of the data at the vertices ``v`` (b,) from their neighbours ``rows`` (b, k).

    Every vertex of the batch has the same number ``k`` of neighbours and the same kind of fit, so that the
    fits are solved together by one batched least-squares call; each fit is independent of the others.
    """
    d = x.shape[1]
    iu = torch.triu_indices(d, d)
    # The fit is solved in the coordinates q = H^{-1} offsets, H = diag(extent of each axis), with the rows
    # weighted by 1 / |q|: each weighted linear row is the unit vector q / |q| and its right-hand side the slope
    # (f_w - f_v) / |q|. They are computed from the mantissas and exponents of the offsets, so that nothing
    # overflows or underflows on the way however the axes and the neighbours differ in scale, even where the
    # offsets themselves overflow, between vertices two rings apart near opposite ends of the float64 range.
    mantissa, exponent = _split_difference(x[rows], x[v][:, None])  # (b, k, d)
    exponent = torch.where(mantissa == 0, -(2**20), exponent)  # zeros never set the extent
    extent_exponent = exponent.amax(dim=1, keepdim=True)  # H_j = extent_mantissa_j * 2**extent_exponent_j
    extent_mantissa = torch.where(exponent == extent_exponent, mantissa.abs(), 0.0).amax(dim=1, keepdim=True)
    exponent = torch.where(mantissa == 0, -(2**20), exponent - extent_exponent)
    row_exponent = exponent.amax(dim=2, keepdim=True)  # q_k = 2**row_exponent_k * p_k, at most 0
    p = torch.where(mantissa == 0, 0.0, _times_power_of_two(mantissa / extent_mantissa, exponent - row_exponent))
    length = p.norm(dim=2, keepdim=True)  # in [1/2, 2 sqrt(d)]
    unit = p / length
    if quadratic:
        q = _times_power_of_two(p, row_exponent)  # entries of q are at most 1 in magnitude
        design = torch.cat([unit, unit[:, :, iu[0]] * q[:, :, iu[1]]], dim=2)
    else:
        design = unit
    # Slopes as mantissa * 2**exponent, scaled by one power of two per output so that the largest is near one.
    slope_mantissa, slope_exponent = _split_difference(f[rows], f[v][:, None])  # (b, k, n_outputs)
    slope_mantissa = slope_mantissa / length
    slope_exponent = torch.where(slope_mantissa == 0, -(2**20), slope_exponent - row_exponent)
    top = slope_exponent.amax(dim=1, keepdim=True)  # (b, 1, n_outputs)
    top = torch.where(top < -(2**19), 0, top)  # data constant on the neighbourhood
    rhs = _times_power_of_two(slope_mantissa, slope_exponent - top)
    solution: torch.Tensor = torch.linalg.lstsq(design, rhs, driver="gelsd").solution
    # The linear coefficients are the gradient times H, divided by 2**top.
    shift = top - extent_exponent.transpose(1, 2)
    return _times_power_of_two(solution[:, :d] / extent_mantissa.transpose(1, 2), shift)


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
    d = mesh.dimension
    n_quadratic = d + d * (d + 1) // 2
    out = torch.zeros(mesh.n_vertices, d, f.shape[1], dtype=torch.float64)
    # Fits with the same number of neighbours and the same kind are solved together.
    groups: dict[tuple[bool, int], tuple[list[int], list[list[int]]]] = {}
    for v in torch.unique(mesh.simplices).tolist():
        rows, quadratic = _neighbourhood(v, ring, max_rings, n_quadratic)
        vertices, neighbours = groups.setdefault((quadratic, len(rows)), ([], []))
        vertices.append(v)
        neighbours.append(rows)
    for (quadratic, _), (vertices, neighbours) in groups.items():
        v = torch.tensor(vertices, dtype=torch.long)
        out[v] = _gradients_at(v, torch.tensor(neighbours, dtype=torch.long), mesh.vertices, f, quadratic)
    return out


def _check_gradients(mesh: SimplicialMesh, gradients: Any, f: torch.Tensor) -> torch.Tensor:
    """Read vertex gradients of shape ``(n_vertices, d, n_outputs)``, or estimate them from ``f`` if ``None``."""
    if gradients is None:
        return estimate_gradients(mesh, f)
    g = _as_real_tensor(gradients)
    if g.shape != (mesh.n_vertices, mesh.dimension, f.shape[1]) or g.is_complex() or not torch.isfinite(g).all():
        raise ValueError("gradients must be finite and real with shape (n_vertices, d, n_outputs).")
    return g.to(device="cpu").to(dtype=torch.float64)


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
    monotone interpolant. The model is C0; it is not C1 in general, even in one dimension, because
    clamping can change the slope independently on either side of a vertex. For a C1 interpolant
    in two dimensions, use :func:`macro_interpolant` with ``split=CloughTocher(), continuity=C1()``.

    In one dimension, each endpoint slope is limited to the interval between zero and three times
    the secant slope of that interval. This gives the square part of the Fritsch–Carlson monotonicity
    region :cite:p:`fritsch1980monotone`. The default gradient estimates are not PCHIP slopes.
    Supplying PCHIP slopes as ``gradients`` gives the PCHIP interpolant up to float64 roundoff:
    the interior slopes use the Fritsch–Butland weighted harmonic mean :cite:p:`fritsch1984method`,
    and the endpoints use PCHIP's one-sided rule. Those slopes already satisfy the limiter.

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
    g = _check_gradients(mesh, gradients, f)
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


@dataclass(frozen=True)
class NoSplit:
    """Interpolate on the simplices of the mesh as they are (a ``split`` for :func:`macro_interpolant`)."""


@dataclass(frozen=True)
class CloughTocher:
    """Split every triangle into three at its incenter: the Clough–Tocher split (a ``split`` for :func:`macro_interpolant`).

    Two dimensions only. The incenter rather than the centroid is used because the segment joining the split points
    of two triangles sharing an edge then crosses that edge strictly inside it, which :class:`C1` relies on to keep
    every piece within the range of the data of its triangle. It is the incenter after each axis is divided by a
    power of two near the longest side of the mesh along it, which keeps that property (an affine map does) and
    makes the split, like the mesh, independent of the units of the axes. Where those powers of two are equal for both
    axes (e.g. on :func:`lattice_mesh`), it is the Euclidean incenter.
    """


@dataclass(frozen=True)
class C0:
    """Continuous joins between neighbouring pieces (a ``continuity`` for :func:`macro_interpolant`)."""


@dataclass(frozen=True)
class C1:
    """Continuously differentiable joins between neighbouring pieces (a ``continuity`` for :func:`macro_interpolant`)."""


# Values are kept below 2**_HEADROOM where the C1 construction adds, subtracts or averages a few of them, which leaves
# a factor 2**24 below the float64 overflow at 2**1024.
_HEADROOM = 1000
_OTHERS = ((1, 2), (0, 2), (0, 1))  # the vertices of a triangle other than vertex k, in ascending order
_TURNS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


class _Pieces(NamedTuple):
    """A mesh split into pieces: the pieces, the simplex of the mesh containing each, and the map of points into them."""

    mesh: SimplicialMesh
    parent: torch.Tensor
    transfer: Callable[[Points], Points]
    weights: torch.Tensor | None  # barycentric coordinates of the split point of each simplex


def _no_split(mesh: SimplicialMesh) -> _Pieces:
    """The mesh as its own single piece per simplex."""
    return _Pieces(mesh, torch.arange(mesh.n_simplices), lambda points: points, None)


def _clough_tocher_split(mesh: SimplicialMesh) -> _Pieces:
    """Split triangle ``t`` into the pieces ``3t + k`` spanned by the side opposite vertex ``k`` and the incenter.

    The incenter of triangle ``t`` becomes vertex ``n_vertices + t`` of the pieces.
    """
    if mesh.dimension != 2:
        raise ValueError(
            f"CloughTocher splits triangles: the mesh must be two-dimensional, not {mesh.dimension}-dimensional; "
            f"supported: {_supported_constructions()}."
        )
    x, s = mesh.vertices.detach(), mesh.simplices
    n, m = mesh.n_vertices, mesh.n_simplices
    others = torch.tensor(_OTHERS)
    order = _coordinate_order(x[s])
    corners = x[s.gather(1, order)]
    sides = corners[:, others[:, 1]] - corners[:, others[:, 0]]  # (m, 3, 2): side k is opposite corner k
    if not torch.isfinite(sides).all():
        raise ValueError("the vertices of a triangle are too far apart: their differences overflow float64.")
    # The incenter weighs each vertex by the length of the opposite side. The lengths are measured with every axis
    # divided by a power of two near the longest side along it in the mesh, so that, like the mesh itself, the split
    # does not depend on the units of the axes; the segment joining the split points of two neighbours still crosses
    # their common edge, as an affine map keeps it so. They are then taken relative to a power of two near the
    # longest side of the triangle, so that they neither overflow nor underflow.
    mantissa, exponent = torch.frexp(sides)
    exponent = torch.where(mantissa == 0, -(2**20), exponent.to(torch.long))  # zeros never set a scale
    exponent = exponent - exponent.amax(dim=(0, 1))
    exponent = exponent - exponent.amax(dim=(1, 2), keepdim=True)
    unit = _times_power_of_two(mantissa, exponent)
    lengths = torch.hypot(unit[..., 0], unit[..., 1])
    weights = lengths / lengths.sum(dim=1, keepdim=True)
    centers = corners[:, 0] + weights[:, 1:2] * sides[:, 2] + weights[:, 2:3] * sides[:, 1]
    weights = torch.empty_like(weights).scatter_(1, order, weights)  # back to the order of the vertices in s
    center_ids = n + torch.arange(m)
    simplices = torch.stack([torch.stack([s[:, a], s[:, b], center_ids], dim=1) for a, b in _OTHERS], dim=1)
    try:
        pieces = _SplitMesh(torch.cat([x, centers]), simplices.reshape(3 * m, 3), unmatched=mesh._unmatched)
    except ValueError as error:
        raise ValueError(
            "CloughTocher cannot split a triangle: its split point rounds onto or next to one of its sides (e.g. a "
            "triangle far thinner than its distance from the origin, or much thinner across than its neighbours "
            "after scaling each axis to the mesh)."
        ) from error
    simplex_of_cell = torch.full((int(mesh.cell_ids.max()) + 1,), -1, dtype=torch.long)
    simplex_of_cell[mesh.cell_ids] = torch.arange(m)

    def transfer(points: Points) -> Points:
        # The piece containing a point is the one whose apex (the incenter) has the largest coordinate c there,
        # i.e. the k minimizing lambda_k / w_k.
        t = simplex_of_cell[points.cell_ids.cpu()]
        lam = points.parameters[:, :3]
        w = weights[t].to(lam)
        ratio = lam / w
        k = ratio.argmin(dim=1, keepdim=True).cpu()
        c = ratio.gather(1, k.to(lam.device))
        pair = others[k[:, 0]].to(lam.device)
        mu = torch.cat([lam.gather(1, pair) - c * w.gather(1, pair), c], dim=1).clamp_min(0)
        return Points(mu / mu.sum(dim=1, keepdim=True), pieces.cell_ids[3 * t + k[:, 0]])

    return _Pieces(pieces, torch.arange(m).repeat_interleave(3), transfer, weights)


def _coordinate_order(corners: torch.Tensor) -> torch.Tensor:
    """The order (m, 3) of the corners (m, 3, 2) of each triangle by their coordinates, the first axis first.

    Sums over the corners taken in this order round the same however the vertices are numbered, so that the rays
    seen by the limiter, which is sensitive to their rounding where the feasible gradients form a thin wedge (e.g.
    at a vertex on a straight boundary), do not depend on the numbering.
    """
    order = torch.argsort(corners[..., 1], dim=1, stable=True)
    return order.gather(1, torch.argsort(corners[..., 0].gather(1, order), dim=1, stable=True))


class _SplitMesh(SimplicialMesh):
    """The mesh of the pieces of a parent mesh, which conform exactly where the simplices of the parent do.

    Its unmatched faces are those of the parent, as its new faces join pieces of one simplex. Testing them again would
    only add rounding: two long sides of a sliver, never compared as sides of one simplex, may then seem to overlap.
    """

    def __init__(self, vertices: torch.Tensor, simplices: torch.Tensor, *, unmatched: int):
        super().__init__(vertices, simplices, strict=False)
        self._unmatched = unmatched

    def _count_unmatched_faces(self) -> int:
        # SimplicialMesh.__init__ warns when this is non-zero; the parent's count, already warned about there, is
        # assigned after it.
        return 0


def _increments(dx: torch.Tensor, g: torch.Tensor, scale: torch.Tensor, origin: torch.Tensor, clamp: bool) -> torch.Tensor:
    """``dx . g / 3`` (rays, outputs) in units of ``2**scale``, for ray displacements ``dx`` (rays, d) and gradients ``g``.

    Each term is formed as the product of the mantissas of ``dx`` and ``g`` times a power of two, so that it neither
    overflows nor underflows before it is placed, and the terms are summed in units of the largest, so that an
    increment is finite whenever its value is, also where its terms overflow alone and cancel. Each term is rounded
    once, as in a float64 dot product, so an increment is accurate up to a few units of rounding of its largest term
    (not of its value where the terms cancel), and scaling ``dx`` or ``g`` by a power of two scales it exactly
    whether or not its terms overflow. With ``clamp``, a
    gradient (of a vertex and output) whose increments would overflow is first divided by a power of two, which keeps
    its direction: the limiter shrinks it into the data range anyway.
    """
    mantissa, exponent = torch.frexp(g)
    dx_mantissa, dx_exponent = torch.frexp(dx)
    term_mantissa, term_exponent = torch.frexp(dx_mantissa[:, :, None] * mantissa)
    term_exponent = term_exponent.to(torch.long) + exponent.to(torch.long) + dx_exponent.to(torch.long)[:, :, None] - scale
    term_exponent = torch.where(term_mantissa == 0, -(2**20), term_exponent)
    top = term_exponent.amax(dim=1)
    if clamp:
        largest = torch.full((int(origin.max()) + 1, top.shape[1]), -(2**20), dtype=torch.long)
        largest = largest.scatter_reduce(0, origin[:, None].expand_as(top), top, "amax")
        shift = (largest - _HEADROOM).clamp_min(0)[origin]
        term_exponent, top = term_exponent - shift[:, None, :], top - shift
    total = _times_power_of_two(term_mantissa, term_exponent - top[:, None, :]).sum(dim=1)
    return _times_power_of_two(total / 3, top)


def _unclamped_increments(
    dx: torch.Tensor, g: torch.Tensor, scale: torch.Tensor, origin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The increments of :func:`_increments` without clamp, and the power of two added to ``scale`` for them.

    Increments beyond 2**_HEADROOM raise the scale of their output as such data do, so that the controls they make
    stay finite in the sums of the construction (the limiter keeps them in the data range with clamp).
    """
    increment = _increments(dx, g, scale, origin, False)
    _, top = torch.frexp(increment.abs().amax(dim=0))
    extra = (top.to(torch.long) - _HEADROOM).clamp_min(0)
    if extra.any():
        increment = _increments(dx, g, scale + extra, origin, False)
    return increment, extra


def _nearest_feasible(
    q: torch.Tensor, r: torch.Tensor, target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """The point of ``{q h : lower <= q h <= upper}`` nearest to ``target``, for QR factors ``q`` (N, K, 2), ``r`` (N, 2, 2).

    ``target`` lies in the span of ``q``, so this is the nearest point ``h`` of a convex polygon to ``h0 = q^T target``,
    which contains ``h = 0``. It is ``h0`` itself or lies on an edge: the part of a line ``q_k . h = lower_k`` or
    ``upper_k`` inside the polygon, a segment found by intersecting the line with the other slabs, on which the point
    nearest to ``h0`` is its projection clamped to the segment. The candidates are tried together, in ``O(K**2)``.

    The rays ``a = q r`` are known only up to rounding, ``eps |a|`` (and the QR factorization rounds them so): turned
    by that much, they move ``q h = a g`` by up to ``eps |r| |g|`` for ``g = r^-1 h``, which can be far more than
    ``eps |h|`` for ill-conditioned rays. The bounds are relaxed by such rounding, so that, e.g., the line between two
    nearly opposite rays of a vertex on a straight boundary is not lost to it.
    """
    # In units of a power of two per row near its largest value, so that the squared distances neither underflow nor
    # overflow.
    _, exponent = torch.frexp(torch.cat([target, lower, upper], dim=1).abs().amax(dim=1, keepdim=True))
    exponent = exponent.to(torch.long)
    target, lower, upper = (_times_power_of_two(t, -exponent) for t in (target, lower, upper))
    n, k = target.shape
    best = torch.zeros(n, 2, dtype=torch.float64)
    # Rows per chunk, so that a chunk holds about 2**20 pairs of a line (2K per row) and a slab (K per row): its
    # float64 temporaries take a few hundred megabytes.
    chunk = max(1, 2**20 // (2 * k * k))
    for begin in range(0, n, chunk):
        rows = slice(begin, begin + chunk)
        best[rows] = _nearest_feasible_chunk(q[rows], r[rows], target[rows], lower[rows], upper[rows])
    return _times_power_of_two(torch.einsum("nkd,nd->nk", q, best), exponent)


def _nearest_feasible_chunk(
    q: torch.Tensor, r: torch.Tensor, target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """The coordinates ``h`` (N, 2) of :func:`_nearest_feasible` for rows whose values are already of order one."""
    h0 = torch.einsum("nkd,nk->nd", q, target)
    lines = q.repeat(1, 2, 1)  # (N, 2K, 2): line l lies on bound l // K (lower, then upper) of slab l % K
    level = torch.cat([lower, upper], dim=1)
    direction = torch.stack([-lines[..., 1], lines[..., 0]], dim=2)
    # The projection of h0 onto each line, as its parts across and along the line, without cancellation against h0.
    along = _dots(direction, h0[:, None])[..., 0]
    base = (lines * level[..., None] + direction * along[..., None]) / lines.square().sum(dim=2, keepdim=True)
    # Along line l, slab j holds b + t a with a = q_j . direction and b = q_j . base: an interval of t, or all or no t
    # where a = 0 (a line parallel to the slab). Its own slab holds the whole line, up to rounding. The slabs are
    # widened by the rounding tolerated in the candidates below, for points of the size of the base, so that a line
    # along a slab that rounding puts just outside it is not lost.
    a, b = _dots(direction, q), _dots(base, q)
    size = _ray_rounding_size(r, base)
    lo = lower[:, None] - _feasibility_slack(size, lower[:, None])
    hi = upper[:, None] + _feasibility_slack(size, upper[:, None])
    k = q.shape[1]
    own = torch.arange(2 * k)[:, None] % k == torch.arange(k)
    parallel = (a == 0) | own
    inside = ((b >= lo) & (b <= hi)) | own
    safe = torch.where(parallel, 1.0, a)
    t1, t2 = (lo - b) / safe, (hi - b) / safe
    t_lo, j_lo = torch.where(parallel, torch.where(inside, -math.inf, math.inf), torch.minimum(t1, t2)).max(dim=2)
    t_hi, j_hi = torch.where(parallel, torch.where(inside, math.inf, -math.inf), torch.maximum(t1, t2)).min(dim=2)
    # The projection clamped to [t_lo, t_hi]: at an end, the corner where the line meets the slab bounding it there,
    # solved directly (Cramer's rule) rather than as base + t direction, which loses accuracy for small a.
    use_lo = t_lo > 0
    j = torch.where(use_lo, j_lo, j_hi)
    a_j = a.gather(2, j[..., None])[..., 0]
    at_lower = (a_j > 0) == use_lo  # the end lies on the lower bound of slab j
    c = torch.where(at_lower, lower.gather(1, j), upper.gather(1, j))
    qj = q.gather(1, j[..., None].expand(-1, -1, 2))
    det = lines[..., 0] * qj[..., 1] - lines[..., 1] * qj[..., 0]
    corner = (
        torch.stack([level * qj[..., 1] - c * lines[..., 1], lines[..., 0] * c - qj[..., 0] * level], dim=2) / det[..., None]
    )
    end = torch.where((use_lo | (t_hi < 0))[..., None], corner, base)
    candidates = torch.cat([h0[:, None], end], dim=1)  # (N, 1 + 2K, 2)
    values = _dots(candidates, q)
    # Only rounding is tolerated, also on a narrow (or zero-width) slab: a larger violation clipped afterwards would make
    # the increments of the vertex inconsistent with any one gradient, and break C1 there.
    size = _ray_rounding_size(r, candidates)
    lower, upper = lower[:, None], upper[:, None]
    feasible = ((values >= lower - _feasibility_slack(size, lower)) & (values <= upper + _feasibility_slack(size, upper))).all(
        dim=2
    )
    distance = torch.where(feasible, (candidates - h0[:, None]).square().sum(dim=2), math.inf)  # NaN is never feasible
    nearest = distance.argmin(dim=1)
    found = torch.isfinite(distance.gather(1, nearest[:, None]))[:, 0] & (
        distance.gather(1, nearest[:, None])[:, 0] < h0.square().sum(dim=1)
    )
    return torch.where(found[:, None], candidates[torch.arange(len(q)), nearest], torch.zeros_like(h0))


def _ray_rounding_size(r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``|r| |r^-1 h|`` (N, L, 1) for points ``h`` (N, L, 2), with 1-norms: the size of the rounding of ``q h = a g``."""
    g: torch.Tensor = torch.linalg.solve_triangular(r[:, None], h[..., None], upper=True)[..., 0]
    return r.abs().sum(dim=(1, 2))[:, None, None] * g.abs().sum(dim=2, keepdim=True)


def _dots(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dot products (N, I, J) of planar vectors ``u`` (N, I, 2) and ``v`` (N, J, 2), each rounded the same way."""
    return u[:, :, None, 0] * v[:, None, :, 0] + u[:, :, None, 1] * v[:, None, :, 1]


def _feasibility_slack(size: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """The rounding tolerated in a value of magnitude about ``size`` compared with ``bound``.

    A value is a few products and sums of numbers of magnitude ``size``, and a corner solves a 2 x 2 system for it: 32
    roundings comfortably cover both.
    """
    return 32 * torch.finfo(torch.float64).eps * (size.abs() + bound.abs())


def _limit_increments(
    dx: torch.Tensor, increment: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, origin: torch.Tensor
) -> torch.Tensor:
    """Project the increments of each vertex (and output) onto those of the gradients that keep them in bounds.

    The increments ``dx . g / 3`` along the rays of a vertex are ``A g`` with ``A = dx / 3``. Where some of them leave
    ``[lower, upper]``, ``g`` is replaced by the feasible gradient minimizing ``|A (g - g_0)|``, i.e. the increments
    by their Euclidean projection onto the feasible increments; this is invariant under affine maps of the domain.
    The rays of a vertex are processed together, grouped with the vertices with as many rays.
    """
    out = increment.clone()
    bad = (increment < lower) | (increment > upper)
    # The rays of a vertex in the order of their directions, so that the factorization below, and how its rounding
    # resolves a nearly flat set of feasible gradients, does not depend on the order of the simplices.
    order = torch.argsort(dx[:, 1], stable=True)
    order = order[torch.argsort(dx[order, 0], stable=True)]
    order = order[torch.argsort(origin[order], stable=True)]
    counts = torch.bincount(origin)
    starts = counts.cumsum(0) - counts
    for k in torch.unique(counts[counts > 0]).tolist():
        rays = order[starts[counts == k][:, None] + torch.arange(k)]  # (V, K)
        vertex, output = bad[rays].any(dim=1).nonzero(as_tuple=True)
        if not len(vertex):
            continue
        a = dx[rays]
        _, exponent = torch.frexp(a.abs().amax(dim=1))  # equilibrate the axes before the QR factorization
        factors = torch.linalg.qr(_times_power_of_two(a, -exponent.to(torch.long)[:, None, :]))
        rows, o = rays[vertex], output[:, None]
        out[rows, o] = _nearest_feasible(
            factors.Q[vertex], factors.R[vertex], increment[rows, o], lower[rows, o], upper[rows, o]
        )
    return out


def _plane_coordinates(u: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(alpha, beta)`` with ``u = alpha a + beta b`` for planar vectors of shape ``(k, 2)``, by Cramer's rule.

    Each axis is first divided by a power of two near its largest component, which leaves ``alpha`` and ``beta``
    unchanged (an affine map does) and keeps the products from overflowing or underflowing.
    """
    stacked = torch.stack([u, a, b], dim=1)
    _, exponent = torch.frexp(stacked.abs().amax(dim=1))
    u, a, b = _times_power_of_two(stacked, -exponent.to(torch.long)[:, None, :]).unbind(dim=1)
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return (u[:, 0] * b[:, 1] - u[:, 1] * b[:, 0]) / det, (a[:, 0] * u[:, 1] - a[:, 1] * u[:, 0]) / det


def _split_point_coordinates(
    x: torch.Tensor, s: torch.Tensor, pairs: torch.Tensor, to_split: torch.Tensor, deep: torch.Tensor, shallow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(lam, mu, nu)`` with ``C_shallow = lam C_deep + mu V_a + nu V_b``, for rows ``deep`` and ``shallow`` of ``pairs``.

    Rows ``deep`` and ``shallow`` are the edge ``(V_a, V_b)`` in two triangles ``deep // 3`` and ``shallow // 3``, and
    ``to_split[t, k]`` is ``C - V`` at vertex ``k`` of triangle ``t``, so that only differences of vertices are used.
    """
    va, vb = pairs[deep, 0], pairs[deep, 1]
    ab = x[vb] - x[va]

    def from_vertex(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # C - V of triangle t at its vertex v
        return to_split[t, (s[t] == v[:, None]).to(torch.long).argmax(dim=1)]

    lam, nu = _plane_coordinates(from_vertex(va, shallow // 3), from_vertex(va, deep // 3), ab)
    _, mu = _plane_coordinates(from_vertex(vb, shallow // 3), from_vertex(vb, deep // 3), -ab)
    return lam, mu, nu


def _clough_tocher_c1(mesh: SimplicialMesh, pieces: _Pieces, f: torch.Tensor, gradients: Any, clamp: bool) -> BezierModel:
    r"""The C1 Clough–Tocher interpolant, with the range of every triangle's data kept on it if ``clamp``.

    On piece ``(V_a, V_b, C)`` of a triangle with split point :math:`C = \sum_v w_v V_v`, the controls are
    :math:`f_a`, :math:`e_{ab} = f_a + g_a \cdot (V_b - V_a) / 3`, :math:`F_a = f_a + g_a \cdot (C - V_a) / 3`, their
    mirror images, :math:`Q_{ab}` next to the edge, :math:`E_a = w_a F_a + w_b Q_{ab} + w_c Q_{ac}` (``c`` the third
    vertex) and :math:`\sum_v w_v E_v` at ``C`` :cite:p:`clough1965finite,farin2002curves`. These are C1 inside the
    triangle for any ``Q``. Each ``Q`` is the reduced Clough–Tocher value, whose derivative across the edge (along the
    line joining the split points on either side, or towards ``C`` on the boundary) is linear along it, moved to the
    nearest pair satisfying the C1 condition across the edge. With ``clamp``, the gradients are first limited so that
    the controls next to every vertex lie in the range of each triangle containing them, and ``Q`` is moved within
    that range on both sides, which is possible because the line joining the split points crosses the edge inside
    it. All other controls are then convex combinations, so the convex-hull property keeps every piece in the range
    of its triangle.
    """
    assert pieces.weights is not None
    g = _check_gradients(mesh, gradients, f)
    x, s, w = mesh.vertices.detach(), mesh.simplices, pieces.weights
    m, n_outputs = mesh.n_simplices, f.shape[1]
    others = torch.tensor(_OTHERS)
    data_lo, data_hi = f[s].amin(dim=1), f[s].amax(dim=1)
    # Data beyond 2**_HEADROOM are divided by a power of two per output, so that their differences and sums are finite.
    _, scale = torch.frexp(f.abs().amax(dim=0))
    scale = (scale.to(torch.long) - _HEADROOM).clamp_min(0)
    f = _times_power_of_two(f, -scale)
    lo, hi = f[s].amin(dim=1), f[s].amax(dim=1)

    # Rays from a vertex to a point a third of the way to which a control lies: ray 2e (2e + 1) runs from the first
    # (second) end of edge e to the other, ray 2E + 3t + a from vertex a of triangle t to its split point.
    pairs = s[:, others].reshape(3 * m, 2)  # row 3t + k: the edge of triangle t opposite vertex k
    edges, edge_of = torch.unique(pairs, dim=0, return_inverse=True)
    n_edges = len(edges)
    origin = torch.cat([edges.flatten(), s.flatten()])
    # C - V_a = sum_v w_v (V_v - V_a), from differences of the data vertices rather than the rounded vertex C of the
    # pieces, so that, like the pieces located by transfer, it does not move when the mesh is translated. Finite: the
    # sides are (checked by the split), and this is a convex combination of the sides from V_a. Summed over the
    # vertices v in the order of their coordinates, as the split does.
    corners = x[s]
    by_coordinates = _coordinate_order(corners)
    v_weights, v_corners = w.gather(1, by_coordinates), x[s.gather(1, by_coordinates)]
    to_split = (v_weights[:, None, :, None] * (v_corners[:, None] - corners[:, :, None])).sum(dim=2)  # (m, 3, 2): from a
    along = x[edges[:, 1]] - x[edges[:, 0]]
    dx = torch.cat([torch.stack([along, -along], dim=1).reshape(2 * n_edges, 2), to_split.reshape(3 * m, 2)])
    triangle = torch.arange(m).repeat_interleave(3)
    index = edge_of[:, None].expand(-1, n_outputs)
    edge_lo = torch.full((n_edges, n_outputs), -math.inf, dtype=torch.float64).scatter_reduce(0, index, lo[triangle], "amax")
    edge_hi = torch.full((n_edges, n_outputs), math.inf, dtype=torch.float64).scatter_reduce(0, index, hi[triangle], "amin")
    ray_lo = torch.cat([edge_lo.repeat_interleave(2, dim=0), lo.repeat_interleave(3, dim=0)])
    ray_hi = torch.cat([edge_hi.repeat_interleave(2, dim=0), hi.repeat_interleave(3, dim=0)])

    if clamp:
        increment = _increments(dx, g[origin], scale, origin, clamp)
        lower, upper = ray_lo - f[origin], ray_hi - f[origin]
        increment = _limit_increments(dx, increment, lower, upper, origin).clamp(lower, upper)
    else:
        # lo and hi, used only with clamp, keep the scale of the data.
        increment, extra = _unclamped_increments(dx, g[origin], scale, origin)
        scale, f = scale + extra, _times_power_of_two(f, -extra)
    ordinate = f[origin] + increment
    if clamp:
        ordinate = ordinate.clamp(ray_lo, ray_hi)
    toward_c = ordinate[2 * n_edges :]  # row 3t + a: F_a of triangle t

    row = torch.arange(3 * m)
    first_vertex = 3 * triangle + others[row % 3, 0]
    second_vertex = 3 * triangle + others[row % 3, 1]
    fa, fb = f[pairs[:, 0]], f[pairs[:, 1]]
    ea, eb = ordinate[2 * edge_of], ordinate[2 * edge_of + 1]
    fa_c, fb_c = toward_c[first_vertex], toward_c[second_vertex]

    # The direction of the derivative across each edge, as barycentric differences (beta_a, beta_b, beta_C): towards
    # C from the middle of a boundary edge; along C_shallow - C_deep across an interior edge with a deep and a shallow
    # side, where C_shallow = lam C_deep + mu V_a + nu V_b. Then (mu, nu, lam - 1) on the deep side is lam times its
    # value on the shallow side.
    beta = torch.tensor([-0.5, -0.5, 1.0], dtype=torch.float64).repeat(3 * m, 1)
    count = torch.bincount(edge_of, minlength=n_edges)
    order = torch.argsort(edge_of, stable=True)
    start = (count.cumsum(0) - count)[count == 2]
    deep, shallow = order[start], order[start + 1]  # the two rows of each interior edge
    # -lam is the ratio of the distances of the two split points from the edge. The deep side is the one whose split
    # point is the farther (the shallow side the nearer), so that |lam| <= 1 and mu + nu = 1 - lam <= 2 keep the
    # terms below of the data's size next to a much thinner neighbour.
    lam, _, _ = _split_point_coordinates(x, s, pairs, to_split, deep, shallow)
    swap = lam.abs() > 1
    deep, shallow = torch.where(swap, shallow, deep), torch.where(swap, deep, shallow)
    lam, mu, nu = (c[:, None] for c in _split_point_coordinates(x, s, pairs, to_split, deep, shallow))  # (E, 1)
    t_deep, t_shallow = deep // 3, shallow // 3
    beta[deep] = beta[shallow] = torch.cat([mu, nu, lam - 1], dim=1)
    # beta_a / beta_C and beta_b / beta_C, both in [-1, 0] (beta_C = -(beta_a + beta_b), and mu, nu >= 0).
    ra, rb = beta[:, :1] / beta[:, 2:], beta[:, 1:2] / beta[:, 2:]
    # The reduced Clough-Tocher value: the derivative along beta across the edge is a quadratic along it, with controls
    # proportional to beta . (f_a, e_ab, F_a), beta . (e_ab, e_ba, Q) and beta . (e_ba, f_b, F_b) (in the order a, b, C);
    # Q makes it linear, the middle control the mean of the others.
    q = (fa_c + fb_c) / 2 + (ra * (fa + eb) + rb * (ea + fb)) / 2 - (ra * ea + rb * eb)
    # The C1 condition across an interior edge is Q_shallow = lam Q_deep + r: take the nearest such pair.
    r = mu * ea[deep] + nu * eb[deep]
    q_deep = (q[deep] + lam * (q[shallow] - r)) / (1 + lam.square())
    if clamp:
        # Both in range: Q_deep between q_low and q_high, which r / (1 - lam), a convex combination of e_a and e_b, is.
        q_low = torch.maximum(lo[t_deep], (hi[t_shallow] - r) / lam)
        q_high = torch.minimum(hi[t_deep], (lo[t_shallow] - r) / lam)
        q_deep = torch.minimum(torch.maximum(q_deep, q_low), q_high)
    q[deep], q[shallow] = q_deep, lam * q_deep + r
    if clamp:
        q = q.clamp(lo[triangle], hi[triangle])

    q = q.reshape(m, 3, n_outputs)
    toward_c3 = toward_c.reshape(m, 3, n_outputs)
    weight = w[:, :, None]
    # E_a = w_a F_a + w_b Q_ab + w_c Q_ac, where Q_ab belongs to the edge opposite c.
    inner = torch.stack(
        [weight[:, a] * toward_c3[:, a] + weight[:, b] * q[:, c] + weight[:, c] * q[:, b] for a, b, c in _TURNS], dim=1
    )
    center = (weight * inner).sum(dim=1)
    if clamp:
        inner, center = inner.clamp(lo[:, None], hi[:, None]), center.clamp(lo, hi)
    inner = inner.reshape(3 * m, n_outputs)
    values: dict[tuple[int, ...], torch.Tensor] = {
        (3, 0, 0): fa,
        (0, 3, 0): fb,
        (0, 0, 3): center[triangle],
        (2, 1, 0): ea,
        (1, 2, 0): eb,
        (2, 0, 1): fa_c,
        (0, 2, 1): fb_c,
        (1, 0, 2): inner[first_vertex],
        (0, 1, 2): inner[second_vertex],
        (1, 1, 1): q.reshape(3 * m, n_outputs),
    }
    space = pieces.mesh.space
    controls = torch.tensor([space.cells[c].controls for c in pieces.mesh.cell_ids.tolist()], dtype=torch.long)
    coefficients = torch.zeros(space.n_controls, n_outputs, dtype=torch.float64)
    for k, (alpha,) in enumerate(_indices((2,), (_DEGREE,))):
        # A shared control gets bitwise equal values from every piece, so duplicate rows are harmless.
        coefficients[controls[:, k]] = values[alpha]
    coefficients = _times_power_of_two(coefficients, scale)
    if clamp:
        # Into the ranges of the data themselves, against the rounding of the scaled data: a control shared by pieces
        # of several triangles (on an edge or at a vertex) lies in the range of each, which every value above does.
        rows = controls.flatten()[:, None].expand(-1, n_outputs)
        parents = pieces.parent.repeat_interleave(controls.shape[1])
        low = torch.full_like(coefficients, -math.inf).scatter_reduce(0, rows, data_lo[parents], "amax")
        high = torch.full_like(coefficients, math.inf).scatter_reduce(0, rows, data_hi[parents], "amin")
        coefficients = coefficients.clamp(low, high)
    if not torch.isfinite(coefficients).all():
        raise ValueError("the interpolant is not finite: the vertex data or gradients cause its controls to overflow float64.")
    return BezierModel(space, coefficients)


def _hermite_c0(mesh: SimplicialMesh, pieces: _Pieces, f: torch.Tensor, gradients: Any, clamp: bool) -> BezierModel:
    """The C0 interpolant of :func:`hermite_interpolant` on the unsplit mesh."""
    return hermite_interpolant(mesh, f, gradients=gradients, clamp=clamp)


_SPLITS: dict[type, Callable[[SimplicialMesh], _Pieces]] = {NoSplit: _no_split, CloughTocher: _clough_tocher_split}
_CONSTRUCTIONS: dict[tuple[type, type], Callable[[SimplicialMesh, _Pieces, torch.Tensor, Any, bool], BezierModel]] = {
    (NoSplit, C0): _hermite_c0,
    (CloughTocher, C1): _clough_tocher_c1,
}
_CONTINUITIES = tuple(dict.fromkeys(continuity for _, continuity in _CONSTRUCTIONS))


def _supported_constructions() -> str:
    """The supported pairs of a split and a continuity, for error messages."""
    return ", ".join(f"split={a.__name__}() with continuity={b.__name__}()" for a, b in _CONSTRUCTIONS)


@dataclass(frozen=True, eq=False)
class MacroInterpolant:
    """A piecewise cubic interpolant on a mesh, defined on pieces of its simplices (see :func:`macro_interpolant`).

    Attributes
    ----------
    mesh
        The mesh carrying the data.
    pieces
        The mesh of the pieces, on whose :attr:`~SimplicialMesh.space` the model is defined. It is ``mesh`` itself
        without a split.
    parent
        ``parent[k]`` is the simplex of ``mesh`` containing piece ``k``.
    model
        The float64 model. Evaluate it at :meth:`locate`, or at :meth:`transfer` of points of ``mesh.space``.
    """

    mesh: SimplicialMesh
    pieces: SimplicialMesh
    parent: torch.Tensor
    model: BezierModel
    _transfer: Callable[[Points], Points] = field(repr=False, kw_only=True)

    def transfer(self, points: Points) -> Points:
        """Map points of ``mesh.space`` to the pieces containing them, preserving their order.

        Parameters
        ----------
        points
            Barycentric coordinates in simplices of ``mesh`` and their cell IDs (from :attr:`SimplicialMesh.cell_ids`),
            e.g. from :meth:`SimplicialMesh.locate`.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in ``pieces.space``.

        Raises
        ------
        ValueError
            If ``points`` are not valid points of ``mesh.space`` (see :meth:`~torch_bsf.core.BezierSpace.validate`),
            or a cell ID is not that of a simplex of ``mesh`` (e.g. a face, or a simplex replaced by :func:`refine`).

        Examples
        --------
        The interpolant of the first coordinate reproduces it, here at the centroids of the triangles:

        >>> import torch
        >>> from torch_bsf.core import Points
        >>> from torch_bsf.interpolation import C1, CloughTocher, lattice_mesh, macro_interpolant
        >>> mesh = lattice_mesh(2, 2)
        >>> interpolant = macro_interpolant(mesh, mesh.vertices[:, 0], split=CloughTocher(), continuity=C1())
        >>> centroids = Points(torch.full((mesh.n_simplices, 3), 1 / 3, dtype=torch.float64), mesh.cell_ids)
        >>> values = interpolant.model(interpolant.transfer(centroids))[:, 0]
        >>> torch.allclose(values, mesh.vertices[mesh.simplices, 0].mean(dim=1))
        True
        """
        if not torch.isin(points.cell_ids.cpu(), self.mesh.cell_ids).all():
            raise ValueError("points must lie in simplices of the mesh: their cell IDs must be in mesh.cell_ids.")
        self.mesh.space.validate(points)
        return self._transfer(points)

    def locate(self, x: Any, *, tol: float = 1e-8, batch_size: int | None = None) -> Points:
        """Find the piece containing each point and its barycentric coordinates there.

        The points are located in :attr:`mesh` by :meth:`SimplicialMesh.locate`, and then in the pieces of their
        simplex by :meth:`transfer`.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.
        tol
            Tolerance on the barycentric coordinates, as in :meth:`SimplicialMesh.locate`.
        batch_size
            Number of points located at a time, as in :meth:`SimplicialMesh.locate`, which also chooses the
            default.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in ``pieces.space``, in the order of ``x``.

        Raises
        ------
        ValueError
            If ``tol`` or a point is invalid, or a point is outside the mesh (see :meth:`SimplicialMesh.locate`).
        """
        return self.transfer(self.mesh.locate(x, tol=tol, batch_size=batch_size))

    def __call__(self, x: Any, *, tol: float = 1e-8, batch_size: int | None = None) -> torch.Tensor:
        """Evaluate the interpolant at points ``x``.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.
        tol, batch_size
            As in :meth:`locate`.

        Returns
        -------
        torch.Tensor
            Float64 predictions of shape ``(n_points, n_outputs)``.
        """
        prediction: torch.Tensor = self.model(self.locate(x, tol=tol, batch_size=batch_size))
        return prediction


def macro_interpolant(
    mesh: SimplicialMesh,
    values: Any,
    *,
    split: NoSplit | CloughTocher = NoSplit(),
    continuity: C0 | C1 = C0(),
    gradients: Any = None,
    clamp: bool = True,
) -> MacroInterpolant:
    r"""Interpolate vertex data by a piecewise cubic Bézier model on pieces of the simplices of a mesh.

    ``split`` says how each simplex is divided into pieces, and ``continuity`` how smoothly neighbouring pieces
    join. The supported constructions are

    * ``NoSplit()`` with ``C0()`` (any dimension): :func:`hermite_interpolant`;
    * ``CloughTocher()`` with ``C1()`` (two dimensions): the Clough–Tocher macro-element, continuously
      differentiable, on three cubic pieces per triangle.

    Both interpolate the data and reproduce quadratic data without clamping when the gradients do. With
    ``clamp=True``, on every simplex of ``mesh`` and for every output, the prediction lies between the smallest and
    largest data at its vertices. For the C1 construction, the gradients are first limited for this, each projected
    onto the gradients keeping the controls next to its vertex in the range of every triangle containing them,
    which keeps the interpolant C1. In the spirit of :cite:p:`fritsch1980monotone`, this is a two-dimensional
    range-restricted C1 interpolant. The set of such gradients can shrink abruptly as the mesh moves. For example,
    take a vertex whose datum is the largest datum of the triangle on one side of each of two of its edges and the
    smallest of the triangle on the other side. Its gradient must be orthogonal to both edges: while they are
    parallel, it is free along their common normal; once they are not, it must vanish. The clamped C1 interpolant
    therefore need not depend continuously on the vertex coordinates.

    The construction is not differentiated with respect to the mesh: like every :class:`~torch_bsf.core.BezierModel`,
    the model copies its coefficients without their autograd history, as for :func:`hermite_interpolant`, and the
    split points are computed from the vertices without it too. Derivatives with respect to the vertices therefore
    pass only through :meth:`SimplicialMesh.locate`.

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``. For the C1 construction, an output
        with data beyond about 2**1000 (or, without ``clamp``, controls beyond it) is processed in units of a power
        of two up to 2**24, so that its sums stay finite; its subnormal data are then interpolated only up to an
        absolute error below 2**-1050.
    split
        How each simplex is divided into pieces: :class:`NoSplit` or :class:`CloughTocher`.
    continuity
        How smoothly neighbouring pieces join: :class:`C0` or :class:`C1`.
    gradients
        Vertex gradients of shape ``(n_vertices, d, n_outputs)``. Estimated with :func:`estimate_gradients` if
        ``None``. With ``clamp`` and the C1 construction, a gradient whose controls exceed about 2**1000 (near
        float64 overflow) is divided by a power of two before it is limited.
    clamp
        Whether to keep the prediction on every simplex within the range of its vertex data.

    Returns
    -------
    MacroInterpolant
        The interpolant: call it at points, or evaluate its ``model`` at its :meth:`~MacroInterpolant.locate`.

    Raises
    ------
    TypeError
        If ``split`` or ``continuity`` is not one of the classes above.
    ValueError
        If the construction is not supported, the mesh is not two-dimensional for :class:`CloughTocher` or has a
        triangle whose split point rounds onto one of its sides, the data or supplied gradients are invalid, or
        the interpolant is not finite in float64 (see :func:`hermite_interpolant`).

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import C1, CloughTocher, lattice_mesh, macro_interpolant
    >>> mesh = lattice_mesh(2, 3)
    >>> x = mesh.vertices
    >>> interpolant = macro_interpolant(mesh, x[:, 0] * x[:, 1], split=CloughTocher(), continuity=C1())
    >>> interpolant.pieces.n_simplices == 3 * mesh.n_simplices
    True
    >>> y = interpolant(x)
    >>> torch.allclose(y[:, 0], x[:, 0] * x[:, 1])
    True
    """
    # Exact types, as the constructions are looked up by them.
    if type(split) not in _SPLITS or type(continuity) not in _CONTINUITIES:
        splits, continuities = (" or ".join(c.__name__ for c in options) for options in (_SPLITS, _CONTINUITIES))
        raise TypeError(f"split must be {splits}, and continuity {continuities}.")
    construction = _CONSTRUCTIONS.get((type(split), type(continuity)))
    if construction is None:
        raise ValueError(
            f"split={split!r} with continuity={continuity!r} is not supported; supported: {_supported_constructions()}."
        )
    f = mesh._check_values(values)
    pieces = _SPLITS[type(split)](mesh)
    model = construction(mesh, pieces, f, gradients, bool(clamp))
    return MacroInterpolant(mesh, pieces.mesh, pieces.parent, model, _transfer=pieces.transfer)


def _refinement_scores(
    cubic: torch.Tensor, values: torch.Tensor, volume_mantissa: torch.Tensor, volume_exponent: torch.Tensor
) -> torch.Tensor:
    """Volume-weighted mean correction, retaining tiny residuals and avoiding intermediate overflow.

    Ordinary rows keep the vectorized reductions. Near either float64 boundary, compute the score from
    the finite centroid predictions and vertex data as exact rationals, rounding only the final score.
    This also combines a correction outside float64 with a small volume before rounding either of them.
    """
    if not torch.isfinite(cubic).all():
        raise ValueError("the interpolant has a non-finite centroid prediction.")
    linear = values.mean(dim=1)
    correction = (cubic - linear).abs().mean(dim=1)
    mantissa, exponent = torch.frexp(correction)
    score = _times_power_of_two(mantissa * volume_mantissa, exponent.to(torch.long) + volume_exponent)
    score = torch.where(correction == 0, 0.0, score)
    entries = torch.cat([cubic, values.flatten(1)], dim=1).abs()
    info = torch.finfo(torch.float64)
    # A factor-two margin bounds both vertex/output sums; small terms must survive
    # division by their counts until a possibly huge volume has been applied.
    count = max(values.shape[1], values.shape[2])
    risky = (entries >= info.max / (2 * count)).any(dim=1)
    risky |= ((entries > 0) & (entries < info.tiny * count)).any(dim=1)
    risky |= ~torch.isfinite(score) | ((score == 0) & (correction > 0))
    for row in risky.nonzero().flatten().tolist():
        columns = zip(cubic[row].tolist(), values[row].T.tolist(), strict=True)
        exact = (
            sum(
                (
                    abs(Fraction(prediction) - sum(map(Fraction, column), Fraction(0)) / len(column))
                    for prediction, column in columns
                ),
                Fraction(0),
            )
            / cubic.shape[1]
        )
        volume = Fraction(volume_mantissa[row].item()) * Fraction(2) ** int(volume_exponent[row])
        score[row] = _rounded(exact * volume)
    return score


def refinement_indicator(
    mesh: SimplicialMesh,
    values: Any,
    *,
    gradients: Any = None,
    clamp: bool = True,
    split: NoSplit | CloughTocher = NoSplit(),
    continuity: C0 | C1 = C0(),
) -> torch.Tensor:
    """Score simplices for refinement by the cubic correction to linear interpolation.

    The score of a simplex is its volume times the mean over outputs of the absolute difference between
    the cubic interpolant (:func:`macro_interpolant`, by default :func:`hermite_interpolant`) and linear
    interpolation at its centroid. It is large where the data bend
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
    split, continuity
        The construction of the cubic interpolant, as in :func:`macro_interpolant`.

    Returns
    -------
    torch.Tensor
        Nonnegative float64 scores of shape ``(n_simplices,)``, one per simplex of ``mesh``.
        Finite centroid predictions and extreme vertex data use exact reductions near the float64
        limits, applying the volume before final rounding. A score beyond float64 is positive infinity.

    Raises
    ------
    TypeError
        If ``split`` or ``continuity`` is not one of the classes of :func:`macro_interpolant`.
    ValueError
        If :func:`macro_interpolant` cannot construct a finite interpolant, including when the
        vertex data or gradients cause its controls to overflow float64, or if a centroid
        prediction is non-finite.

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
    interpolant = macro_interpolant(mesh, f, split=split, continuity=continuity, gradients=gradients, clamp=clamp)
    n = mesh.dimension + 1
    centroid = Points(torch.full((mesh.n_simplices, n), 1 / n, dtype=torch.float64), mesh.cell_ids)
    with torch.no_grad():
        cubic: torch.Tensor = interpolant.model(interpolant.transfer(centroid))
    return _refinement_scores(cubic, f[mesh.simplices], mesh._volume_mantissa, mesh._volume_exponent)
