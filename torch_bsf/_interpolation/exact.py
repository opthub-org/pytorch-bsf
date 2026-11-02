"""Exact and power-of-two arithmetic of :mod:`torch_bsf.interpolation`.

Products and differences scaled by powers of two beyond the float64 range, the rational inverse of a simplex,
the exact barycentric expressions of :class:`~torch_bsf.interpolation.SimplicialMesh`, and the checks of its
simplices and of real tensor arguments.
"""

import math
from fractions import Fraction
from typing import Any

import torch

# PyTorch 2.3 imports Dynamo on the first custom-autograd transform. Its nested-tensor
# initialization uses random meta operations, which fail if that import occurs inside vmap.
# Initialize it before transforms; PyTorch 2.4 replaced the random module-level dummy.
if torch.__version__.startswith("2.3."):
    import torch._dynamo


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
