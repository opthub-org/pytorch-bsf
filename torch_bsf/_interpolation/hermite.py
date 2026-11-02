"""Gradient estimation and the cubic Hermite interpolant of :mod:`torch_bsf.interpolation`."""

import math
from fractions import Fraction
from typing import Any

import torch

from torch_bsf._interpolation.exact import _as_real_tensor, _split_difference, _times_power_of_two
from torch_bsf._interpolation.mesh import _DEGREE, SimplicialMesh
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierModel


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
