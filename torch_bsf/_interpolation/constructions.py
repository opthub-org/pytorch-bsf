"""Splits, continuities and the Clough–Tocher C1 construction of :func:`~torch_bsf.interpolation.macro_interpolant`."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from torch_bsf._interpolation.exact import _times_power_of_two
from torch_bsf._interpolation.hermite import _check_gradients, hermite_interpolant
from torch_bsf._interpolation.mesh import _DEGREE, SimplicialMesh
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierModel, Points


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
