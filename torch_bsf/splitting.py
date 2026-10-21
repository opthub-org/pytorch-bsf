r"""Splitting Bézier simplices and Bézier simploids via the de Casteljau algorithm.

This module splits a Bézier simplex along a single edge by inserting a new
vertex, and computes the control points of the two resulting sub-simplices
with the de Casteljau algorithm.  A Bézier simploid, a tensor product of
Bézier simplices, is split along an edge of one of its factors in the same
way; the other factors are left as they are.

The split can be chosen explicitly or determined automatically by a
:data:`SplitCriterion`.

Examples
--------
Split a Bézier curve at its midpoint:

>>> import torch
>>> from torch_bsf import BezierSimploid
>>> from torch_bsf.splitting import split
>>> bs = BezierSimploid.rand((1,), 3, n_values=3)
>>> bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
>>> bs_A.simploid_type == bs.simploid_type and bs_A.degree == bs.degree
True

Use the longest-edge criterion to choose the split automatically:

>>> from torch_bsf.splitting import longest_edge_criterion, split_by_criterion
>>> bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
"""

from collections.abc import Callable, Sequence
from functools import cache
from itertools import product
from typing import Literal

import torch
import torch.nn.functional as F

from torch_bsf.simploidal.bezier import BezierSimploid
from torch_bsf.simploidal.combinatorics import SimploidType, _check_type, _indices, _positions

#: Type of a split criterion: a callable that accepts a :class:`~torch_bsf.simploidal.BezierSimploid` and returns
#: ``(factor, i, j, s)``: the factor to split, the vertices ``i`` and ``j`` of its edge and the split parameter.
#: For a Bézier simplex, ``factor`` is ``0``.
SplitCriterion = Callable[[BezierSimploid], tuple[int, int, int, float]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_edge(simploid_type: SimploidType, factor: int, i: int, j: int) -> None:
    if not 0 <= factor < len(simploid_type):
        raise ValueError(f"factor must satisfy 0 <= factor < {len(simploid_type)}, but factor={factor}.")
    n_vertices = simploid_type[factor] + 1
    if not (0 <= i < n_vertices and 0 <= j < n_vertices and i != j):
        raise ValueError(
            f"Edge indices must satisfy 0 <= i, j < {n_vertices} and i != j for factor {factor}, but i={i}, j={j}."
        )


def _check_s(s: float) -> None:
    if not (0.0 < s < 1.0):
        raise ValueError(f"Split parameter s must be in (0, 1), but s={s}.")


@cache
def _edge_tables(
    simploid_type: SimploidType, degree: tuple[int, ...], factor: int, i: int, j: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """The components ``alpha_i`` and ``alpha_j`` of factor ``factor`` of every control point index,
    and the rows of ``alpha + e_i - e_j`` and ``alpha + e_j - e_i`` in that factor (``-1`` if not an index).
    """
    pos = _positions(simploid_type, degree)
    alpha_i: list[int] = []
    alpha_j: list[int] = []
    shift_ij: list[int] = []
    shift_ji: list[int] = []
    for alpha in _indices(simploid_type, degree):
        m = alpha[factor]
        alpha_i.append(m[i])
        alpha_j.append(m[j])
        for src, dst, shifts in ((j, i, shift_ij), (i, j, shift_ji)):
            if m[src] == 0:
                shifts.append(-1)
                continue
            n = list(m)
            n[src] -= 1
            n[dst] += 1
            shifts.append(pos[alpha[:factor] + (tuple(n),) + alpha[factor + 1 :]])
    return tuple(alpha_i), tuple(alpha_j), tuple(shift_ij), tuple(shift_ji)


def _de_casteljau(b: torch.Tensor, keep: torch.Tensor, shift: torch.Tensor, n: int, w_keep: float) -> torch.Tensor:
    r"""One sub-model of a split: :math:`n` de Casteljau steps along an edge.

    Each step replaces the rows with ``shift >= 0`` by ``w_keep`` times themselves plus ``1 - w_keep`` times
    their shifted rows, and the rows with ``keep == r`` are taken after step :math:`r`.
    """
    result = torch.empty_like(b)
    update = shift >= 0
    c = b
    result[keep == 0] = c[keep == 0]
    for r in range(1, n + 1):
        c = c.clone()
        c[update] = w_keep * c[update] + (1.0 - w_keep) * c[shift[update]]
        result[keep == r] = c[keep == r]
    return result


def _split_core(model: BezierSimploid, factor: int, i: int, j: int, s: float) -> tuple[BezierSimploid, BezierSimploid]:
    b = model.control_points.detach()
    tables = _edge_tables(model.simploid_type, model.degree, factor, i, j)
    alpha_i, alpha_j, shift_ij, shift_ji = (torch.tensor(x, dtype=torch.long, device=b.device) for x in tables)
    n = model.degree[factor]
    with torch.no_grad():
        # A: vertex j -> new vertex; c^r[alpha] = s c^{r-1}[alpha] + (1-s) c^{r-1}[alpha + e_i - e_j].
        b_a = _de_casteljau(b, alpha_j, shift_ij, n, s)
        # B: vertex i -> new vertex; c^r[alpha] = (1-s) c^{r-1}[alpha] + s c^{r-1}[alpha + e_j - e_i].
        b_b = _de_casteljau(b, alpha_i, shift_ji, n, 1.0 - s)
    return (
        BezierSimploid(model.simploid_type, model.degree, b_a),
        BezierSimploid(model.simploid_type, model.degree, b_b),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split(model: BezierSimploid, i: int, j: int, s: float = 0.5, *, factor: int = 0) -> tuple[BezierSimploid, BezierSimploid]:
    r"""Split a Bézier simplex or simploid along edge ``(i, j)`` using the de Casteljau algorithm.

    A new vertex is inserted on the edge between vertex ``i`` and vertex ``j``
    of the parameter domain at the relative position ``s``.  The original
    simplex is thereby subdivided into two sub-simplices that together cover
    the entire original domain.  For a Bézier simploid, the edge is one of the
    simplex of factor ``factor``, and the other factors are left as they are.

    Parameters
    ----------
    model
        The Bézier simplex or simploid to split.
    i, j
        Indices of the two vertices that define the split edge
        (0-indexed within the factor, ``i ≠ j``).
    s
        Split parameter in the open interval ``(0, 1)``.  ``s = 0.5``
        produces a midpoint split.  The new vertex is located at
        :math:`(1-s)\,v_i + s\,v_j` in the parameter domain.
    factor
        The factor to split; ``0`` for a Bézier simplex.

    Returns
    -------
    (bs_A, bs_B) : tuple[BezierSimploid, BezierSimploid]
        Two models of the same type and degree as ``model``:

        * **bs_A** — replaces vertex :math:`j` with the new vertex.  It covers
          the sub-domain :math:`\{t : t_j / (t_i + t_j) \le s\}`.
        * **bs_B** — replaces vertex :math:`i` with the new vertex.  It covers
          the sub-domain :math:`\{t : t_j / (t_i + t_j) \ge s\}`.

        Here :math:`t_i` and :math:`t_j` are the coordinates of the factor;
        :func:`reparametrize` maps points of ``model`` to those of the two pieces.

    Raises
    ------
    ValueError
        If ``factor``, ``i``, ``j`` or ``s`` is out of range.

    Notes
    -----
    The algorithm runs :math:`n` de Casteljau steps along the chosen edge
    direction, where :math:`n` is the degree of the factor.

    At each step :math:`r = 1, \ldots, n` the control-point matrix is updated
    as

    .. math::

        c^{(r)}_\alpha =
        \begin{cases}
            s \cdot c^{(r-1)}_\alpha
                + (1-s) \cdot c^{(r-1)}_{\alpha + e_i - e_j}
            & \text{if } \alpha_j \ge 1 \\
            c^{(r-1)}_\alpha & \text{otherwise,}
        \end{cases}

    where :math:`\alpha` is the multi-index of the factor, and rows with
    :math:`\alpha_j = r` are saved as control points of **bs_A**.  An
    analogous recursion with the roles of :math:`i` and :math:`j` swapped gives
    **bs_B**.  The two pieces agree on their common face.

    Examples
    --------
    Split the identity Bézier curve at the midpoint:

    >>> from torch_bsf import BezierSimploid
    >>> from torch_bsf.splitting import split
    >>> bs = BezierSimploid((1,), 1, {(1, 0): [0.0], (0, 1): [1.0]})
    >>> bs_A, bs_B = split(bs, i=0, j=1, s=0.5)
    >>> bs_A[(0, 1)].item()  # split-point value
    0.5
    >>> bs_B[(1, 0)].item()  # same split point from other side
    0.5

    Split a bilinear patch along its second factor:

    >>> import torch
    >>> patch = BezierSimploid.rand((1, 1), 1, n_values=1)
    >>> bs_A, bs_B = split(patch, 0, 1, 0.5, factor=1)
    >>> torch.allclose(bs_A(torch.tensor([[0.3, 0.7, 0.0, 1.0]])), patch(torch.tensor([[0.3, 0.7, 0.5, 0.5]])))
    True
    """
    _check_edge(model.simploid_type, factor, i, j)
    _check_s(s)
    return _split_core(model, factor, i, j, s)


def reparametrize(
    t: torch.Tensor,
    i: int,
    j: int,
    s: float,
    part: Literal["A", "B"],
    *,
    simploid_type: Sequence[int] | None = None,
    factor: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Re-parameterise points of a split model to one of its two pieces.

    After splitting edge ``(i, j)`` at ``s``, each point of the original
    domain belongs to one of the two pieces.  This function converts the
    original parameters to the local parameters of a piece.

    Parameters
    ----------
    t
        Parameters of shape ``(N, n_params)`` of the original model.
    i, j
        Edge vertex indices used in :func:`split`.
    s
        Split parameter.
    part
        ``"A"`` for the piece covering :math:`\{t : t_j / (t_i + t_j) \le s\}`,
        or ``"B"`` for the complementary region.
    simploid_type
        The type of the original model; ``None`` for a simplex (whose
        barycentric coordinates are all of ``t``).
    factor
        The factor that was split, as in :func:`split`.

    Returns
    -------
    (u, mask) : tuple[torch.Tensor, torch.Tensor]
        * **u** — local parameters of the piece, of the same shape as ``t``.
        * **mask** — boolean tensor of shape ``(N,)`` indicating which input
          points belong to the requested piece.  Points where
          :math:`t_i = t_j = 0` belong to both pieces and are included
          in both masks.

    Notes
    -----
    Writing :math:`t_k` for the coordinates of the factor, piece A transforms them as

    .. math::

        u_j = \frac{t_j}{s}, \quad
        u_i = t_i - \frac{1-s}{s}\,t_j, \quad
        u_k = t_k \; (k \ne i, j),

    and piece B as

    .. math::

        u_i = \frac{t_i}{1-s}, \quad
        u_j = t_j - \frac{s}{1-s}\,t_i, \quad
        u_k = t_k \; (k \ne i, j).

    The coordinates of the other factors are unchanged.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.splitting import reparametrize
    >>> t = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    >>> u, mask = reparametrize(t, 0, 1, 0.5, "A")
    >>> mask.tolist(), [round(x, 6) for x in u[0].tolist()]
    ([True, False], [0.6, 0.4])
    """
    if t.ndim != 2:
        raise ValueError(f"`t` must be a 2-D tensor of shape (N, n_params), but got shape {tuple(t.shape)}.")
    if simploid_type is None:
        if t.shape[1] < 2:
            raise ValueError(f"`t` must have at least 2 columns (n_params >= 2), but got shape {tuple(t.shape)}.")
        st: SimploidType = (t.shape[1] - 1,)
    else:
        st = _check_type(simploid_type)
        if t.shape[1] != sum(a + 1 for a in st):
            raise ValueError(f"`t` must have {sum(a + 1 for a in st)} columns for type {st}, but got {tuple(t.shape)}.")
    _check_edge(st, factor, i, j)
    if not t.is_floating_point():
        raise ValueError(
            f"`t` must be a floating-point tensor, but got dtype {t.dtype}. "
            f"Convert `t` to a float dtype (e.g. t.float()) before calling reparametrize."
        )
    _check_s(s)
    offset = sum(a + 1 for a in st[:factor])
    ci, cj = offset + i, offset + j

    ti = t[:, ci]
    tj = t[:, cj]
    denom = ti + tj
    safe = denom != 0  # points where t_i + t_j > 0

    u = t.clone()
    if part == "A":
        # belongs to A when t_j / (t_i + t_j) <= s  (or t_i = t_j = 0)
        mask = (~safe) | (tj <= s * denom)
        u[safe, cj] = tj[safe] / s
        u[safe, ci] = ti[safe] - (1.0 - s) * tj[safe] / s
    elif part == "B":
        # belongs to B when t_j / (t_i + t_j) >= s  (or t_i = t_j = 0)
        mask = (~safe) | (tj >= s * denom)
        u[safe, ci] = ti[safe] / (1.0 - s)
        u[safe, cj] = tj[safe] - s * ti[safe] / (1.0 - s)
    else:
        raise ValueError("part must be exactly 'A' or 'B'")

    return u, mask


def _edges(simploid_type: SimploidType) -> list[tuple[int, int, int]]:
    """All edges ``(factor, i, j)``, ``i < j``, in lexicographic order."""
    return [(k, i, j) for k, a in enumerate(simploid_type) for i in range(a + 1) for j in range(i + 1, a + 1)]


def _check_splittable(model: BezierSimploid) -> None:
    if not model.simploid_type:
        raise ValueError("Splitting requires a simploid with at least one factor, but the type is ().")


def longest_edge_criterion(model: BezierSimploid, s: float = 0.5) -> tuple[int, int, int, float]:
    r"""Select the edge with the greatest value-space length and split it at ``s``.

    The "length" of edge :math:`(i, j)` of a Bézier simplex is the Euclidean
    distance between its values at the two vertices of the parameter domain:

    .. math::

        \ell_{ij} = \|B(e_i) - B(e_j)\|_2
                  = \|b_{n \cdot e_i} - b_{n \cdot e_j}\|_2

    where :math:`n` is the degree and :math:`e_k` is the :math:`k`-th unit
    vector.  An edge of a factor of a Bézier simploid is parallel to an edge for
    every vertex of the other factors, and its length is the greatest length of them.
    Ties go to the first edge in the order of factors, then of ``(i, j)``.

    Parameters
    ----------
    model
        The Bézier simplex or simploid.
    s
        Split parameter included verbatim in the returned tuple
        (defaults to ``0.5``).

    Returns
    -------
    (factor, i, j, s)
        The edge with the greatest value-space length, together with ``s``.

    Examples
    --------
    >>> from torch_bsf import BezierSimploid
    >>> from torch_bsf.splitting import longest_edge_criterion
    >>> bs = BezierSimploid((2,), 2, {
    ...     (2, 0, 0): [0.0], (1, 1, 0): [0.0], (1, 0, 1): [0.0],
    ...     (0, 2, 0): [1.0], (0, 1, 1): [0.0], (0, 0, 2): [3.0],
    ... })
    >>> longest_edge_criterion(bs)
    (0, 0, 2, 0.5)
    """
    _check_splittable(model)
    st, ds = model.simploid_type, model.degree
    cp = model.control_points.detach()
    pos = _positions(st, ds)

    def corner(vertices: Sequence[int]) -> torch.Tensor:
        index = tuple(tuple(d if m == v else 0 for m in range(a + 1)) for a, d, v in zip(st, ds, vertices, strict=True))
        return cp[pos[index]]

    best, best_length = (0, 0, 1), -1.0
    for k, i, j in _edges(st):
        others = [range(a + 1) for a in st[:k] + st[k + 1 :]]
        length = max(
            float(torch.linalg.vector_norm(corner(v[:k] + (i,) + v[k:]) - corner(v[:k] + (j,) + v[k:])))
            for v in product(*others)
        )
        if length > best_length:
            best, best_length = (k, i, j), length
    return (*best, s)


def max_error_criterion(params: torch.Tensor, values: torch.Tensor, grid_size: int = 10) -> SplitCriterion:
    r"""Build a criterion that minimises the combined approximation error.

    For each candidate edge ``(i, j)`` of each factor and split parameter
    ``s`` drawn from a uniform grid over ``(0, 1)``, the combined
    mean-squared error

    .. math::

        E(i, j, s) = \mathrm{MSE}_A + \mathrm{MSE}_B

    is computed, where :math:`\mathrm{MSE}_A` and :math:`\mathrm{MSE}_B`
    are evaluated on the portions of the data that fall within each
    piece.  The candidate minimising ``E`` is returned.

    Parameters
    ----------
    params
        Parameter vectors of shape ``(N, n_params)``.
    values
        Target value vectors of shape ``(N, n_values)``.
    grid_size
        Number of candidate ``s`` values in the grid search (default ``10``).
        Larger values give a finer search at higher cost.

    Returns
    -------
    SplitCriterion
        A callable ``criterion(model) -> (factor, i, j, s)``.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import BezierSimploid
    >>> from torch_bsf.splitting import max_error_criterion
    >>> params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> values = torch.tensor([[0.0], [0.5], [1.0]])
    >>> bs = BezierSimploid.rand((1,), 1, n_values=1)
    >>> criterion = max_error_criterion(params, values)
    >>> criterion(bs)[:3]
    (0, 0, 1)
    """
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    params_t = torch.as_tensor(params)
    values_t = torch.as_tensor(values)

    def criterion(model: BezierSimploid) -> tuple[int, int, int, float]:
        _check_splittable(model)
        _check_split_data(params_t, values_t, model)
        # Move data to the same device and dtype as the model so that forward
        # passes and mse_loss calls don't raise device/dtype mismatch errors.
        cp = model.control_points
        p = params_t.to(device=cp.device, dtype=cp.dtype)
        v = values_t.to(device=cp.device, dtype=cp.dtype)

        best, best_error = (0, 0, 1, 0.5), float("inf")
        # linspace(0, 1, grid_size+2)[1:-1] gives `grid_size` evenly-spaced
        # values strictly inside (0, 1), with the midpoint 0.5 included when
        # grid_size is odd (e.g. grid_size=1 → [0.5], grid_size=3 → [0.25, 0.5, 0.75]).
        s_candidates = torch.linspace(0.0, 1.0, grid_size + 2)[1:-1].tolist()
        for k, i, j in _edges(model.simploid_type):
            for s in s_candidates:
                error = _split_error(model, p, v, k, i, j, s)
                if error < best_error:
                    best, best_error = (k, i, j, s), error
        return best

    return criterion


def _check_split_data(params: torch.Tensor, values: torch.Tensor, model: BezierSimploid) -> None:
    """Check that ``params`` and ``values`` are 2D data of the same size matching ``model``."""
    if params.ndim != 2:
        raise ValueError(f"`params` must be a 2D tensor of shape (N, {model.n_params}), but got shape {tuple(params.shape)}.")
    if values.ndim != 2:
        raise ValueError(f"`values` must be a 2D tensor of shape (N, {model.n_values}), but got shape {tuple(values.shape)}.")
    if params.shape[0] != values.shape[0]:
        raise ValueError(
            f"`params` and `values` must have the same number of samples N, but got {params.shape[0]} and {values.shape[0]}."
        )
    if params.shape[1] != model.n_params:
        raise ValueError(
            f"`params` must have shape (N, {model.n_params}) for the given model, but got shape {tuple(params.shape)}."
        )
    if values.shape[1] != model.n_values:
        raise ValueError(
            f"`values` must have shape (N, {model.n_values}) for the given model, but got shape {tuple(values.shape)}."
        )


def _split_error(model: BezierSimploid, params: torch.Tensor, values: torch.Tensor, k: int, i: int, j: int, s: float) -> float:
    """The sum of the mean-squared errors of the two pieces of ``split(model, i, j, s, factor=k)`` on their data.

    A piece that contains none of the data contributes nothing.
    """
    bs_a, bs_b = _split_core(model, k, i, j, s)
    error = 0.0
    with torch.no_grad():
        pieces: tuple[tuple[BezierSimploid, Literal["A", "B"]], ...] = ((bs_a, "A"), (bs_b, "B"))
        for piece, part in pieces:
            u, mask = reparametrize(params, i, j, s, part, simploid_type=model.simploid_type, factor=k)
            if mask.any():
                error += float(F.mse_loss(piece(u[mask]), values[mask]).item())
    return error


def split_by_criterion(model: BezierSimploid, criterion: SplitCriterion) -> tuple[BezierSimploid, BezierSimploid]:
    """Split a Bézier simplex or simploid using a split criterion.

    Convenience wrapper that calls ``criterion(model)`` to obtain the factor,
    edge and split parameter, then delegates to :func:`split`.

    Parameters
    ----------
    model
        The Bézier simplex or simploid to split.
    criterion
        A :data:`SplitCriterion` callable that returns ``(factor, i, j, s)``.

    Returns
    -------
    (bs_A, bs_B) : tuple[BezierSimploid, BezierSimploid]
        The two pieces; see :func:`split` for details.

    Examples
    --------
    >>> from torch_bsf import BezierSimploid
    >>> from torch_bsf.splitting import longest_edge_criterion, split_by_criterion
    >>> bs = BezierSimploid.rand((1,), 2, n_values=3)
    >>> bs_A, bs_B = split_by_criterion(bs, longest_edge_criterion)
    >>> bs_A.degree == bs.degree
    True
    """
    factor, i, j, s = criterion(model)
    return split(model, i, j, s, factor=factor)
