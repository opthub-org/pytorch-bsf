r"""Simploid types, control point indices, Bernstein bases and sampling."""

from functools import lru_cache
from itertools import product as iproduct
from math import factorial
from numbers import Integral
from typing import Sequence, TypeAlias, cast

import torch

#: The type :math:`(a_1, \ldots, a_n)` of a simploid; ``()`` is a vertex.
SimploidType: TypeAlias = tuple[int, ...]

#: The index of a control point of a Bézier simploid: a tuple of multi-indices, one per factor.
SimploidIndex: TypeAlias = tuple[tuple[int, ...], ...]

#: A degeneracy map: for each factor of a degenerate simploid, the factor of its base it comes
#: from (``-1`` for an inserted edge factor) and the surjection on the vertices of that factor.
DegeneracyMap: TypeAlias = tuple[tuple[int, tuple[int, ...]], ...]


def _simplex_indices(n_vertices: int, degree: int) -> list[tuple[int, ...]]:
    """The multi-indices of length ``n_vertices`` summing to ``degree``, in reverse lexicographic order."""
    if n_vertices == 1:
        return [(degree,)]
    return [(i,) + rest for i in range(degree, -1, -1) for rest in _simplex_indices(n_vertices - 1, degree - i)]


def _multinomial(degree: int, index: Sequence[int]) -> float:
    r"""The multinomial coefficient :math:`\binom{D}{\boldsymbol\alpha} = D! / (\alpha_1! \cdots \alpha_M!)`."""
    r: float = factorial(degree)
    for i in index:
        r /= factorial(i)
    return r


def _check_type(simploid_type: Sequence[int]) -> SimploidType:
    t = tuple(int(a) for a in simploid_type)
    if any(a < 1 for a in t):
        raise ValueError(f"Every component of a simploid type must be positive, but {t} is given.")
    return t


def _check_degree(degree: int | Sequence[int], length: int) -> tuple[int, ...]:
    if isinstance(degree, int) and degree < 0:
        raise ValueError(f"degree must be non-negative, but {degree} is given.")
    ds = (int(degree),) * length if isinstance(degree, int) else tuple(int(d) for d in degree)
    if len(ds) != length:
        raise ValueError(f"degree must have {length} components, but {ds} is given.")
    if any(d < 0 for d in ds):
        raise ValueError(f"degree must be non-negative, but {ds} is given.")
    return ds


@lru_cache(maxsize=None)
def _indices(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[SimploidIndex, ...]:
    return tuple(iproduct(*(tuple(_simplex_indices(a + 1, d)) for a, d in zip(simploid_type, degree))))


@lru_cache(maxsize=None)
def _positions(simploid_type: SimploidType, degree: tuple[int, ...]) -> dict[SimploidIndex, int]:
    return {alpha: pos for pos, alpha in enumerate(_indices(simploid_type, degree))}


def _canonical_index(simploid_type: SimploidType, index: object) -> SimploidIndex:
    """A control point index as a tuple of tuples of ints.

    A Bézier simplex (a single factor) also accepts its multi-index alone, e.g. ``(2, 1, 0)`` for ``((2, 1, 0),)``.
    Raises :class:`TypeError` or :class:`ValueError` if ``index`` is not a sequence of multi-indices.
    """
    items = tuple(cast(Sequence, index))
    if len(simploid_type) == 1 and items and all(isinstance(v, Integral) for v in items):
        items = (items,)
    return tuple(tuple(int(v) for v in m) for m in items)


def simploid_indices(simploid_type: Sequence[int], degree: int | Sequence[int]) -> list[SimploidIndex]:
    r"""Lists the control point indices of a Bézier simploid in canonical order.

    The order is the lexicographic order of the tuples of multi-indices, the
    first factor varying slowest (Algorithm 4 of :cite:t:`peltier2018simploidal`);
    the multi-indices of each factor are in reverse lexicographic order, e.g. ``(2, 0), (1, 1), (0, 2)``.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.

    Returns
    -------
    The indices.

    Examples
    --------
    >>> from torch_bsf.simploidal import simploid_indices
    >>> simploid_indices((1, 1), 1)
    [((1, 0), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, 1), (0, 1))]
    >>> simploid_indices((), 3)
    [()]
    """
    t = _check_type(simploid_type)
    return list(_indices(t, _check_degree(degree, len(t))))


def simploid_n_params(simploid_type: Sequence[int]) -> int:
    r"""The length :math:`\sum_i (a_i + 1)` of the parameter vector of a simploid.

    Examples
    --------
    >>> from torch_bsf.simploidal import simploid_n_params
    >>> simploid_n_params((2, 1))
    5
    """
    return sum(a + 1 for a in _check_type(simploid_type))


def bernstein_basis(simploid_type: Sequence[int], degree: int | Sequence[int], t: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the tensor-product Bernstein basis of a Bézier simploid.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.
    t
        Parameters of shape ``(batch, sum(a_i + 1))``: the concatenated
        barycentric coordinates of every factor.

    Returns
    -------
    The basis of shape ``(batch, n_indices)``, columns in the order of :func:`simploid_indices`.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.simploidal import bernstein_basis
    >>> bernstein_basis((1, 1), 1, torch.tensor([[0.5, 0.5, 1.0, 0.0]]))
    tensor([[0.5000, 0.0000, 0.5000, 0.0000]])
    """
    st = _check_type(simploid_type)
    ds = _check_degree(degree, len(st))
    if t.dim() != 2 or t.shape[1] != sum(a + 1 for a in st):
        raise ValueError(f"t must have shape (batch, {sum(a + 1 for a in st)}), but {tuple(t.shape)} is given.")
    dtype = t.dtype if t.is_floating_point() else torch.get_default_dtype()
    t = t.to(dtype)
    return _tensor_product_basis(t, [_factor_tensors(a, d, dtype, t.device) for a, d in zip(st, ds)])


@lru_cache(maxsize=None)
def _factor_tables(a: int, d: int) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    """The multi-indices of a factor of dimension ``a`` and degree ``d`` and their multinomial coefficients."""
    idx = tuple(_simplex_indices(a + 1, d))
    return idx, tuple(_multinomial(d, i) for i in idx)


def _factor_tensors(a: int, d: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The powers ``(n_k, a + 1)`` and coefficients ``(n_k,)`` of the Bernstein basis of a factor, as tensors."""
    idx, coeffs = _factor_tables(a, d)
    return (
        torch.tensor(idx, dtype=torch.long, device=device).reshape(len(idx), a + 1),
        torch.tensor(coeffs, dtype=dtype, device=device),
    )


def _tensor_product_basis(t: torch.Tensor, factors: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """The tensor-product Bernstein basis of ``t`` given the powers and coefficients of every factor."""
    basis = torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)
    offset = 0
    for powers, coeffs in factors:
        width = powers.shape[1]
        block = t[:, offset : offset + width]
        factor = coeffs.to(t.dtype) * torch.pow(block.unsqueeze(1), powers.unsqueeze(0)).prod(dim=-1)
        basis = (basis.unsqueeze(2) * factor.unsqueeze(1)).reshape(t.shape[0], -1)
        offset += width
    return basis


@lru_cache(maxsize=None)
def _adjacent_pairs(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """The pairs ``(p, q)``, ``p < q``, of positions of adjacent control points in :func:`simploid_indices` order.

    Two control points are adjacent if their multi-indices differ in a single factor, by moving one unit
    from one component to another (an :math:`L^1` distance of 2 in that factor).
    """
    pos = _positions(simploid_type, degree)
    pairs = []
    for alpha, p in pos.items():
        for k, m in enumerate(alpha):
            for i in range(len(m)):
                if m[i] == 0:
                    continue
                for j in range(len(m)):
                    if j == i:
                        continue
                    n = list(m)
                    n[i] -= 1
                    n[j] += 1
                    q = pos[alpha[:k] + (tuple(n),) + alpha[k + 1 :]]
                    if p < q:
                        pairs.append((p, q))
    return tuple(sorted(pairs))


def sample_simploid(simploid_type: Sequence[int], n_samples: int, seed: int | None = None) -> torch.Tensor:
    r"""Samples points uniformly on a simploid.

    Each factor is sampled from the flat Dirichlet distribution.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    n_samples
        The number of samples.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Parameters of shape ``(n_samples, sum(a_i + 1))``.

    Examples
    --------
    >>> from torch_bsf.simploidal import sample_simploid
    >>> t = sample_simploid((2, 1), 4, seed=0)
    >>> t.shape
    torch.Size([4, 5])
    >>> bool(((t[:, :3].sum(dim=1) - 1).abs() < 1e-6).all())
    True
    """
    st = _check_type(simploid_type)
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, but {n_samples} is given.")
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    blocks = [torch.empty(n_samples, 0)]
    for a in st:
        e = -torch.log(torch.rand(n_samples, a + 1, generator=generator).clamp_min(1e-12))
        blocks.append(e / e.sum(dim=1, keepdim=True))
    return torch.cat(blocks, dim=1).to(torch.get_default_dtype())


def _face_index(alpha: SimploidIndex, i: int, j: int) -> SimploidIndex:
    """The index in the face ``d^i_j`` of a control point index whose ``alpha[i][j]`` is zero."""
    a = alpha[i]
    if len(a) == 2:
        return alpha[:i] + alpha[i + 1 :]
    return alpha[:i] + (a[:j] + a[j + 1 :],) + alpha[i + 1 :]


def _pullback_index(alpha: SimploidIndex, dmap: DegeneracyMap, base_length: int) -> SimploidIndex:
    """The control point index of the base simploid that equals ``alpha`` of a degenerate simploid (Property 2)."""
    beta: list[tuple[int, ...]] = [()] * base_length
    for a, (src, g) in zip(alpha, dmap):
        if src < 0:
            continue
        b = [0] * (max(g) + 1)
        for x, v in enumerate(g):
            b[v] += a[x]
        beta[src] = tuple(b)
    return tuple(beta)
