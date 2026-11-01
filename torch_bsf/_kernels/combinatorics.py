r"""Simploid types, control point indices, Bernstein bases and sampling."""

from collections.abc import Iterator, Sequence
from functools import cache, lru_cache
from itertools import product as iproduct
from math import comb, factorial, prod
from numbers import Integral
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

#: The type :math:`(a_1, \ldots, a_n)` of a simploid; ``()`` is a vertex.
SimploidType: TypeAlias = tuple[int, ...]

_EdgeKeys: TypeAlias = npt.NDArray[np.int64] | npt.NDArray[np.uint32]

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
    try:
        return factorial(degree) / prod(factorial(i) for i in index)
    except OverflowError as exc:
        raise ValueError("degree produces multinomial coefficients outside float64 range.") from exc


def _structural_integer(value: object, name: str, minimum: int = 0) -> int:
    """Accept Python/NumPy integral values without coercing floats, booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}, but {value!r} is given.")
    return int(value)


def _check_type(simploid_type: Sequence[int]) -> SimploidType:
    t = tuple(_structural_integer(a, f"dimension[{i}]") for i, a in enumerate(simploid_type))
    if any(a < 1 for a in t):
        raise ValueError(f"Every component of a simploid type must be positive, but {t} is given.")
    return t


def _check_degree(degree: int | Sequence[int], length: int) -> tuple[int, ...]:
    if isinstance(degree, Integral) or not hasattr(degree, "__iter__") or isinstance(degree, (str, bytes)):
        ds = (_structural_integer(degree, "degree"),) * length
    else:
        ds = tuple(_structural_integer(d, f"degree[{i}]") for i, d in enumerate(cast(Sequence[int], degree)))
    if len(ds) != length:
        raise ValueError(f"degree must have {length} components, but {ds} is given.")
    if any(d < 0 for d in ds):
        raise ValueError(f"degree must be non-negative, but {ds} is given.")
    return ds


@cache
def _indices(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[SimploidIndex, ...]:
    return tuple(iproduct(*(tuple(_simplex_indices(a + 1, d)) for a, d in zip(simploid_type, degree, strict=True))))


@cache
def _positions(simploid_type: SimploidType, degree: tuple[int, ...]) -> dict[SimploidIndex, int]:
    return {alpha: pos for pos, alpha in enumerate(_indices(simploid_type, degree))}


def _canonical_index(simploid_type: SimploidType, index: object) -> SimploidIndex:
    """A control point index as a tuple of tuples of ints.

    A Bézier simplex (a single factor) also accepts its multi-index alone, e.g. ``(2, 1, 0)`` for ``((2, 1, 0),)``.
    Raises :class:`TypeError` or :class:`ValueError` if ``index`` is not a sequence of multi-indices.
    """
    if isinstance(index, (str, bytes)):
        raise ValueError("control index must be a sequence of integer multi-indices.")
    items = tuple(cast(Sequence[Any], index))
    if len(simploid_type) == 1 and items and all(not hasattr(v, "__iter__") or isinstance(v, (str, bytes)) for v in items):
        items = (items,)
    return tuple(
        tuple(_structural_integer(v, f"control index[{i}][{j}]") for j, v in enumerate(m)) for i, m in enumerate(items)
    )


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
    >>> from torch_bsf._kernels import simploid_indices
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
    >>> from torch_bsf._kernels import simploid_n_params
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
    >>> from torch_bsf._kernels import bernstein_basis
    >>> bernstein_basis((1, 1), 1, torch.tensor([[0.5, 0.5, 1.0, 0.0]]))
    tensor([[0.5000, 0.0000, 0.5000, 0.0000]])
    """
    st = _check_type(simploid_type)
    ds = _check_degree(degree, len(st))
    if t.dim() != 2 or t.shape[1] != sum(a + 1 for a in st):
        raise ValueError(f"t must have shape (batch, {sum(a + 1 for a in st)}), but {tuple(t.shape)} is given.")
    dtype = t.dtype if t.is_floating_point() else torch.get_default_dtype()
    t = t.to(dtype)
    return _tensor_product_basis(t, [_factor_tensors(a, d, dtype, t.device) for a, d in zip(st, ds, strict=True)], ds)


@cache
def _factor_tables(a: int, d: int) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    """The multi-indices of a factor of dimension ``a`` and degree ``d`` and their multinomial coefficients."""
    idx = tuple(_simplex_indices(a + 1, d))
    return idx, tuple(_multinomial(d, i) for i in idx)


def _factor_tensors(a: int, d: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Polynomial factor tensors, or empty tensors encoding recurrence width and degree in their shapes."""
    # Empty coefficient metadata encodes recurrence degree without a device
    # scalar read in evaluation (including after module .to(dtype/device)).
    coeffs = () if d >= 16 else _factor_tables(a, d)[1]
    recurrent = d >= 16 or max(coeffs, default=0) > torch.finfo(torch.float16).max
    if recurrent:
        # Only the width and degree are needed; no power table is read.
        return (
            torch.empty((0, a + 1), dtype=torch.long, device=device),
            torch.empty((d, 0), dtype=dtype, device=device),
        )
    idx = _factor_tables(a, d)[0]
    return (
        torch.tensor(idx, dtype=torch.long, device=device).reshape(len(idx), a + 1),
        torch.tensor(coeffs, dtype=dtype, device=device),
    )


def _tensor_product_basis(
    t: torch.Tensor, factors: Sequence[tuple[torch.Tensor, torch.Tensor]], degrees: Sequence[int]
) -> torch.Tensor:
    """The tensor-product Bernstein basis of ``t`` given the powers and coefficients of every factor."""
    basis = torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)
    offset = 0
    for (powers, coeffs), degree in zip(factors, degrees, strict=True):
        width = powers.shape[1]
        block = t[:, offset : offset + width]
        if coeffs.numel() == 0:
            factor = _recursive_basis(block, coeffs.shape[0])
        else:
            # Build integer powers with multiplication: tensor-exponent pow's
            # higher backward evaluates 0 * 0**negative at zero coordinates.
            # The small coordinate table avoids elevating the entire basis.
            # Scalar zero powers retain a zero coordinate gradient even for
            # entirely constant bases; unlike tensor exponents, they have a
            # nonsingular backward at zero.
            values = [block.pow(0)]
            if degree:
                values.append(block)
            for _ in range(1, degree):
                values.append(values[-1] * block)
            table = torch.stack(values, dim=2)
            # Explicit multiplication also avoids prod backward dividing by tiny
            # coordinates, which can spoil higher derivatives near a face.
            coordinates = table.gather(2, powers.T.unsqueeze(0).expand(len(t), -1, -1)).unbind(dim=1)
            monomials = coordinates[0]
            for coordinate in coordinates[1:]:
                monomials = monomials * coordinate
            factor = coeffs.to(t.dtype) * monomials
        basis = (basis.unsqueeze(2) * factor.unsqueeze(1)).reshape(t.shape[0], -1)
        offset += width
    return basis


@cache
def _predecessors(width: int, degree: int) -> tuple[tuple[int, ...], ...]:
    previous = {alpha: i + 1 for i, alpha in enumerate(_simplex_indices(width, degree - 1))}
    return tuple(
        tuple(previous.get(tuple(v - (j == k) for j, v in enumerate(alpha)), 0) for k in range(width))
        for alpha in _simplex_indices(width, degree)
    )


@cache
def _predecessor_tensor(width: int, degree: int, device: torch.device) -> torch.Tensor:
    """Device-local indices, shared by repeated evaluations of a basis."""
    return torch.tensor(_predecessors(width, degree), device=device)


def _recursive_basis(t: torch.Tensor, degree: int) -> torch.Tensor:
    """Degree elevation using additions/products only, including at zero coordinates."""
    basis = torch.ones((len(t), 1), dtype=t.dtype, device=t.device)
    for d in range(1, degree + 1):
        indices = _predecessor_tensor(t.shape[1], d, t.device)
        padded = torch.cat((torch.zeros_like(basis[:, :1]), basis), dim=1)
        basis = (padded[:, indices] * t[:, None, :]).sum(2)
    return basis


def _simplex_pair_keys(dimension: int, degree: int) -> _EdgeKeys:
    """Encode simplex edges as ``p * n + q`` using reverse-lexicographic composition ranks."""
    n = comb(degree + dimension, dimension)
    if not degree:
        return np.empty(0, dtype=np.int64)
    dtype: np.dtype[np.int32 | np.int64] = np.dtype(
        np.int32 if max(n, comb(degree + dimension, dimension - 1)) <= np.iinfo(np.int32).max else np.int64
    )
    alpha = np.asarray([index[0] for index in _indices((dimension,), (degree,))], dtype=dtype)
    # rank(alpha) = sum_k C(sum(alpha[k + 1:]) + dimension - k - 1, dimension - k).
    # Moving a unit from i to j > i increments the suffix sums k = i, ..., j - 1. Pascal's identity gives
    # the rank increment as a sum of C(suffix[k] + dimension - k - 1, dimension - k - 1).
    suffix = degree - np.cumsum(alpha[:, :-1], axis=1, dtype=dtype)
    del alpha
    increments = np.empty_like(suffix)
    for k in range(dimension):
        table = np.asarray([comb(r + dimension - k - 1, dimension - k - 1) for r in range(degree + 1)], dtype=np.int64)
        increments[:, k] = table[suffix[:, k]]
    np.cumsum(increments, axis=1, dtype=dtype, out=increments)
    key_dtype = np.uint32 if n * n - 1 <= np.iinfo(np.uint32).max else np.int64
    keys = cast(_EdgeKeys, np.empty(comb(degree + dimension - 1, dimension) * comb(dimension + 1, 2), dtype=key_dtype))
    offset = 0
    for i in range(dimension):
        previous = degree if i == 0 else suffix[:, i - 1]
        p = np.flatnonzero(previous != suffix[:, i])
        for j in range(i + 1, dimension + 1):
            q = p + increments[p, j - 1]
            if i:
                q -= increments[p, i - 1]
            keys[offset : offset + len(p)] = p * n + q
            offset += len(p)
    keys.sort()
    return keys


def _product_pair_keys(simploid_type: SimploidType, degree: tuple[int, ...]) -> _EdgeKeys:
    """Generate sorted edge keys in the product's mixed-radix control order."""
    if len(simploid_type) == 1:
        return _simplex_pair_keys(simploid_type[0], degree[0])
    sizes = [comb(a + d, a) for a, d in zip(simploid_type, degree, strict=True)]
    n = prod(sizes)
    keys_by_factor = []
    stride = n
    for a, d, size in zip(simploid_type, degree, sizes, strict=True):
        stride //= size
        local = _simplex_pair_keys(a, d)
        p, q = local // size, local % size
        # The product's control order is mixed radix: preceding factors select a block, following ones a suffix.
        prefix = np.arange(n // (size * stride), dtype=np.int64) * size * stride
        suffix = np.arange(stride, dtype=np.int64)
        keys = (prefix[:, None, None] + p[None, :, None] * stride + suffix) * n
        keys += prefix[:, None, None] + q[None, :, None] * stride + suffix
        keys_by_factor.append(keys.reshape(-1))
    keys = np.concatenate(keys_by_factor) if keys_by_factor else np.empty(0, dtype=np.int64)
    del keys_by_factor
    keys.sort()
    return keys


def _make_adjacent_pairs(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Convert sorted integer keys after releasing the generation temporaries."""
    keys = _product_pair_keys(simploid_type, degree)
    n = prod(comb(a + d, a) for a, d in zip(simploid_type, degree, strict=True))
    # Reuse one Python integer per control, as the old position dictionary did. Converting endpoint arrays with
    # tolist() would create two Python integers per edge and more than double the wide-simplex memory use.
    endpoints = np.arange(n, dtype=np.int64).astype(object)
    return tuple(_edge_tuple_chunks(keys, endpoints, n))


def _edge_tuple_chunks(keys: _EdgeKeys, endpoints: npt.NDArray[np.object_], n: int) -> Iterator[tuple[int, int]]:
    """Bound temporary object arrays while yielding the already sorted edges."""
    for start in range(0, len(keys), 65536):
        chunk = keys[start : start + 65536]
        yield from zip(endpoints[chunk // n], endpoints[chunk % n], strict=True)


@lru_cache(maxsize=128)
def _small_adjacent_pairs(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Cache only small immutable edge tuples; large results must die with their owning spaces."""
    return _make_adjacent_pairs(simploid_type, degree)


def _adjacent_pairs(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """The pairs ``(p, q)``, ``p < q``, of positions of adjacent control points in :func:`simploid_indices` order.

    Two control points are adjacent if their multi-indices differ in a single factor, by moving one unit
    from one component to another (an :math:`L^1` distance of 2 in that factor). Results with at most 4096
    edges share a bounded cache; larger tuples are not retained after their caller releases them.
    """
    sizes = [comb(a + d, a) for a, d in zip(simploid_type, degree, strict=True)]
    n = prod(sizes)
    count = sum(
        comb(a + d - 1, a) * comb(a + 1, 2) * (n // size) if d else 0
        for a, d, size in zip(simploid_type, degree, sizes, strict=True)
    )
    return _small_adjacent_pairs(simploid_type, degree) if count <= 4096 else _make_adjacent_pairs(simploid_type, degree)


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
    >>> from torch_bsf._kernels import sample_simploid
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
    for a, (src, g) in zip(alpha, dmap, strict=True):
        if src < 0:
            continue
        b = [0] * (max(g) + 1)
        for x, v in enumerate(g):
            b[v] += a[x]
        beta[src] = tuple(b)
    return tuple(beta)
