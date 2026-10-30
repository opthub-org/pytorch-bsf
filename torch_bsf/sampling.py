"""Domain-aware random and deterministic sampling, using local random generators.

Skeleton sampling and its optimal ratio follow the asymptotic risk analysis of
inductive skeleton fitting by Tanaka et al. (AAAI 2020), extended to every
dimension and degree and to products of simplices.
"""

import math
import numbers
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations, product
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from torch_bsf._kernels.combinatorics import _indices, _simplex_indices
from torch_bsf._skeleton import (
    _blocks,
    _control_faces,
    _faces_with_interior_controls,
    _interior_and_boundary,
    _skeleton_cell,
)
from torch_bsf._validation import _choice
from torch_bsf.core import BezierSpace, Points, _integer
from torch_bsf.geometry import Simploid


def _generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _probabilities(space: BezierSpace, weights: Mapping[int | Simploid, float] | None) -> torch.Tensor:
    probability = torch.zeros(space.n_cells, dtype=torch.float64)
    if weights is None:
        probability[list(space.main_cells)] = 1.0
    else:
        for cell, weight in weights.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("cell_weights must be finite and nonnegative.")
            probability[space.cell_id(cell)] += weight
    if not torch.isfinite(probability.sum()) or not probability.sum() > 0:
        raise ValueError("cell_weights must have a positive sum.")
    return probability / probability.sum()


def _exponentials(generator: torch.Generator, n: int, width: int, method: str) -> torch.Tensor:
    """Standard exponential draws whose normalized blocks are uniform on simplices."""
    if method == "random":
        uniforms = torch.rand(n, width, generator=generator, dtype=torch.float64)
    else:
        local_seed = int(torch.randint(2**31, (1,), generator=generator))
        engine = torch.quasirandom.SobolEngine(width, scramble=True, seed=local_seed)
        uniforms = torch.empty(n, width, dtype=torch.float64)
        if n:
            # SobolEngine caches its first point in the default dtype; reconstruct it before rounding.
            uniforms[0] = engine.quasi.to(torch.float64) / 2**engine.MAXBIT
            if n > 1:
                engine.fast_forward(1)
                uniforms[1:] = engine.draw(n - 1, dtype=torch.float64)
    return -uniforms.clamp_min(torch.finfo(torch.float64).tiny).log()


def sample(
    space: BezierSpace,
    n: int,
    *,
    method: Literal["random", "sobol"] = "random",
    seed: int | None = None,
    cell_weights: Mapping[int | Simploid, float] | None = None,
) -> Points:
    """Sample a space, uniformly within each selected cell's simplex factors.

    Cell probabilities default to equal weights on maximal cells, rather than
    an unspecified geometric volume. Sobol coordinates use exponential
    normalization of scrambled quasi-random uniforms. Cell selection remains
    categorical random sampling. Neither method resets global random state.

    Parameters
    ----------
    space : BezierSpace
        Compiled domain to sample, with cell-local barycentric coordinate blocks.
    n : int
        Nonnegative number of coordinate rows to draw.
    method : {"random", "sobol"}, optional
        Uniform source used before exponential normalization within each factor.
    seed : int, optional
        Seed for a local generator; omitted to seed it nondeterministically.
    cell_weights : mapping, optional
        Nonnegative weights by cell ID or builder handle, normalized to probabilities; defaults to equal maximal-cell weights.

    Returns
    -------
    Points
        CPU float64 coordinates with explicit cell IDs and padding to the space width.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import sample
    >>> sample(BezierSpace.simplex(2, degree=3), 5, seed=0).parameters.shape
    torch.Size([5, 3])
    """
    _integer(n, "n")
    _choice(method, "method", ("random", "sobol"), message="method must be 'random' or 'sobol'.")
    generator = _generator(seed)
    probability = _probabilities(space, cell_weights)
    ids = torch.multinomial(probability, n, replacement=True, generator=generator) if n else torch.empty(0, dtype=torch.long)
    parameters = torch.zeros(n, space.width, dtype=torch.float64)
    for cid in torch.unique(ids).tolist():
        info = space.cells[cid]
        rows = torch.nonzero(ids == cid).flatten()
        if not info.width:
            continue
        exponentials = _exponentials(generator, len(rows), info.width, method)
        offset = 0
        for dimension in info.dimensions:
            block = exponentials[:, offset : offset + dimension + 1]
            parameters[rows, offset : offset + dimension + 1] = block / block.sum(1, keepdim=True)
            offset += dimension + 1
    return Points(parameters, ids)


def grid(space: BezierSpace, resolution: int = 10, *, cells: Sequence[int | Simploid] | None = None) -> Points:
    """Create tensor-product barycentric grids on the selected cells.

    ``resolution=0`` gives a cell's barycenter. Boundary points of different
    cells remain distinct parameter-domain points even when their images agree.

    Parameters
    ----------
    space : BezierSpace
        Compiled domain on which to construct cell-local grids.
    resolution : int, optional
        Nonnegative lattice degree per factor; zero selects each cell's barycenter.
    cells : sequence of int or Simploid, optional
        Ordered cell IDs or original builder handles; defaults to the maximal cells.

    Returns
    -------
    Points
        Concatenated CPU float64 grid rows with explicit cell IDs and padded coordinate widths.
    """
    _integer(resolution, "resolution")
    selected = list(space.main_cells) if cells is None else [space.cell_id(c) for c in cells]
    coordinates, identifiers = [], []
    for cid in selected:
        info = space.cells[cid]
        if resolution:
            indices = _indices(info.dimensions, (resolution,) * len(info.dimensions))
            t = torch.tensor([[v / resolution for factor in alpha for v in factor] for alpha in indices], dtype=torch.float64)
            t = t.reshape(len(indices), info.width)
        else:
            t = torch.tensor([[1 / (d + 1) for d in info.dimensions for _ in range(d + 1)]], dtype=torch.float64)
        coordinates.append(F.pad(t, (0, space.width - info.width)))
        identifiers.append(torch.full((len(t),), cid, dtype=torch.long))
    if not coordinates:
        return Points(torch.empty(0, space.width), torch.empty(0, dtype=torch.long))
    return Points(torch.cat(coordinates), torch.cat(identifiers))


_RISK_RELATIVE_ERROR = 1e-6
"""Largest first-order relative error bound accepted by :func:`skeleton_risk_coefficients`."""

_UNIT_ROUNDOFF = torch.finfo(torch.float64).eps / 2


def _face_gram(
    rows: torch.Tensor, columns: torch.Tensor, face: Sequence[int], dimensions: Sequence[int], degree: Sequence[int]
) -> torch.Tensor:
    r"""Gram matrix of Bernstein polynomials under the uniform distribution on a face.

    ``rows`` and ``columns`` hold concatenated multi-indices of the simploid with factor ``dimensions`` and
    ``degree``, and the uniform distribution on a face is the product of the uniform distributions on its
    factors. On a :math:`k`-simplex, :math:`E[t^\alpha] = k!\,\alpha! / (|\alpha| + k)!`, so the factor of the
    entry for :math:`a, b` is :math:`\prod_i \binom{a_i + b_i}{a_i} / (\binom{2D}{D} \binom{2D + k}{k})`.
    Each binomial, and the product of the exact integer denominators, is rounded once, so an entry has a
    relative error of at most ``(2 * len(face) + 2) * U``. This bound, which the conditioning check of
    :func:`skeleton_risk_coefficients` relies on, is why the lgamma-based Gram matrix of
    ``torch_bsf._kernels.sgd`` is not reused.
    """
    face = list(face)
    top = max(degree)
    binomials = torch.tensor(
        [[float(math.comb(i, j)) for j in range(top + 1)] for i in range(2 * top + 1)], dtype=torch.float64
    )
    # Multiplying one coordinate at a time keeps memory at O(len(rows) * len(columns)) for wide simplices.
    numerators = torch.ones(len(rows), len(columns), dtype=torch.float64)
    for i in face:
        a, b = rows[:, i, None], columns[None, :, i]
        numerators *= binomials[a + b, a.expand(-1, len(columns))]
    denominator = 1
    for block, d in zip(_blocks(dimensions), degree, strict=True):
        k = len(set(block).intersection(face)) - 1
        denominator *= math.comb(2 * d, d) * math.comb(2 * d + k, k)
    return numerators / float(denominator)


def _factor_grams(dimensions: Sequence[int], degree: Sequence[int]) -> list[torch.Tensor]:
    """Gram matrices of each factor's Bernstein polynomials under the uniform distribution on that factor.

    Under the uniform distribution on the product, the Gram matrix on the whole simploid is, up to the order of the
    controls, their Kronecker product, so its condition number is the product of theirs. On a simplex, the only
    factor's matrix is the whole one.
    """
    grams = []
    for a, d in zip(dimensions, degree, strict=True):
        alpha = torch.tensor(_simplex_indices(a + 1, d), dtype=torch.long)
        grams.append(_face_gram(alpha, alpha, range(a + 1), (a,), (d,)))
    return grams


def _shapes(dimensions: Sequence[int], m: int) -> list[tuple[int, ...]]:
    """Vertex counts per factor of the faces of dimension ``m``, in lexicographic order."""
    if len(dimensions) <= 1:
        return [(m + 1,)[: len(dimensions)]] if 0 <= m <= sum(dimensions) else []
    return [(s, *rest) for s in range(1, min(dimensions[0], m) + 2) for rest in _shapes(dimensions[1:], m - s + 1)]


def _n_faces(dimensions: Sequence[int], sizes: Sequence[int]) -> int:
    """The number of faces with ``sizes`` vertices per factor."""
    return math.prod(math.comb(a + 1, s) for a, s in zip(dimensions, sizes, strict=True))


def _sampled_shapes(dimensions: Sequence[int], degree: Sequence[int], m: int) -> list[tuple[int, ...]]:
    """Shapes of the faces of the ``m``-skeleton that receive its rows.

    These are the faces with interior controls (at most ``degree`` vertices in every factor), or every face of
    dimension ``m`` when none has interior controls. On a simplex, the faces of one dimension are all alike.
    """
    shapes = _shapes(dimensions, m)
    return [sizes for sizes in shapes if all(s <= d for s, d in zip(sizes, degree, strict=True))] or shapes


def _n_skeleton_faces(dimensions: Sequence[int], degree: Sequence[int], m: int) -> int:
    """The number of faces of the ``m``-skeleton that receive its rows."""
    return sum(_n_faces(dimensions, sizes) for sizes in _sampled_shapes(dimensions, degree, m))


def skeleton_risk_coefficients(space: BezierSpace) -> tuple[float, ...]:
    r"""Asymptotic risk coefficients of inductive skeleton fitting on a simplex or simploid.

    Inductive skeleton fitting estimates the controls interior to each face by
    least squares on that face's observations, holding the controls of lower
    faces at their estimates (see :func:`torch_bsf.fitting.fit_skeleton_regression`).
    With :math:`N^{(m)}` observations on the :math:`m`-skeleton, split equally
    among its faces and drawn uniformly on each, outputs of dimension :math:`L`
    and independent noise of variance :math:`\sigma^2`, the mean integrated
    squared error over the uniform domain satisfies

    .. math:: R = \sigma^2 L \sum_{m} a_m / N^{(m)} + o(1 / N).

    Coefficient ``m`` is :math:`a_m`. It is zero for skeletons whose faces
    have no interior controls (on a simplex, :math:`m \geq` degree). This
    extends the numerical tables of Tanaka et al. (AAAI 2020) to every
    dimension and degree.

    On a simploid, a product of simplices, a face is a product of one face per
    factor and the :math:`m`-skeleton consists of the faces of dimension
    :math:`m`. A face has interior controls when no factor has more vertices
    than its degree, and the :math:`N^{(m)}` observations are split equally
    among the faces of dimension :math:`m` with interior controls, as
    :func:`sample_skeleton` does. Observations and the error are uniform on
    the product.

    Parameters
    ----------
    space
        A single simplex or simploid of positive dimension and degree in every factor.

    Returns
    -------
    tuple of float
        :math:`a_0, \dots, a_n` for a domain of dimension :math:`n`, which is
        :math:`M - 1` for a simplex with :math:`M` vertices.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If ``space`` is not a single simplex or simploid of positive dimension
        and degree in every factor, or if its Bernstein Gram matrices are too
        ill-conditioned to bound the float64 relative error by ``1e-6``. For a
        simplex, the largest accepted degree falls with the dimension: 14 for 2
        vertices, 12 for 3, 11 for 4, 9 for 5, 8 for 6 and 7, and 7 for 8. On
        a simploid, the condition number of the Gram matrix on the whole
        simploid is the product of those of its factors' Gram matrices, so it
        also grows with the number of factors: 12 segments of degree 1 are
        rejected.

    Notes
    -----
    Time and memory grow with the square of the number of controls,
    :math:`\binom{D + M - 1}{D}` for a simplex and the product of these over the
    factors of a simploid, so wide domains are costly even at low degree.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import skeleton_risk_coefficients
    >>> [round(a, 4) for a in skeleton_risk_coefficients(BezierSpace.simplex(2, degree=3))]
    [0.1464, 3.5357, 1.0]
    """
    cell = _skeleton_cell(space)
    dimensions, degree = cell.dimensions, cell.degree
    width, n_controls, levels = cell.width, space.n_controls, sum(dimensions) + 1

    def check(kappa: float) -> None:
        """Reject condition numbers exceeding the float64 relative-error budget.

        Parameters
        ----------
        kappa : float
            Largest Gram-matrix condition number encountered so far.
        """
        # Bernstein Gram matrices are ill-conditioned in the degree. Their entries have relative errors of at most
        # (2 * width + 2) * U. To first order, each of the at most ``levels`` face dimensions of solves then adds
        # (2 * width + 2 + n) * kappa * U relative error (entry perturbation plus backward-stable elimination),
        # and the final traces add n * U, kappa being the largest condition number of the Gram matrices.
        bound = (levels * (2 * width + 2 + n_controls) * kappa + n_controls) * _UNIT_ROUNDOFF
        if not bound <= _RISK_RELATIVE_ERROR:
            raise ValueError(f"degree is too large for the risk coefficients in float64 (condition number {kappa:.1e}).")

    # Reject hopeless spaces (high degrees, or many factors on a simploid) before allocating anything of size
    # n_controls: the largest denominator must fit in float64; the small Gram block of the controls interior to one
    # edge of the factor of highest degree, among those whose largest condition number enters the bound, must
    # already pass it; and so must the Gram matrix on the whole simploid. Its condition number, which grows with the
    # number of factors, is the product of those of the factors' matrices (see _factor_grams), of C(d + a, a) rows
    # each, and this product replaces the condition number of the whole matrix in the final check.
    factors = zip(dimensions, degree, strict=True)
    if math.prod(math.comb(2 * d, d) * math.comb(2 * d + a, a) for a, d in factors) > sys.float_info.max:
        check(math.inf)
    top = max(degree)
    if top >= 2:
        edge = torch.tensor([[j, top - j] for j in range(1, top)], dtype=torch.long)
        check(torch.linalg.cond(_face_gram(edge, edge, (0, 1), (1,), (top,))).item())
    factor_grams = _factor_grams(dimensions, degree)
    full_condition = math.prod(torch.linalg.cond(g).item() for g in factor_grams)
    check(full_condition)
    alpha = torch.tensor([sum(index, ()) for index in _indices(dimensions, degree)], dtype=torch.long)
    control_faces = _control_faces(dimensions, degree)
    # Control errors as linear maps of the independent face noises: on face I,
    # e_T = eta_I - G_TT^{-1} G_TB e_B, where T are the controls interior to I and B those on its boundary.
    errors = torch.zeros(len(alpha), len(alpha), dtype=torch.float64)
    face_grams: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for face in _faces_with_interior_controls(dimensions, degree):
        interior, boundary = (torch.tensor(k, dtype=torch.long) for k in _interior_and_boundary(control_faces, face))
        gram = _face_gram(alpha[interior], alpha[torch.cat([interior, boundary])], face, dimensions, degree)
        g_tt, g_tb = gram[:, : len(interior)], gram[:, len(interior) :]
        errors[interior] = -torch.linalg.solve(g_tt, g_tb @ errors[boundary])
        errors[interior[:, None], interior] = torch.eye(len(interior), dtype=torch.float64)
        face_grams.append((len(face) - len(dimensions), interior, g_tt))
    check(max([full_condition] + [torch.linalg.cond(g_tt).item() for _, _, g_tt in face_grams]))
    # On a simplex, the only factor's Gram matrix is the whole one.
    full = factor_grams[0] if len(factor_grams) == 1 else _face_gram(alpha, alpha, range(width), dimensions, degree)
    coefficients = [0.0] * levels
    n_faces = [_n_skeleton_faces(dimensions, degree, m) for m in range(levels)]
    for m, interior, g_tt in face_grams:
        # n_I = N^(m) / K_m observations, with K_m faces of dimension m, give Cov(eta_I) = sigma^2 G_TT^{-1} / n_I.
        propagated = errors[:, interior]
        trace = torch.linalg.solve(g_tt, propagated.T @ full @ propagated).trace().item()
        coefficients[m] += n_faces[m] * trace
    return tuple(coefficients)


def optimal_skeleton_ratio(space: BezierSpace) -> tuple[float, ...]:
    r"""Skeleton sample fractions that minimize the asymptotic risk of inductive skeleton fitting.

    Minimizing :math:`\sum_m a_m / N^{(m)}` subject to :math:`\sum_m N^{(m)} = N`
    gives :math:`N^{(m)} / N = \sqrt{a_m} / \sum_k \sqrt{a_k}` and the minimum
    :math:`(\sum_m \sqrt{a_m})^2 / N` (Cauchy-Schwarz), where :math:`a_m` are
    :func:`skeleton_risk_coefficients`.

    Parameters
    ----------
    space
        A single simplex or simploid of positive dimension and degree in every factor.

    Returns
    -------
    tuple of float
        Nonnegative fractions for the 0-, 1-, ..., n-skeletons of a domain of
        dimension n (n = M - 1 for a simplex with M vertices), summing to one.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        As :func:`skeleton_risk_coefficients`, including degrees too high for
        float64 and simploids with too many factors.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import optimal_skeleton_ratio
    >>> [round(r, 4) for r in optimal_skeleton_ratio(BezierSpace.simplex(1, degree=2))]
    [0.4142, 0.5858]
    """
    roots = [math.sqrt(a) for a in skeleton_risk_coefficients(space)]
    return tuple(r / sum(roots) for r in roots)


def _real_entries(ratio: object) -> bool:
    """Whether a non-tensor ratio is a NumPy array or a sequence of real, non-boolean numbers."""
    if isinstance(ratio, np.ndarray):
        return ratio.dtype.kind in "iuf"

    def real(entry: object) -> bool:
        if isinstance(entry, torch.Tensor):
            return entry.ndim == 0 and entry.dtype != torch.bool and not entry.is_complex()
        return isinstance(entry, numbers.Real) and not isinstance(entry, bool)

    return isinstance(ratio, Sequence) and not isinstance(ratio, str) and all(real(entry) for entry in ratio)


def _skeleton_fractions(ratio: Sequence[float] | torch.Tensor | None, space: BezierSpace, levels: int) -> list[float]:
    """Validated fractions of the ``levels`` skeletons summing to one, the optimal ones by default."""
    if ratio is None:
        try:
            return list(optimal_skeleton_ratio(space))
        except ValueError as error:
            raise ValueError(f"{error} Pass ratio explicitly.") from error
    message = f"ratio must have {levels} finite nonnegative real entries with a positive sum."
    if isinstance(ratio, torch.Tensor):
        if ratio.dtype == torch.bool or ratio.is_complex():
            raise ValueError(message)
        try:
            tensor = ratio.detach().cpu().to(torch.float64)
        except (RuntimeError, TypeError, NotImplementedError) as error:  # e.g. a meta tensor has no data to copy
            raise ValueError(message) from error
    else:
        if not _real_entries(ratio):
            raise ValueError(message)
        # Converting directly to float64 avoids float32 overflow of large entries and int64 overflow of huge integers.
        try:
            tensor = torch.as_tensor(ratio, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(message) from error
    if tensor.shape != (levels,) or not torch.isfinite(tensor).all() or (tensor < 0).any() or not tensor.max() > 0:
        raise ValueError(message)
    # Scaling by the maximum first keeps the sum finite (at most levels) for entries near the float64 limit.
    tensor = tensor / tensor.max()
    return (tensor / tensor.sum()).tolist()


def _skeleton_counts(n: int, fractions: Sequence[float]) -> list[int]:
    """Largest-remainder rounding of ``n`` by fractions summing to one."""
    quotas = [n * f for f in fractions]
    counts = [math.floor(q) for q in quotas]
    by_remainder = sorted(range(len(fractions)), key=lambda m: (counts[m] - quotas[m], m))
    for m in by_remainder[: n - sum(counts)]:
        counts[m] += 1
    return counts


def _face_counts(
    dimensions: Sequence[int], degree: Sequence[int], m: int, count: int, generator: torch.Generator
) -> list[tuple[tuple[int, ...], int]]:
    """Faces of the ``m``-skeleton that receive its rows, with their counts: an equal split, the remainder on random faces."""
    blocks, shapes = _blocks(dimensions), _sampled_shapes(dimensions, degree, m)
    faces_per_shape = [_n_faces(dimensions, sizes) for sizes in shapes]
    n_faces = sum(faces_per_shape)
    if 2 * count < n_faces:
        # Every face gets at most one row. Rejection sampling of distinct faces costs O(count) draws on average
        # and avoids listing all faces of a wide simplex. Drawing a shape in proportion to its number of faces,
        # then each factor's vertices uniformly, makes every face equally likely.
        weights = torch.tensor(faces_per_shape, dtype=torch.float64) if len(shapes) > 1 else None
        chosen: set[tuple[int, ...]] = set()
        while len(chosen) < count:
            sizes = shapes[0] if weights is None else shapes[int(torch.multinomial(weights, 1, generator=generator))]
            face: tuple[int, ...] = ()
            for block, size in zip(blocks, sizes, strict=True):
                vertices = torch.randperm(len(block), generator=generator)[:size].tolist()
                face += tuple(sorted(block.start + v for v in vertices))
            chosen.add(face)
        return [(face, 1) for face in sorted(chosen)]
    faces = sorted(
        sum(parts, ())
        for sizes in shapes
        for parts in product(*(combinations(block, size) for block, size in zip(blocks, sizes, strict=True)))
    )
    per_face = torch.full((n_faces,), count // n_faces, dtype=torch.long)
    per_face[torch.randperm(n_faces, generator=generator)[: count % n_faces]] += 1
    return [(face, rows) for face, rows in zip(faces, per_face.tolist(), strict=True) if rows]


def sample_skeleton(
    space: BezierSpace,
    n: int,
    *,
    ratio: Sequence[float] | torch.Tensor | None = None,
    method: Literal["random", "sobol"] = "random",
    seed: int | None = None,
) -> Points:
    """Stratified sample of the skeletons of a simplex or simploid for inductive skeleton fitting.

    The ``n`` rows are split among the 0-skeleton (vertices), 1-skeleton
    (edges), ... in proportion to ``ratio`` by largest-remainder rounding.
    Each skeleton's count is split equally among its faces; when it does not
    divide, randomly chosen faces receive one extra row. Rows are uniform on
    their face, with exact zeros outside it, and are ordered by face dimension.

    A face of a simploid is a product of one face per simplex factor, and its
    rows are uniform on each factor's face. The faces of one skeleton can differ
    in shape: the edges of a prism are triangle edges at an end of the interval
    and intervals at a vertex of the triangle. A skeleton's rows then go only to
    its faces with interior controls (no factor has more vertices than its
    degree), unless none has any. On a simplex, either every face of a
    skeleton has interior controls or none has.

    Parameters
    ----------
    space
        A single simplex or simploid of positive dimension and degree in every factor.
    n
        Number of rows.
    ratio
        Relative sizes of the 0-, 1-, ..., n-skeletons of a domain of
        dimension n: n + 1 finite nonnegative real entries with a positive sum,
        that is, M entries for a simplex with M vertices. Defaults to
        :func:`optimal_skeleton_ratio`, which places no rows on skeletons
        without interior controls. Rows on faces without interior controls
        (with more vertices than the degree in some factor) are not used by
        :func:`~torch_bsf.fitting.fit_skeleton_regression`.
    method
        ``"random"`` or ``"sobol"`` coordinates within each face, as in :func:`sample`.
    seed
        Seed of the local generator; global random state is not used.

    Returns
    -------
    Points
        Barycentric rows of the single cell.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If ``space`` is not a single simplex or simploid of positive dimension
        and degree in every factor, ``n`` is not a nonnegative integer,
        ``method`` is unknown, or ``ratio`` is malformed. With the default
        ``ratio``, also if :func:`skeleton_risk_coefficients` rejects the space
        (too high a degree, or too many factors on a simploid); pass ``ratio``
        then.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import sample_skeleton
    >>> points = sample_skeleton(BezierSpace.simplex(2, degree=2), 100, seed=0)
    >>> (points.parameters > 0).sum(1).bincount().tolist()
    [0, 26, 74]
    """
    cell = _skeleton_cell(space)
    _integer(n, "n")
    _choice(method, "method", ("random", "sobol"), message="method must be 'random' or 'sobol'.")
    counts = _skeleton_counts(n, _skeleton_fractions(ratio, space, sum(cell.dimensions) + 1))
    generator = _generator(seed)
    # Each factor's coordinates on a face are normalized to sum to one; ends[j] bounds factor j's coordinates.
    ends = torch.tensor([block.stop for block in _blocks(cell.dimensions)])
    chunks = [torch.zeros(0, cell.width, dtype=torch.float64)]
    for m, count in enumerate(counts):
        if not count:
            continue  # An empty level draws nothing; listing its face shapes can cost 2**n_factors on many factors.
        for face, rows in _face_counts(cell.dimensions, cell.degree, m, count, generator):
            exponentials = _exponentials(generator, rows, len(face), method)
            chunk = torch.zeros(rows, cell.width, dtype=torch.float64)
            if len(ends) == 1:
                # A single simplex needs no per-factor sums; this path keeps its draws and values unchanged.
                chunk[:, face] = exponentials / exponentials.sum(1, keepdim=True)
            else:
                factor = torch.bucketize(torch.tensor(face), ends, right=True)
                sums = torch.zeros(rows, len(ends), dtype=torch.float64).index_add_(1, factor, exponentials)
                chunk[:, face] = exponentials / sums[:, factor]
            chunks.append(chunk)
    return Points(torch.cat(chunks))
