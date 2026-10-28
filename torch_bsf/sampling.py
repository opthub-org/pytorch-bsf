"""Domain-aware random and deterministic sampling, using local random generators.

Skeleton sampling and its optimal ratio follow the asymptotic risk analysis of
inductive skeleton fitting by Tanaka et al. (AAAI 2020), extended to every
dimension and degree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from torch_bsf._kernels.combinatorics import _indices, _simplex_indices
from torch_bsf._skeleton import _control_faces, _faces_with_interior_controls, _interior_and_boundary, _skeleton_shape
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
        uniforms = torch.quasirandom.SobolEngine(width, scramble=True, seed=local_seed).draw(n, dtype=torch.float64)
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


def _face_gram(rows: torch.Tensor, columns: torch.Tensor, face: Sequence[int], degree: int) -> torch.Tensor:
    r"""Gram matrix of Bernstein polynomials under the uniform distribution on a face.

    On a :math:`k`-simplex, :math:`E[t^\alpha] = k!\,\alpha! / (|\alpha| + k)!`, so the entry for
    :math:`a, b` is :math:`\prod_i \binom{a_i + b_i}{a_i} / (\binom{2D}{D} \binom{2D + k}{k})`.
    Each binomial is rounded once, so an entry has a relative error of at most ``(2 * len(face) + 2) * U``.
    This bound, which the conditioning check of :func:`skeleton_risk_coefficients` relies on, is why the
    lgamma-based Gram matrix of ``torch_bsf._kernels.sgd`` is not reused.
    """
    face = list(face)
    binomials = torch.tensor(
        [[float(math.comb(i, j)) for j in range(degree + 1)] for i in range(2 * degree + 1)], dtype=torch.float64
    )
    a, b = rows[:, face], columns[:, face]
    numerators = binomials[a[:, None] + b[None], a[:, None].expand(-1, len(b), -1)].prod(2)
    k = len(face) - 1
    return numerators / float(math.comb(2 * degree, degree) * math.comb(2 * degree + k, k))


def skeleton_risk_coefficients(space: BezierSpace) -> tuple[float, ...]:
    r"""Asymptotic risk coefficients of inductive skeleton fitting on a simplex.

    Inductive skeleton fitting estimates the controls interior to each face by
    least squares on that face's observations, holding the controls of lower
    faces at their estimates (see :func:`torch_bsf.fitting.fit_skeleton_regression`).
    With :math:`N^{(m)}` observations on the :math:`m`-skeleton, split equally
    among its faces and drawn uniformly on each, outputs of dimension :math:`L`
    and independent noise of variance :math:`\sigma^2`, the mean integrated
    squared error over the uniform simplex satisfies

    .. math:: R = \sigma^2 L \sum_{m} a_m / N^{(m)} + o(1 / N).

    Coefficient ``m`` is :math:`a_m`. It is zero for skeletons whose faces
    have no interior controls (:math:`m \geq` degree). This extends the
    numerical tables of Tanaka et al. (AAAI 2020) to every dimension and degree.

    Parameters
    ----------
    space
        A single simplex of positive dimension and degree.

    Returns
    -------
    tuple of float
        :math:`a_0, \dots, a_{M-1}` for a simplex with :math:`M` vertices.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If ``space`` is not a single simplex of positive dimension and degree,
        or if its Bernstein Gram matrices are too ill-conditioned to bound the
        float64 relative error by ``1e-6``. The largest accepted degree falls
        with the dimension: 14 for 2 vertices, 12 for 3, 11 for 4, 9 for 5,
        8 for 6 and 7, and 7 for 8.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import skeleton_risk_coefficients
    >>> [round(a, 4) for a in skeleton_risk_coefficients(BezierSpace.simplex(2, degree=3))]
    [0.1464, 3.5357, 1.0]
    """
    width, degree = _skeleton_shape(space)
    alpha = torch.tensor(_simplex_indices(width, degree), dtype=torch.long)
    control_faces = _control_faces(width, degree)
    # Control errors as linear maps of the independent face noises: on face I,
    # e_T = eta_I - G_TT^{-1} G_TB e_B, where T are the controls interior to I and B those on its boundary.
    errors = torch.zeros(len(alpha), len(alpha), dtype=torch.float64)
    face_grams: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for face in _faces_with_interior_controls(width, degree):
        interior, boundary = (torch.tensor(k, dtype=torch.long) for k in _interior_and_boundary(control_faces, face))
        gram = _face_gram(alpha[interior], alpha[torch.cat([interior, boundary])], face, degree)
        g_tt, g_tb = gram[:, : len(interior)], gram[:, len(interior) :]
        errors[interior] = -torch.linalg.solve(g_tt, g_tb @ errors[boundary])
        errors[interior[:, None], interior] = torch.eye(len(interior), dtype=torch.float64)
        face_grams.append((len(face), interior, g_tt))
    full = _face_gram(alpha, alpha, range(width), degree)
    # Bernstein Gram matrices are ill-conditioned in the degree. Their entries have relative errors of at most
    # (2 * width + 2) * U. To first order, each of the at most width levels of solves then adds
    # (2 * width + 2 + n) * kappa * U relative error (entry perturbation plus backward-stable elimination),
    # and the final traces add n * U, kappa being the largest condition number of the Gram matrices.
    kappa = max(torch.linalg.cond(g).item() for g in [full] + [g_tt for _, _, g_tt in face_grams])
    bound = (width * (2 * width + 2 + len(alpha)) * kappa + len(alpha)) * _UNIT_ROUNDOFF
    if not bound <= _RISK_RELATIVE_ERROR:
        raise ValueError(f"degree is too large for the risk coefficients in float64 (condition number {kappa:.1e}).")
    coefficients = [0.0] * width
    for size, interior, g_tt in face_grams:
        # n_I = N^(m) / C(M, m + 1) observations give Cov(eta_I) = sigma^2 G_TT^{-1} / n_I.
        propagated = errors[:, interior]
        trace = torch.linalg.solve(g_tt, propagated.T @ full @ propagated).trace().item()
        coefficients[size - 1] += math.comb(width, size) * trace
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
        A single simplex of positive dimension and degree.

    Returns
    -------
    tuple of float
        Nonnegative fractions for the 0-, 1-, ..., (M-1)-skeletons, summing to one.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        As :func:`skeleton_risk_coefficients`, including degrees too high for float64.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import optimal_skeleton_ratio
    >>> [round(r, 4) for r in optimal_skeleton_ratio(BezierSpace.simplex(1, degree=2))]
    [0.4142, 0.5858]
    """
    roots = [math.sqrt(a) for a in skeleton_risk_coefficients(space)]
    return tuple(r / sum(roots) for r in roots)


def _skeleton_fractions(ratio: Sequence[float] | torch.Tensor | None, space: BezierSpace, width: int) -> list[float]:
    """Validated skeleton fractions summing to one, the optimal ones by default."""
    if ratio is None:
        try:
            return list(optimal_skeleton_ratio(space))
        except ValueError as error:
            raise ValueError(f"{error} Pass ratio explicitly.") from error
    message = f"ratio must have {width} finite nonnegative real entries with a positive sum."
    if isinstance(ratio, torch.Tensor):
        if ratio.dtype == torch.bool or ratio.is_complex():
            raise ValueError(message)
        try:
            tensor = ratio.detach().cpu().to(torch.float64)
        except (RuntimeError, TypeError, NotImplementedError) as error:  # e.g. a meta tensor has no data to copy
            raise ValueError(message) from error
    else:
        # Converting directly to float64 avoids float32 overflow of large entries and int64 overflow of huge integers.
        try:
            tensor = torch.as_tensor(ratio, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(message) from error
        if tensor.shape == (width,) and any(isinstance(entry, (bool, np.bool_)) for entry in ratio):
            raise ValueError(message)
    if tensor.shape != (width,) or not torch.isfinite(tensor).all() or (tensor < 0).any() or not tensor.max() > 0:
        raise ValueError(message)
    # Scaling by the maximum first keeps the sum finite (at most width) for entries near the float64 limit.
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


def _face_counts(width: int, size: int, count: int, generator: torch.Generator) -> list[tuple[tuple[int, ...], int]]:
    """Faces of one skeleton with their row counts: an equal split, the remainder on random distinct faces."""
    n_faces = math.comb(width, size)
    if 2 * count < n_faces:
        # Every face gets at most one row. Rejection sampling of distinct faces costs O(count) draws on average
        # and avoids listing all C(width, size) faces of a wide simplex.
        chosen: set[tuple[int, ...]] = set()
        while len(chosen) < count:
            chosen.add(tuple(sorted(torch.randperm(width, generator=generator)[:size].tolist())))
        return [(face, 1) for face in sorted(chosen)]
    faces = list(combinations(range(width), size))
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
    """Stratified sample of a simplex's skeletons for inductive skeleton fitting.

    The ``n`` rows are split among the 0-skeleton (vertices), 1-skeleton
    (edges), ... in proportion to ``ratio`` by largest-remainder rounding.
    Each skeleton's count is split equally among its faces; when it does not
    divide, randomly chosen faces receive one extra row. Rows are uniform on
    their face, with exact zeros outside it, and are ordered by face size.

    Parameters
    ----------
    space
        A single simplex of positive dimension and degree.
    n
        Number of rows.
    ratio
        Relative sizes of the 0-, 1-, ..., (M-1)-skeletons of a simplex with
        M vertices: M finite nonnegative real entries with a positive sum.
        Defaults to :func:`optimal_skeleton_ratio`, which places no rows on
        skeletons without interior controls. Rows on such skeletons (faces with
        more than ``degree`` vertices) are not used by
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
        If ``space`` is not a single simplex of positive dimension and degree,
        ``n`` is not a nonnegative integer, ``method`` is unknown, or ``ratio``
        is malformed. With the default ``ratio``, also if the degree is too
        high for :func:`skeleton_risk_coefficients`; pass ``ratio`` then.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import sample_skeleton
    >>> points = sample_skeleton(BezierSpace.simplex(2, degree=2), 100, seed=0)
    >>> (points.parameters > 0).sum(1).bincount().tolist()
    [0, 26, 74]
    """
    width, _ = _skeleton_shape(space)
    _integer(n, "n")
    _choice(method, "method", ("random", "sobol"), message="method must be 'random' or 'sobol'.")
    counts = _skeleton_counts(n, _skeleton_fractions(ratio, space, width))
    generator = _generator(seed)
    blocks = [torch.zeros(0, width, dtype=torch.float64)]
    for size, count in enumerate(counts, start=1):
        for face, rows in _face_counts(width, size, count, generator) if count else []:
            exponentials = _exponentials(generator, rows, size, method)
            block = torch.zeros(rows, width, dtype=torch.float64)
            block[:, face] = exponentials / exponentials.sum(1, keepdim=True)
            blocks.append(block)
    return Points(torch.cat(blocks))
