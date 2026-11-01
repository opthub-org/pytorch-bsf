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
from typing import Literal, overload

import numpy as np
import torch
from torch.nn import functional as F

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._skeleton import (
    _blocks,
    _control_faces,
    _faces_with_interior_controls,
    _interior_and_boundary,
    _single_skeleton,
    _skeleton_cell,
    _skeleton_stages,
)
from torch_bsf._skeleton_risk import _closed_risk_coefficients
from torch_bsf._validation import _choice
from torch_bsf.core import BezierSpace, Points, _integer
from torch_bsf.geometry import Simploid
from torch_bsf.results import SkeletonAllocation

_CellWeights = Mapping[int, float] | Mapping[Simploid, float] | Mapping[int | Simploid, float]


def _generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _probabilities(space: BezierSpace, weights: _CellWeights | None) -> torch.Tensor:
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
    cell_weights: _CellWeights | None = None,
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


def _gram_condition(dimensions: Sequence[int], degree: Sequence[int]) -> float:
    r"""Condition number of the Gram matrix of Bernstein polynomials under the uniform distribution on a simploid.

    On a :math:`k`-simplex, the Gram matrix of degree :math:`D` has the eigenvalues
    :math:`D!\,(D + k)! / ((D - j)!\,(D + j + k)!)` for :math:`j = 0, \dots, D`, up to a common factor
    (Kirby, Numerische Mathematik 135, 2017), so its condition number is :math:`\binom{2D + k}{D}`. On a simploid,
    the Gram matrix is, up to the order of the controls, the Kronecker product of its factors' matrices, so its
    condition number is the product of theirs. The caller ensures that the product fits in float64.
    """
    return float(math.prod(math.comb(2 * d + a, d) for a, d in zip(dimensions, degree, strict=True)))


def _face_trace_sums(dimensions: Sequence[int], degree: Sequence[int]) -> tuple[list[float], float]:
    r"""Risk traces of the faces of a simploid, summed by face dimension, and the largest condition number of their blocks.

    The trace of a face :math:`I` with interior controls :math:`T` is
    :math:`\mathrm{tr}(G^I_{TT}{}^{-1} E_I^\top G E_I)`, where :math:`G^I` and :math:`G` are the Gram matrices under
    the uniform distributions on :math:`I` and on the simploid, and :math:`E_I` maps the noise of the least-squares
    estimate of the controls :math:`T` to the errors of all controls. With :math:`n_I` observations on :math:`I`, that
    noise has the covariance :math:`\sigma^2 G^I_{TT}{}^{-1} / n_I`, so the face adds :math:`\sigma^2 L / n_I` times
    its trace to the risk of :func:`skeleton_risk_coefficients`; the sums leave out :math:`1 / n_I`. The condition
    number is the largest one of the blocks :math:`G^I_{TT}`. On a simplex, vertex symmetry reduces the noise
    columns to one representative face per dimension; all control rows are retained.
    """
    alpha = torch.tensor([sum(index, ()) for index in _indices(dimensions, degree)], dtype=torch.long)
    control_faces = _control_faces(dimensions, degree)
    faces = _faces_with_interior_controls(dimensions, degree)
    # Vertex permutations preserve a simplex's uniform Gram matrices and traces. Keep one face's noise per
    # dimension, but propagate it to ALL controls, including controls outside that representative face.
    representatives = {len(face): face for face in reversed(faces)} if len(dimensions) == 1 else {}
    noises = [i for i, face in enumerate(control_faces) if not representatives or face == representatives[len(face)]]
    positions = torch.full((len(alpha),), -1, dtype=torch.long)
    positions[noises] = torch.arange(len(noises))
    # Built first, so that its temporaries are freed before the error matrix is allocated.
    full = _face_gram(alpha, alpha, range(alpha.shape[1]), dimensions, degree)
    errors = torch.zeros(len(alpha), len(noises), dtype=torch.float64)
    face_grams: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
    condition = 1.0
    for face in faces:
        interior, boundary = (torch.tensor(k, dtype=torch.long) for k in _interior_and_boundary(control_faces, face))
        closure = torch.cat([interior, boundary])
        gram = _face_gram(alpha[interior], alpha[closure], face, dimensions, degree)
        g_tt, g_tb = gram[:, : len(interior)], gram[:, len(interior) :]
        # Boundary errors depend only on noises in the closure of this face; other columns stay zero.
        columns = positions[closure]
        columns = columns[columns >= 0]
        if len(columns):
            errors[interior[:, None], columns] = -torch.linalg.solve(g_tt, g_tb @ errors[boundary[:, None], columns])
        own = positions[interior]
        active = own >= 0
        errors[interior[active], own[active]] = 1.0
        # Check every original block, even when symmetry removes its noise columns.
        condition = max(condition, torch.linalg.cond(g_tt).item())
        if not representatives or face == representatives[len(face)]:
            multiplicity = math.comb(dimensions[0] + 1, len(face)) if representatives else 1
            face_grams.append((len(face) - len(dimensions), own, g_tt, multiplicity))
    return _batched_face_traces(full, errors, face_grams, sum(dimensions) + 1), condition


def _batched_face_traces(
    full: torch.Tensor,
    errors: torch.Tensor,
    face_grams: list[tuple[int, torch.Tensor, torch.Tensor, int]],
    levels: int,
) -> list[float]:
    """Contract the propagated errors with the full Gram matrix once per face dimension."""
    sums = [0.0] * levels
    for m in range(levels):
        blocks = [(columns, gram, count) for level, columns, gram, count in face_grams if level == m]
        if not blocks:
            continue
        propagated = errors[:, torch.cat([columns for columns, _, _ in blocks])]
        weighted = full @ propagated
        offset = 0
        for columns, gram, count in blocks:
            stop = offset + len(columns)
            trace = torch.linalg.solve(gram, propagated[:, offset:stop].T @ weighted[:, offset:stop]).trace().item()
            sums[m] += count * trace
            offset = stop
    return sums


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


def _float_weight(weight: float, name: str) -> float:
    """Convert real weights before tensor accumulation, retaining the distinction between zero and underflow."""
    message = f"{name} must contain finite nonnegative real weights representable in float64."
    if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or weight < 0:
        raise ValueError(message)
    try:
        converted = float(weight)
    except OverflowError as error:
        raise ValueError(message) from error
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(message)
    if weight > 0 and converted == 0:
        raise ValueError(f"{name} underflow: a positive weight is not representable in float64.")
    return converted


def _evaluation_probabilities(
    space: BezierSpace, weights: _CellWeights | None, name: str = "evaluation_weights"
) -> torch.Tensor:
    """Normalize chart evaluation weights, combining ID/handle aliases and rejecting invalid entries."""
    if weights is not None and not isinstance(weights, Mapping):
        raise ValueError(f"{name} must be a mapping of finite nonnegative real weights.")
    converted = None if weights is None else {cell: _float_weight(w, name) for cell, w in weights.items()}
    try:
        probability = _probabilities(space, converted)
    except ValueError as error:
        raise ValueError(str(error).replace("cell_weights", name)) from error
    if converted is not None and any(w > 0 and probability[space.cell_id(cell)] == 0 for cell, w in converted.items()):
        raise ValueError(f"{name} underflow: normalized positive weights are not representable in float64.")
    return probability


_ASSEMBLED_RISK_CONTROLS = 512
_ASSEMBLED_RISK_ENTRIES = 4_000_000


def _assembled_gram(space: BezierSpace, cid: int) -> torch.Tensor:
    """Scatter local moments, retaining cross terms of every repeated global control."""
    cell = space.cells[cid]
    indices = torch.tensor([sum(alpha, ()) for alpha in _indices(cell.dimensions, cell.degree)], dtype=torch.long)
    local = (
        _face_gram(indices, indices, tuple(range(cell.width)), cell.dimensions, cell.degree)
        if cell.dimensions
        else torch.ones(1, 1, dtype=torch.float64)
    )
    columns = torch.tensor(cell.controls, dtype=torch.long)
    gram = torch.zeros(space.n_controls, space.n_controls, dtype=torch.float64)
    gram.index_put_((columns[:, None], columns[None, :]), local, accumulate=True)
    return gram


def _assembled_risk_system(
    space: BezierSpace,
) -> tuple[list[tuple[int, tuple[int, ...]]], torch.Tensor, list[torch.Tensor], float]:
    """Build the unit triangular population system with local proper-block solves, checking roundoff."""
    stages = _skeleton_stages(space)
    p = space.n_controls
    if p > _ASSEMBLED_RISK_CONTROLS or sum(len(c.controls) ** 2 for c in space.cells) > _ASSEMBLED_RISK_ENTRIES:
        raise ValueError("assembled risk exceeds the dense limit (512 global controls or 4000000 local Gram entries).")
    width = max(cell.width for cell in space.cells)
    levels = max(sum(cell.dimensions) for cell in space.cells) + 1
    # Check local conditioning before allocating moment tables or dense matrices.
    try:
        condition = max(_gram_condition(cell.dimensions, cell.degree) for cell in space.cells)
    except OverflowError:
        condition = math.inf
    # A moment entry has relative error <= (2*width+2)*U; backward-stable local
    # solves/reductions add O(p*U), amplified by the local Gram condition number.
    # Dependencies span at most `levels` dimensions. The factor eight budgets
    # both moment/whitening paths, solve and trace rounding, and higher orders.
    scale = 8 * levels * (2 * width + p + 2) * _UNIT_ROUNDOFF
    if not scale * condition <= _RISK_RELATIVE_ERROR:
        raise ValueError("assembled risk is unreliable in float64: local Gram condition number is too large.")
    system = torch.eye(p, dtype=torch.float64)
    blocks = []
    for cid, proper in stages:
        gram = _assembled_gram(space, cid)
        h = gram[list(proper)][:, list(proper)]
        chol = torch.linalg.cholesky(h)
        system[list(proper)] = torch.cholesky_solve(gram[list(proper)], chol)
        # The proper block is mathematically the identity; assigning it exactly preserves unit triangularity.
        system[torch.tensor(proper)[:, None], torch.tensor(proper)[None, :]] = torch.eye(len(proper), dtype=torch.float64)
        blocks.append(chol)
    order = torch.tensor([i for _, proper in stages for i in proper], dtype=torch.long)
    ordered = system[order][:, order]
    inverse = torch.linalg.solve_triangular(ordered, torch.eye(p, dtype=torch.float64), upper=False, unitriangular=True)
    propagation = torch.empty_like(inverse)
    propagation[order[:, None], order[None, :]] = inverse
    # One propagation condition factor covers the unit triangular solve; a second
    # covers conversion of its normwise perturbation to the quadratic trace energy.
    # This is a conservative first-order reliability screen, not an all-degree certificate.
    bound = scale * condition * max(1.0, torch.linalg.cond(propagation).item()) ** 2
    if not math.isfinite(bound) or bound > _RISK_RELATIVE_ERROR:
        raise ValueError("assembled risk is unreliable in float64: propagated roundoff exceeds 1e-6.")
    return stages, propagation, blocks, bound


def _assembled_coefficients(space: BezierSpace, weights: torch.Tensor) -> dict[int, float]:
    """Evaluate the propagated cell traces as nonnegative squared norms under the chosen chart measure."""
    if ((weights > 0) & (weights < sys.float_info.min)).any():
        raise ValueError("assembled risk underflow: evaluation weights have insufficient float64 relative precision.")
    try:
        stages, propagation, blocks, _ = _assembled_risk_system(space)
        traces = torch.zeros(len(stages), dtype=torch.float64)
        for cid, weight in enumerate(weights):
            if not weight:
                continue
            cell = space.cells[cid]
            indices = torch.tensor([sum(a, ()) for a in _indices(cell.dimensions, cell.degree)], dtype=torch.long)
            gram = (
                _face_gram(indices, indices, range(cell.width), cell.dimensions, cell.degree)
                if cell.dimensions
                else torch.ones(1, 1, dtype=torch.float64)
            )
            root = torch.linalg.cholesky(gram).T
            for k, ((_, proper), block) in enumerate(zip(stages, blocks, strict=True)):
                evaluated = root @ propagation[list(cell.controls)][:, list(proper)]
                whitened = torch.linalg.solve_triangular(block, evaluated.T, upper=False)
                unweighted = whitened.square().sum()
                contribution = weight * unweighted
                if unweighted > 0 and contribution < sys.float_info.min:
                    raise ValueError("assembled risk underflow: a positive trace has insufficient float64 relative precision.")
                traces[k] += contribution
        if not torch.isfinite(traces).all():
            raise ValueError("assembled risk is not finite in float64.")
        return {cid: value for (cid, _), value in zip(stages, traces.tolist(), strict=True)}
    except torch.linalg.LinAlgError as error:
        raise ValueError("assembled risk is unreliable in float64: a Gram factorization failed.") from error


def _cell_risk_coefficients(space: BezierSpace, weights: torch.Tensor) -> dict[int, float]:
    """Use proved closed forms when applicable; otherwise retain the dense identified-set algorithm."""
    if ((weights > 0) & (weights < sys.float_info.min)).any():
        raise ValueError("assembled risk underflow: evaluation weights have insufficient float64 relative precision.")
    n_factors = max((len(cell.dimensions) for cell in space.cells), default=1)

    def factor_coefficients(a: int, d: int) -> tuple[float, ...]:
        """Bound compounded factor errors and positive chart accumulation before using incidence products."""
        n_controls = math.comb(a + d, d)
        try:
            condition = _gram_condition((a,), (d,))
        except OverflowError:
            condition = math.inf
        bound = ((a + 1) * (2 * (a + 1) + 2 + n_controls) * condition + n_controls) * _UNIT_ROUNDOFF
        # Factor relative errors add in products; four budgets the subsequent
        # divisions, products, positive reductions and higher-order terms.
        if 4 * (n_factors * bound + space.n_cells * _UNIT_ROUNDOFF) > _RISK_RELATIVE_ERROR:
            raise ValueError("incidence risk is unreliable in float64: factor roundoff exceeds 1e-6.")
        if n_controls > _ASSEMBLED_RISK_CONTROLS:
            raise ValueError("incidence risk exceeds the dense limit of 512 controls per simplex factor.")
        factor = BezierSpace.simplex(a, degree=d)
        return skeleton_risk_coefficients(factor)

    closed = _closed_risk_coefficients(space, weights, factor_coefficients, _ASSEMBLED_RISK_ENTRIES)
    return _assembled_coefficients(space, weights) if closed is None else closed


@overload
def skeleton_risk_coefficients(  # noqa: D418 - numpydoc validates overload docstrings.
    space: BezierSpace,
    *,
    evaluation_weights: _CellWeights | None = None,
    allocation: Literal["dimension"] = "dimension",
) -> tuple[float, ...]:
    """Risk coefficients indexed by dimension.

    Parameters
    ----------
    space
        Compiled simploidal space.
    evaluation_weights
        Optional nonnegative chart weights for risk evaluation.
    allocation
        Select dimension-indexed results.

    Returns
    -------
    tuple of float
        Risk coefficients for the selected allocation.
    """
    ...


@overload
def skeleton_risk_coefficients(  # noqa: D418 - numpydoc validates overload docstrings.
    space: BezierSpace,
    *,
    evaluation_weights: _CellWeights | None = None,
    allocation: Literal["cell"],
) -> dict[int, float]:
    """Risk coefficients indexed by cell.

    Parameters
    ----------
    space
        Compiled simploidal space.
    evaluation_weights
        Optional nonnegative chart weights for risk evaluation.
    allocation
        Select cell-indexed results.

    Returns
    -------
    dict of int to float
        Risk coefficients for the selected allocation.
    """
    ...


def skeleton_risk_coefficients(
    space: BezierSpace,
    *,
    evaluation_weights: _CellWeights | None = None,
    allocation: Literal["dimension", "cell"] = "dimension",
) -> tuple[float, ...] | dict[int, float]:
    r"""Asymptotic risk coefficients of inductive skeleton fitting on a simploidal space.

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
    :func:`sample_skeleton` does with ``allocation="dimension"``. Observations and the error are uniform on
    the product.

    Parameters
    ----------
    space
        A simplex, simploid or assembled set with positive degrees in positive-dimensional factors.

    evaluation_weights
        Nonnegative chart weights by cell ID or builder handle, normalized to a
        probability measure; omitted cells have zero weight. Defaults to equal
        weights on maximal cells. ID/handle aliases add. Unknown cells, nonfinite
        or negative weights and a nonfinite or zero total are rejected.
    allocation
        ``"dimension"`` returns the usual tuple; ``"cell"`` returns a dictionary
        of active cell IDs to coefficients and requires addressable faces via
        ``BezierSpace.on_set`` (or a zero-dimensional space).

    Returns
    -------
    tuple of float or dict of int to float
        :math:`a_0, \dots, a_n` for a domain of dimension :math:`n`, which is
        :math:`M - 1` for a simplex with :math:`M` vertices.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If a positive-dimensional factor has degree zero, weights are invalid,
        numerical reliability checks fail, or a computation limit is exceeded.
        For the original bare simplex algorithm, the largest accepted degree
        falls with the dimension: 14 for 2
        vertices, 12 for 3, 11 for 4, 9 for 5, 8 for 6 and 7, and 7 for 8. On
        a bare simploid, the condition number of the Gram matrix on the whole
        simploid is the product of those of its factors' Gram matrices, so it
        also grows with the number of factors: 12 segments of degree 1 are
        rejected.

    Notes
    -----
    On an assembled set, cell coefficients satisfy ``risk = sum(c_s / q_s)``
    up to the common noise/output/budget factor. Dimension coefficients are
    ``a_m = K_m * sum(c_s for dim(s) == m)``. The evaluator is independent of
    sampling allocation and may include lower cells. The model must be correctly
    specified, with independent homoscedastic noise, uniform within-cell designs
    and unregularized least squares. Graphs, linear factor degrees and regular
    complexes use the closed graph, corner-collision and weighted-incidence
    formulas, without global dense matrices. Incidences retain conditioned
    factor computations with at most 512 controls per factor. Nonlinear
    identified sets retain dense propagation, limited to 512 global controls
    and 4,000,000 local Gram entries, with local/propagated conditioning checks. Unrepresentable
    positive weights and subnormal weighted risk contributions raise a numerical
    reliability error instead of being classified as mathematical zeros. Zero coefficients
    do not remove active cells from the estimator. See :ref:`skeleton-risk-assembled`.

    On a face of a single simploid, the Gram matrices and the propagated errors are
    Kronecker products of those of its factor faces, so a face's trace
    :math:`\mathrm{tr}(G^I_{TT}{}^{-1} E_I^\top G E_I)` is the product of
    theirs. The coefficients are therefore computed factor by factor:
    :math:`a_m = K_m\, [x^m] \prod_j p_j(x)`, where :math:`K_m` is the number of
    faces of dimension :math:`m` with interior controls and the coefficient of
    :math:`x^s` in :math:`p_j` sums the traces of the :math:`s`-faces of factor
    :math:`j` on that simplex. Vertex permutation symmetry permits one
    representative noise source per face dimension, propagated to all controls.
    Matrix products are batched by dimension and propagation uses only noises
    in each face closure. The dense Gram matrix still needs quadratic memory
    in a factor's :math:`\binom{D + k}{D}` controls. At fixed degree, its
    construction and the trace contractions take quadratic rather than cubic
    time in that count, apart from the coordinate and face bookkeeping.
    The number of controls of the product does not enter these matrices.

    A simplex is its own only factor. The condition number of a
    factor's Gram matrix is :math:`\binom{2D + k}{D}` in closed form, so a
    space that the conditioning check rejects is rejected before anything of
    the size of a factor is allocated. Accepted wide simplices remain costly:
    the widest ones, such as :math:`M = 22` vertices (:math:`k = 21`) at
    degree 4 with 12650 controls, still need a dense float64 Gram matrix of
    1.3 GB and similarly sized temporaries during its construction. If an allocation fails, PyTorch raises ``RuntimeError``; on
    systems that overcommit memory, such as Linux by default, the process may
    be terminated instead.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import skeleton_risk_coefficients
    >>> [round(a, 4) for a in skeleton_risk_coefficients(BezierSpace.simplex(2, degree=3))]
    [0.1464, 3.5357, 1.0]
    """
    single = _single_skeleton(space)
    _choice(allocation, "allocation", ("dimension", "cell"))
    weights = _evaluation_probabilities(space, evaluation_weights)
    if not single or allocation == "cell":
        coefficients = _cell_risk_coefficients(space, weights)
        if allocation == "cell":
            return coefficients
        levels = max(sum(c.dimensions) for c in space.cells) + 1
        groups = [[v for cid, v in coefficients.items() if sum(space.cells[cid].dimensions) == m] for m in range(levels)]
        return tuple(len(values) * math.fsum(values) for values in groups)
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
        # On a simplex, this is the computation below. On K >= 2 factors, the computation runs on each factor, which
        # has at most as many face dimensions, coordinates and controls as the product and condition numbers of at
        # most kappa / 3 ** (K - 1), since every factor's Gram matrix has a condition number of at least
        # C(2 + a, 1) >= 3. The relative errors of the K factor traces of a face then add up to at most
        # K / 3 ** (K - 1) <= 2 / 3 of the first term, and the other third covers the roundings of the sums over
        # faces and of the products of factor traces.
        bound = (levels * (2 * width + 2 + n_controls) * kappa + n_controls) * _UNIT_ROUNDOFF
        if not bound <= _RISK_RELATIVE_ERROR:
            raise ValueError(f"degree is too large for the risk coefficients in float64 (condition number {kappa:.1e}).")

    # Reject hopeless spaces (high degrees, or many factors on a simploid) before anything of the size of a factor is
    # allocated: the largest denominator must fit in float64, which also bounds the closed form of _gram_condition;
    # the small Gram block of the controls interior to one edge of the factor of highest degree, among those whose
    # largest condition number enters the bound, must already pass it; and so must the Gram matrix on the whole
    # simploid, whose condition number has a closed form (see _gram_condition) that grows with the number of factors.
    # In exact arithmetic, the edge block passes whenever the closed form does (see the check after the computation);
    # it is checked first so that a space rejected at a high degree reports the condition number of that block.
    factors = list(zip(dimensions, degree, strict=True))
    if math.prod(math.comb(2 * d, d) * math.comb(2 * d + a, a) for a, d in factors) > sys.float_info.max:
        check(math.inf)
    top = max(degree)
    if top >= 2:
        edge = torch.tensor([[j, top - j] for j in range(1, top)], dtype=torch.long)
        check(torch.linalg.cond(_face_gram(edge, edge, (0, 1), (1,), (top,))).item())
    full_condition = _gram_condition(dimensions, degree)
    check(full_condition)
    # On a face I = I_1 x ... x I_K, the Gram matrices under the uniform distribution and the errors that eta_I passes
    # on (fixed by the orthogonality of the error on every larger face to its interior controls) are, up to the
    # order of the controls, Kronecker products of those of the factor faces I_j on their simplices. The face has
    # interior controls exactly when every I_j does, and its trace and the condition number of its block G_TT are the
    # products of theirs. The sum of the traces of the faces of dimension m is then the coefficient of x^m in the
    # product of the factors' polynomials of trace sums by face dimension.
    trace_sums, block_condition = [1.0], 1.0
    for a, d in factors:
        sums, condition = _face_trace_sums((a,), (d,))
        trace_sums = np.convolve(trace_sums, sums).tolist()
        block_condition *= condition
    # In exact arithmetic, each block G_TT is a principal submatrix of the Gram matrix on its face and no worse
    # conditioned (Cauchy interlacing), and the closed form of _gram_condition grows with the face's dimension in every
    # factor, so no block is worse conditioned than the Gram matrix on the whole simploid; this check guards rounding.
    check(max(full_condition, block_condition))
    # The N^(m) rows are split equally among the K_m faces of dimension m with interior controls, so n_I = N^(m) / K_m
    # and a_m is K_m times the sum of the traces. Levels without interior controls have no traces; skipping them also
    # skips the count of all their faces, which exceeds float64 on wide simplices of degree 1.
    return tuple(_n_skeleton_faces(dimensions, degree, m) * t if t else 0.0 for m, t in enumerate(trace_sums))


def optimal_skeleton_ratio(
    space: BezierSpace,
    *,
    evaluation_weights: _CellWeights | None = None,
    allocation: Literal["auto", "dimension", "cell"] = "auto",
) -> SkeletonAllocation:
    r"""Choose a skeleton allocation, preferring the individual-cell asymptotic optimum.

    Parameters
    ----------
    space
        Compiled simplex, simploid or assembled domain with positive factor degrees.
    evaluation_weights
        Risk measure, normalized independently of sampling shares as in
        :func:`skeleton_risk_coefficients`.
    allocation
        ``"auto"`` prefers cell shares proportional to ``sqrt(c_s)``. It selects
        equal-within-dimension shares only when zero-risk active cells prevent
        a cell minimum but every active dimension has positive coefficient.
        Bare simplex/simploid constructors directly use their original fast
        dimension path. ``"cell"`` and ``"dimension"`` select a strict method
        without fallback. Cell mode requires addressable faces via ``on_set``.

    Returns
    -------
    SkeletonAllocation
        One stable result type for every strategy. Inspect ``strategy`` and
        ``fallback_reason``; pass the whole result as ``sample_skeleton``'s
        ``allocation`` argument to preserve within-dimension fractions and
        the corresponding rounding rule. ``ratio`` gives dimension totals;
        ``cell_weights`` gives active-cell fractions on addressable spaces.

    Raises
    ------
    TypeError
        If ``space`` is not a BezierSpace.
    ValueError
        For invalid inputs, unreliable numerical evaluation, work-limit errors,
        or a zero coefficient in an active dimension. Strict cell allocation
        also rejects zero-coefficient active cells. Zero-coefficient cases
        describe a mathematical infimum, not an attained positive-allocation optimum.
        Numerical and input errors never trigger fallback.

    Examples
    --------
    >>> from torch_bsf import BezierSpace, SimploidalSet
    >>> from torch_bsf.sampling import optimal_skeleton_ratio, sample_skeleton
    >>> domain = SimploidalSet()
    >>> _ = domain.add_simplex(1)
    >>> space = BezierSpace.on_set(domain, degree=2)
    >>> plan = optimal_skeleton_ratio(space)
    >>> plan.strategy, plan.fallback_reason
    ('cell', None)
    >>> [round(r, 4) for r in plan.ratio]
    [0.4142, 0.5858]
    >>> len(sample_skeleton(space, 8, allocation=plan, seed=0))
    8
    """
    single = _single_skeleton(space)
    _choice(allocation, "allocation", ("auto", "dimension", "cell"))
    if single and allocation != "cell":
        coefficients = skeleton_risk_coefficients(space, evaluation_weights=evaluation_weights)
        roots = [math.sqrt(a) for a in coefficients]
        ratio = tuple(r / sum(roots) for r in roots)
        return SkeletonAllocation(space, "dimension", ratio, {}, "single_cell_fast_path" if allocation == "auto" else None)
    cells = skeleton_risk_coefficients(space, evaluation_weights=evaluation_weights, allocation="cell")
    stages = list(cells)
    levels = max(sum(cell.dimensions) for cell in space.cells) + 1
    groups = [[cid for cid in stages if sum(space.cells[cid].dimensions) == m] for m in range(levels)]
    grouped = [len(group) * math.fsum(cells[cid] for cid in group) for group in groups]
    zero_cells = any(c <= 0 for c in cells.values())
    if allocation == "cell" and zero_cells:
        raise ValueError(
            "zero-risk active cells have only an infimum under positive allocation; pass cell_weights explicitly."
        )
    if allocation != "dimension" and not zero_cells:
        roots_by_cell = {cid: math.sqrt(c) for cid, c in cells.items()}
        total = math.fsum(roots_by_cell.values())
        shares = {cid: root / total for cid, root in roots_by_cell.items()}
        ratio = tuple(math.fsum(shares[cid] for cid in group) for group in groups)
        return SkeletonAllocation(space, "cell", ratio, shares)
    if any(group and coefficient <= 0 for group, coefficient in zip(groups, grouped, strict=True)):
        raise ValueError("zero-risk active groups have only an infimum under positive allocation; pass ratio explicitly.")
    roots = [math.sqrt(a) for a in grouped]
    ratio = tuple(root / math.fsum(roots) for root in roots)
    shares = {cid: ratio[m] / len(group) for m, group in enumerate(groups) for cid in group}
    return SkeletonAllocation(space, "dimension", ratio, shares, "zero_cell_coefficients" if allocation == "auto" else None)


def _real_entries(ratio: object) -> bool:
    """Whether a non-tensor ratio is a NumPy array or a sequence of real, non-boolean numbers."""
    if isinstance(ratio, np.ndarray):
        return ratio.dtype.kind in "iuf"

    def real(entry: object) -> bool:
        if isinstance(entry, torch.Tensor):
            return entry.ndim == 0 and entry.dtype != torch.bool and not entry.is_complex()
        return isinstance(entry, numbers.Real) and not isinstance(entry, bool)

    return isinstance(ratio, Sequence) and not isinstance(ratio, str) and all(real(entry) for entry in ratio)


def _skeleton_fractions(ratio: Sequence[float] | torch.Tensor | None, levels: int) -> list[float]:
    """Normalize explicit dimension shares after the allocation policy has been resolved."""
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
    evaluation_weights: _CellWeights | None = None,
    cell_weights: _CellWeights | None = None,
    allocation: Literal["auto", "dimension", "cell"] | SkeletonAllocation = "auto",
) -> Points:
    """Stratified sample of a simploidal space for inductive skeleton fitting.

    Automatic allocation prefers individual-cell optimal shares and falls back
    to equal-within-dimension shares only when zero-risk cells can be rescued.
    Explicit ``ratio`` or dimension allocation first rounds each dimension's
    budget and splits it equally among active faces, with random remainder
    ties. Cell allocation uses one largest-remainder rounding in fitting order. Rows are uniform on
    their face, with exact zeros outside it, and are ordered by face dimension.

    A face of a simploid is a product of one face per simplex factor, and its
    rows are uniform on each factor's face. The faces of one skeleton can differ
    in shape: the edges of a prism are triangle edges at an end of the interval
    and intervals at a vertex of the triangle. A skeleton's rows then go only to
    its faces with interior controls (no factor has more vertices than its
    degree). For the original single-cell constructors, an explicit ratio may
    also sample a dimension with no interior controls, using all its faces;
    fitting ignores those rows. Assembled spaces sample active cells only and
    reject positive shares on dimensions without any active cells. On a single
    simplex, either every face of a skeleton has interior controls or none has.

    Parameters
    ----------
    space
        A simplex, simploid or assembled set with positive degrees in positive-dimensional factors.
    n
        Number of rows.
    ratio
        Relative sizes of the 0-, 1-, ..., n-skeletons of a domain of
        dimension n: n + 1 finite nonnegative real entries with a positive sum,
        that is, M entries for a simplex with M vertices. When omitted,
        ``allocation`` chooses the automatic plan, with no rows on skeletons
        without interior controls. Rows on faces without interior controls
        (with more vertices than the degree in some factor) are not used by
        :func:`~torch_bsf.fitting.fit_skeleton_regression`.
    method
        ``"random"`` or ``"sobol"`` coordinates within each face, as in :func:`sample`.
    seed
        Seed of the local generator; global random state is not used.

    evaluation_weights
        Evaluation measure used for automatic optimization. Validated
        independently of explicit sampling shares; see :func:`skeleton_risk_coefficients`.
    cell_weights
        Explicit active-cell shares by cell ID or builder handle, normalized to
        probabilities. Mutually exclusive with ``ratio``. Requires addressable
        faces (``BezierSpace.on_set`` or a zero-dimensional space). Pass the
        ``cell_weights`` field from :func:`optimal_skeleton_ratio` here.

    allocation
        ``"auto"`` (default), strict ``"cell"`` or strict ``"dimension"``
        strategy, or a reusable :class:`~torch_bsf.SkeletonAllocation` from
        :func:`optimal_skeleton_ratio`. Plans cannot be combined with explicit
        ``ratio`` or ``cell_weights`` and must match the space. Explicit shares
        override automatic optimization; contradictory strict methods raise.

    Returns
    -------
    Points
        Exactly ``n`` CPU float64 rows. Assembled rows carry active nondegenerate
        cell IDs and zero padding. Largest-remainder rounding preserves the
        budget, even for ``n=0`` or budgets below the active-cell count; it does
        not guarantee sufficient rows or rank for fitting. Dimension shares are
        split equally within each dimension, with randomized remainder ties.
        Pass a plan to inspect its strategy and fallback reason before sampling.
        On assembled spaces, positive shares on empty dimensions or inactive cells are rejected.
        Explicit shares bypass risk conditioning and dense-computation limits.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If a positive-dimensional factor has degree zero, ``n`` is not a nonnegative integer,
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
    single = _single_skeleton(space)
    _integer(n, "n")
    _choice(method, "method", ("random", "sobol"), message="method must be 'random' or 'sobol'.")
    if evaluation_weights is not None:
        _evaluation_probabilities(space, evaluation_weights)
    ratio, cell_weights = _sampling_allocation(space, ratio, cell_weights, allocation, evaluation_weights)
    if not single or cell_weights is not None:
        return _sample_assembled_skeleton(space, n, ratio, method, seed, evaluation_weights, cell_weights)
    cell = _skeleton_cell(space)
    counts = _skeleton_counts(n, _skeleton_fractions(ratio, sum(cell.dimensions) + 1))
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


def _sampling_allocation(
    space: BezierSpace,
    ratio: Sequence[float] | torch.Tensor | None,
    cell_weights: _CellWeights | None,
    allocation: Literal["auto", "dimension", "cell"] | SkeletonAllocation,
    evaluation_weights: _CellWeights | None,
) -> tuple[Sequence[float] | torch.Tensor | None, _CellWeights | None]:
    """Resolve an explicit plan or automatic policy, without silently overriding explicit shares."""
    if isinstance(allocation, SkeletonAllocation):
        if ratio is not None or cell_weights is not None:
            raise ValueError("allocation plans and explicit ratio/cell_weights are mutually exclusive.")
        if not space.compatible(allocation.space):
            raise ValueError("allocation plan belongs to an incompatible space.")
        _choice(allocation.strategy, "strategy", ("cell", "dimension"))
        return (None, allocation.cell_weights) if allocation.strategy == "cell" else (allocation.ratio, None)
    _choice(allocation, "allocation", ("auto", "dimension", "cell"))
    if ratio is not None or cell_weights is not None:
        if allocation == "cell" and ratio is not None:
            raise ValueError("ratio is a dimension allocation; select allocation='dimension' or 'auto'.")
        if allocation == "dimension" and cell_weights is not None:
            raise ValueError("cell_weights require allocation='cell' or 'auto'.")
        return ratio, cell_weights
    try:
        plan = optimal_skeleton_ratio(space, evaluation_weights=evaluation_weights, allocation=allocation)
    except ValueError as error:
        raise ValueError(f"{error} Pass ratio explicitly or supply cell_weights.") from error
    return (None, plan.cell_weights) if plan.strategy == "cell" else (plan.ratio, None)


def _assembled_counts(
    space: BezierSpace,
    stages: list[tuple[int, tuple[int, ...]]],
    n: int,
    ratio: Sequence[float] | torch.Tensor | None,
    evaluation_weights: _CellWeights | None,
    cell_weights: _CellWeights | None,
    generator: torch.Generator,
) -> list[int]:
    """Round either explicit cell shares or dimension shares followed by equal within-dimension shares."""
    if cell_weights is not None:
        if ratio is not None:
            raise ValueError("ratio and cell_weights are mutually exclusive.")
        probabilities = _evaluation_probabilities(space, cell_weights, "cell_weights")
        active = {cid for cid, _ in stages}
        if any(w > 0 and cid not in active for cid, w in enumerate(probabilities)):
            raise ValueError("cell_weights may assign positive mass only to cells with proper controls.")
        return _skeleton_counts(n, [probabilities[cid].item() for cid, _ in stages])
    levels = max(sum(c.dimensions) for c in space.cells) + 1
    fractions = _skeleton_fractions(ratio, levels)
    groups = [[k for k, (cid, _) in enumerate(stages) if sum(space.cells[cid].dimensions) == m] for m in range(levels)]
    if any(f > 0 and not group for f, group in zip(fractions, groups, strict=True)):
        raise ValueError("ratio assigns mass to a dimension without active cells.")
    counts = [0] * len(stages)
    for group, count in zip(groups, _skeleton_counts(n, fractions), strict=True):
        if not group or not count:
            continue
        extra = set(torch.randperm(len(group), generator=generator)[: count % len(group)].tolist())
        for j, k in enumerate(group):
            counts[k] = count // len(group) + int(j in extra)
    return counts


def _sample_assembled_skeleton(
    space: BezierSpace,
    n: int,
    ratio: Sequence[float] | torch.Tensor | None,
    method: str,
    seed: int | None,
    evaluation_weights: _CellWeights | None,
    cell_weights: _CellWeights | None,
) -> Points:
    """Draw exactly n rows, tagged with their active nondegenerate cells and padded to the space width."""
    stages = _skeleton_stages(space)
    generator = _generator(seed)
    counts = _assembled_counts(space, stages, n, ratio, evaluation_weights, cell_weights, generator)
    parameters = torch.zeros(n, space.width, dtype=torch.float64)
    ids = torch.empty(n, dtype=torch.long)
    offset = 0
    for (cid, _), count in zip(stages, counts, strict=True):
        cell = space.cells[cid]
        if count and cell.width:
            exponentials = _exponentials(generator, count, cell.width, method)
            for block in _blocks(cell.dimensions):
                t = exponentials[:, block]
                parameters[offset : offset + count, block] = t / t.sum(1, keepdim=True)
        ids[offset : offset + count] = cid
        offset += count
    return Points(parameters, ids)
