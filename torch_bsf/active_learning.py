"""Acquisition functions with explicit, mathematically relevant inputs."""

from __future__ import annotations

from fractions import Fraction

import torch

from torch_bsf.core import Points, _integer
from torch_bsf.ensemble import Ensemble


def _exact_disagreement(predictions: torch.Tensor) -> Fraction:
    """Return K squared times the variance sum, exactly for supplied floats."""
    score = Fraction(0)
    for output in predictions.T.tolist():
        values = [Fraction(value) for value in output]
        total = sum(values, Fraction(0))
        score += len(values) * sum((value * value for value in values), Fraction(0)) - total * total
    return score


def _resolve_close_disagreements(
    predictions: torch.Tensor, order: torch.Tensor, mantissas: torch.Tensor, exponents: torch.Tensor
) -> torch.Tensor:
    """Resolve groups whose approximate scores differ by reduction roundoff."""
    values = mantissas[order]
    powers = exponents[order]
    aligned = values[1:] * torch.pow(values.new_tensor(2.0), powers[1:] - powers[:-1])
    # Generous forward-error allowance for centering, variance, and output
    # summation; exact comparisons also retain terms below float64 precision.
    tolerance = 64 * torch.finfo(torch.float64).eps * predictions.shape[0] * predictions.shape[2]
    close = ((values[:-1] - aligned) <= tolerance * values[:-1]) & (values[1:] != 0)
    if not close.any():
        return order
    indices = order.tolist()
    start = 0
    for end in range(1, len(indices) + 1):
        if end < len(indices) and close[end - 1]:
            continue
        if end - start > 1:
            indices[start:end] = sorted(
                indices[start:end], key=lambda index: (-_exact_disagreement(predictions[:, index]), index)
            )
        start = end
    return torch.tensor(indices, dtype=torch.int64)


def _disagreement_order(predictions: torch.Tensor) -> torch.Tensor:
    """Compare variance sums without materializing potentially unrepresentable scores."""
    predictions = predictions.detach().cpu().double()
    if not torch.isfinite(predictions).all():
        raise ValueError("committee predictions must be finite.")
    values, exponents = torch.frexp(predictions)
    _, scale = torch.frexp(predictions.abs().amax(dim=0))
    # Normalize each candidate/output by an exact power of two. Its exponent
    # is restored below, so this does not change relative output weights.
    shifts = torch.where(values == 0, 0, exponents - scale)
    normalized = values * torch.pow(values.new_tensor(2.0), shifts)
    # Remove a reference first: identical members must have exactly zero
    # disagreement, including large common offsets and subnormal predictions.
    variance = (normalized - normalized[:1]).var(dim=0, unbiased=False)
    mantissas, adjustments = torch.frexp(variance)
    exponents = 2 * scale + adjustments
    # A zero term must not set the scale of a tiny, nonzero output variance.
    exponents = torch.where(mantissas == 0, -10000, exponents)
    largest = exponents.amax(dim=1)
    totals = (mantissas * torch.pow(mantissas.new_tensor(2.0), exponents - largest[:, None])).sum(dim=1)
    mantissas, adjustments = torch.frexp(totals)
    exponents = torch.where(totals == 0, -10000, largest + adjustments)
    # The output count is common to every candidate, so ranking variance sums
    # also ranks their means. Stable lexicographic sorting preserves ties.
    order = torch.argsort(mantissas, descending=True, stable=True)
    order = order[torch.argsort(exponents[order], descending=True, stable=True)]
    return _resolve_close_disagreements(predictions, order, mantissas, exponents)


def query_by_committee(ensemble: Ensemble, candidates: Points, n: int = 1) -> Points:
    """Select candidates by mean population variance across committee outputs.

    Ties preserve candidate order. Finite predictions are scored on CPU in float64
    with exponent scaling to avoid variance overflow and underflow. This
    preserves relative output units and common-scalar ranking invariance,
    subject to rounding in the supplied predictions. Nearly tied nonzero
    scores use exact rational comparisons of those predictions, at additional
    CPU cost. Nonfinite predictions
    raise ``ValueError``. Returned points retain their input gradients.
    """
    _integer(n, "n", 1)
    if n > len(candidates):
        raise ValueError("n exceeds the number of candidates.")
    with torch.no_grad():
        order = _disagreement_order(ensemble.members(candidates))
    return candidates.take(order[:n])


def farthest_points(candidates: Points, observed: Points, n: int = 1, *, chunk_size: int = 1024) -> Points:
    """Greedily cover parameter coordinates, comparing distances within cells.

    Unobserved cells are selected first. No dummy ensemble or duplicated
    geometry/dimension arguments are required for this density acquisition.
    Equal distances select the earliest remaining candidate. ``chunk_size``
    bounds both axes of each temporary distance matrix, using at most
    ``chunk_size ** 2`` float64 distances. Smaller chunks reduce memory at
    the cost of more distance-kernel calls; arithmetic work is unchanged.

    Parameters
    ----------
    candidates : Points
        Candidate parameter coordinates with their cell IDs.
    observed : Points
        Existing observations in the same coordinate layout.
    n : int, default=1
        Number of distinct candidate rows to select.
    chunk_size : int, default=1024
        Positive upper bound on each distance-matrix dimension.

    Returns
    -------
    Points
        Selected candidates in greedy order, retaining input gradients.

    Examples
    --------
    >>> from torch_bsf import Points
    >>> from torch_bsf.active_learning import farthest_points
    >>> farthest_points(Points([[0.0], [1.0]]), Points([[0.0]]), chunk_size=1).parameters.tolist()
    [[1.0]]
    """
    _integer(n, "n", 1)
    _integer(chunk_size, "chunk_size", 1)
    if n > len(candidates) or observed.parameters.shape[1] != candidates.parameters.shape[1]:
        raise ValueError("candidate count or coordinate width is invalid.")
    # Selection indices are discrete. Keep no distance graph across chunks;
    # the returned Points still gather the original differentiable parameters.
    coordinates = candidates.parameters.detach().cpu().double()
    selected: list[int] = []
    existing = observed.parameters.detach().cpu().double()
    existing_ids = observed.cell_ids.cpu()
    candidate_ids = candidates.cell_ids.cpu()
    distances = torch.full((len(candidates),), float("inf"), dtype=torch.float64)
    for cid in torch.unique(candidate_ids).tolist():
        rows = torch.nonzero(candidate_ids == cid).flatten()
        other = existing[existing_ids == cid]
        if len(other):
            # Retain the dense call's algorithm choice even for tiny chunks:
            # PyTorch otherwise switches at 25 rows, changing roundoff/ties.
            mode = "use_mm_for_euclid_dist" if len(rows) > 25 or len(other) > 25 else "donot_use_mm_for_euclid_dist"
            for batch in rows.split(chunk_size):
                nearest = distances[batch]
                for observations in other.split(chunk_size):
                    block = torch.cdist(coordinates[batch], observations, compute_mode=mode)
                    nearest = torch.minimum(nearest, block.min(1).values)
                distances[batch] = nearest
    for _ in range(n):
        index = int(distances.argmax())
        selected.append(index)
        rows = torch.nonzero(candidate_ids == candidate_ids[index]).flatten()
        for batch in rows.split(chunk_size):
            distance = (coordinates[batch] - coordinates[index]).norm(dim=1)
            distances[batch] = torch.minimum(distances[batch], distance)
        distances[selected] = -1
    return candidates.take(torch.tensor(selected))
