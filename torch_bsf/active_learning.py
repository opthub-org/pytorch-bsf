"""Acquisition functions with explicit, mathematically relevant inputs."""

from __future__ import annotations

import torch

from torch_bsf.core import Points, _integer
from torch_bsf.ensemble import Ensemble


def query_by_committee(ensemble: Ensemble, candidates: Points, n: int = 1) -> Points:
    """Select the largest predictive disagreement from supplied domain points."""
    _integer(n, "n", 1)
    if n > len(candidates):
        raise ValueError("n exceeds the number of candidates.")
    with torch.no_grad():
        disagreement = ensemble.members(candidates).var(0, unbiased=False).mean(1)
    return candidates.take(torch.argsort(disagreement, descending=True, stable=True)[:n])


def farthest_points(candidates: Points, observed: Points, n: int = 1, *, chunk_size: int = 1024) -> Points:
    """Greedily cover parameter coordinates, comparing distances within cells.

    Unobserved cells are selected first. No dummy ensemble or duplicated
    geometry/dimension arguments are required for this density acquisition.
    Equal distances select the earliest remaining candidate. ``chunk_size``
    bounds both axes of each temporary distance matrix, using at most
    ``chunk_size ** 2`` float64 distances. Smaller chunks reduce memory at
    the cost of more distance-kernel calls; arithmetic work is unchanged.
    Distances use direct coordinate differences to retain small separations
    between nearby points, independently of the candidate/observation count.

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
            for batch in rows.split(chunk_size):
                nearest = distances[batch]
                for observations in other.split(chunk_size):
                    # Squared-norm expansion loses small separations through
                    # cancellation. Always use coordinate differences, even
                    # above PyTorch's 25-row matrix-multiplication threshold.
                    block = torch.cdist(coordinates[batch], observations, compute_mode="donot_use_mm_for_euclid_dist")
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
