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


def farthest_points(candidates: Points, observed: Points, n: int = 1) -> Points:
    """Greedily cover parameter coordinates, comparing distances within cells.

    Unobserved cells are selected first. No dummy ensemble or duplicated
    geometry/dimension arguments are required for this density acquisition.
    """
    _integer(n, "n", 1)
    if n > len(candidates) or observed.parameters.shape[1] != candidates.parameters.shape[1]:
        raise ValueError("candidate count or coordinate width is invalid.")
    coordinates = candidates.parameters.cpu().double()
    selected: list[int] = []
    existing = observed.parameters.cpu().double()
    existing_ids = observed.cell_ids.cpu()
    distances = torch.full((len(candidates),), float("inf"), dtype=torch.float64)
    for cid in torch.unique(candidates.cell_ids).tolist():
        rows = torch.nonzero(candidates.cell_ids.cpu() == cid).flatten()
        other = existing[existing_ids == cid]
        if len(other):
            distances[rows] = torch.cdist(coordinates[rows], other).min(1).values
    for _ in range(n):
        index = int(distances.argmax())
        selected.append(index)
        same = candidates.cell_ids.cpu() == candidates.cell_ids[index].item()
        distances[same] = torch.minimum(distances[same], (coordinates[same] - coordinates[index]).norm(dim=1))
        distances[selected] = -1
    return candidates.take(torch.tensor(selected))
