"""Closed assembled risk formulas from graph, incidence and corner-collision theorems."""

import math
import sys
from collections.abc import Callable

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._skeleton import _skeleton_stages
from torch_bsf.core import BezierSpace


def _risk_contribution(weight: float, trace: float) -> float:
    """Retain structural zeros, rejecting positive contributions without normal float64 precision."""
    value = weight * trace
    if not math.isfinite(value):
        raise ValueError("assembled risk is not finite in float64.")
    if trace < 0:
        raise ValueError("closed risk is unreliable in float64: a negative trace was computed.")
    if weight > 0 and trace > 0 and value < sys.float_info.min:
        raise ValueError("assembled risk underflow: a positive trace has insufficient float64 relative precision.")
    return value


def _graph_coefficients(space: BezierSpace, weights: list[float], owners: dict[int, int]) -> dict[int, float]:
    """Graph endpoint norms, loop cross terms and proper-edge ranks, without a matrix solve."""
    contributions: dict[int, list[float]] = {cid: [] for cid in owners.values()}
    for cid, (cell, weight) in enumerate(zip(space.cells, weights, strict=True)):
        if not weight:
            continue
        if not cell.dimensions:
            contributions[cid].append(weight)
            continue
        degree = cell.degree[0]
        left, right = owners[cell.controls[0]], owners[cell.controls[-1]]
        if left == right:
            loop = 2 / ((degree + 1) * (degree + 1 + (-1) ** degree))
            contributions[left].append(_risk_contribution(weight, loop))
        else:
            endpoint = 1 / (degree * (degree + 2))
            for vertex in (left, right):
                contributions[vertex].append(_risk_contribution(weight, endpoint))
        if degree > 1:
            contributions[cid].append(_risk_contribution(weight, degree - 1))
    return {cid: math.fsum(values) for cid, values in contributions.items()}


def _linear_coefficients(
    space: BezierSpace, weights: list[float], owners: dict[int, int], pair_limit: int
) -> dict[int, float]:
    """Sum Dirichlet corner-pair moments, including repeated corners and collapsed boundaries."""
    contributions: dict[int, list[float]] = {cid: [] for cid in owners.values()}
    for cell, weight in zip(space.cells, weights, strict=True):
        if not weight:
            continue
        fibers: dict[int, list[int]] = {}
        for i, control in enumerate(cell.controls):
            fibers.setdefault(control, []).append(i)
        if len(fibers) == 1:
            contributions[owners[cell.controls[0]]].append(weight)
            continue  # The assembled basis is identically one, even with many identified corners.
        if sum(len(fiber) ** 2 for fiber in fibers.values()) > pair_limit:
            raise ValueError("linear risk exceeds the local corner-pair limit; pass explicit sampling shares.")
        corners = [tuple(alpha.index(1) for alpha in index) for index in _indices(cell.dimensions, cell.degree)]
        denominator = math.prod((a + 1) * (a + 2) for a in cell.dimensions)
        for control, fiber in fibers.items():
            numerator = sum(
                2 ** sum(a == b for a, b in zip(corners[i], corners[j], strict=True)) for i in fiber for j in fiber
            )
            contributions[owners[control]].append(_risk_contribution(weight, numerator / denominator))
    return {cid: math.fsum(values) for cid, values in contributions.items()}


def _regular_coefficients(
    space: BezierSpace,
    weights: list[float],
    owners: dict[int, int],
    factor_coefficients: Callable[[int, int], tuple[float, ...]],
) -> dict[int, float]:
    """Sum single-chart face traces over incidences, with no global Gram matrix or propagation solve."""
    contributions: dict[int, list[float]] = {cid: [] for cid in owners.values()}
    factors: dict[tuple[int, int], tuple[float, ...]] = {}
    for cell, weight in zip(space.cells, weights, strict=True):
        if not weight:
            continue
        traces = []
        for a, d in zip(cell.dimensions, cell.degree, strict=True):
            key = (a, d)
            if key not in factors:
                values = factor_coefficients(a, d)
                factors[key] = tuple(value / math.comb(a + 1, m + 1) ** 2 for m, value in enumerate(values))
            traces.append(factors[key])
        # Every local control is distinct on a regular chart. Its global owner
        # identifies one unique face; all that owner's proper controls have the same support.
        seen = set()
        for control, index in zip(cell.controls, _indices(cell.dimensions, cell.degree), strict=True):
            owner = owners[control]
            if owner in seen:
                continue
            seen.add(owner)
            value = math.prod(trace[sum(v > 0 for v in alpha) - 1] for trace, alpha in zip(traces, index, strict=True))
            contributions[owner].append(_risk_contribution(weight, value))
    return {cid: math.fsum(values) for cid, values in contributions.items()}


def _closed_risk_coefficients(
    space: BezierSpace,
    weights: torch.Tensor,
    factor_coefficients: Callable[[int, int], tuple[float, ...]],
    pair_limit: int,
) -> dict[int, float] | None:
    """Select only proved closed-form cases; return None for general identified nonlinear sets."""
    stages = _skeleton_stages(space)
    owners = {control: cid for cid, proper in stages for control in proper}
    values = weights.tolist()
    if all(sum(cell.dimensions) <= 1 for cell in space.cells):
        result = _graph_coefficients(space, values, owners)
    elif all(d == 1 for cell in space.cells for d in cell.degree):
        result = _linear_coefficients(space, values, owners, pair_limit)
    elif all(len(set(cell.controls)) == len(cell.controls) for cell in space.cells):
        result = _regular_coefficients(space, values, owners, factor_coefficients)
    else:
        return None
    return {cid: result[cid] for cid, _ in stages}
