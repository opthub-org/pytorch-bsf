"""Training-only local interpolation-bias acquisition for parameterized curves.

Interior observations are withheld from a local cubic interpolant while chart
endpoints remain available. Scalar PCHIP competes with cubic interpolation by
the same input-volume-weighted residuals. Residuals are scaled from the larger
leave-one-out hole to each neighboring cell, then weighted by cell measure.
These are conditional diagnostics, not unbiased CV errors or posterior bounds:
the supplied structural hypothesis was learned from all observed responses.
No response evaluator, test design, function identity or truth is accepted.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator

from .evidence import GapEvidence, _inputs


def _normalized(y: torch.Tensor) -> tuple[NDArray[np.float64], float]:
    maximum = float(y.abs().max())
    if not maximum:
        return y.numpy().copy(), 1.0
    centered = y / maximum
    centered = centered - centered.mean(0)
    spread = float(centered.norm(dim=1).max())
    if not spread:
        return centered.numpy(), 1.0
    return (centered / spread).numpy(), maximum * spread


def _residuals(
    x: NDArray[np.float64], y: NDArray[np.float64], *, shape_preserving: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate each interior target without using its own response."""
    errors, orders = np.zeros(len(x)), np.full(len(x), 4.0)
    for i in range(1, len(x) - 1):
        indices = np.arange(max(0, i - 3), min(len(x), i + 4))
        indices = indices[indices != i]
        # Local input normalization conditions very small adaptive neighborhoods.
        left, width = x[indices[0]], x[indices[-1]] - x[indices[0]]
        train, at = (x[indices] - left) / width, (x[i] - left) / width
        interpolator = PchipInterpolator if shape_preserving else CubicSpline
        prediction = interpolator(train, y[indices], axis=0)(at)
        errors[i] = np.linalg.norm(np.asarray(prediction) - y[i])
        orders[i] = min(3 if shape_preserving else 4, len(indices))
    if not np.isfinite(errors).all():
        raise ArithmeticError("Local leave-one-out interpolation produced nonfinite errors.")
    errors[errors < 128 * np.finfo(float).eps] = 0
    return errors, orders


def _chart_scores(x: NDArray[np.float64], y: NDArray[np.float64], noise: NDArray[np.float64]) -> NDArray[np.float64]:
    """Choose a local interpolation family using observed interior residuals."""
    widths = np.diff(x)
    result = np.zeros(len(widths))
    if len(x) < 3:
        return result
    mass = (widths[:-1] + widths[1:]) / 2
    candidates = []
    for shape_preserving in (False, True) if y.shape[1] == 1 else (False,):
        errors, orders = _residuals(x, y, shape_preserving=shape_preserving)
        errors[1:-1] = np.maximum(0, errors[1:-1] - np.maximum(noise[:-1], noise[1:]))
        score = float(np.dot(mass / mass.sum(), errors[1:-1] ** 2))
        candidates.append((score, errors, orders))
    _, errors, orders = min(candidates, key=lambda candidate: candidate[0])
    for i in range(1, len(x) - 1):
        hole = x[i + 1] - x[i - 1]
        for cell in (i - 1, i):
            estimated_error = errors[i] * (widths[cell] / hole) ** orders[i]
            result[cell] = max(result[cell], widths[cell] * estimated_error**2)
    return result


def _validate_evidence(points: torch.Tensor, evidence: Sequence[GapEvidence], target: float) -> None:
    if isinstance(target, bool) or not math.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive.")
    if len(evidence) != max(0, len(points) - 1):
        raise ValueError("Evidence must align with every adjacent input interval.")
    for i, row in enumerate(evidence):
        if row.left != float(points[i]) or row.right != float(points[i + 1]):
            raise ValueError("Evidence coordinates do not match the supplied observations.")


def scores(x: Any, y: Any, evidence: Sequence[GapEvidence], target: float) -> torch.Tensor:
    """Return one dimensionless input-integrated bias score per adjacent cell.

    Positive excess brackets isolate interpolation neighborhoods but do not
    themselves assert disconnected topology. They receive an explicit jump
    refinement score. Localized retained candidates get zero acquisition score
    here; the caller must preserve independent exploration and cut audits.
    Vector responses use a common cubic operator, preserving orthogonal-output
    invariance. Noise floors only remove already-accounted empirical noise;
    this function supplies no stochastic calibration or replication policy.
    """
    points, values, _ = _inputs(x, y, None)
    _validate_evidence(points, evidence, target)
    if len(points) < 2:
        return torch.zeros(0, dtype=torch.float64)
    z = ((points - points[0]) / (points[-1] - points[0])).numpy()
    normalized, scale = _normalized(values)
    barriers = [i for i, row in enumerate(evidence) if row.excess > row.threshold or row.status == "jump_candidate"]
    noise = np.array([row.noise_floor / scale for row in evidence])
    result = np.zeros(len(points) - 1)
    bounds = [0] + [i + 1 for i in barriers] + [len(points)]
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        if stop - start > 1:
            result[start : stop - 1] = _chart_scores(z[start:stop], normalized[start:stop], noise[start : stop - 1])
    for i, row in enumerate(evidence):
        if row.excess > row.threshold:
            result[i] = (z[i + 1] - z[i]) * (row.excess / scale) ** 2
        if row.status == "jump_candidate" and row.right - row.left <= target:
            result[i] = 0
    if not np.isfinite(result).all():
        raise ArithmeticError("Local cross-validation acquisition score overflowed.")
    return torch.from_numpy(result)


def _ranked_cell(candidates: list[tuple[int, float]], scores: torch.Tensor, x: torch.Tensor) -> float:
    """Break roundoff-level risk/width ties by input order, independent of units."""
    best = max(float(scores[i]) for i, _ in candidates)
    tolerance = abs(best) * torch.finfo(torch.float64).eps * 64
    tied = [(i, point) for i, point in candidates if float(scores[i]) >= best - tolerance]
    widths = (x[1:] - x[:-1]) / (x[-1] - x[0])
    widest = max(float(widths[i]) for i, _ in tied)
    tolerance = abs(widest) * torch.finfo(torch.float64).eps * 64
    return min((i, point) for i, point in tied if float(widths[i]) >= widest - tolerance)[1]
