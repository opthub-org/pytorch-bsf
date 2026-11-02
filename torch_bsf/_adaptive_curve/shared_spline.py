"""Conditional input-volume PRESS for shared C0 shape and chart offsets.

The fixed portfolio uses Bernstein charts with independent shape and offset
ridge grids. Inferred chart locations remain fixed during validation; these
scores are not calibrated end-to-end structural risks. No evaluator is used.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from .evidence import _inputs
from .polynomial import _basis, _Spline
from .spectral import _mass


def _split(controls: torch.Tensor, at: float) -> tuple[torch.Tensor, torch.Tensor]:
    left, right, level = [controls[0]], [controls[-1]], controls
    while len(level) > 1:
        level = (1 - at) * level[:-1] + at * level[1:]
        left.append(level[0])
        right.append(level[-1])
    return torch.stack(left), torch.stack(right[::-1])


def restrict_spline(source: _Spline, left: float, right: float) -> _Spline:
    """Restrict an existing C0 Bernstein curve exactly, without fitting values."""
    if not float(source.knots[0]) <= left <= right <= float(source.knots[-1]):
        raise ValueError("Restriction must lie within the source chart.")
    if left == right:
        value = source.predict(source.knots.new_tensor([left]))
        return _Spline(source.knots.new_tensor([left]), 0, torch.zeros_like(value), value[0])
    knots = [left]
    pieces: list[torch.Tensor] = []
    degree = source.degree
    for i in range(len(source.knots) - 1):
        a, b = float(source.knots[i]), float(source.knots[i + 1])
        low, high = max(a, left), min(b, right)
        if high <= low:
            continue
        controls = source.controls[i * degree : i * degree + degree + 1]
        t0, t1 = (low - a) / (b - a), (high - a) / (b - a)
        prefix = _split(controls, t1)[0] if t1 < 1 else controls
        selected = _split(prefix, t0 / t1)[1] if t0 > 0 else prefix
        pieces.append(selected if not pieces else selected[1:])
        knots.append(high)
    return _Spline(source.knots.new_tensor(knots), degree, torch.cat(pieces), source.center.clone())


@dataclass
class _Fit:
    """A fixed-design ridge fit and its conditional validation score."""

    coefficients: torch.Tensor
    score: float
    condition: float
    degrees_of_freedom: float


def _ridge(design: torch.Tensor, values: torch.Tensor, mass: torch.Tensor, penalty: torch.Tensor) -> tuple[Any, ...] | None:
    weights = mass.sqrt()
    weighted = design * weights[:, None]
    scales = weighted.square().sum(0).sqrt()
    if not (scales > 0).all():
        return None
    augmented = torch.cat((weighted / scales, torch.diag(penalty.sqrt() / scales)))
    u, singular, vh = torch.linalg.svd(augmented, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if float(singular[-1]) > 0 else math.inf
    if condition > 1e8:
        return None
    coefficients = (vh.T @ ((u[: len(values)].T @ (values * weights[:, None])) / singular[:, None])) / scales[:, None]
    influence = ((design / scales) @ vh.T) / singular
    leverage = influence.square().sum(1) * mass
    if not torch.isfinite(coefficients).all():
        return None
    return coefficients, leverage, condition


def _score(
    design: torch.Tensor, values: torch.Tensor, mass: torch.Tensor, penalty: torch.Tensor, validation: torch.Tensor
) -> _Fit | None:
    fitted = _ridge(design, values, mass, penalty)
    if fitted is None:
        return None
    coefficients, leverage, condition = fitted
    residual = design @ coefficients - values
    errors = []
    for i in torch.nonzero(validation).flatten().tolist():
        if float(1 - leverage[i]) <= 1e-7:
            training = torch.arange(len(values)) != i
            fold = _ridge(design[training], values[training], mass[training], penalty)
            if fold is None:
                return None
            error = design[i] @ fold[0] - values[i]
        else:
            error = residual[i] / (1 - leverage[i])
        errors.append(error.square().sum() * mass[i])
    score = float(torch.stack(errors).sum() / mass[validation].sum())
    if not math.isfinite(score):
        return None
    return _Fit(coefficients, score, condition, float(leverage.sum()))


def _contrast(regions: int) -> torch.Tensor:
    matrix = torch.zeros((regions, max(0, regions - 1)), dtype=torch.float64)
    for j in range(regions - 1):
        matrix[: j + 1, j] = 1 / math.sqrt((j + 1) * (j + 2))
        matrix[j + 1, j] = -(j + 1) / math.sqrt((j + 1) * (j + 2))
    return matrix


def _partition(x: torch.Tensor, gaps: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    boundaries: list[float] = []
    for pair in gaps:
        left, right = (float(pair.left), float(pair.right)) if hasattr(pair, "left") else tuple(map(float, pair))
        if not left < right or bool(((x > left) & (x < right)).any()):
            raise ValueError("Gap intervals must be ordered, positive and free of observed interior points.")
        if not bool((x == left).any()) or not bool((x == right).any()):
            raise ValueError("Gap endpoints must be observed coordinates.")
        if boundaries and left < boundaries[-1]:
            raise ValueError("Gap intervals must be sorted without overlap.")
        boundaries.append(right)
    labels = torch.searchsorted(x.new_tensor(boundaries), x.contiguous(), right=True)
    groups = [torch.nonzero(labels == i).flatten() for i in range(len(boundaries) + 1)]
    if any(not len(group) for group in groups):
        raise ValueError("Every retained chart needs observed support.")
    return labels, groups


def _weighted_charts(points: torch.Tensor, groups: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mass = points.new_zeros(len(points))
    validation = torch.ones(len(points), dtype=torch.bool)
    for group in groups:
        mass[group] = _mass(points[group]) * (points[group[-1]] - points[group[0]])
        validation[group[0]] = validation[group[-1]] = False
    mass /= mass.sum()
    return mass, validation


def _bases(points: torch.Tensor) -> list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    bases = []
    for degree in range(7):
        for pieces in (1,) if degree == 0 else range(1, 5):
            indices = torch.linspace(0, len(points) - 1, pieces + 1).round().long().unique()
            if len(indices) != pieces + 1:
                continue
            knots = points[indices]
            bernstein = _basis(points, knots, degree)
            helmert = _contrast(bernstein.shape[1])
            shape = bernstein @ helmert
            bases.append((degree, pieces, knots, helmert, shape))
    return bases


def _portfolio(shape: torch.Tensor, offsets: torch.Tensor) -> list[tuple[str, float, float, torch.Tensor, torch.Tensor]]:
    global_design = torch.cat((shape.new_ones((len(shape), 1)), shape), 1)
    candidates = []
    for shape_penalty in (0.0,) if not shape.shape[1] else (0.0, 0.001, 0.01, 0.1):
        penalty = global_design.new_full((global_design.shape[1],), shape_penalty)
        penalty[0] = 0
        candidates.append(("global_c0", shape_penalty, 0.0, global_design, penalty))
        if offsets.shape[1]:
            design = torch.cat((global_design, offsets), 1)
            for offset_penalty in (0.0, 0.001, 0.01, 0.1, 1.0):
                regularization = torch.cat((penalty, design.new_full((offsets.shape[1],), offset_penalty)))
                candidates.append(("shared_c0_offsets", shape_penalty, offset_penalty, design, regularization))
    return candidates


def fit_spline_charts(x: Any, y: Any, gaps: Any = ()) -> tuple[list[_Spline], dict[str, Any]]:
    """Select degree0..6/pieces1..4 using observed interior PRESS only."""
    points, values, _ = _inputs(x, y, None)
    labels, groups = _partition(points, gaps)
    if any(len(group) < 3 for group in groups):
        raise ValueError("Spline pooling requires at least three observations per retained chart.")
    mass, validation = _weighted_charts(points, groups)
    chart_contrast = _contrast(len(groups))
    offsets = chart_contrast[labels]
    candidates, summaries = [], []
    for degree, pieces, knots, helmert, shape in _bases(points):
        for family, shape_penalty, offset_penalty, design, penalty in _portfolio(shape, offsets):
            fitted = _score(design, values, mass, penalty, validation)
            if fitted is None:
                continue
            candidates.append((family, degree, pieces, knots, helmert, shape_penalty, offset_penalty, fitted))
            summaries.append(
                {
                    "family": family,
                    "degree": degree,
                    "pieces": pieces,
                    "shape_penalty": shape_penalty,
                    "offset_penalty": offset_penalty,
                    "cv_mse": fitted.score,
                    "effective_df": fitted.degrees_of_freedom,
                    "condition": fitted.condition,
                }
            )
    if not candidates:
        raise ArithmeticError("No full-rank C0 pooling model is available.")
    minimum = min(row[-1].score for row in candidates)
    variation = float(((values - (values * mass[:, None]).sum(0)).square().sum(1) * mass).sum())
    tolerance = max(minimum * 1e-8, variation * torch.finfo(torch.float64).eps ** 2 * 256)
    selected = min(
        (row for row in candidates if row[-1].score <= minimum + tolerance),
        key=lambda row: (row[-1].degrees_of_freedom, row[1], row[2]),
    )
    family, degree, pieces, knots, helmert, shape_penalty, offset_penalty, fitted = selected
    mean = fitted.coefficients[0]
    number_shape = helmert.shape[1]
    controls = helmert @ fitted.coefficients[1 : number_shape + 1]
    chart_coefficients = fitted.coefficients[number_shape + 1 :]
    models = []
    for region, group in enumerate(groups):
        center = mean.clone()
        if family == "shared_c0_offsets":
            center += chart_contrast[region] @ chart_coefficients
        global_shape = _Spline(knots.clone(), degree, controls.clone(), center)
        models.append(restrict_spline(global_shape, float(points[group[0]]), float(points[group[-1]])))
    return models, {
        "family": family,
        "degree": degree,
        "pieces": pieces,
        "shape_penalty": shape_penalty,
        "offset_penalty": offset_penalty,
        "weighted_cv_mse": fitted.score,
        "effective_df": fitted.degrees_of_freedom,
        "selection_tolerance": tolerance,
        "chart_validation_masses": [float(mass[g[1:-1]].sum()) for g in groups],
        "selection_scope": "Observed CV conditional on already inferred charts; not end-to-end risk calibration.",
        "selection_measure": "covered input-volume mass, common interior leave-one-out rows",
        "chart_endpoints_retained": True,
        "no_oracle_calls": True,
        "candidates": summaries,
    }
