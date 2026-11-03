"""Conditional common-row convex stacking of shared C0 and chart-offset GP.

One output-rotation-invariant weight minimizes common interior-row PRESS.
The result is compiled from the exact polynomial blend; fitted-kernel
conversion remains empirically checked, not uniformly certified. No query
session or evaluator is accepted by this postprocessor.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .component_kernel import ComponentKernelModel, _contrasts, _covariance, fit_component_kernel
from .export import _compile_result
from .kernel import _fixed_fit
from .polynomial import StructureLearningResult, _Spline
from .pooled_prediction import gap_contrasts
from .shared_spline import _bases, _contrast, _partition, _portfolio, _ridge, _weighted_charts, restrict_spline
from .support_converter import component_splines


def _c0_residuals(result: StructureLearningResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Recover signed common-row OOF errors of the selected fixed C0 family."""
    diagnostic = result.diagnostics.get("spline_pooling", {})
    if diagnostic.get("selected_predictor") not in ("global_c0", "shared_c0_offsets"):
        return None
    points, values = result.inputs[:, 0], result.values
    labels, groups = _partition(points, result.gaps)
    mass, validation = _weighted_charts(points, groups)
    base = next((row for row in _bases(points) if row[:2] == (diagnostic["degree"], diagnostic["pieces"])), None)
    if base is None:
        return None
    offsets = _contrast(len(groups))[labels]
    specification = next(
        (
            row
            for row in _portfolio(base[-1], offsets)
            if row[:3] == (diagnostic["family"], diagnostic["shape_penalty"], diagnostic["offset_penalty"])
        ),
        None,
    )
    if specification is None:
        return None
    design, penalty = specification[-2:]
    fitted = _ridge(design, values, mass, penalty)
    if fitted is None:
        return None
    coefficients, leverage, _ = fitted
    residuals = values - design @ coefficients
    for i in torch.nonzero(validation).flatten().tolist():
        if float(1 - leverage[i]) <= 1e-7:
            use = torch.arange(len(values)) != i
            fold = _ridge(design[use], values[use], mass[use], penalty)
            if fold is None:
                return None
            residuals[i] = values[i] - design[i] @ fold[0]
        else:
            residuals[i] /= 1 - leverage[i]
    return residuals, mass, validation


def _gp_residuals(model: ComponentKernelModel, result: StructureLearningResult, mass: torch.Tensor) -> torch.Tensor | None:
    operator, diagnostic = model._smooth, model.diagnostics
    unit = operator.unit
    labels, _ = _partition(result.inputs[:, 0], result.gaps)
    covariance = _covariance(unit, unit, labels.numpy(), labels.numpy(), operator.kind, operator.length, diagnostic["eta"])
    trend = np.column_stack((np.ones(len(unit)), 2 * unit - 1))[:, : operator.degree + 1]
    normalized = (model.values.numpy() - operator.center) / operator.scale
    fitted = _fixed_fit(covariance, normalized, trend, diagnostic["ridge"] / (len(unit) * mass.numpy()))
    return torch.from_numpy(fitted.loo_residuals * operator.scale) if fitted is not None else None


def _weight(first: torch.Tensor, second: torch.Tensor, mass: torch.Tensor) -> tuple[float, float]:
    delta = second - first
    denominator = float((delta.square().sum(1) * mass).sum())
    numerator = -float(((first * delta).sum(1) * mass).sum())
    weight = min(1.0, max(0.0, numerator / denominator)) if denominator else 0.0
    score = float(((first + weight * delta).square().sum(1) * mass).sum() / mass.sum())
    return weight, score


def _elevate(controls: torch.Tensor, degree: int) -> torch.Tensor:
    while len(controls) <= degree:
        old = len(controls) - 1
        ratio = torch.arange(1, old + 1, dtype=torch.float64)[:, None] / (old + 1)
        controls = torch.cat((controls[:1], ratio * controls[:-1] + (1 - ratio) * controls[1:], controls[-1:]))
    return controls


def _blend(first: _Spline, second: _Spline, weight: float) -> _Spline:
    knots = torch.unique(torch.cat((first.knots, second.knots)))
    degree = max(first.degree, second.degree)
    pieces = []
    for i, (left, right) in enumerate(zip(knots[:-1], knots[1:], strict=True)):
        a, b = restrict_spline(first, float(left), float(right)), restrict_spline(second, float(left), float(right))
        controls = (1 - weight) * _elevate(a.controls, degree) + weight * _elevate(b.controls, degree)
        pieces.append(controls if i == 0 else controls[1:])
    return _Spline(knots, degree, torch.cat(pieces), (1 - weight) * first.center + weight * second.center)


def refit_prediction(result: StructureLearningResult, *, connected_control: bool = False) -> StructureLearningResult:
    """Fit one observed-only convex mixture, retaining unsupported native paths."""
    if (
        result.values.shape[1] == 1
        or (not result.gaps and not connected_control)
        or (result.diagnostics.get("noise_variance") or 0) > 0
    ):
        return result
    diagnostics: dict[str, Any] = {
        "selected_predictor": "c0_native_fallback",
        "stack_weight": 0.0,
        "conversion_profile": "support_partition_vector_matern",
    }
    updated = result
    try:
        residuals = _c0_residuals(result)
        if residuals is None:
            return replace(
                result,
                diagnostics={**result.diagnostics, "stacked_gap_prediction": {**diagnostics, "reason": "no_common_c0_cv"}},
            )
        first, mass, validation = residuals
        model = fit_component_kernel(result.inputs[:, 0], result.values, result.gaps)
        second = _gp_residuals(model, result, mass) if model is not None else None
        if model is None or second is None:
            return replace(
                result,
                diagnostics={**result.diagnostics, "stacked_gap_prediction": {**diagnostics, "reason": "no_admissible_gp"}},
            )
        weight, cv = _weight(first[validation], second[validation], mass[validation])
        baseline = float((first[validation].square().sum(1) * mass[validation]).sum() / mass[validation].sum())
        diagnostics.update(model.diagnostics)
        diagnostics.update(stack_weight=weight, stack_cv_mse=cv, c0_common_row_cv_mse=baseline)
        thresholds = [gap.threshold for gap in result.gaps]
        predicted = result.predict(result.inputs).numpy()
        native_contrasts = _contrasts(predicted, result.values.numpy(), model.groups, thresholds)
        diagnostics["c0_gap_consistency"] = native_contrasts
        tolerance = result.diagnostics.get("spline_pooling", {}).get("selection_tolerance", 0)
        if weight and cv + tolerance < baseline and all(row["admissible"] for row in native_contrasts):
            models = [_blend(a, b, weight) for a, b in zip(result._models, component_splines(model), strict=True)]
            diagnostics.update(model.diagnostics)
            proposed = _compile_result(replace(result, _models=models, _compiled=None, diagnostics=dict(result.diagnostics)))
            actual = _contrasts(proposed.predict(result.inputs).numpy(), result.values.numpy(), model.groups, thresholds)
            if not all(row["admissible"] for row in actual):
                raise ArithmeticError("Actual BSF mixture failed its retained-gap direction constraint.")
            updated = proposed
            diagnostics["selected_predictor"] = "cv_convex_c0_component_gp"
    except (ArithmeticError, RuntimeError, ValueError) as failure:
        diagnostics["fallback_error"] = f"{type(failure).__name__}: {failure}"
    diagnostics["retained_actual_bsf_gap_contrasts"] = gap_contrasts(updated)
    return replace(updated, diagnostics={**updated.diagnostics, "stacked_gap_prediction": diagnostics})
