"""Select a shared vector shape from observed data, preserving BSF routing.

Scalar, connected and empirically noisy results retain their original fitted
operator. Conditional observed-data CV can replace a deterministic split
vector predictor; unavailable or inferior pooling retains the native model.
"""

import math
from dataclasses import replace
from typing import Any

import torch

from .export import _compile_result
from .polynomial import StructureLearningResult
from .shared_spline import fit_spline_charts


def _native_score(result: StructureLearningResult, masses: list[float]) -> float | None:
    capacities = result.diagnostics.get("capacity", [])
    if len(capacities) != len(masses):
        return None
    x = result.inputs[:, 0]
    if any(int(((x >= model.knots[0]) & (x <= model.knots[-1])).sum()) < 6 for model in result._models):
        return None
    numerator = 0.0
    for capacity, mass in zip(capacities, masses, strict=True):
        if capacity.get("cv_available") is False:
            return None
        score = capacity.get("weighted_cv_mse", capacity.get("cv_mse"))
        if score is None or not math.isfinite(float(score)):
            return None
        numerator += float(score) * mass
    return numerator / sum(masses) if sum(masses) > 0 else None


def gap_contrasts(result: StructureLearningResult) -> list[dict[str, Any]]:
    """Expose response geometry separately from disconnected input charts."""
    contrasts = []
    for gap in result.gaps:
        points = torch.tensor([[gap.left], [gap.right]], dtype=torch.float64)
        indices = torch.searchsorted(result.inputs[:, 0].contiguous(), points[:, 0])
        observed = result.values[indices[1]] - result.values[indices[0]]
        predicted = result.predict(points)
        delta = predicted[1] - predicted[0]
        original, amplitude = float(observed.norm()), float(delta.norm())
        ratio = amplitude / original if original else None
        cosine = float(torch.dot(observed / original, delta / amplitude)) if original and amplitude else None
        contrasts.append(
            {
                "left": gap.left,
                "right": gap.right,
                "observed_endpoint_norm": original,
                "predicted_endpoint_norm": amplitude,
                "predicted_observed_ratio": ratio,
                "direction_cosine": cosine,
                "near_coinciding_outputs": ratio is not None and ratio < 0.05,
                "near_coinciding_rule": "Descriptive norm ratio<0.05; never a topology certificate or selection rule.",
            }
        )
    return contrasts


def refit_prediction(result: StructureLearningResult) -> StructureLearningResult:
    """Choose shared C0 shape by training CV, retaining native fallback."""
    if result.values.shape[1] == 1 or not result.gaps or (result.diagnostics.get("noise_variance") or 0.0) > 0:
        return result
    diagnostics: dict[str, Any]
    try:
        models, diagnostics = fit_spline_charts(result.inputs[:, 0], result.values, result.gaps)
        native = _native_score(result, diagnostics["chart_validation_masses"])
        diagnostics["native_observed_cv_mse"] = native
        keep_native = native is None or diagnostics["weighted_cv_mse"] + diagnostics["selection_tolerance"] >= native
        if keep_native:
            diagnostics["selected_predictor"] = "native_chart_models"
            updated = replace(result, diagnostics={**result.diagnostics, "spline_pooling": diagnostics})
        else:
            diagnostics["selected_predictor"] = diagnostics["family"]
            updated = _compile_result(
                replace(
                    result, _models=models, _compiled=None, diagnostics={**result.diagnostics, "spline_pooling": diagnostics}
                )
            )
    except (ArithmeticError, RuntimeError, ValueError) as failure:
        diagnostics = {"selected_predictor": "native_chart_models", "fallback_error": f"{type(failure).__name__}: {failure}"}
        updated = replace(result, diagnostics={**result.diagnostics, "spline_pooling": diagnostics})
    updated.diagnostics["spline_pooling"]["retained_gap_contrasts"] = gap_contrasts(updated)
    return updated
