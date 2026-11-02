"""Conditional scalar kernel refitting with the native BSF as fallback.

This postprocessor has no query/session interface. Observed CV can replace
each scalar chart independently; vector and noisy results retain identity.
"""

import math
from dataclasses import replace
from typing import Any

from .export import _compile_result
from .kernel import fit_kernel_predictor
from .polynomial import StructureLearningResult


def _score(capacity: dict[str, Any]) -> float | None:
    value = capacity.get("weighted_cv_mse", capacity.get("cv_mse"))
    return float(value) if value is not None and math.isfinite(float(value)) else None


def refit_prediction(result: StructureLearningResult) -> StructureLearningResult:
    """Compare per-chart observed CV risk, retaining native masks and fallback."""
    if result.values.shape[1] != 1 or (result.diagnostics.get("noise_variance") or 0) > 0:
        return result
    native_capacity = result.diagnostics.get("capacity", [])
    if len(native_capacity) != len(result._models):
        return result
    models, capacities, comparisons = [], [], []
    for spline, capacity in zip(result._models, native_capacity, strict=True):
        selected = spline
        updated_capacity = dict(capacity)
        take = (result.inputs[:, 0] >= spline.knots[0]) & (result.inputs[:, 0] <= spline.knots[-1])
        native_score = _score(capacity)
        comparison: dict[str, Any] = {
            "left": float(spline.knots[0]),
            "right": float(spline.knots[-1]),
            "native_observed_cv_mse": native_score,
            "selected": "native",
            "selection_policy": "cv",
        }
        kernel = fit_kernel_predictor(result.inputs[take, 0], result.values[take], noise=0)
        if kernel is not None:
            proposed = kernel.diagnostics["weighted_cv_mse"]
            comparison["kernel_observed_cv_mse"] = proposed
            comparison["kernel_diagnostics"] = kernel.diagnostics
            if native_score is not None and proposed < native_score:
                try:
                    selected = kernel.as_spline()
                except (ArithmeticError, RuntimeError, ValueError) as failure:
                    comparison["conversion_rejection"] = type(failure).__name__ + ": " + str(failure)
                else:
                    comparison["selected"] = "stationary_kernel"
                    updated_capacity = dict(kernel.diagnostics)
        models.append(selected)
        capacities.append(updated_capacity)
        comparisons.append(comparison)
    if not any(row["selected"] != "native" for row in comparisons):
        return replace(result, diagnostics={**result.diagnostics, "kernel_prediction": comparisons})
    updated = replace(
        result,
        _models=models,
        _compiled=None,
        diagnostics={
            **result.diagnostics,
            "capacity": capacities,
            "native_capacity": native_capacity,
            "kernel_prediction": comparisons,
        },
    )
    return _compile_result(updated)
