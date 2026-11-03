"""Common response units for comparisons outside safe squared-loss scales."""

import math
from typing import Any

import torch

from .polynomial import _Spline


def response_scale(values: torch.Tensor, noise: float = 0.0) -> float:
    """Return one power-of-two scale, preserving ordinary response arithmetic."""
    # Reserve exponent headroom for squared residuals and weighted reductions.
    # A shared power of two avoids separate output-coordinate rescalings and
    # makes normalization exact unless an individual value is subnormal.
    magnitude = max(float(values.abs().max()), math.sqrt(noise))
    if magnitude == 0 or 1e-100 <= magnitude <= 1e100:
        return 1.0
    return math.ldexp(1.0, math.frexp(magnitude)[1] - 1)


def restore_score(diagnostics: dict[str, Any], key: str, scale: float) -> None:
    """Keep a usable normalized CV score when physical squared units overflow."""
    score = float(diagnostics[key])
    diagnostics.update(normalized_cv_mse=score, cv_response_scale=scale)
    diagnostics[key] = score * scale * scale


def comparison_score(diagnostics: dict[str, Any], scale: float) -> float | None:
    """Express a model's observed CV loss in common, representable units."""
    normalized = diagnostics.get("normalized_cv_mse")
    if normalized is not None:
        ratio = float(diagnostics["cv_response_scale"]) / scale
        result = float(normalized) * ratio * ratio
    else:
        score = diagnostics.get("weighted_cv_mse", diagnostics.get("cv_mse"))
        if score is None:
            return None
        result = float(score) / scale / scale
    return result if math.isfinite(result) else None


def restore_spline(spline: _Spline, scale: float) -> _Spline:
    """Restore response units after a normalized operator conversion check."""
    controls, center = spline.controls * scale, spline.center * scale
    if not torch.isfinite(controls).all() or not torch.isfinite(center).all():
        raise ArithmeticError("Fitted Bernstein coefficients exceed finite response units.")
    return _Spline(spline.knots.clone(), spline.degree, controls, center)


def restore_conversion(diagnostics: dict[str, Any], normalized: dict[str, Any], scale: float) -> None:
    """Report conversion error in physical and explicitly normalized units."""
    diagnostics.update(normalized, conversion_response_scale=scale)
    for name in ("conversion_checked_max_error", "conversion_checked_tolerance"):
        if name in normalized:
            diagnostics["normalized_" + name] = normalized[name]
            diagnostics[name] = normalized[name] * scale
