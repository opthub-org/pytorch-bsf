"""Observed-data model selection; fitted operators use no evaluator calls."""

from typing import Any

import torch

from .advanced import fit_predictor as fit_advanced_predictor
from .rational import fit_rational


def _fit_prediction(x: torch.Tensor, y: torch.Tensor, noise: float) -> Any:
    """Select a polynomial or pole-screened rational family by observed CV."""
    connected = fit_advanced_predictor(x, y, noise=noise)
    rational = fit_rational(x, y)
    base_score = connected.diagnostics.get("weighted_cv_mse")
    rational_score = (
        rational.diagnostics["cv_mse"] + noise * rational.diagnostics["support_points"] / max(1, len(x) - 2)
        if rational is not None
        else None
    )
    connected.diagnostics["rational_weighted_cv_mse"] = rational_score
    if rational is not None and base_score is not None and rational_score is not None and rational_score < base_score:
        # Conversion can reject a near-pole operator that needs impractically
        # small polynomial pieces. The connected polynomial family then stays.
        try:
            rational.as_spline()
        except ArithmeticError as failure:
            connected.diagnostics["rational_conversion_rejection"] = str(failure)
        else:
            rational.diagnostics["polynomial_weighted_cv_mse"] = base_score
            return rational
    return connected
