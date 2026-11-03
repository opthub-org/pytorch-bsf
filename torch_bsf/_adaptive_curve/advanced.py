"""Alternative training-only LOOCV predictor with exact polynomial export.

A fixed candidate family combines
conditioned global Chebyshev regression, C0 polynomial regression, true
cubic interpolation, linear interpolation, and scalar-only PCHIP. LOOCV
retains endpoints and weights validation errors by input Voronoi mass.
High-leverage polynomial cases use actual leave-row refits, not unstable
division of nearly zero PRESS residuals. No oracle or test labels enter.

CubicSpline/PchipInterpolator follow SciPy1.14.1's documented API. PCHIP's
componentwise sign decisions are not rotation-equivariant for vector data;
it is deliberately excluded when there is more than one output coordinate.
All selected functions are polynomials by pieces with exact Bernstein form.
"""

import math
from typing import Any

import numpy as np
import scipy
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator

from .._numeric import _finite_mean
from .response_units import response_scale
from .spectral import (
    SpectralCurveModel,
    _Candidate,
    _chebyshev,
    _designs,
    _linear_predict,
    _mass,
    _restore_response_units,
    _solve,
    _validated,
)
from .spectral import fit_predictor as _previous


def _direct_loo(design: torch.Tensor, y: torch.Tensor, mass: torch.Tensor, max_condition: float) -> tuple[float, float] | None:
    error = total = condition = 0.0
    for i in range(1, len(y) - 1):
        train = torch.arange(len(y)) != i
        center = _finite_mean(y[train], 0)
        fit = _solve(design[train], y[train] - center, mass[train], max_condition)
        if fit is None:
            return None
        coefficients, current_condition = fit
        residual = design[i] @ coefficients + center - y[i]
        error += float(residual.square().sum() * mass[i])
        total += float(mass[i])
        condition = max(condition, current_condition)
    return error / total, condition


def _press(
    design: torch.Tensor, y: torch.Tensor, mass: torch.Tensor, max_condition: float
) -> tuple[torch.Tensor, float, float, bool] | None:
    """Exact linear-model LOOCV, using direct refits when leverage is high."""
    if design.shape[1] >= len(y) or not torch.isfinite(y).all():
        return None
    weight = (mass / mass.max()).sqrt()
    weighted = design * weight[:, None]
    scales = torch.linalg.vector_norm(weighted, dim=0)
    if not (scales > 0).all():
        return None
    normalized = weighted / scales
    u, singular, vh = torch.linalg.svd(normalized, full_matrices=False)
    if singular[-1] <= singular[0] / max_condition:
        return None
    coefficients = (vh.T @ ((u.T @ (y * weight[:, None])) / singular[:, None])) / scales[:, None]
    denominator = 1 - u.square().sum(1)
    condition = float(singular[0] / singular[-1])
    # Endpoint values are retained. These are interpolation-validation errors,
    # not a claim about extrapolation or independently noisy endpoint labels.
    direct = design.shape[1] > len(y) // 2 or float(denominator[1:-1].min()) < 1e-5
    if direct:
        result = _direct_loo(design, y, mass, max_condition)
        if result is None:
            return None
        score, fold_condition = result
        condition = max(condition, fold_condition)
    else:
        residual = (design @ coefficients - y)[1:-1] / denominator[1:-1, None]
        score = float((residual.square().sum(1) * mass[1:-1]).sum() / mass[1:-1].sum())
        # Removing a row changes A^T A by rank one. This bounds the condition
        # of every retained-column-scaled fold without using its heldout y.
        bound = condition / math.sqrt(float(denominator[1:-1].min()))
        if bound > max_condition:
            result = _direct_loo(design, y, mass, max_condition)
            if result is None:
                return None
            score, fold_condition = result
            condition = max(condition, fold_condition)
            direct = True
        else:
            condition = bound
    if not torch.isfinite(coefficients).all() or not math.isfinite(score):
        return None
    return coefficients, score, condition, direct


def _interpolator(kind: str, x: torch.Tensor, y: torch.Tensor) -> Any:
    if kind == "cubic":
        return CubicSpline(x.numpy(), y.numpy(), axis=0, bc_type="not-a-knot", extrapolate=False)
    return PchipInterpolator(x.numpy(), y.numpy(), axis=0, extrapolate=False)


def _cubic_candidate(kind: str, x: torch.Tensor, y: torch.Tensor, mass: torch.Tensor) -> tuple[_Candidate, str]:
    error = total = 0.0
    for i in range(1, len(x) - 1):
        train = torch.arange(len(x)) != i
        center = _finite_mean(y[train], 0)
        fitted = _interpolator(kind, x[train], y[train] - center)
        residual = torch.from_numpy(np.asarray(fitted(float(x[i]))).copy()) + center - y[i]
        error += float(residual.square().sum() * mass[i])
        total += float(mass[i])
    center = _finite_mean(y, 0)
    centered = y - center
    fitted = _interpolator(kind, x, centered)
    slopes = torch.from_numpy(np.asarray(fitted(x.numpy(), 1)).copy())
    width = (x[1:] - x[:-1])[:, None]
    controls = torch.stack((centered[:-1], centered[:-1] + width * slopes[:-1] / 3, centered[1:] - width * slopes[1:] / 3), 1)
    flat = torch.cat((controls.reshape(-1, y.shape[1]), centered[-1:]))
    return _Candidate("spline", 3, x.clone(), flat, center, error / total, 1.0, len(flat)), kind


def _linear_candidate(x: torch.Tensor, y: torch.Tensor, mass: torch.Tensor) -> _Candidate:
    error = total = 0.0
    for i in range(1, len(x) - 1):
        train = torch.arange(len(x)) != i
        residual = _linear_predict(x[train], y[train], x[i : i + 1])[0] - y[i]
        error += float(residual.square().sum() * mass[i])
        total += float(mass[i])
    center = _finite_mean(y, 0)
    return _Candidate("linear", 1, x.clone(), y - center, center, error / total, 1.0, len(x))


def fit_predictor(
    x: Any,
    y: Any,
    noise: float = 0.0,
    *,
    bounds: Any = None,
    max_degree: int = 48,
    max_pieces: int = 16,
    max_condition: float = 1e8,
) -> SpectralCurveModel:
    """Fit the fixed LOOCV candidate family using observed responses only."""
    inputs, values, limits = _validated(x, y, noise, bounds, max_degree, max_pieces, max_condition)
    unit = response_scale(values, noise)
    if unit != 1:
        result = fit_predictor(
            inputs,
            values / unit,
            noise / unit / unit,
            bounds=bounds,
            max_degree=max_degree,
            max_pieces=max_pieces,
            max_condition=max_condition,
        )
        return _restore_response_units(result, values, unit)
    return _fit_observed(inputs, values, noise, limits, bounds, max_degree, max_pieces, max_condition)


def _fit_observed(
    inputs: torch.Tensor,
    values: torch.Tensor,
    noise: float,
    limits: tuple[float, float],
    bounds: Any,
    max_degree: int,
    max_pieces: int,
    max_condition: float,
) -> SpectralCurveModel:
    """Run the unchanged portfolio in safe response units."""
    if len(inputs) < 6:
        result = _previous(
            inputs, values, noise, bounds=bounds, max_degree=max_degree, max_pieces=max_pieces, max_condition=max_condition
        )
        result.diagnostics["family"] = "linear_small_sample"
        return result
    coordinates = inputs[:, 0]
    mass = _mass(coordinates)
    center = _finite_mean(values, 0)
    centered = values - center
    candidates: list[tuple[_Candidate, str]] = [(_linear_candidate(coordinates, values, mass), "linear")]
    failed_interpolators = []
    for kind in ("cubic", "pchip") if values.shape[1] == 1 else ("cubic",):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate, name = _cubic_candidate(kind, coordinates, values, mass)
            if not torch.isfinite(candidate.coefficients).all() or not math.isfinite(candidate.score):
                raise FloatingPointError("nonfinite interpolation coefficients/validation score")
            candidates.append((candidate, name))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            failed_interpolators.append({"family": kind, "error": str(error)})
    for candidate, _ in candidates:
        candidate.score += noise
    designs = [
        (kind, degree, knots, design)
        for kind, degree, knots, design in _designs(coordinates, max_degree, max_pieces)
        if kind != "chebyshev"
    ]
    for degree in sorted(
        set(range(min(max_degree, 32, len(inputs) - 2) + 1)) | {min(max_degree, len(inputs) - 2), 36, 40, 48}
    ):
        if degree <= max_degree and degree <= len(inputs) - 2:
            designs.append(
                (
                    "chebyshev",
                    degree,
                    coordinates[[0, -1]],
                    _chebyshev(coordinates, degree, float(coordinates[0]), float(coordinates[-1])),
                )
            )
    rejected = direct_fits = 0
    for kind, degree, knots, design in designs:
        fitted = _press(design, centered, mass, max_condition)
        if fitted is None:
            rejected += 1
            continue
        coefficients, score, condition, direct = fitted
        direct_fits += int(direct)
        score += noise * design.shape[1] / (len(inputs) - 2)
        candidate = _Candidate(kind, degree, knots, coefficients, center, score, condition, design.shape[1])
        candidates.append((candidate, kind))
    best = min(model.score for model, _ in candidates)
    scale = float((centered.square().sum(1) * mass).sum())
    tolerance = max(scale * torch.finfo(torch.float64).eps ** 2 * 256, best * 1e-8)
    selected, family = min((row for row in candidates if row[0].score <= best + tolerance), key=lambda row: row[0].n_controls)
    diagnostics = {
        "kind": selected.kind,
        "family": family,
        "degree": selected.degree,
        "n_controls_native": selected.n_controls,
        "weighted_cv_mse": selected.score,
        "condition": None if family in ("cubic", "pchip") else selected.condition,
        "condition_limit": max_condition,
        "condition_scope": "normalized WLS design; interpolation matrix condition is not measured",
        "cv_available": True,
        "cv_endpoints_retained": True,
        "selection_measure": "interior leave-one-out errors weighted by input Voronoi mass",
        "rejected_condition_or_rank_candidates": rejected,
        "explicit_loo_candidates": direct_fits,
        "failed_interpolators": failed_interpolators,
        "scalar_pchip_only": True,
        "scipy_version": scipy.__version__,
        "no_oracle_calls": True,
        "candidates": [
            {
                "family": name,
                "degree": row.degree,
                "controls": row.n_controls,
                "score": row.score,
                "condition": None if name in ("cubic", "pchip") else row.condition,
            }
            for row, name in candidates
        ],
    }
    return SpectralCurveModel(inputs, values, diagnostics, selected, limits)
