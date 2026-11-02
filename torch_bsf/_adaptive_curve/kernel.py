"""Stationary scalar kernels with stable dual prediction and observed CV.

The fixed798-candidate kernel/ridge/trend grid selects by conditional input-
volume PRESS. Cached Cholesky prediction weights avoid cancellation of large
dual coefficients. No evaluator is used and no confidence bound is claimed.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpocon

from .evidence import _inputs
from .polynomial import _Spline
from .response_units import response_scale, restore_conversion, restore_score, restore_spline
from .spectral import _mass, _rows


def _kernel(first: NDArray[np.float64], second: NDArray[np.float64], kind: str, length: float) -> NDArray[np.float64]:
    radius = np.abs(first[:, None] - second[None, :]) / length
    if kind == "rbf":
        raw = np.exp(-0.5 * radius**2)
    elif kind == "matern32":
        z = math.sqrt(3) * radius
        raw = (1 + z) * np.exp(-z)
    elif kind == "matern52":
        z = math.sqrt(5) * radius
        raw = (1 + z + z**2 / 3) * np.exp(-z)
    else:
        raise ValueError("Unknown stationary kernel.")
    result: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    return result


@dataclass
class _FixedFit:
    """Fixed GLS fit, deleted-row errors and observed-data diagnostics."""

    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    loo_residuals: NDArray[np.float64]
    training_residuals: NDArray[np.float64]
    condition: float
    likelihood: float
    effective_df: float


@dataclass
class _Operator:
    """Common fitted operator coordinates; evaluated by the stable subclass."""

    support: NDArray[np.float64]
    kind: str
    length: float
    degree: int
    fit: _FixedFit
    center: NDArray[np.float64]
    scale: float


@dataclass
class _StableOperator(_Operator):
    """Cached Cholesky factors and residuals for dual prediction weights."""

    factor: tuple[NDArray[np.float64], bool]
    mean_residual: NDArray[np.float64]

    def predict(self, normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use cached dual weights, avoiding cancellation of large alpha rows."""
        unit = (normalized + 1) / 2
        polynomial = np.column_stack((np.ones(len(unit)), normalized))[:, : self.degree + 1]
        cross = _kernel(unit, (self.support + 1) / 2, self.kind, self.length)
        weights: NDArray[np.float64] = cho_solve(self.factor, cross.T, check_finite=True).T
        result: NDArray[np.float64] = (weights @ self.mean_residual + polynomial @ self.fit.beta) * self.scale + self.center
        return result


def _fixed_fit(
    covariance: NDArray[np.float64],
    values: NDArray[np.float64],
    trend: NDArray[np.float64],
    diagonal_noise: NDArray[np.float64],
    max_condition: float = 1e12,
) -> _FixedFit | None:
    """Fit one fixed kernel and mean; analytic PRESS includes mean refitting."""
    try:
        matrix = covariance + np.diag(diagonal_noise)
        factor = cho_factor(matrix, lower=True, check_finite=True)
        reciprocal, status = dpocon(factor[0], float(np.linalg.norm(matrix, 1)), uplo="L")
        if status or not math.isfinite(reciprocal) or reciprocal <= 0 or 1 / reciprocal > max_condition:
            return None
        inverse = cho_solve(factor, np.eye(len(matrix)), check_finite=True)
        projected = cho_solve(factor, trend, check_finite=True)
        gram = trend.T @ projected
        gram_factor = cho_factor(gram, lower=True, check_finite=True)
        correction = cho_solve(gram_factor, projected.T, check_finite=True)
        precision = inverse - projected @ correction
        diagonal = np.diag(precision)
        if not np.isfinite(precision).all() or not (diagonal > 0).all():
            return None
        beta = cho_solve(gram_factor, projected.T @ values, check_finite=True)
        alpha = cho_solve(factor, values - trend @ beta, check_finite=True)
        residuals = alpha / diagonal[:, None]
        training = diagonal_noise[:, None] * alpha
        energy = float(np.sum(values * alpha))
        variance = max(energy / len(values), float(np.finfo(np.float64).tiny))
        likelihood = 0.5 * (len(values) * math.log(variance) + 2 * float(np.log(np.diag(factor[0])).sum()))
        # The fitted mean is unpenalized; H=I-D*Q also includes its leverage.
        degrees = float(len(values) - diagonal_noise @ diagonal)
        if not np.isfinite(alpha).all() or not np.isfinite(beta).all() or not np.isfinite(residuals).all():
            return None
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return _FixedFit(alpha, beta, residuals, training, 1 / reciprocal, likelihood, degrees)


def _search(
    unit: NDArray[np.float64], normalized: NDArray[np.float64], mass: NDArray[np.float64]
) -> tuple[list[tuple[float, str, float, float, int, _FixedFit]], int]:
    rows: list[tuple[float, str, float, float, int, _FixedFit]] = []
    validation = np.arange(1, len(unit) - 1)
    rejected = 0
    for kind in ("rbf", "matern32", "matern52"):
        for length in np.exp2(np.linspace(-8.0, 1.0, 19)):
            covariance = _kernel(unit, unit, kind, float(length))
            for ridge in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
                # An almost coincident initial row can have subnormal mass.
                # Nonfinite regularization is rejected by the existing fit
                # checks; it should not emit a warning from this candidate.
                with np.errstate(over="ignore"):
                    diagonal = ridge / (len(unit) * mass)
                for degree in (0, 1):
                    trend = np.column_stack((np.ones(len(unit)), 2 * unit - 1))[:, : degree + 1]
                    fitted = _fixed_fit(covariance, normalized, trend, diagonal)
                    if fitted is None:
                        rejected += 1
                        continue
                    losses = np.square(fitted.loo_residuals[validation])
                    cv = float((losses.sum(axis=1) * mass[validation]).sum() / mass[validation].sum())
                    if math.isfinite(cv):
                        rows.append((cv, kind, float(length), ridge, degree, fitted))
    return rows, rejected


@dataclass
class KernelCurveModel:
    """One trained operator; polynomial conversion uses no evaluator calls."""

    inputs: torch.Tensor
    values: torch.Tensor
    diagnostics: dict[str, Any]
    _operator: _StableOperator
    _spline: _Spline | None = field(default=None, repr=False)
    _response_unit: float = field(default=1.0, repr=False)

    def predict(self, points: Any) -> torch.Tensor:
        x = _rows(points, "points")
        left, right = float(self.inputs[0, 0]), float(self.inputs[-1, 0])
        if x.shape[1] != 1 or ((x[:, 0] < left) | (x[:, 0] > right)).any():
            raise ValueError("Prediction points must lie within observed scalar support.")
        normalized = ((x[:, 0] - left) / (right - left) * 2 - 1).numpy()
        return torch.from_numpy(self._operator.predict(normalized)) * self._response_unit

    def as_spline(self) -> _Spline:
        """Convert the fitted operator with empirical degree12 checks, not a certificate."""
        from .operator_spline import approximate_operator

        if self._spline is None:
            if self._response_unit != 1:
                normalized = KernelCurveModel(self.inputs, self.values / self._response_unit, {}, self._operator)
                self._spline = restore_spline(normalized.as_spline(), self._response_unit)
                restore_conversion(self.diagnostics, normalized.diagnostics, self._response_unit)
            else:
                self._spline = approximate_operator(self.inputs, self.values, self._operator, self.predict, self.diagnostics)
        return self._spline


def fit_kernel_predictor(x: Any, y: Any, *, noise: float = 0.0) -> KernelCurveModel | None:
    """Select the fixed kernel family by observed scalar CV only."""
    points, values, _ = _inputs(x, y, None)
    if isinstance(noise, bool) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")
    if len(points) < 6 or values.shape[1] != 1 or noise > 0:
        return None
    response_unit = response_scale(values)
    if response_unit != 1:
        result = fit_kernel_predictor(points[:, None], values / response_unit)
        if result is not None:
            result.values = values
            result._response_unit = response_unit
            restore_score(result.diagnostics, "weighted_cv_mse", response_unit)
            result.diagnostics["weighted_training_mse"] = (
                result.diagnostics["weighted_training_mse"] * response_unit * response_unit
            )
        return result
    width = float(points[-1] - points[0])
    if not math.isfinite(width) or width <= 0:
        return None
    unit: NDArray[np.float64] = ((points - points[0]) / width).numpy()
    if not (np.diff(unit) > 0).all():
        return None
    amplitude = float(values.abs().max())
    if not amplitude or amplitude > math.sqrt(float(np.finfo(np.float64).max)):
        return None
    scaled = values.numpy() / amplitude
    center = scaled.mean(axis=0)
    normalized = scaled - center
    mass = _mass(points).numpy()
    rows, rejected = _search(unit, normalized, mass)
    if not rows:
        return None
    score, kind, length, ridge, degree, fitted = min(rows, key=lambda row: (row[0], row[5].effective_df))
    policy = "cv"
    covariance = _kernel(unit, unit, kind, length)
    factor = cho_factor(covariance + np.diag(ridge / (len(points) * mass)), lower=True, check_finite=True)
    trend = np.column_stack((np.ones(len(points)), 2 * unit - 1))[:, : degree + 1]
    operator = _StableOperator(
        2 * unit - 1,
        kind,
        length,
        degree,
        fitted,
        center * amplitude,
        amplitude,
        (factor[0], bool(factor[1])),
        normalized - trend @ fitted.beta,
    )
    diagnostics = {
        "family": "stationary_kernel",
        "linear_solver": "direct Cholesky GLS and cached dual prediction weights; unchanged kernel/grid/CV",
        "kernel": kind,
        "length_scale_domain_fraction": length,
        "ridge": ridge,
        "mean_degree": degree,
        "weighted_cv_mse": score * amplitude**2,
        "profiled_negative_log_likelihood": fitted.likelihood,
        "selection_policy": policy,
        "weighted_training_mse": float((np.square(fitted.training_residuals).sum(axis=1) * mass).sum()) * amplitude**2,
        "condition": fitted.condition,
        "effective_df": fitted.effective_df,
        "cv_available": True,
        "cv_endpoints_retained": True,
        "selection_measure": (
            "Input Voronoi mass on common interior deleted rows; joint constant/linear mean refitted analytically."
        ),
        "selection_scope": "Conditional geometry and hyperparameters; no unbiased adaptive-risk or confidence claim.",
        "candidate_count": len(rows),
        "rejected_condition_or_numeric": rejected,
        "no_oracle_calls": True,
    }
    return KernelCurveModel(points[:, None].clone(), values.clone(), diagnostics, operator)
