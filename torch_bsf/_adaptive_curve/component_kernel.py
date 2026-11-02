"""Observed-chart random-intercept kernels with conditional training CV.

The covariance adds a positive-semidefinite same-chart indicator to a
stationary kernel. Fixed input-volume masses and observed chart boundaries
remain fixed during PRESS; no end-to-end risk calibration is claimed.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from .evidence import _inputs
from .kernel import _fixed_fit, _FixedFit, _kernel
from .operator_spline import approximate_operator
from .polynomial import _Spline
from .shared_spline import _partition, _weighted_charts, restrict_spline
from .spectral import _rows

KINDS = ("rbf", "matern32", "matern52")
LENGTHS = tuple(float(x) for x in np.exp2(np.linspace(-8.0, 1.0, 19)))
ETAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BASE_RIDGES = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
EXPANSION_RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _covariance(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    labels: NDArray[np.int64],
    other_labels: NDArray[np.int64],
    kind: str,
    length: float,
    eta: float,
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = _kernel(first, second, kind, length) + eta * (labels[:, None] == other_labels[None, :])
    return result


@dataclass
class _SmoothPosterior:
    """Stable cached dual predictor for the shared smooth posterior."""

    support: NDArray[np.float64]
    unit: NDArray[np.float64]
    kind: str
    length: float
    degree: int
    factor: tuple[NDArray[np.float64], bool]
    residual: NDArray[np.float64]
    beta: NDArray[np.float64]
    center: NDArray[np.float64]
    scale: float

    def predict(self, normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        at = (normalized + 1) / 2
        cross = _kernel(at, self.unit, self.kind, self.length)
        weights = cho_solve(self.factor, cross.T, check_finite=True).T
        trend = np.column_stack((np.ones(len(at)), normalized))[:, : self.degree + 1]
        result: NDArray[np.float64] = (weights @ self.residual + trend @ self.beta) * self.scale + self.center
        return result


@dataclass
class ComponentKernelModel:
    """Stable shared posterior plus one random intercept per inferred chart."""

    inputs: torch.Tensor
    values: torch.Tensor
    groups: list[torch.Tensor]
    offsets: NDArray[np.float64]
    diagnostics: dict[str, Any]
    _smooth: _SmoothPosterior = field(repr=False)
    _models: list[_Spline] | None = field(default=None, repr=False)

    def _smooth_predict(self, points: Any) -> torch.Tensor:
        at = _rows(points, "prediction inputs")
        if at.shape[1] != 1:
            raise ValueError("Prediction inputs must have one coordinate.")
        left, right = float(self.inputs[0, 0]), float(self.inputs[-1, 0])
        if ((at < left) | (at > right)).any():
            raise ValueError("Prediction inputs must lie inside the observed bounds.")
        normalized = ((at[:, 0] - left) / (right - left) * 2 - 1).numpy()
        return torch.from_numpy(self._smooth.predict(normalized))

    def predict(self, points: Any) -> torch.Tensor:
        """Return NaNs inside each open excluded interval."""
        at = _rows(points, "prediction inputs")
        result = self._smooth_predict(at)
        covered = torch.zeros(len(at), dtype=torch.bool)
        for i, group in enumerate(self.groups):
            use = (at[:, 0] >= self.inputs[group[0], 0]) & (at[:, 0] <= self.inputs[group[-1], 0])
            result[use] += torch.from_numpy(self.offsets[i])
            covered |= use
        result[~covered] = math.nan
        return result

    def as_splines(self) -> list[_Spline]:
        """Approximate only the fitted smooth operator, without response queries."""
        if self._models is None:
            smooth = approximate_operator(self.inputs, self.values, self._smooth, self._smooth_predict, self.diagnostics)
            models = []
            maximum = 0.0
            for i, group in enumerate(self.groups):
                left, right = float(self.inputs[group[0], 0]), float(self.inputs[group[-1], 0])
                chart = restrict_spline(smooth, left, right)
                chart.center += torch.from_numpy(self.offsets[i])
                # Check restriction, physical mapping and the added offset in
                # the actual controls, not just the smooth pre-conversion.
                check = torch.unique(torch.cat((self.inputs[group, 0], torch.linspace(left, right, 129, dtype=torch.float64))))
                error = float((chart.predict(check) - self.predict(check[:, None])).norm(dim=1).max())
                maximum = max(maximum, error)
                models.append(chart)
            if not math.isfinite(maximum) or maximum > self.diagnostics["conversion_checked_tolerance"]:
                raise ArithmeticError("Restricted component Bernstein model failed its checked tolerance.")
            self.diagnostics["restricted_conversion_checked_max_error"] = maximum
            self.diagnostics["chart_polynomial_pieces"] = [len(model.knots) - 1 for model in models]
            self._models = models
        return self._models


def _contrasts(
    predicted: NDArray[np.float64], values: NDArray[np.float64], groups: list[torch.Tensor], thresholds: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for i, threshold in enumerate(thresholds):
        left, right = int(groups[i][-1]), int(groups[i + 1][0])
        observed = values[right] - values[left]
        actual = predicted[right] - predicted[left]
        original, amplitude = float(np.linalg.norm(observed)), float(np.linalg.norm(actual))
        projection = float(np.dot(actual, observed / original)) if original else 0.0
        rows.append(
            {
                "observed_endpoint_norm": original,
                "predicted_endpoint_norm": amplitude,
                "predicted_projected_jump": projection,
                "detector_threshold": threshold,
                "direction_cosine": projection / amplitude if amplitude else None,
                "predicted_observed_ratio": amplitude / original if original else None,
                "admissible": projection > threshold,
            }
        )
    return rows


def _risk(fit: _FixedFit, mass: NDArray[np.float64], validation: NDArray[np.bool_]) -> float:
    return float((np.square(fit.loo_residuals[validation]).sum(1) * mass[validation]).sum() / mass[validation].sum())


def _eta_search(
    matrices: dict[tuple[str, float], NDArray[np.float64]],
    features: NDArray[np.float64],
    trends: list[NDArray[np.float64]],
    mass: NDArray[np.float64],
    validation: NDArray[np.bool_],
    indicator: NDArray[np.float64],
    groups: list[torch.Tensor],
    thresholds: list[float],
    scale: float,
    ridge: float,
) -> tuple[dict[str, Any] | None, int, int]:
    diagonal = ridge / (len(mass) * mass)
    rejected, inadmissible = 0, 0
    best: dict[str, Any] | None = None
    for (kind, length), smooth in matrices.items():
        for eta in ETAS:
            covariance = smooth + eta * indicator
            for degree, trend in enumerate(trends):
                fitted = _fixed_fit(covariance, features, trend, diagonal)
                if fitted is None:
                    rejected += 1
                    continue
                cv = _risk(fitted, mass, validation)
                if not math.isfinite(cv) or (best is not None and cv >= best["score"]):
                    continue
                contrasts = _contrasts((features - fitted.training_residuals) * scale, features * scale, groups, thresholds)
                if not all(row["admissible"] for row in contrasts):
                    inadmissible += 1
                    continue
                best = dict(score=cv, kind=kind, length=length, ridge=ridge, degree=degree, eta=eta)
    return best, rejected, inadmissible


def _expanded_nugget(
    matrices: dict[tuple[str, float], NDArray[np.float64]],
    features: NDArray[np.float64],
    trends: list[NDArray[np.float64]],
    mass: NDArray[np.float64],
    validation: NDArray[np.bool_],
) -> tuple[tuple[float, float] | None, int, list[dict[str, Any]]]:
    best: tuple[float, float] | None = None
    rejected, trace = 0, []
    for ridge in (*BASE_RIDGES, *EXPANSION_RIDGES):
        if ridge > BASE_RIDGES[-1] and best is not None and best[1] < ridge / 10:
            break
        diagonal = ridge / (len(mass) * mass)
        for covariance in matrices.values():
            for trend in trends:
                fitted = _fixed_fit(covariance, features, trend, diagonal)
                if fitted is None:
                    rejected += 1
                    continue
                errors = np.square(fitted.loo_residuals[validation]).sum(1)
                cv = float((errors * mass[validation]).sum() / mass[validation].sum())
                if math.isfinite(cv) and (best is None or cv < best[0]):
                    best = (cv, ridge)
        trace.append(
            {"largest_tested_ridge": ridge, "selected_ridge": best[1] if best else None, "best_cv": best[0] if best else None}
        )
    return best, rejected, trace


def fit_component_kernel(x: Any, y: Any, gaps: Any = (), *, noise: float = 0.0) -> ComponentKernelModel | None:
    """Fit the declared expanding-grid GP from observed rows only."""
    points, values, _ = _inputs(x, y, None)
    if isinstance(noise, bool) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")
    gaps = tuple(gaps)
    labels, groups = _partition(points, gaps)
    if len(points) < 6 or noise > 0 or any(len(group) < 3 for group in groups):
        return None
    thresholds = [float(gap.threshold) for gap in gaps]
    if any(not math.isfinite(threshold) or threshold < 0 for threshold in thresholds):
        raise ValueError("Gap thresholds must be finite and nonnegative.")
    mass, validation = _weighted_charts(points, groups)
    unit: NDArray[np.float64] = ((points - points[0]) / (points[-1] - points[0])).numpy()
    if not np.isfinite(unit).all() or not (np.diff(unit) > 0).all():
        return None
    response = values.numpy()
    center = response.mean(0)
    amplitude = float(np.linalg.norm(response - center, axis=1).max())
    if not math.isfinite(amplitude) or amplitude <= 0:
        return None
    normalized = (response - center) / amplitude
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    features = u * singular
    matrices = {(kind, length): _kernel(unit, unit, kind, length) for kind in KINDS for length in LENGTHS}
    trends = [np.column_stack((np.ones(len(unit)), 2 * unit - 1))[:, : degree + 1] for degree in (0, 1)]
    first, rejected, trace = _expanded_nugget(matrices, features, trends, mass.numpy(), validation.numpy())
    if first is None:
        return None
    indicator = (labels.numpy()[:, None] == labels.numpy()[None, :]).astype(np.float64)
    selected, additional, inadmissible = _eta_search(
        matrices, features, trends, mass.numpy(), validation.numpy(), indicator, groups, thresholds, amplitude, first[1]
    )
    if selected is None:
        return None
    kind, length, ridge, degree, eta = (selected[key] for key in ("kind", "length", "ridge", "degree", "eta"))
    covariance = _covariance(unit, unit, labels.numpy(), labels.numpy(), kind, length, eta)
    diagonal = ridge / (len(unit) * mass.numpy())
    trend = trends[degree]
    fitted = _fixed_fit(covariance, normalized, trend, diagonal)
    if fitted is None:
        return None
    factor = cho_factor(covariance + np.diag(diagonal), lower=True, check_finite=True)
    residual = normalized - trend @ fitted.beta
    indicators = (labels.numpy()[:, None] == np.arange(len(groups))[None, :]).astype(np.float64)
    offsets = cho_solve(factor, eta * indicators, check_finite=True).T @ residual * amplitude
    operator = _SmoothPosterior(2 * unit - 1, unit, kind, length, degree, factor, residual, fitted.beta, center, amplitude)
    predicted = operator.predict(2 * unit - 1) + offsets[labels.numpy()]
    contrasts = _contrasts(predicted, response, groups, thresholds)
    if not all(row["admissible"] for row in contrasts):
        return None
    diagnostics = {
        "family": "expanded_stationary_plus_chart_random_intercept_gp",
        **selected,
        "weighted_cv_mse": float(selected["score"] * amplitude**2),
        "condition": fitted.condition,
        "effective_df": fitted.effective_df,
        "ridge_expansion_trace": trace,
        "ridge_expansion_cap": EXPANSION_RIDGES[-1],
        "ridge_expansion_stop": "cap" if trace[-1]["selected_ridge"] == EXPANSION_RIDGES[-1] else "interior_best",
        "search_rejections": rejected + additional,
        "gap_consistency_rejections": inadmissible,
        "chart_validation_masses": [float(mass[group[1:-1]].sum()) for group in groups],
        "selection_measure": "covered input-volume mass, common interior leave-one-out rows",
        "cv_conditional_on_observed_charts": True,
        "retained_gap_contrasts": contrasts,
        "new_oracle_rows": 0,
    }
    return ComponentKernelModel(points[:, None], values, groups, offsets, diagnostics, operator)
