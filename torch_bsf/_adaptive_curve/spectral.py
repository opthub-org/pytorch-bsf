"""Training-only selection of conditioned spectral and C0 polynomial models.

Interleaved validation retains endpoints and uses input Voronoi masses to
target uniform input-volume risk rather than adaptive observation density.
Chebyshev degrees and C0 spline capacities compete on the same heldout rows.
No oracle, boundary labels, function identity, or external test responses are
accepted. Validation cannot certify behavior between observations.
"""

import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any

import torch

from .polynomial import _basis, _Spline


def _rows(value: Any, name: str, *, allow_vector: bool = False) -> torch.Tensor:
    raw = torch.as_tensor(value)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(f"{name} must contain real numbers.")
    result = raw.detach().cpu().double() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float64)
    if allow_vector and result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or not len(result) or not result.shape[1] or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty matrix.")
    return result.clone()


def _chebyshev(x: torch.Tensor, degree: int, left: float, right: float) -> torch.Tensor:
    t = 2 * ((x - left) / (right - left)) - 1
    columns = [torch.ones_like(t)]
    if degree:
        columns.append(t)
    for _ in range(2, degree + 1):
        columns.append(2 * t * columns[-1] - columns[-2])
    return torch.stack(columns, 1)


def _mass(x: torch.Tensor) -> torch.Tensor:
    """Positive input-volume quadrature on the observed interval."""
    if len(x) == 1:
        return torch.ones(1, dtype=torch.float64)
    width = x[-1] - x[0]
    edges = torch.cat((x[:1], x[:-1] / 2 + x[1:] / 2, x[-1:]))
    return (edges[1:] - edges[:-1]) / width


def _solve(
    design: torch.Tensor, centered: torch.Tensor, mass: torch.Tensor, max_condition: float
) -> tuple[torch.Tensor, float] | None:
    if design.shape[1] > len(design):
        return None
    weight = (mass / mass.max()).sqrt()
    weighted = design * weight[:, None]
    scales = torch.linalg.vector_norm(weighted, dim=0)
    if not (scales > 0).all():
        return None
    normalized = weighted / scales
    solved = torch.linalg.lstsq(normalized, centered * weight[:, None], driver="gelsd", rcond=1 / max_condition)
    if int(solved.rank) != design.shape[1] or not torch.isfinite(solved.solution).all():
        return None
    condition = float(solved.singular_values[0] / solved.singular_values[-1])
    return solved.solution / scales[:, None], condition


@dataclass
class _Candidate:
    kind: str
    degree: int
    knots: torch.Tensor
    coefficients: torch.Tensor
    center: torch.Tensor
    score: float
    condition: float
    n_controls: int

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "chebyshev":
            return _chebyshev(x, self.degree, float(self.knots[0]), float(self.knots[-1])) @ self.coefficients + self.center
        return _basis(x, self.knots, self.degree) @ self.coefficients + self.center


def _linear_predict(x: torch.Tensor, y: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    index = torch.searchsorted(x, at.contiguous()).clamp(1, len(x) - 1)
    fraction = ((at - x[index - 1]) / (x[index] - x[index - 1]))[:, None]
    return y[index - 1] * (1 - fraction) + y[index] * fraction


def _folds(n: int) -> list[torch.Tensor]:
    groups = []
    for offset in range(3):
        valid = torch.arange(n) % 3 == offset
        valid[0] = valid[-1] = False
        if valid.any():
            groups.append(valid)
    return groups


def _score(
    design: torch.Tensor,
    y: torch.Tensor,
    mass: torch.Tensor,
    folds: list[torch.Tensor],
    noise: float,
    max_condition: float,
) -> tuple[float, float] | None:
    numerator = denominator = condition = 0.0
    for validation in folds:
        training = ~validation
        center = y[training].mean(0)
        result = _solve(design[training], y[training] - center, mass[training], max_condition)
        if result is None:
            return None
        coefficients, current_condition = result
        residual = design[validation] @ coefficients + center - y[validation]
        numerator += float((residual.square().sum(1) * mass[validation]).sum())
        denominator += float(mass[validation].sum())
        condition = max(condition, current_condition)
    score = numerator / denominator + noise * design.shape[1] / max(1, len(y) - math.ceil(len(y) / 3))
    return score, condition


def _designs(x: torch.Tensor, max_degree: int, max_pieces: int) -> list[tuple[str, int, torch.Tensor, torch.Tensor]]:
    result = []
    for degree in sorted({0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, max_degree}):
        if degree <= max_degree and degree + 1 <= max(2, 2 * len(x) // 3 - 1):
            result.append(("chebyshev", degree, x[[0, -1]], _chebyshev(x, degree, float(x[0]), float(x[-1]))))
    for pieces in (2, 3, 4, 6, 8, 12, 16):
        if pieces > max_pieces:
            continue
        knots = x[torch.linspace(0, len(x) - 1, pieces + 1).round().long().unique()]
        if len(knots) != pieces + 1:
            continue
        for degree in (1, 2, 3, 5):
            if pieces * degree + 1 <= max(2, 2 * len(x) // 3 - 1):
                result.append(("spline", degree, knots, _basis(x, knots, degree)))
    return result


def _bernstein_controls(coefficients: torch.Tensor, pieces: int) -> torch.Tensor:
    """Convert exactly represented Chebyshev polynomial before float rounding.

    High precision controls are subdivided before returningfloat64. This
    avoids enormous alternating global Bernstein coefficients destroying
    evaluation accuracy for otherwise well-conditioned spectral fits.
    """
    degree = len(coefficients) - 1
    with localcontext() as context:
        context.prec = 128
        matrix = []
        for k in range(degree + 1):
            row = []
            for j in range(degree + 1):
                integer = sum(
                    (-1) ** (j - r) * math.comb(2 * j, 2 * r) * math.comb(degree - j, k - r)
                    for r in range(max(0, k - degree + j), min(j, k) + 1)
                )
                row.append(Decimal(integer) / Decimal(math.comb(degree, k)))
            matrix.append(row)
        original = [[Decimal.from_float(float(v)) for v in row] for row in coefficients]
        remaining = [
            [sum((matrix[k][j] * original[j][d] for j in range(degree + 1)), Decimal(0)) for d in range(coefficients.shape[1])]
            for k in range(degree + 1)
        ]
        output = []
        for current in range(pieces):
            if current == pieces - 1:
                output.append(remaining)
                break
            t = Decimal(1) / Decimal(pieces - current)
            level = remaining
            left, right = [level[0]], [level[-1]]
            while len(level) > 1:
                level = [
                    [(1 - t) * a + t * b for a, b in zip(u, v, strict=True)]
                    for u, v in zip(level[:-1], level[1:], strict=True)
                ]
                left.append(level[0])
                right.append(level[-1])
            output.append(left)
            remaining = list(reversed(right))
        controls = [row for i, piece in enumerate(output) for row in (piece if i == 0 else piece[1:])]
        return torch.tensor([[float(value) for value in row] for row in controls], dtype=torch.float64)


@dataclass
class SpectralCurveModel:
    """Selected native predictor with an exact polynomial BSF representation."""

    inputs: torch.Tensor
    values: torch.Tensor
    diagnostics: dict[str, Any]
    _selected: _Candidate = field(repr=False)
    _bounds: tuple[float, float] = field(repr=False)
    _exported: _Spline | None = field(default=None, repr=False)

    def predict(self, points: Any) -> torch.Tensor:
        """Predict bounded inputs, clamping beyond observed endpoint support."""
        raw = torch.as_tensor(points)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError("prediction inputs must be real.")
        x = raw.detach().cpu().double() if isinstance(points, torch.Tensor) else torch.as_tensor(points, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != 1 or not torch.isfinite(x).all():
            raise ValueError("prediction inputs must be finite with shape (n, 1).")
        if ((x < self._bounds[0]) | (x > self._bounds[1])).any():
            raise ValueError("prediction inputs must lie within bounds.")
        clipped = x[:, 0].clamp(float(self.inputs[0, 0]), float(self.inputs[-1, 0]))
        return self._selected.predict(clipped)

    def as_spline(self) -> _Spline:
        """Return Bernstein controls for existing export_curve, without a fit."""
        if self._exported is None:
            selected = self._selected
            if selected.kind == "chebyshev":
                pieces = max(1, math.ceil(selected.degree / 6))
                knots = torch.linspace(float(selected.knots[0]), float(selected.knots[-1]), pieces + 1, dtype=torch.float64)
                controls = _bernstein_controls(selected.coefficients, pieces)
                self._exported = _Spline(knots, selected.degree, controls, selected.center.clone())
            else:
                self._exported = _Spline(
                    selected.knots.clone(), selected.degree, selected.coefficients.clone(), selected.center.clone()
                )
        return self._exported


def _bounds(coordinates: torch.Tensor, bounds: Any) -> tuple[float, float]:
    if bounds is not None:
        raw_bounds = torch.as_tensor(bounds)
        if raw_bounds.is_complex() or raw_bounds.dtype == torch.bool or raw_bounds.shape != (2,):
            raise ValueError("bounds must contain two finite real limits.")
    limits = (
        (float(coordinates[0]), float(coordinates[-1]))
        if bounds is None
        else tuple(torch.as_tensor(bounds, dtype=torch.float64).tolist())
    )
    if (
        len(limits) != 2
        or not all(math.isfinite(v) for v in limits)
        or limits[0] > coordinates[0]
        or limits[1] < coordinates[-1]
    ):
        raise ValueError("bounds must contain the observed input support.")
    return limits


def _validated(
    x: Any, y: Any, noise: float, bounds: Any, max_degree: int, max_pieces: int, max_condition: float
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    inputs = _rows(x, "x", allow_vector=True)
    values = _rows(y, "y")
    if inputs.shape[1] != 1 or len(inputs) != len(values):
        raise ValueError("x must have one column aligned with y.")
    if isinstance(noise, bool) or not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")
    for name, value in (("max_degree", max_degree), ("max_pieces", max_pieces)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(max_condition, bool) or not math.isfinite(max_condition) or max_condition <= 1:
        raise ValueError("max_condition must be finite and greater than one.")
    order = torch.argsort(inputs[:, 0], stable=True)
    inputs, values = inputs[order], values[order]
    coordinates = inputs[:, 0]
    if len(inputs) > 1 and not (coordinates[1:] > coordinates[:-1]).all():
        raise ValueError("x coordinates must be unique.")
    if not math.isfinite(float(coordinates[-1] - coordinates[0])):
        raise ValueError("observed input width must be finite.")
    return inputs, values, _bounds(coordinates, bounds)


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
    """Select using observed labels only; no queries or gap inference occur.

    ``noise`` is estimated total vector variance of one observed response.
    Voronoi weights approximate uniform-volume risk on observed input support.
    The CV score is empirical, not a confidence interval or population bound.
    Exact duplicate coordinates are rejected; aggregate replicates first.
    """
    inputs, values, limits = _validated(x, y, noise, bounds, max_degree, max_pieces, max_condition)
    coordinates = inputs[:, 0]
    center = values.mean(0)
    centered = values - center
    if len(inputs) < 6:
        selected = _Candidate(
            "linear", 1 if len(inputs) > 1 else 0, coordinates.clone(), centered.clone(), center, 0.0, 1.0, len(inputs)
        )
        return SpectralCurveModel(inputs, values, {"kind": "linear", "cv_available": False}, selected, limits)
    mass, folds = _mass(coordinates), _folds(len(inputs))
    linear_error = total_mass = 0.0
    for valid in folds:
        residual = _linear_predict(coordinates[~valid], values[~valid], coordinates[valid]) - values[valid]
        linear_error += float((residual.square().sum(1) * mass[valid]).sum())
        total_mass += float(mass[valid].sum())
    linear_error /= total_mass
    candidates = [
        _Candidate("linear", 1, coordinates.clone(), centered.clone(), center, linear_error + noise, 1.0, len(inputs))
    ]
    rejected = 0
    for kind, degree, knots, design in _designs(coordinates, max_degree, max_pieces):
        score = _score(design, centered, mass, folds, noise, max_condition)
        solved = _solve(design, centered, mass, max_condition)
        if score is None or solved is None:
            rejected += 1
            continue
        candidates.append(
            _Candidate(kind, degree, knots, solved[0], center, score[0], max(score[1], solved[1]), design.shape[1])
        )
    best = min(row.score for row in candidates)
    scale = float((centered.square().sum(1) * mass).sum())
    tolerance = max(scale * torch.finfo(torch.float64).eps ** 2 * 256, best * 1e-8)
    selected = min((row for row in candidates if row.score <= best + tolerance), key=lambda row: row.n_controls)
    diagnostics = {
        "kind": selected.kind,
        "degree": selected.degree,
        "n_controls_native": selected.n_controls,
        "weighted_cv_mse": selected.score,
        "linear_weighted_cv_mse": candidates[0].score,
        "condition": selected.condition,
        "condition_limit": max_condition,
        "rejected_condition_or_rank_candidates": rejected,
        "cv_available": True,
        "cv_folds": len(folds),
        "cv_endpoints_retained": True,
        "selection_measure": "input Voronoi quadrature on observed support; not observation-row average",
        "no_oracle_calls": True,
        "candidates": [
            {
                "kind": row.kind,
                "degree": row.degree,
                "controls": row.n_controls,
                "score": row.score,
                "condition": row.condition,
            }
            for row in candidates
        ],
    }
    return SpectralCurveModel(inputs, values, diagnostics, selected, limits)
