"""Training-only rational approximation compiled to connected BSF polynomials.

The shared-weight barycentric construction follows the AAA algorithm of
Nakatsukasa, Sete and Trefethen (2018, doi:10.1137/16M1106122). Real poles in
the observed interval cause rejection. Approximation of the fitted rational
operator uses no oracle responses. Its conversion tolerance is a numerical
check, not a certified uniform error bound or evidence of a discontinuity.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.linalg import eigvals

from .polynomial import _Spline
from .response_units import response_scale, restore_conversion, restore_score, restore_spline
from .spectral import _bernstein_controls, _mass, _rows


@dataclass
class _Rational:
    support: np.ndarray[Any, np.dtype[np.float64]]
    values: np.ndarray[Any, np.dtype[np.float64]]
    weights: np.ndarray[Any, np.dtype[np.float64]]
    center: np.ndarray[Any, np.dtype[np.float64]]

    def predict(self, at: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        distance = at[:, None] - self.support[None, :]
        equal = distance == 0
        safe = np.where(equal, 1.0, distance)
        terms = self.weights[None, :] / safe
        with np.errstate(divide="ignore", invalid="ignore"):
            result = terms @ self.values / terms.sum(axis=1)[:, None]
        for i in np.flatnonzero(equal.any(axis=1)):
            result[i] = self.values[np.flatnonzero(equal[i])[0]]
        returned: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(result + self.center, dtype=np.float64)
        return returned

    def has_real_pole(self, left: float, right: float) -> bool:
        n = len(self.support)
        a = np.zeros((n + 1, n + 1), dtype=np.float64)
        a[0, 1:], a[1:, 0] = self.weights, 1.0
        a[1:, 1:] = np.diag(self.support)
        b = np.diag(np.r_[0.0, np.ones(n)])
        poles = eigvals(a, b, check_finite=True)
        finite = poles[np.isfinite(poles)]
        real = finite.real[np.abs(finite.imag) <= 1e-10 * np.maximum(1.0, np.abs(finite.real))]
        return bool(((real >= left) & (real <= right)).any())


def _aaa(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]]) -> _Rational:
    center = y.mean(axis=0)
    values = y - center
    scale = max(float(np.max(np.linalg.norm(values, axis=1))), np.finfo(np.float64).tiny)
    normalized = values / scale
    approximation = np.zeros_like(normalized)
    support: list[int] = []
    remaining = np.ones(len(x), dtype=bool)
    weights = np.ones(1)
    for _ in range(min(16, max(1, len(x) // 2))):
        error = np.linalg.norm(normalized - approximation, axis=1)
        error[~remaining] = -1
        j = int(np.argmax(error))
        support.append(j)
        remaining[j] = False
        z, f = x[support], normalized[support]
        cauchy = 1.0 / (x[remaining, None] - z[None, :])
        loewner = (normalized[remaining, None, :] - f[None, :, :]) * cauchy[:, :, None]
        matrix = loewner.transpose(0, 2, 1).reshape(-1, len(support))
        weights = np.linalg.svd(matrix, full_matrices=False)[2][-1]
        model = _Rational(z, values[support], weights, center)
        approximation = (model.predict(x) - center) / scale
        if not np.isfinite(approximation).all() or not np.isfinite(weights).all():
            raise ArithmeticError("Rational weights or training predictions are nonfinite.")
        if np.max(np.linalg.norm(normalized - approximation, axis=1)) <= 1e-11:
            break
    return _Rational(x[support], values[support], weights, center)


@dataclass
class RationalCurveModel:
    """An accepted fitted rational operator with observed-label CV diagnostics."""

    inputs: torch.Tensor
    values: torch.Tensor
    diagnostics: dict[str, Any]
    _rational: _Rational
    _spline: _Spline | None = None
    _response_unit: float = 1.0

    def predict(self, points: Any) -> torch.Tensor:
        x = _rows(points, "points")
        if x.shape[1] != 1 or ((x[:, 0] < self.inputs[0, 0]) | (x[:, 0] > self.inputs[-1, 0])).any():
            raise ValueError("prediction points must lie within the observed one-dimensional support.")
        left, right = float(self.inputs[0, 0]), float(self.inputs[-1, 0])
        z = ((x[:, 0] - left) / (right - left) * 2 - 1).numpy()
        return torch.from_numpy(self._rational.predict(z)) * self._response_unit

    def as_spline(self) -> _Spline:
        """Approximate the fitted operator by checked degree12 connected pieces."""
        if self._spline is not None:
            return self._spline
        if self._response_unit != 1:
            normalized = RationalCurveModel(self.inputs, self.values / self._response_unit, {}, self._rational)
            self._spline = restore_spline(normalized.as_spline(), self._response_unit)
            restore_conversion(self.diagnostics, normalized.diagnostics, self._response_unit)
            return self._spline
        degree = 12
        nodes = np.cos(np.pi * np.arange(degree + 1) / degree)
        checks = np.cos(np.pi * (np.arange(2 * degree + 3) + 0.5) / (2 * degree + 3))
        response = self.values.numpy()
        amplitude = float(np.max(np.linalg.norm(response - response.mean(axis=0), axis=1)))
        magnitude = float(np.linalg.norm(response, axis=1).max())
        tolerance = max(
            amplitude * 1e-10, magnitude * float(np.finfo(np.float64).eps) * 2048, float(np.finfo(np.float64).tiny)
        )
        pending = [(-1.0, 1.0, 0)]
        pieces = []
        maximum_error = 0.0
        physical_left, physical_right = float(self.inputs[0, 0]), float(self.inputs[-1, 0])
        observed = (self.inputs[:, 0].numpy() - physical_left) / (physical_right - physical_left) * 2 - 1
        required = np.unique(np.r_[observed, self._rational.support])
        validation_points = []
        while pending:
            left, right, depth = pending.pop()
            z = (left + right) / 2 + (right - left) / 2 * nodes
            values = self._rational.predict(z)
            coefficients = np.polynomial.chebyshev.chebfit(nodes, values, degree)
            at = (left + right) / 2 + (right - left) / 2 * checks
            at = np.unique(np.r_[left, right, at, required[(required >= left) & (required <= right)]])
            controls = _bernstein_controls(torch.from_numpy(coefficients), 1)
            controls[0] = torch.from_numpy(self._rational.predict(np.array([left]))[0])
            controls[-1] = torch.from_numpy(self._rational.predict(np.array([right]))[0])
            t = (at - left) / (right - left)
            design = np.column_stack([math.comb(degree, j) * t**j * (1 - t) ** (degree - j) for j in range(degree + 1)])
            approximation = design @ controls.numpy()
            error = float(np.linalg.norm(approximation - self._rational.predict(at), axis=1).max())
            if not math.isfinite(error):
                raise ArithmeticError("Rational-to-polynomial conversion produced nonfinite errors.")
            if error > tolerance:
                if depth >= 16:
                    raise ArithmeticError("Rational-to-polynomial conversion did not meet its checked tolerance.")
                middle = left / 2 + right / 2
                pending.extend(((middle, right, depth + 1), (left, middle, depth + 1)))
                continue
            maximum_error = max(maximum_error, error)
            # Pin shared endpoints to one evaluation of the same operator;
            # rounding of separate fits must not create disconnected charts.
            pieces.append((left, right, controls))
            validation_points.extend(at.tolist())
            if len(pieces) + len(pending) > 1024:
                raise ArithmeticError("Rational conversion exceeded its polynomial-piece limit.")
        pieces.sort(key=lambda item: item[0])
        knots = torch.tensor([pieces[0][0], *[piece[1] for piece in pieces]], dtype=torch.float64)
        knots = physical_left + (knots + 1) / 2 * (physical_right - physical_left)
        knots[0], knots[-1] = physical_left, physical_right
        if not (knots[1:] > knots[:-1]).all():
            raise ArithmeticError("Polynomial pieces collapse at the represented physical input scale.")
        controls = torch.cat([pieces[0][2], *[piece[2][1:] for piece in pieces[1:]]])
        spline = _Spline(knots, degree, controls, torch.zeros(self.values.shape[1], dtype=torch.float64))
        actual = np.unique(
            np.r_[
                self.inputs[:, 0].numpy(),
                physical_left + (np.array(validation_points) + 1) / 2 * (physical_right - physical_left),
            ]
        )
        actual = np.clip(actual, physical_left, physical_right)
        expected = self.predict(torch.from_numpy(actual[:, None]))
        actual_error = float((spline.predict(torch.from_numpy(actual)) - expected).norm(dim=1).max())
        if not math.isfinite(actual_error) or actual_error > tolerance:
            raise ArithmeticError("Actual Bernstein conversion failed its observed/support/verification-point check.")
        self._spline = spline
        self.diagnostics.update(
            polynomial_pieces=len(pieces),
            conversion_checked_max_error=max(maximum_error, actual_error),
            conversion_checked_tolerance=tolerance,
            conversion_is_certified=False,
            conversion_oracle_rows=0,
        )
        return self._spline


def _validated_fit(inputs: torch.Tensor, values: torch.Tensor) -> RationalCurveModel | None:
    coordinates = inputs[:, 0]
    z = ((coordinates - coordinates[0]) / (coordinates[-1] - coordinates[0]) * 2 - 1).numpy()
    if not np.isfinite(z).all() or not (np.diff(z) > 0).all():
        return None
    response = values.numpy()
    mass = _mass(coordinates).numpy()
    error = 0.0
    for held in range(1, len(inputs) - 1):
        use = np.arange(len(inputs)) != held
        model = _aaa(z[use], response[use])
        if model.has_real_pole(-1.0, 1.0):
            return None
        predicted = model.predict(z[held : held + 1])[0]
        if not np.isfinite(predicted).all():
            return None
        error += float(np.square(predicted - response[held]).sum()) * float(mass[held])
    model = _aaa(z, response)
    if model.has_real_pole(-1.0, 1.0):
        return None
    return RationalCurveModel(
        inputs,
        values,
        {"kind": "aaa_rational", "cv_mse": error / float(mass[1:-1].sum()), "support_points": len(model.support)},
        model,
    )


def fit_rational(x: Any, y: Any) -> RationalCurveModel | None:
    """Fit and validate shared vector barycentric weights using observed data.

    Leave-one-interior-row-out predictions use at most16 support points and
    reject real poles on each training interval. Failure returns None so a
    connected polynomial model remains available. The returned CV score is
    weighted by original input Voronoi mass, approximating uniform input risk.
    """
    inputs, values = _rows(x, "x", allow_vector=True), _rows(y, "y")
    if inputs.shape[1] != 1 or len(inputs) != len(values):
        raise ValueError("Inputs must be unique ordered one-dimensional rows aligned with outputs.")
    if len(inputs) < 8:
        return None
    if not (inputs[1:, 0] > inputs[:-1, 0]).all():
        raise ValueError("Inputs must be increasing and unique.")
    unit = response_scale(values)
    if unit != 1:
        normalized = fit_rational(inputs, values / unit)
        if normalized is not None:
            normalized.values = values
            normalized._response_unit = unit
            restore_score(normalized.diagnostics, "cv_mse", unit)
        return normalized
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _validated_fit(inputs, values)
    except (np.linalg.LinAlgError, ArithmeticError):
        return None
