"""Compile interval charts to actual BSF geometry without refitting.

The domain records the candidate topology of the input charts. Distinct chart
components remain distinct even when their embedded outputs intersect or
coincide. Nothing here certifies the topology of the output image.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from torch_bsf.core import BezierModel, BezierSpace, Points
from torch_bsf.geometry import Simploid, SimploidalSet

from ..adaptive_curve import CurveInterval as IntervalCell
from .polynomial import CurveGap, CurveLearningResult, StructureLearningResult, _Spline


@dataclass(frozen=True)
class BSFCurveExport:
    """Fresh simploidal domain, exact exported model, and parameter routing.

    Segments share endpoint controls within each source spline only. Gaps
    separate components, and singleton charts are zero-dimensional cells.
    Prediction retains the source result's endpoint-clamping convention.
    """

    domain: SimploidalSet
    model: BezierModel
    intervals: tuple[IntervalCell, ...]
    degree: int
    bounds: tuple[float, float]
    gaps: tuple[CurveGap, ...]
    topology_scope: str = "input-chart topology candidate; output-image topology is not certified"

    def predict(self, points: Any) -> torch.Tensor:
        """Predict original-coordinate inputs, leaving unresolved gaps as NaNs."""
        raw = torch.as_tensor(points)
        if raw.is_complex():
            raise ValueError("prediction inputs must be real.")
        x = raw.detach().cpu().double() if isinstance(points, torch.Tensor) else torch.as_tensor(points, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != 1 or not torch.isfinite(x).all():
            raise ValueError("prediction inputs must be finite with shape (n, 1).")
        if ((x < self.bounds[0]) | (x > self.bounds[1])).any():
            raise ValueError("prediction inputs must lie within original bounds.")
        # A declared grid can leave its first/last observed key inside bounds.
        # The source predictor extends those endpoint values to the raw bounds.
        routed = x.clamp(self.intervals[0].left, self.intervals[-1].right)
        intervals = tuple((cell.left, cell.right) for cell in self.intervals)
        ids = tuple(cell.cell_id for cell in self.intervals)
        parameters, valid = _route(self.model.space, intervals, ids, routed)
        values = torch.full((len(x), self.model.n_outputs), math.nan, dtype=torch.float64)
        if valid.any():
            with torch.no_grad():
                values[valid] = self.model(parameters.take(torch.nonzero(valid).flatten()))
        return values


def _elevate_controls(controls: torch.Tensor, target: int) -> torch.Tensor:
    """Raise a univariate Bernstein degree exactly, up to float64 rounding."""
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise ValueError("target degree must be a nonnegative integer.")
    if (
        controls.is_complex()
        or controls.ndim != 2
        or not len(controls)
        or not controls.shape[1]
        or not torch.isfinite(controls).all()
    ):
        raise ValueError("controls must be a finite nonempty matrix.")
    original = len(controls) - 1
    if target < original:
        raise ValueError("degree elevation cannot lower the degree.")
    values = controls.detach().cpu().double()
    if target == original:
        return values.clone()
    raised = torch.zeros(target + 1, values.shape[1], dtype=torch.float64)
    for k in range(target + 1):
        denominator = math.comb(target, k)
        for j in range(max(0, k - target + original), min(original, k) + 1):
            weight = math.comb(original, j) * math.comb(target - original, k - j) / denominator
            raised[k] += weight * values[j]
    if not torch.isfinite(raised).all():
        raise ArithmeticError("degree elevation overflowed.")
    return raised


def _charts(result: CurveLearningResult) -> list[_Spline]:
    if not isinstance(result, CurveLearningResult):
        raise TypeError("result must be a CurveLearningResult.")
    if not result._models:
        raise ValueError("the result has no charts to export.")
    lower, upper = result._bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or not math.isfinite(upper - lower) or lower >= upper:
        raise ValueError("original bounds must be finite and increasing.")
    charts = []
    last = lower
    for source in result._models:
        if isinstance(source.degree, bool) or not isinstance(source.degree, int) or source.degree < 0:
            raise ValueError("chart degree must be a nonnegative integer.")
        if source.knots.is_complex() or source.controls.is_complex() or source.center.is_complex():
            raise ValueError("chart knots and coefficients must be real.")
        knots = source.knots.detach().cpu().double().clone()
        controls = source.controls.detach().cpu().double().clone()
        center = source.center.detach().cpu().double().clone()
        if (
            knots.ndim != 1
            or not len(knots)
            or not torch.isfinite(knots).all()
            or (knots[1:] <= knots[:-1]).any()
            or float(knots[0]) < last
            or float(knots[-1]) > upper
        ):
            raise ValueError("chart knots must be finite, increasing, ordered, and within original bounds.")
        expected = 1 if len(knots) == 1 else (len(knots) - 1) * source.degree + 1
        if (
            controls.ndim != 2
            or controls.shape != (expected, result.values.shape[1])
            or center.shape != (result.values.shape[1],)
            or not torch.isfinite(controls).all()
            or not torch.isfinite(center).all()
        ):
            raise ValueError("chart coefficients and center must be finite and aligned with its knots and degree.")
        charts.append(_Spline(knots, source.degree, controls, center))
        last = float(knots[-1])
    return charts


def _domain(charts: list[_Spline]) -> tuple[SimploidalSet, list[tuple[int, int, Simploid]]]:
    domain = SimploidalSet()
    records = []
    for chart_index, chart in enumerate(charts):
        vertices = [domain.add_vertex() for _ in chart.knots]
        if len(vertices) == 1:
            records.append((chart_index, 0, vertices[0]))
        for piece in range(len(vertices) - 1):
            cell = domain.add_simplex(1)
            vertices[piece] = domain.identify(vertices[piece], cell.face(0, 1))
            vertices[piece + 1] = domain.identify(vertices[piece + 1], cell.face(0, 0))
            records.append((chart_index, piece, cell))
    return domain, records


def export_curve(result: CurveLearningResult) -> BSFCurveExport:
    """Convert polynomial charts without fitting or oracle evaluations.

    A common degree accommodates BSF's shared endpoint controls. Lower-degree
    pieces are elevated by the Bernstein identity, preserving their maps.
    The fresh model is independent of later source-tensor mutations.
    """
    charts = _charts(result)
    degree = max(1, *(chart.degree for chart in charts))
    domain, records = _domain(charts)
    space = BezierSpace.on_set(domain, degree=degree)
    coefficients = torch.full((space.n_controls, result.values.shape[1]), math.nan, dtype=torch.float64)
    mapping = []
    for chart_index, piece, handle in records:
        chart = charts[chart_index]
        left = float(chart.knots[piece])
        right = float(chart.knots[piece + 1]) if len(chart.knots) > 1 else left
        mapping.append(IntervalCell(left, right, space.cell_id(handle), chart_index, piece, handle))
        if len(chart.knots) == 1:
            coefficients[space.control_index((), cell=handle)] = chart.controls[0] + chart.center
        else:
            local = chart.controls[piece * chart.degree : (piece + 1) * chart.degree + 1]
            raised = _elevate_controls(local, degree) + chart.center
            for k, value in enumerate(raised):
                coefficients[space.control_index((degree - k, k), cell=handle)] = value
    if not torch.isfinite(coefficients).all():
        raise ArithmeticError("exported chart coefficients are unavailable or overflowed.")
    return BSFCurveExport(domain, BezierModel(space, coefficients), tuple(mapping), degree, result._bounds, tuple(result.gaps))


def _route(
    space: BezierSpace, intervals: tuple[tuple[float, float], ...], ids: tuple[int, ...], x: torch.Tensor
) -> tuple[Points, torch.Tensor]:
    parameters = torch.zeros(len(x), space.width, dtype=torch.float64)
    cell_ids = torch.full((len(x),), ids[0], dtype=torch.long)
    valid = torch.zeros(len(x), dtype=torch.bool)
    for (left, right), cell_id in zip(intervals, ids, strict=True):
        selected = (x[:, 0] >= left) & (x[:, 0] <= right)
        if space.cells[cell_id].width:
            u = (x[selected, 0] - left) / (right - left)
            parameters[selected] = torch.stack((1 - u, u), 1)
        cell_ids[selected] = cell_id
        valid |= selected
    return Points(parameters, cell_ids), valid


def _compile_result(result: StructureLearningResult) -> StructureLearningResult:
    compiled = export_curve(result)
    points = torch.unique(torch.cat((result.inputs[:, 0], torch.linspace(*result._bounds, 257, dtype=torch.float64))))[:, None]
    source, actual = result.predict(points), compiled.predict(points)
    finite = torch.isfinite(source)
    if not torch.equal(finite, torch.isfinite(actual)):
        raise ArithmeticError("Actual BSF export changed the source prediction support.")
    error = float((source[finite] - actual[finite]).abs().max()) if finite.any() else 0.0
    scale = max(float(compiled.model.coefficients.detach().abs().max()), torch.finfo(torch.float64).tiny)
    if error > scale * 1e-10:
        raise ArithmeticError("Actual BSF export failed its polynomial-equivalence check.")
    result.diagnostics["actual_bsf"] = {
        "components": compiled.domain.betti_numbers()[0],
        "controls": compiled.model.space.n_controls,
        "degree": compiled.degree,
        "source_polynomial_max_error_on_checked_points": error,
        "finite_masks_equal": True,
        "checked_inputs": len(points),
        "additional_oracle_rows": 0,
        "native_predictor": "actual exported BSF",
    }
    result._compiled = compiled
    return result
