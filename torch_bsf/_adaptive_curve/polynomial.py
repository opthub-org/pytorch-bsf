"""Univariate C0 polynomial charts and charged initialization helpers."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ..adaptive_curve import CurveGap
from .queries import QuerySession

if TYPE_CHECKING:
    from .export import BSFCurveExport


@dataclass
class _Spline:
    knots: torch.Tensor
    degree: int
    controls: torch.Tensor
    center: torch.Tensor

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.knots) == 1 or self.degree == 0:
            return self.controls[:1].expand(len(x), -1) + self.center
        indices = torch.searchsorted(self.knots, x.contiguous(), right=True).sub(1).clamp(0, len(self.knots) - 2)
        t = ((x - self.knots[indices]) / (self.knots[indices + 1] - self.knots[indices])).clamp(0, 1)
        # Query memory depends on local degree and output size, not on the
        # number of pieces. Keep the global design only for regression.
        result = torch.zeros((len(x), self.controls.shape[1]), dtype=torch.float64)
        for j in range(self.degree + 1):
            weight = math.comb(self.degree, j) * t.pow(j) * (1 - t).pow(self.degree - j)
            result += weight[:, None] * self.controls[indices * self.degree + j]
        return result + self.center


@dataclass
class CurveLearningResult:
    """Observed samples, selective predictor, and honest experimental diagnostics."""

    inputs: torch.Tensor
    values: torch.Tensor
    gaps: tuple[CurveGap, ...]
    status: str
    diagnostics: dict[str, Any]
    _models: list[_Spline] = field(repr=False)
    _bounds: tuple[float, float] = field(repr=False)

    def predict(self, points: Any) -> torch.Tensor:
        """Predict inside observed charts, returning NaNs across proposed gaps.

        Input bounds are checked. Grid users should pass canonical inputs when
        evaluating the discrete protocol; interpolation is only a surrogate.
        """
        raw = torch.as_tensor(points)
        if raw.is_complex():
            raise ValueError("prediction inputs must be real.")
        x = raw.detach().cpu().double() if isinstance(points, torch.Tensor) else torch.as_tensor(points, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != 1 or not torch.isfinite(x).all():
            raise ValueError("prediction inputs must be finite with shape (n, 1).")
        if ((x < self._bounds[0]) | (x > self._bounds[1])).any():
            raise ValueError("prediction inputs must lie within bounds.")
        result = torch.full((len(x), self.values.shape[1]), math.nan, dtype=torch.float64)
        for model in self._models:
            use = (x[:, 0] >= model.knots[0]) & (x[:, 0] <= model.knots[-1])
            result[use] = model.predict(x[use, 0])
        # A declared grid can have its first/last key inside the raw bounds.
        if self._models:
            first, last = self._models[0], self._models[-1]
            below, above = x[:, 0] < first.knots[0], x[:, 0] > last.knots[-1]
            result[below] = first.predict(first.knots[:1]).expand(int(below.sum()), -1)
            result[above] = last.predict(last.knots[-1:]).expand(int(above.sum()), -1)
        return result


def _basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Shared endpoint controls enforce C0 continuity across local charts."""
    if len(knots) == 1:
        return torch.ones((len(x), 1), dtype=torch.float64)
    pieces = len(knots) - 1
    indices = torch.searchsorted(knots, x.contiguous(), right=True).sub(1).clamp(0, pieces - 1)
    t = ((x - knots[indices]) / (knots[indices + 1] - knots[indices])).clamp(0, 1)
    design = torch.zeros((len(x), pieces * degree + 1), dtype=torch.float64)
    for j in range(degree + 1):
        design[torch.arange(len(x)), indices * degree + j] = math.comb(degree, j) * t.pow(j) * (1 - t).pow(degree - j)
    return design


def _midpoints(session: QuerySession, x: torch.Tensor) -> list[tuple[int, float]]:
    proposed = session.canonical((x[:-1] / 2 + x[1:] / 2)[:, None])[:, 0]
    return [(i, float(point)) for i, point in enumerate(proposed) if x[i] < point < x[i + 1]]


def _settings(
    session: QuerySession,
    initial_samples: int,
    location_tolerance: float | None,
    max_pieces: int,
    exploration_every: int,
    noise_replicates: int,
    min_relative_jump: float,
) -> tuple[float, float, float, float]:
    if len(session.bounds) != 1:
        raise ValueError("Adaptive curve fitting supports exactly one input dimension.")
    for name, value, minimum in (
        ("initial_samples", initial_samples, 2),
        ("max_pieces", max_pieces, 1),
        ("exploration_every", exploration_every, 1),
        ("noise_replicates", noise_replicates, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(min_relative_jump, bool) or not math.isfinite(min_relative_jump) or min_relative_jump <= 0:
        raise ValueError("min_relative_jump must be finite and positive.")
    lower, upper = session.bounds[0].tolist()
    width = upper - lower
    if not math.isfinite(width):
        raise ValueError("the input range must have a finite width.")
    target = width / 256 if location_tolerance is None else location_tolerance
    if isinstance(target, bool) or not math.isfinite(target) or target <= 0:
        raise ValueError("location_tolerance must be finite and positive.")
    target = max(float(target), float(session.resolution[0]))
    return lower, upper, width, target


def _seed(session: QuerySession, initial_samples: int) -> None:
    existing, _, _ = session.observations()
    seeded = {tuple(row) for row in existing.tolist()}
    lower, upper = session.bounds[0].tolist()
    initial = session.canonical(
        torch.linspace(lower, upper, min(initial_samples, session.budget), dtype=torch.float64)[:, None]
    )
    initial = torch.unique(initial, dim=0)
    missing = [row for row in initial if tuple(row.tolist()) not in seeded]
    initial = torch.stack(missing) if missing else initial[:0]
    initial = initial[: session.budget - session.cost]
    if len(initial):
        session.query(initial)


def _noise_pilot(session: QuerySession, repeats: int) -> None:
    x, _, _ = session.observations()
    if session.deterministic or not len(x):
        return
    anchors = sorted({0, len(x) // 2, len(x) - 1})
    for _ in range(repeats - 1):
        for index in anchors:
            if session.cost >= session.budget:
                return
            session.query(x[index : index + 1])


@dataclass
class StructureLearningResult(CurveLearningResult):
    """A curve result whose deployed predictor is the actual compiled BSF."""

    _compiled: "BSFCurveExport | None" = field(default=None, repr=False)

    def predict(self, points: Any) -> torch.Tensor:
        """Predict with the compiled simploidal model once it is available."""
        return super().predict(points) if self._compiled is None else self._compiled.predict(points)
