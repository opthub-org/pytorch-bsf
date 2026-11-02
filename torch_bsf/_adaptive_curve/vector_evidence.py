"""Cross-fitted coherent vector-jump evidence, alongside the frozen detector.

An outer local regression estimates a candidate response direction without
using either bracket endpoint. The endpoint contrast validates that direction.
Projected derivative magnitudes, curvature, residuals, and agreement across
outer windows guard against mistaking arbitrary high-dimensional roughness for
a jump. All tests are finite-scale empirical diagnostics; adaptive selection
and unknown deterministic roughness preclude a nominal confidence claim.
Scalar observations delegate exactly to the frozen detector.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from .evidence import GapEvidence, MultiscaleGapDetector, Status, _norm


@dataclass
class _Side:
    coefficients: torch.Tensor
    residual: torch.Tensor
    inverse: torch.Tensor
    score: float
    degrees_of_freedom: int


@dataclass
class _Outer:
    left: _Side
    right: _Side
    radius: float
    degree: int
    window: int
    smallest_side: int
    mean_direction: torch.Tensor

    @property
    def direction(self) -> torch.Tensor:
        return self.right.coefficients[0] - self.left.coefficients[0]

    @property
    def score(self) -> float:
        return (self.left.score + self.right.score) / 2


def _side(z: torch.Tensor, y: torch.Tensor, degree: int) -> _Side | None:
    if len(z) < degree + 2:
        return None
    design = torch.stack([z**power for power in range(degree + 1)], 1)
    fit = torch.linalg.lstsq(design, y, driver="gelsd", rcond=1e-10)
    if int(fit.rank) != degree + 1:
        return None
    inverse = torch.linalg.pinv(design, rtol=1e-10)
    leverage = (design * inverse.T).sum(1)
    if bool((leverage >= 1 - 1e-8).any()):
        return None
    residual = y - design @ fit.solution
    loo = residual / (1 - leverage)[:, None]
    score = float(loo.square().sum(1).mean())
    if not math.isfinite(score):
        return None
    return _Side(fit.solution, residual, inverse, score, len(z) - degree - 1)


def _outer_models(x: torch.Tensor, y: torch.Tensor, i: int) -> list[_Outer]:
    """Both endpoints i/i+1 are absent from every fit and model-selection score."""
    middle = float(x[i] / 2 + x[i + 1] / 2)
    models, seen = [], set()
    for window in (3, 5, 7):
        left = torch.arange(max(0, i - window), i)
        right = torch.arange(i + 2, min(len(x), i + 2 + window))
        signature = (tuple(left.tolist()), tuple(right.tolist()))
        if min(len(left), len(right)) < 3 or signature in seen:
            continue
        seen.add(signature)
        radius = max(float((x[left] - middle).abs().max()), float((x[right] - middle).abs().max()))
        choices = []
        for degree in (1, 2):
            first = _side((x[left] - middle) / radius, y[left], degree)
            second = _side((x[right] - middle) / radius, y[right], degree)
            if first is not None and second is not None:
                choices.append(
                    _Outer(
                        first, second, radius, degree, window, min(len(left), len(right)), y[right].mean(0) - y[left].mean(0)
                    )
                )
        if choices:
            models.append(min(choices, key=lambda model: model.score))
    return models


def _projected_statistics(model: _Outer, u: torch.Tensor, width: float) -> tuple[float, float, float]:
    """Absolute sided Taylor continuation plus forecast residual floor."""
    bounds, variances, total_residual = [], [], 0.0
    for sign, side in ((-1, model.left), (1, model.right)):
        t = sign * width / (2 * model.radius)
        row = u.new_tensor([t**power for power in range(model.degree + 1)])
        derivative = float(side.coefficients[1] @ u) / model.radius
        curvature = float(side.coefficients[2] @ u) / model.radius**2 if model.degree == 2 else 0.0
        bounds.append(width * abs(derivative + sign * width * curvature) + width**2 * abs(curvature))
        projected = side.residual @ u
        variance = float(projected.square().sum()) / side.degrees_of_freedom
        forecast_leverage = float((row @ side.inverse).square().sum())
        variances.append(variance * (1 + forecast_leverage))
        total_residual += float(side.residual.square().sum())
    projected_residual = sum(float((side.residual @ u).square().sum()) for side in (model.left, model.right))
    orthogonal_fraction = max(0.0, 1 - projected_residual / total_residual) if total_residual > 0 else 0.0
    return max(bounds), 3 * math.sqrt(sum(variances)), orthogonal_fraction


class CrossFittedVectorGapDetector(MultiscaleGapDetector):
    """A compatible vector evidence arm; base scalar arithmetic is unchanged.

    Extra evidence is considered only when most outer-model residual energy
    lies orthogonal to a coherent candidate direction. Clean scalar lifts
    therefore retain the original safeguards. ``last_projection_diagnostics``
    exposes model support/floors without claiming statistical calibration.
    """

    def __init__(
        self,
        *,
        continuation_factor: float = 3.0,
        min_relative_jump: float = 0.01,
        min_refinements: int = 2,
        cache_outer: bool = True,
    ) -> None:
        if not isinstance(cache_outer, bool):
            raise ValueError("cache_outer must be a bool")
        self.cache_outer = cache_outer
        self._model_cache: dict[tuple[Any, ...], list[_Outer]] = {}
        self.outer_model_fits = 0
        self.outer_model_cache_hits = 0
        super().__init__(
            continuation_factor=continuation_factor, min_relative_jump=min_relative_jump, min_refinements=min_refinements
        )
        self.last_projection_diagnostics: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset observation history and vector diagnostics for a new stream."""
        super().reset()
        self.last_projection_diagnostics = []
        self._model_cache = {}
        self.outer_model_fits = 0
        self.outer_model_cache_hits = 0

    def _persistence(
        self, key: tuple[float, float], u: torch.Tensor, current: float, floor: float
    ) -> tuple[float | None, float | None, bool]:
        chain = self._chains[key]
        if not chain:
            return None, None, True
        ancestor = chain[min(max(1, self.min_refinements), len(chain)) - 1]
        previous = float((self._values[ancestor[1]][0] - self._values[ancestor[0]][0]) @ u)
        if previous <= floor:
            return None, None, False
        ratio = (current + floor) / (previous - floor)
        exponent = math.log(ratio) / (math.log(key[1] - key[0]) - math.log(ancestor[1] - ancestor[0]))
        return ratio, exponent, current > 0

    def _assess(self, y: torch.Tensor, i: int, base: GapEvidence, models: list[_Outer]) -> tuple[GapEvidence, dict[str, Any]]:
        supported = [model for model in models if model.smallest_side >= 4 and _norm(model.mean_direction) > base.threshold]
        diagnostic: dict[str, Any] = {"outer_model_count": len(models), "base_excess": base.excess}
        if not supported or len(models) < 2:
            return base, {**diagnostic, "reason": "insufficient_independent_outer_context"}
        selected = min(supported, key=lambda model: model.score)
        # Sided means avoid letting polynomial extrapolation rotate a response
        # direction toward lexical jitter on unevenly spaced numeric strings.
        # The polynomial still supplies the absolute continuation/remainder.
        u = selected.mean_direction / _norm(selected.mean_direction)
        strengths = [float(model.direction @ u) for model in models]
        direction_norms = [_norm(model.mean_direction) for model in models]
        coherence = min(
            float(model.mean_direction @ u) / norm if norm else 0.0
            for model, norm in zip(models, direction_norms, strict=True)
        )
        center_strength = float(selected.direction @ u)
        stability = max(abs(value - center_strength) for value in strengths)
        projected = float((y[i + 1] - y[i]) @ u)
        bound, empirical_floor, orthogonal = _projected_statistics(selected, u, base.right - base.left)
        floor = max(base.noise_floor, empirical_floor)
        excess = max(0.0, min(projected, center_strength) - self.continuation_factor * bound - floor - stability)
        diagnostic.update(
            window=selected.window,
            degree=selected.degree,
            outer_gcv=selected.score,
            coherence=coherence,
            stability_penalty=stability,
            projected_endpoint_change=projected,
            continuation_bound=bound,
            residual_floor=floor,
            orthogonal_residual_fraction=orthogonal,
            projected_excess=excess,
        )
        if orthogonal < 0.5 or excess <= max(base.threshold, base.excess) or base.status == "jump_candidate":
            return base, diagnostic
        ratio, exponent, aligned = self._persistence((base.left, base.right), u, projected, floor)
        status: Status = "unresolved"
        reason = "projection_needs_nested_queries"
        if base.nested_scales >= self.min_refinements and coherence >= 0.5 and aligned:
            status, reason = "jump_candidate", "cross_fitted_coherent_vector_change"
            if exponent is not None and exponent > 0.5:
                status, reason = "unresolved", "projection_contracting_change"
        diagnostic["reason"] = reason
        return replace(
            base,
            local_continuation_bound=bound,
            excess=excess,
            noise_floor=floor,
            status=status,
            contraction=ratio,
            contraction_exponent=exponent,
            direction_agreement=coherence,
            reason=reason,
        ), diagnostic

    def update(self, x: Any, y: Any, *, counts: Any = None, noise_variance: float = 0.0) -> tuple[GapEvidence, ...]:
        """Use current observed values only; never request evaluator responses."""
        base = super().update(x, y, counts=counts, noise_variance=noise_variance)
        self.last_projection_diagnostics = []
        if not base or self._values[self._x[0]][0].numel() == 1:
            return base
        points = torch.tensor(self._x, dtype=torch.float64)
        values = torch.stack([self._values[point][0] for point in self._x])
        result = []
        live: dict[tuple[Any, ...], list[_Outer]] = {}
        for i, row in enumerate(base):
            context = self._x[max(0, i - 7) : min(len(points), i + 9)]
            key = (row.left, row.right, tuple((point, self._values[point][1]) for point in context))
            if self.cache_outer and key in self._model_cache:
                models = self._model_cache[key]
                self.outer_model_cache_hits += 1
            else:
                models = _outer_models(points, values, i)
                self.outer_model_fits += 1
            live[key] = models
            evidence, diagnostic = self._assess(values, i, row, models)
            result.append(evidence)
            self.last_projection_diagnostics.append(diagnostic)
        self._model_cache = live
        return tuple(result)
