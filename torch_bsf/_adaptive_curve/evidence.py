"""Research-only continuation checks for vector-valued one-dimensional data.

Large approximation error is not itself evidence for a disconnected chart.
An adjacent change is compared with the *norms* of neighboring secants, then
with its own observed ancestors after genuinely new nested samples arrive.
This is empirical finite-scale evidence, not a continuity test or a confidence
bound. A narrower smooth transition or an unsampled pulse remains possible.
"""

import bisect
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

Status = Literal["continuous_like", "unresolved", "jump_candidate"]
_Bracket = tuple[float, float]


@dataclass(frozen=True)
class GapEvidence:
    """Observable change evidence; ``probe`` is uncharged and may be unavailable."""

    left: float
    right: float
    raw_change: float
    local_continuation_bound: float
    excess: float
    threshold: float
    noise_floor: float
    status: Status
    nested_scales: int
    contraction: float | None
    contraction_exponent: float | None
    direction_agreement: float | None
    probe: float | None
    reason: str


def _norm(value: torch.Tensor) -> float:
    maximum = float(value.abs().max())
    result = maximum * float(torch.linalg.vector_norm(value / maximum)) if maximum else 0.0
    if not math.isfinite(result):
        raise ArithmeticError("Response differences exceed the finite float64 range.")
    return result


def _variation(y: torch.Tensor) -> float:
    maximum = float(y.abs().max())
    if not maximum:
        return 0.0
    normalized = y / maximum
    result = maximum * float((normalized - normalized.mean(0)).square().sum(1).mean().sqrt())
    if not math.isfinite(result):
        raise ArithmeticError("Observed response variation exceeds the finite float64 range.")
    return result


def _inputs(x: Any, y: Any, counts: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_x, raw_y = torch.as_tensor(x), torch.as_tensor(y)
    if raw_x.is_complex() or raw_y.is_complex():
        raise ValueError("Observations must be real.")
    numeric_x = raw_x.detach().cpu().double() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    points = numeric_x.reshape(-1) if raw_x.ndim in (1, 2) else numeric_x
    if raw_x.ndim == 2 and raw_x.shape[1] != 1:
        raise ValueError("Only one-dimensional inputs are supported.")
    values = raw_y.detach().cpu().double() if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float64)
    if (
        points.ndim != 1
        or values.ndim != 2
        or len(points) != len(values)
        or not values.shape[1]
        or not torch.isfinite(points).all()
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Finite sorted inputs and aligned NxD values are required.")
    if len(points) > 1 and (not bool((points[1:] > points[:-1]).all()) or not math.isfinite(float(points[-1] - points[0]))):
        raise ValueError("Inputs must increase strictly with finite total span.")
    versions = torch.ones(len(points), dtype=torch.int64) if counts is None else torch.as_tensor(counts).cpu()
    if (
        versions.shape != (len(points),)
        or versions.dtype == torch.bool
        or versions.is_complex()
        or not torch.isfinite(versions).all()
        or (versions < 1).any()
        or (versions > torch.iinfo(torch.int64).max).any()
        or (versions != versions.round()).any()
    ):
        raise ValueError("counts must be aligned positive integers.")
    return points, values, versions.to(torch.int64)


class MultiscaleGapDetector:
    """Track one append-only observation stream without querying its oracle.

    ``min_refinements=2`` requires two actual nested bracket refinements before
    proposing a cut. Zero is useful for retrospective *provisional* screening.
    Relative thresholds are calibrated from observed vector variation. Noise
    floors use vector noise variance of an independent response row, as in
    QuerySession; they are diagnostics without statistical error control.
    Candidates must remain reversible and be audited by the query planner.
    """

    def __init__(self, *, continuation_factor: float = 3.0, min_relative_jump: float = 0.01, min_refinements: int = 2) -> None:
        if (
            isinstance(continuation_factor, bool)
            or not math.isfinite(continuation_factor)
            or continuation_factor < 1
            or isinstance(min_relative_jump, bool)
            or not math.isfinite(min_relative_jump)
            or min_relative_jump < 0
        ):
            raise ValueError("continuation_factor>=1 and min_relative_jump>=0 must be finite.")
        if isinstance(min_refinements, bool) or not isinstance(min_refinements, int) or min_refinements < 0:
            raise ValueError("min_refinements must be a nonnegative integer.")
        self.continuation_factor, self.min_relative_jump = continuation_factor, min_relative_jump
        self.min_refinements = min_refinements
        self.reset()

    def reset(self) -> None:
        """Start a separate independent stream; never share evidence across tasks."""
        self._x: list[float] = []
        self._values: dict[float, tuple[torch.Tensor, int]] = {}
        self._chains: dict[_Bracket, tuple[_Bracket, ...]] = {}

    def _history(self, x: torch.Tensor, y: torch.Tensor, counts: torch.Tensor) -> dict[_Bracket, tuple[_Bracket, ...]]:
        coordinates = x.tolist()
        latest = {point: (value.clone(), int(count)) for point, value, count in zip(coordinates, y, counts, strict=True)}
        for point, (old, version) in self._values.items():
            if point not in latest or latest[point][1] < version:
                raise ValueError("Observations must be append-only; reset for a separate stream.")
            if latest[point][1] == version and not torch.equal(old, latest[point][0]):
                raise ValueError("Response means changed without a count increase; reset for a separate stream.")
        chains = {}
        for left, right in zip(coordinates[:-1], coordinates[1:], strict=True):
            key = (left, right)
            if key in self._chains:
                chains[key] = self._chains[key]
            elif self._x:
                i = bisect.bisect_right(self._x, left) - 1
                parent = (self._x[i], self._x[i + 1]) if 0 <= i < len(self._x) - 1 else None
                chains[key] = (
                    (parent,) + self._chains[parent]
                    if parent is not None and parent[0] <= left < right <= parent[1] and right - left < parent[1] - parent[0]
                    else self._chains.get(parent or key, ())
                )
            else:
                chains[key] = ()
        self._x, self._values, self._chains = coordinates, latest, chains
        return chains

    @staticmethod
    def _noise(noise: float, left: int, right: int) -> float:
        return 3 * math.sqrt(noise) * math.sqrt(1 / left + 1 / right)

    @staticmethod
    def _continuation(i: int, widths: list[float], deltas: torch.Tensor, floors: list[float]) -> float:
        """Include observable sided curvature; never cancel opposed derivatives.

        Adjacent secants estimate a first derivative at their cell centers.
        Their difference estimates a second derivative. Integrating that
        one-sided Taylor continuation across the candidate interval adds
        h*(h+h_near)/(h_near+h_far)*||s_near-s_far|| to h*||s_near||.
        This is an empirical remainder estimate, not a smoothness bound.
        """
        bounds = []
        for direction in (-1, 1):
            near, far = i + direction, i + 2 * direction
            if not 0 <= near < len(widths):
                continue
            first = deltas[near] * (widths[i] / widths[near])
            first_noise = floors[near] * (widths[i] / widths[near])
            bound = _norm(first) + first_noise
            if 0 <= far < len(widths):
                second = deltas[far] * (widths[i] / widths[far])
                second_noise = floors[far] * (widths[i] / widths[far])
                bound += ((widths[i] + widths[near]) / (widths[near] + widths[far])) * (
                    _norm(first - second) + first_noise + second_noise
                )
            bounds.append(bound)
        return max(bounds, default=0.0)

    def _contraction(
        self, key: _Bracket, chain: tuple[_Bracket, ...], delta: torch.Tensor, noise: float, current_floor: float
    ) -> tuple[float | None, float | None, float | None]:
        if not chain:
            return None, None, None
        ancestor = chain[min(max(1, self.min_refinements), len(chain)) - 1]
        left, right = self._values[ancestor[0]], self._values[ancestor[1]]
        old_delta = right[0] - left[0]
        previous = _norm(old_delta)
        lower = max(0.0, _norm(delta) - current_floor)
        upper = previous + self._noise(noise, left[1], right[1])
        if not lower or not upper:
            return None, None, None
        ratio = lower / upper
        log_ratio = math.log(lower) - math.log(upper)
        log_width_ratio = math.log(key[1] - key[0]) - math.log(ancestor[1] - ancestor[0])
        exponent = log_ratio / log_width_ratio
        agreement = float(torch.dot(delta / _norm(delta), old_delta / previous)) if previous else None
        return ratio, exponent, agreement

    def update(self, x: Any, y: Any, *, counts: Any = None, noise_variance: float = 0.0) -> tuple[GapEvidence, ...]:
        """Assess every adjacent bracket from current observations, never a truth label."""
        points, values, versions = _inputs(x, y, counts)
        if isinstance(noise_variance, bool) or not math.isfinite(noise_variance) or noise_variance < 0:
            raise ValueError("noise_variance must be finite and nonnegative.")
        chains = self._history(points, values, versions)
        if len(points) < 2:
            return ()
        scale = _variation(values)
        threshold = max(scale * self.min_relative_jump, math.ulp(scale) * 64)
        widths = (points[1:] - points[:-1]).tolist()
        deltas = values[1:] - values[:-1]
        changes = [_norm(delta) for delta in deltas]
        floors = [self._noise(noise_variance, int(versions[i]), int(versions[i + 1])) for i in range(len(widths))]
        result = []
        for i, (left, right) in enumerate(chains):
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(widths)]
            bound = self._continuation(i, widths, deltas, floors)
            excess = max(0.0, changes[i] - floors[i] - self.continuation_factor * bound)
            chain = chains[(left, right)]
            ratio, exponent, agreement = self._contraction((left, right), chain, deltas[i], noise_variance, floors[i])
            status: Status = "continuous_like"
            reason = "observed_neighbor_continuation"
            if noise_variance and changes[i] <= floors[i] + threshold:
                status, reason = "unresolved", "noise_limited"
            if excess > threshold:
                status = "unresolved"
                reason = "needs_nested_queries" if len(neighbors) == 2 else "needs_two_sided_context"
                if len(neighbors) == 2 and len(chain) >= self.min_refinements:
                    status, reason = "jump_candidate", "persistent_finite_scale_change"
                    if exponent is not None and exponent > 0.5:
                        status, reason = "unresolved", "contracting_change"
                    elif agreement is not None and agreement < 0.9:
                        status, reason = "unresolved", "rotating_change"
            probe = left / 2 + right / 2
            result.append(
                GapEvidence(
                    left,
                    right,
                    changes[i],
                    bound,
                    excess,
                    threshold,
                    floors[i],
                    status,
                    len(chain),
                    ratio,
                    exponent,
                    agreement,
                    probe if left < probe < right else None,
                    reason,
                )
            )
        return tuple(result)
