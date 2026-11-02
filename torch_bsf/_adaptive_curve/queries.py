"""Budget and resolution adapters for genuinely queryable embedding evaluators."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

from .._numeric import _finite_mean


def _exact_grid(values: torch.Tensor, bounds: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Snap with exact integer keys when floating-point key ratios overflow."""
    spacing = Fraction.from_float(float(step))
    lower, upper = (Fraction.from_float(value) for value in bounds.tolist())
    minimum, maximum = math.ceil(lower / spacing), math.floor(upper / spacing)
    if minimum > maximum:
        raise ValueError("the declared input grid has no point within bounds.")
    snapped = [
        float(min(max(round(Fraction.from_float(value) / spacing), minimum), maximum) * spacing) for value in values.tolist()
    ]
    return values.new_tensor(snapped)


def _snap_grid(values: torch.Tensor, bounds: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Keep the ordinary grid reduction and repair outward-rounded key limits."""
    ratios = bounds / step
    if not torch.isfinite(ratios).all():
        return _exact_grid(values, bounds, step)
    minimum, maximum = torch.ceil(ratios[0]), torch.floor(ratios[1])
    if not torch.isfinite(torch.stack((minimum * step, maximum * step))).all():
        return _exact_grid(values, bounds, step)
    # Dividing an endpoint can round to an integer whose represented product
    # lies outside bounds. Above 2**53, adding one needs the next float key.
    if minimum * step < bounds[0]:
        minimum = torch.maximum(minimum + 1, torch.nextafter(minimum, minimum.new_tensor(math.inf)))
    if maximum * step > bounds[1]:
        maximum = torch.minimum(maximum - 1, torch.nextafter(maximum, maximum.new_tensor(-math.inf)))
    if minimum > maximum:
        raise ValueError("the declared input grid has no point within bounds.")
    snapped = torch.round(values / step).clamp(minimum, maximum) * step
    if not torch.isfinite(snapped).all() or ((snapped < bounds[0]) | (snapped > bounds[1])).any():
        return _exact_grid(values, bounds, step)
    return snapped


def _scaled_noise_variance(repeats: list[torch.Tensor], denominator: int) -> float:
    """Recover a finite pooled variance when only its unscaled sum overflowed."""
    scale = max(float(rows.abs().max()) for rows in repeats)
    normalized = [rows / scale for rows in repeats]
    squared = math.fsum(float((rows - _finite_mean(rows, 0)).square().sum()) for rows in normalized)
    deviation = scale * math.sqrt(squared / denominator)
    variance = deviation * deviation
    if not math.isfinite(variance):
        raise ArithmeticError("Empirical noise variance exceeds the finite float64 range.")
    return variance


def _response_mean(responses: list[torch.Tensor]) -> torch.Tensor:
    """Keep single-row reductions cheap and guard sums of repeated responses."""
    rows = torch.stack(responses)
    return rows.mean(0) if len(responses) == 1 else _finite_mean(rows, 0)


@dataclass
class _Observation:
    point: torch.Tensor
    responses: list[torch.Tensor]


class QuerySession:
    """Count actual response rows, aggregate replicates, and honor known grids.

    Deterministic cache hits cost no new evaluator call. Stochastic repeats
    always consume budget and update a sample mean. A declared grid is an
    input protocol, not a property inferred from observed responses.
    """

    def __init__(
        self,
        oracle: Callable[[torch.Tensor], Any],
        bounds: Any,
        *,
        budget: int,
        deterministic: bool = True,
        resolution: Any = None,
    ) -> None:
        self.bounds = torch.as_tensor(bounds, dtype=torch.float64).clone()
        if (
            self.bounds.ndim != 2
            or self.bounds.shape[1] != 2
            or not len(self.bounds)
            or not torch.isfinite(self.bounds).all()
            or (self.bounds[:, 1] <= self.bounds[:, 0]).any()
        ):
            raise ValueError("bounds must contain finite increasing coordinate pairs.")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer.")
        if not callable(oracle):
            raise TypeError("oracle must be callable.")
        self.resolution = torch.zeros(len(self.bounds), dtype=torch.float64)
        if resolution is not None:
            self.resolution = torch.as_tensor(resolution, dtype=torch.float64).expand(len(self.bounds)).clone()
            if not torch.isfinite(self.resolution).all() or (self.resolution < 0).any():
                raise ValueError("resolution must be finite and nonnegative.")
        self.oracle, self.budget, self.deterministic = oracle, budget, deterministic
        self.cost = 0
        self.proposed_rows = 0
        self._outputs: int | None = None
        self._observations: dict[tuple[float, ...], _Observation] = {}

    def canonical(self, points: Any) -> torch.Tensor:
        """Return finite input locations snapped to declared zero-aligned grids."""
        raw = torch.as_tensor(points)
        if raw.is_complex():
            raise ValueError("points must be real.")
        x = raw.detach().cpu().double() if isinstance(points, torch.Tensor) else torch.as_tensor(points, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != len(self.bounds) or not torch.isfinite(x).all():
            raise ValueError("points must have shape (n_points, input_dimension).")
        if ((x < self.bounds[:, 0]) | (x > self.bounds[:, 1])).any():
            raise ValueError("points must lie within bounds.")
        x = x.clone()
        for axis, step in enumerate(self.resolution):
            if step > 0:
                x[:, axis] = _snap_grid(x[:, axis], self.bounds[axis], step)
        if not torch.isfinite(x).all() or ((x < self.bounds[:, 0]) | (x > self.bounds[:, 1])).any():
            raise ValueError("resolution must yield finite grid points within bounds.")
        return x

    def query(self, points: Any) -> torch.Tensor:
        """Obtain cached values or new response rows, returning current means."""
        x = self.canonical(points)
        keys = [tuple(row) for row in x.tolist()]
        self.proposed_rows += len(x)
        missing = list(dict.fromkeys(key for key in keys if key not in self._observations)) if self.deterministic else keys
        if self.cost + len(missing) > self.budget:
            raise RuntimeError("oracle query budget exhausted.")
        if missing:
            queried = torch.tensor(missing, dtype=torch.float64)
            self.cost += len(missing)
            returned = self.oracle(queried.clone())
            responses = torch.as_tensor(returned)
            if responses.is_complex():
                raise ValueError("oracle responses must be real.")
            responses = (
                responses.detach().cpu().double()
                if isinstance(returned, torch.Tensor)
                else torch.as_tensor(returned, dtype=torch.float64).cpu()
            )
            if (
                responses.ndim != 2
                or responses.shape[0] != len(missing)
                or not responses.shape[1]
                or not torch.isfinite(responses).all()
                or (self._outputs is not None and responses.shape[1] != self._outputs)
            ):
                raise ValueError("oracle responses must be finite aligned rows with a fixed output count.")
            self._outputs = responses.shape[1]
            for key, point, response in zip(missing, queried, responses, strict=True):
                if key not in self._observations:
                    self._observations[key] = _Observation(point.clone(), [])
                self._observations[key].responses.append(response.clone())
        if not keys:
            if self._outputs is None:
                raise ValueError("an empty query cannot establish the output count.")
            return torch.empty((0, self._outputs), dtype=torch.float64)
        return torch.stack([_response_mean(self._observations[key].responses) for key in keys])

    def observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unique locations, sample means and replicate counts."""
        ordered = sorted(self._observations)
        if not ordered:
            return (
                torch.empty((0, len(self.bounds)), dtype=torch.float64),
                torch.empty((0, self._outputs or 0), dtype=torch.float64),
                torch.empty(0, dtype=torch.int64),
            )
        rows = [self._observations[key] for key in ordered]
        return (
            torch.stack([row.point for row in rows]),
            torch.stack([_response_mean(row.responses) for row in rows]),
            torch.tensor([len(row.responses) for row in rows]),
        )

    def noise_variance(self) -> float | None:
        """Estimate pooled vector noise variance only from independent repeats."""
        numerator, denominator = 0.0, 0
        repeats = []
        for observation in self._observations.values():
            if len(observation.responses) < 2:
                continue
            rows = torch.stack(observation.responses)
            repeats.append(rows)
            numerator += float((rows - _finite_mean(rows, 0)).square().sum())
            denominator += len(rows) - 1
        if not denominator:
            return None
        if not math.isfinite(numerator):
            return _scaled_noise_variance(repeats, denominator)
        return numerator / denominator
