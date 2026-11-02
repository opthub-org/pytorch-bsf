"""A narrowly gated acquisition hint, preserving frozen v4 cut decisions.

Ordinary full-curvature excess and classification are retained. Only when
full curvature vetoes a first-neighbor excess, and the most recent actual
bracket shrink retained amplitude/direction, add an unresolved sampling hint.
No new jump is asserted by this override. The inherited finite-scale detector
can still mistake low-Holder continuous functions for jumps; this variant does
not claim to repair that limitation or certify topology.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .evidence import GapEvidence, MultiscaleGapDetector, _inputs, _norm


@dataclass(frozen=True)
class StableContrastHintEvidence(GapEvidence):
    """Base classification plus explicit, non-topological query evidence."""

    base_excess: float
    full_curvature_bound: float
    first_neighbor_bound: float
    first_neighbor_excess: float
    latest_contraction_exponent: float | None
    latest_direction_agreement: float | None
    acquisition_hint: bool


class StableContrastHintDetector(MultiscaleGapDetector):
    """Keep base cuts and add a one-refinement-stable context query only.

    The first-neighbor bound takes the maximum absolute sided secant. A hint
    requires two-sided context, |latest exponent|<=0.1 and direction>=0.9.
    Unchanged updates, mean replicates and unrelated queries do not manufacture
    a nested parent. Thresholds are fixed for this development variant.
    """

    def _latest(self, key: tuple[float, float], noise: float) -> tuple[float | None, float | None]:
        chain = self._chains[key]
        if not chain:
            return None, None
        parent = chain[0]
        child_values = self._values[key[0]], self._values[key[1]]
        parent_values = self._values[parent[0]], self._values[parent[1]]
        delta = child_values[1][0] - child_values[0][0]
        previous = parent_values[1][0] - parent_values[0][0]
        amplitude, old_amplitude = _norm(delta), _norm(previous)
        lower = max(0.0, amplitude - self._noise(noise, child_values[0][1], child_values[1][1]))
        upper = old_amplitude + self._noise(noise, parent_values[0][1], parent_values[1][1])
        log_width = math.log(key[1] - key[0]) - math.log(parent[1] - parent[0])
        if not lower or not upper or not old_amplitude or log_width >= 0:
            return None, None
        exponent = (math.log(lower) - math.log(upper)) / log_width
        direction = float(torch.dot(delta / amplitude, previous / old_amplitude))
        return exponent, direction

    def update(
        self, x: Any, y: Any, *, counts: Any = None, noise_variance: float = 0.0
    ) -> tuple[StableContrastHintEvidence, ...]:
        """Assess the base detector first; a hint never changes a cut decision."""
        base = super().update(x, y, counts=counts, noise_variance=noise_variance)
        if not base:
            return ()
        points, values, versions = _inputs(x, y, counts)
        widths = (points[1:] - points[:-1]).tolist()
        deltas = values[1:] - values[:-1]
        floors = [self._noise(noise_variance, int(versions[i]), int(versions[i + 1])) for i in range(len(widths))]
        result = []
        for i, row in enumerate(base):
            bounds = []
            for near in (i - 1, i + 1):
                if 0 <= near < len(widths):
                    change = _norm(deltas[near]) + floors[near]
                    bounds.append(change * (widths[i] / widths[near]) if change else 0.0)
            first = max(bounds, default=0.0)
            first_excess = max(0.0, row.raw_change - row.noise_floor - self.continuation_factor * first)
            exponent, direction = self._latest((row.left, row.right), noise_variance)
            hint = (
                0 < i < len(base) - 1
                and row.excess <= row.threshold < first_excess
                and exponent is not None
                and abs(exponent) <= 0.1
                and direction is not None
                and direction >= 0.9
            )
            fields = asdict(row)
            if hint:
                fields.update(
                    excess=first_excess,
                    local_continuation_bound=first,
                    status="unresolved",
                    reason="stable_contrast_context_probe",
                )
            result.append(
                StableContrastHintEvidence(
                    **fields,
                    base_excess=row.excess,
                    full_curvature_bound=row.local_continuation_bound,
                    first_neighbor_bound=first,
                    first_neighbor_excess=first_excess,
                    latest_contraction_exponent=exponent,
                    latest_direction_agreement=direction,
                    acquisition_hint=hint,
                )
            )
        return tuple(result)
