"""Scalar persistence and cross-fitted vector evidence dispatch."""

from typing import Any

from .evidence import GapEvidence, _inputs
from .scalar_evidence import StableContrastHintDetector
from .vector_evidence import CrossFittedVectorGapDetector


class EmbeddingGapDetector:
    """Select scalar persistence or cross-fitted vector evidence from row shape.

    Each child keeps a private observed-data history. Scalar responses use the
    source-bound stable-contrast profile. Multivariate responses use the
    coherent-direction profile, including its conservative fallback for clean
    curves. No source callable, external labels or heldout inputs are inspected.
    """

    def __init__(self, *, continuation_factor: float = 3.0, min_relative_jump: float = 0.01, min_refinements: int = 2) -> None:
        self.scalar = StableContrastHintDetector(
            continuation_factor=continuation_factor, min_relative_jump=min_relative_jump, min_refinements=min_refinements
        )
        self.vector = CrossFittedVectorGapDetector(
            continuation_factor=continuation_factor, min_relative_jump=min_relative_jump, min_refinements=min_refinements
        )
        self.outputs: int | None = None

    def update(self, x: Any, y: Any, *, counts: Any = None, noise_variance: float = 0.0) -> tuple[GapEvidence, ...]:
        """Assess only the current private stream, retaining its output count."""
        points, values, versions = _inputs(x, y, counts)
        outputs = values.shape[1]
        if self.outputs is not None and self.outputs != outputs:
            raise ValueError("A detector stream must retain the same number of output coordinates.")
        self.outputs = outputs
        detector = self.scalar if outputs == 1 else self.vector
        return detector.update(points, values, counts=versions, noise_variance=noise_variance)
