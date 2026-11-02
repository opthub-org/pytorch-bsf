"""Budgeted curve acquisition with central query jitter and continuing audits.

Each call owns its observation history, detector and random generator.
The detector receives only charged observations, never evaluator internals.
"""

import math
from collections.abc import Callable
from typing import Any, Protocol

import torch

from .acquisition import _ranked_cell
from .acquisition import scores as cv_scores
from .evidence import GapEvidence, MultiscaleGapDetector
from .export import _compile_result
from .fitting import _fit_prediction
from .polynomial import CurveGap, StructureLearningResult, _midpoints, _noise_pilot, _seed, _settings
from .queries import QuerySession


class GapDetector(Protocol):
    """One private observation history and its finite-scale evidence."""

    def update(self, x: Any, y: Any, *, counts: Any = None, noise_variance: float = 0.0) -> tuple[GapEvidence, ...]:
        """Update from current charged observations only."""
        ...


def jitter_query(
    session: QuerySession, x: torch.Tensor, point: float, generator: torch.Generator, fraction: float = 0.125
) -> tuple[float, bool]:
    """Jitter an unobserved selected query, falling back to its valid midpoint.

    The unrounded fraction lies in [0.5-fraction,0.5+fraction]. Canonical
    rounding can move it outside that central band on a coarse declared grid;
    the final point must still lie strictly inside the original bracket.
    This function never queries an evaluator or consumes the global RNG.
    """
    if fraction == 0 or bool((x == point).any()):
        return point, False
    i = int(torch.searchsorted(x, x.new_tensor(point), right=True)) - 1
    if not 0 <= i < len(x) - 1 or not float(x[i]) < point < float(x[i + 1]):
        raise ValueError("Selected query must be an interior unobserved bracket point.")
    weight = 0.5 + fraction * (2 * float(torch.rand((), generator=generator, dtype=torch.float64)) - 1)
    candidate = session.canonical((x[i] * (1 - weight) + x[i + 1] * weight).reshape(1, 1))
    actual = float(candidate[0, 0])
    if not float(x[i]) < actual < float(x[i + 1]) or bool((x == actual).any()):
        return point, False
    return actual, actual != point


def _jitter_generator(fraction: float, vector_only: bool, seed: int) -> torch.Generator:
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 0.5:
        raise ValueError("jitter_fraction must be finite and between zero and one half.")
    if not isinstance(vector_only, bool):
        raise ValueError("jitter_vector_only must be boolean.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("jitter_seed must be a nonnegative integer.")
    return torch.Generator().manual_seed(seed)


def _apply_jitter(
    session: QuerySession,
    x: torch.Tensor,
    y: torch.Tensor,
    point: float,
    action: str,
    generator: torch.Generator,
    fraction: float,
    vector_only: bool,
) -> tuple[float, bool]:
    if action == "replicate" or (vector_only and y.shape[1] == 1):
        return point, False
    return jitter_query(session, x, point, generator, fraction)


def _validated_prediction_rule(rule: Any) -> str:
    if not isinstance(rule, str) or rule not in {"cv", "coverage", "balanced"}:
        raise ValueError("prediction_rule must be cv, coverage or balanced.")
    return rule


def _query(
    session: QuerySession,
    x: torch.Tensor,
    y: torch.Tensor,
    counts: torch.Tensor,
    evidence: tuple[GapEvidence, ...],
    gaps: tuple[CurveGap, ...],
    candidates: list[tuple[int, float]],
    *,
    step: int,
    target: float,
    exploration_every: int,
    audit_every: int,
    boundary_every: int,
    localize_when: Callable[[GapEvidence], bool] | None,
    boundary_vector_only: bool,
    prediction_rule: str,
) -> tuple[float, str]:
    zero = torch.zeros(len(x) - 1, dtype=torch.float64)
    if not session.deterministic and step % (2 * exploration_every) == 0:
        return float(x[int(torch.argmin(counts))]), "replicate"
    if step % audit_every == 1 and gaps:
        cuts = {(gap.left, gap.right) for gap in gaps}
        eligible = [(i, point) for i, point in candidates if (float(x[i]), float(x[i + 1])) in cuts]
        return _ranked_cell(eligible or candidates, zero, x), "cut_audit" if eligible else "explore"
    if step % exploration_every == 0:
        return _ranked_cell(candidates, zero, x), "explore"
    if boundary_every and step % boundary_every == 1 and (not boundary_vector_only or y.shape[1] > 1):
        eligible = [
            (i, point)
            for i, point in candidates
            if evidence[i].excess > evidence[i].threshold
            and evidence[i].right - evidence[i].left > target
            and (localize_when is None or localize_when(evidence[i]))
        ]
        if eligible:
            chosen = max(eligible, key=lambda row: evidence[row[0]].excess)
            return chosen[1], "vector_evidence_localization" if boundary_vector_only else "boundary_localization"
    if y.shape[1] == 1 or prediction_rule == "cv":
        score, action = cv_scores(x, y, evidence, target), "local_loo_bias"
    elif prediction_rule == "coverage":
        score, action = zero, "vector_cover"
    else:
        bias = cv_scores(x, y, evidence, target)
        widths = x[1:] - x[:-1]
        score = bias / bias.max() if bool((bias > 0).any()) else zero
        score = score + widths / widths.max()
        action = "vector_balanced"
    return _ranked_cell(candidates, score, x), action


def learn_with_detector(
    session: QuerySession,
    *,
    detector_factory: Callable[..., GapDetector] = MultiscaleGapDetector,
    initial_samples: int = 9,
    location_tolerance: float | None = None,
    exploration_every: int = 4,
    audit_every: int = 8,
    noise_replicates: int = 3,
    min_relative_jump: float = 0.01,
    continuation_factor: float = 3.0,
    min_refinements: int = 2,
    boundary_every: int = 0,
    localize_when: Callable[[GapEvidence], bool] | None = None,
    boundary_vector_only: bool = False,
    jitter_fraction: float = 0.125,
    jitter_vector_only: bool = True,
    jitter_seed: int = 0,
    prediction_rule: str = "coverage",
) -> StructureLearningResult:
    """Fit actual BSF charts using a source-bound detector and charged probes.

    ``location_tolerance`` is a maximum proposed bracket width. By default it
    is twice domain_width/256, corresponding to a conditional midpoint error
    target domain_width/256 for a single contained jump. The target and actual
    finite-width brackets remain distinct from a confidence certificate.
    """
    width = float(session.bounds[0, 1] - session.bounds[0, 0])
    position_target = width / 256
    bracket_limit = 2 * position_target if location_tolerance is None else location_tolerance
    lower, upper, width, target = _settings(
        session, initial_samples, bracket_limit, 32, exploration_every, noise_replicates, min_relative_jump
    )
    if isinstance(audit_every, bool) or not isinstance(audit_every, int) or audit_every < 2:
        raise ValueError("audit_every must be an integer at least two.")
    if isinstance(boundary_every, bool) or not isinstance(boundary_every, int) or boundary_every < 0 or boundary_every == 1:
        raise ValueError("boundary_every must be zero or an integer at least two.")
    if not isinstance(boundary_vector_only, bool):
        raise ValueError("boundary_vector_only must be boolean.")
    prediction_rule = _validated_prediction_rule(prediction_rule)
    generator = _jitter_generator(jitter_fraction, jitter_vector_only, jitter_seed)
    detector = detector_factory(
        continuation_factor=continuation_factor, min_relative_jump=min_relative_jump, min_refinements=min_refinements
    )
    _seed(session, initial_samples)
    _noise_pilot(session, noise_replicates)
    trace: list[dict[str, Any]] = []
    steps = 0
    old_cuts: set[tuple[float, float]] = set()
    status = "budget_exhausted"
    while True:
        inputs, y, counts = session.observations()
        x = inputs[:, 0]
        noise = session.noise_variance() or 0.0
        evidence = detector.update(x, y, counts=counts, noise_variance=noise)
        gaps = tuple(
            CurveGap(row.left, row.right, row.raw_change, row.threshold, row.local_continuation_bound, row.reason)
            for row in evidence
            if row.status == "jump_candidate" and row.right - row.left <= target * (1 + 1e-12)
        )
        cuts = {(gap.left, gap.right) for gap in gaps}
        trace.append(
            {
                "cost": session.cost,
                "gaps": [list(pair) for pair in sorted(cuts)],
                "new_cuts": len(cuts - old_cuts),
                "revoked_cuts": len(old_cuts - cuts),
                "unresolved": sum(row.status == "unresolved" for row in evidence),
            }
        )
        old_cuts = cuts
        if len(x) < 2:
            status = "insufficient_samples"
            break
        if session.cost >= session.budget:
            break
        candidates = _midpoints(session, x)
        if not candidates:
            status = "input_resolution" if session.resolution[0] > 0 else "floating_resolution"
            break
        steps += 1
        point, action = _query(
            session,
            x,
            y,
            counts,
            evidence,
            gaps,
            candidates,
            step=steps,
            target=target,
            exploration_every=exploration_every,
            audit_every=audit_every,
            boundary_every=boundary_every,
            localize_when=localize_when,
            boundary_vector_only=boundary_vector_only,
            prediction_rule=prediction_rule,
        )
        midpoint = point
        point, jittered = _apply_jitter(session, x, y, point, action, generator, jitter_fraction, jitter_vector_only)
        trace[-1].update(
            next_action=action, query_point=point, refinement_step=steps, selected_midpoint=midpoint, query_jittered=jittered
        )
        session.query(torch.tensor([[point]], dtype=torch.float64))
    starts = [0] + [int(torch.searchsorted(x, x.new_tensor(gap.right))) for gap in gaps] + [len(x)]
    models, capacities = [], []
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        if stop > start:
            model = _fit_prediction(x[start:stop], y[start:stop], noise=noise)
            models.append(model.as_spline())
            capacities.append(model.diagnostics)
    diagnostics = {
        "cost": session.cost,
        "unique_inputs": len(x),
        "replicate_rows": session.cost - len(x),
        "noise_variance": session.noise_variance(),
        "noise_confidence_supported": False,
        "topology_certified": False,
        "location_tolerance": target,
        "requested_midpoint_position_tolerance": target / 2,
        "requested_bracket_width_limit": target,
        "largest_interval": float((x[1:] - x[:-1]).max()) if len(x) > 1 else width,
        "capacity": capacities,
        "trace": trace,
        "detector": type(detector).__module__ + "." + type(detector).__name__,
        "acquisition": "cv" if y.shape[1] == 1 else prediction_rule,
        "prediction_rule": prediction_rule,
        "exploration_every": exploration_every,
        "audit_every": audit_every,
        "boundary_every": boundary_every,
        "boundary_vector_only": boundary_vector_only,
        "jitter_fraction": jitter_fraction,
        "jitter_vector_only": jitter_vector_only,
        "jitter_seed": jitter_seed,
        "jittered_queries": sum(row.get("query_jittered", False) for row in trace),
        "audit_queries": sum(row.get("next_action") == "cut_audit" for row in trace),
        "audit_stopping_rule": "no new canonical midpoint; no width/depth floor",
        "continuation_factor": continuation_factor,
        "min_nested_refinements": min_refinements,
        "unresolved": "Empirical evidence; unseen features, sharp continuous transitions and aliases remain possible.",
    }
    return _compile_result(StructureLearningResult(inputs, y, gaps, status, diagnostics, models, (lower, upper)))
