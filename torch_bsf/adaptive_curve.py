"""Budgeted one-parameter curve approximation with provisional gap discovery.

The evaluator supplies scalar or vector responses at requested parameter rows.
Observed-data model selection and continuing gap audits produce actual Bézier
charts. Finite samples cannot certify continuity, discontinuities or the
topology of the embedded image. Install the ``adaptive`` extra to execute a fit.
"""

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from torch_bsf.core import BezierModel
from torch_bsf.geometry import Simploid, SimploidalSet


@dataclass(frozen=True)
class CurveGap:
    """Record a provisional excluded bracket, without a topology certificate.

    Attributes
    ----------
    left : float
        Left observed parameter, retained by the left chart.
    right : float
        Right observed parameter, retained by the right chart.
    jump_estimate : float
        Observed endpoint contrast magnitude in response units.
    threshold : float
        Empirical response-scale threshold used for this proposal.
    sensitivity : float
        Local continuation diagnostic in response units.
    evidence : str
        Finite-scale evidence that produced the proposal.
    """

    left: float
    right: float
    jump_estimate: float
    threshold: float
    sensitivity: float
    evidence: str = "finite_scale_candidate"


@dataclass(frozen=True)
class CurveInterval:
    """Locate an original parameter interval in the compiled Bézier model.

    Attributes
    ----------
    left : float
        Original parameter at the cell's first endpoint.
    right : float
        Original parameter at the second endpoint; equal for a singleton chart.
    cell_id : int
        Compiled cell address accepted by ``Points``.
    chart_index : int
        Connected input-chart index, in increasing parameter order.
    piece_index : int
        Polynomial interval index within its chart.
    handle : Simploid
        Cell handle in the returned mutable domain builder.
    """

    left: float
    right: float
    cell_id: int
    chart_index: int
    piece_index: int
    handle: Simploid = field(repr=False)


@dataclass(frozen=True)
class AdaptiveCurveResult:
    """Retain actual BSF charts, response costs and finite-scale gap evidence.

    Budget exhaustion describes the measurement process, not convergence of
    topology or prediction error. The model is polynomial by pieces. A selected
    rational operator is first approximated and checked numerically before its
    polynomial pieces are exported; that approximation is not a uniform error
    certificate. ``diagnostics['capacity']`` retains those conversion checks.

    Attributes
    ----------
    domain : SimploidalSet
        Candidate input-chart topology; distinct charts remain disconnected
        when their output images touch or intersect.
    model : BezierModel
        Actual compiled model. It accepts barycentric ``Points``; use
        :meth:`predict` for original scalar parameter coordinates.
    intervals : tuple of CurveInterval
        Parameter-to-cell routing, required alongside the model coefficients.
    gaps : tuple of CurveGap
        Provisional excluded open brackets, recomputed after each observation.
    bounds : tuple of float
        Original finite parameter bounds.
    degree : int
        Common compiled Bernstein degree; lower-degree pieces are elevated.
    inputs : torch.Tensor
        Sorted unique observed coordinates, CPU float64 with shape ``(n, 1)``.
    values : torch.Tensor
        Mean responses aligned with inputs, CPU float64 with shape ``(n, d)``.
    counts : torch.Tensor
        Number of charged independent responses at each unique input.
    cost : int
        Actual response rows requested from the evaluator, including repeats.
    proposed_rows : int
        Proposed rows, including deterministic cache hits.
    status : str
        ``budget_exhausted``, ``input_resolution``, ``floating_resolution`` or
        ``insufficient_samples``. None is a topology certificate.
    diagnostics : dict
        Acquisition trace, empirical noise, model selection, conversion and
        gap diagnostics. Noise estimates are not confidence guarantees.
    """

    domain: SimploidalSet
    model: BezierModel
    intervals: tuple[CurveInterval, ...]
    gaps: tuple[CurveGap, ...]
    bounds: tuple[float, float]
    degree: int
    inputs: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    cost: int
    proposed_rows: int
    status: str
    diagnostics: dict[str, Any]

    def predict(self, parameters: Any) -> torch.Tensor:
        """Evaluate original coordinates, returning NaNs inside proposed gaps.

        Parameters
        ----------
        parameters : array-like
            Finite real rows of shape ``(n, 1)`` inside ``bounds``. A declared
            grid affects queries, not these surrogate prediction coordinates.

        Returns
        -------
        torch.Tensor
            Detached CPU float64 predictions with shape ``(n, d)``. Both gap
            endpoints are retained. When a grid's extreme keys lie inside the
            raw bounds, their endpoint values extend to those bounds.
        """
        from torch_bsf._adaptive_curve.export import BSFCurveExport

        exported = BSFCurveExport(self.domain, self.model, self.intervals, self.degree, self.bounds, self.gaps)
        return exported.predict(parameters)


def _validate_options(
    bounds: Sequence[float] | torch.Tensor,
    deterministic: bool,
    resolution: float | None,
    location_tolerance: float | None,
    seed: int,
) -> torch.Tensor:
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a bool.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer between zero and 2**64 - 1.")
    raw_bounds = torch.as_tensor(bounds)
    if raw_bounds.is_complex() or raw_bounds.shape not in ((2,), (1, 2)):
        raise ValueError("bounds must be a real increasing parameter pair.")
    if resolution is not None and (isinstance(resolution, bool) or resolution <= 0):
        raise ValueError("resolution must be positive or None.")
    if location_tolerance is not None and (
        isinstance(location_tolerance, bool) or not math.isfinite(location_tolerance) or location_tolerance <= 0
    ):
        raise ValueError("location_tolerance must be finite and positive.")
    if isinstance(bounds, torch.Tensor):
        return bounds.detach().cpu().double().reshape(1, 2)
    return torch.as_tensor(bounds, dtype=torch.float64).reshape(1, 2)


def fit_adaptive_curve(
    oracle: Callable[[torch.Tensor], Any],
    bounds: Sequence[float] | torch.Tensor,
    *,
    budget: int,
    deterministic: bool = True,
    resolution: float | None = None,
    location_tolerance: float | None = None,
    seed: int = 0,
    initial_inputs: Any = None,
) -> AdaptiveCurveResult:
    """Query a scalar parameter and fit connected or provisionally split charts.

    All initialization, global exploration, localization, continuing audits and
    stochastic repeats consume the same response-row budget. Observed-data
    cross-validation chooses chart predictors and can share a smooth shape
    between vector charts, including a convex spline/kernel mixture. Vector
    queries combine coverage, localization and
    private central jitter; scalar responses retain the scalar persistence
    policy and can select a stationary kernel predictor.

    Parameters
    ----------
    oracle : callable
        Accept an independent CPU float64 tensor of shape ``(n, 1)`` and return
        finite real responses of shape ``(n, d)`` with a fixed positive output
        count. No gradients are retained. Requested rows are charged before
        executing the callback, even if it raises or returns invalid data.
    bounds : sequence of float or torch.Tensor
        Finite increasing pair ``(lower, upper)`` with finite width.
    budget : int
        Positive maximum actual response-row count. Deterministic cache hits
        are free; stochastic repeats each consume another response row.
    deterministic : bool
        Whether identical canonical inputs have fixed responses. False enables
        charged repeats and an empirical pooled noise estimate.
    resolution : float or None
        Known positive zero-aligned query-grid step. None permits continuous
        queries. This is an input protocol supplied by the application.
    location_tolerance : float or None
        Maximum proposed bracket width, enlarged to one declared grid step if
        necessary. None uses twice the domain width divided by 256, with a
        positive float64 lower limit. This is a requested finite-scale
        resolution, not a localization guarantee.
    seed : int
        Integer from zero through ``2**64 - 1`` for a private vector-query jitter
        generator. Scalar acquisitions use no random jitter. The acquisition
        policy does not consume the global random generator.
    initial_inputs : array-like or None
        Optional initial parameter rows of shape ``(n, 1)``. These are queried
        and charged first. The policy then queries any missing points from its
        nine uniform anchors, subject to the remaining budget. None starts
        with those anchors alone. Responses cannot be supplied for free.

    Returns
    -------
    AdaptiveCurveResult
        Actual BSF model, routing metadata, observations, costs and provisional
        gaps. Prediction inside excluded brackets remains NaN.

    Raises
    ------
    ImportError
        If SciPy is unavailable; install ``pytorch-bsf[adaptive]``.
    ValueError
        If arguments or evaluator responses violate the input contract.

    Notes
    -----
    Unknown narrow features, grid aliases, continuous cusps and deterministic
    roughness can defeat finite observations. Input-chart components are a
    modeling proposal, not the topology of the output image. A model archive
    alone does not preserve the original-coordinate interval routing.

    Examples
    --------
    >>> from torch_bsf import fit_adaptive_curve
    >>> result = fit_adaptive_curve(lambda x: x.square(), (0.0, 1.0), budget=9)
    >>> result.cost, result.model.n_outputs
    (9, 1)
    >>> result.predict([[0.25]]).shape
    torch.Size([1, 1])
    """
    pairs = _validate_options(bounds, deterministic, resolution, location_tolerance, seed)
    try:
        importlib.import_module("scipy")
    except ImportError as failure:
        raise ImportError("fit_adaptive_curve requires SciPy; install 'pytorch-bsf[adaptive]'.") from failure

    from torch_bsf._adaptive_curve.detector import EmbeddingGapDetector
    from torch_bsf._adaptive_curve.engine import _default_bracket_limit, learn_with_detector
    from torch_bsf._adaptive_curve.kernel_prediction import refit_prediction as refit_kernel_prediction
    from torch_bsf._adaptive_curve.polynomial import _settings
    from torch_bsf._adaptive_curve.pooled_prediction import refit_prediction
    from torch_bsf._adaptive_curve.queries import QuerySession
    from torch_bsf._adaptive_curve.stacked_prediction import refit_prediction as refit_stacked_prediction

    session = QuerySession(oracle, pairs, budget=budget, deterministic=deterministic, resolution=resolution)
    try:
        width = float(session.bounds[0, 1] - session.bounds[0, 0])
        _settings(
            session, 9, _default_bracket_limit(width) if location_tolerance is None else location_tolerance, 32, 4, 3, 0.01
        )
        if initial_inputs is not None:
            session.query(initial_inputs)
        learned = learn_with_detector(
            session,
            detector_factory=EmbeddingGapDetector,
            location_tolerance=location_tolerance,
            boundary_every=2,
            boundary_vector_only=True,
            jitter_fraction=0.125,
            jitter_vector_only=True,
            jitter_seed=seed,
        )
        learned = (
            refit_kernel_prediction(learned)
            if learned.values.shape[1] == 1
            else refit_stacked_prediction(refit_prediction(learned))
        )
    except Exception as failure:
        failure.add_note(f"Adaptive curve requested {session.cost} response rows from {session.proposed_rows} proposed rows.")
        raise
    if learned.values.shape[1] == 1:
        for row in learned.diagnostics["trace"]:
            row.pop("selected_midpoint", None)
            row.pop("query_jittered", None)
    learned.diagnostics["profile"] = "kernel_cv_scalar" if learned.values.shape[1] == 1 else "coverage_shared_c0_component_gp"
    compiled = learned._compiled
    if compiled is None:
        raise ArithmeticError("Adaptive curve model was not compiled.")
    inputs, values, counts = session.observations()
    return AdaptiveCurveResult(
        compiled.domain,
        compiled.model,
        compiled.intervals,
        compiled.gaps,
        compiled.bounds,
        compiled.degree,
        inputs,
        values,
        counts,
        session.cost,
        session.proposed_rows,
        learned.status,
        learned.diagnostics,
    )
