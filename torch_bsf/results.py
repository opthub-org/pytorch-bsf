"""Explicit numerical results: a computed model is not a convergence certificate."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import torch

from torch_bsf.core import BezierModel, BezierSpace


class ConvergenceWarning(UserWarning):
    """A numerical method stopped without meeting its convergence criteria."""


class ConvergenceError(RuntimeError):
    """Failure requested by ``require_success()``, retaining the numerical result.

    Parameters
    ----------
    result : FitResult
        Failed solve retained on ``result`` with its termination message.
    """

    def __init__(self, result: "FitResult"):
        self.result = result
        super().__init__(result.message)


@dataclass(frozen=True)
class MatrixDiagnostics:
    """Information about one explicitly identified least-squares matrix.

    ``None`` means not computed, never zero. ``method`` distinguishes a direct
    singular-value calculation from an iterative condition estimate. A rank
    below the number of free columns has infinite coefficient condition number.
    """

    shape: tuple[int, int]
    rank: int | None = None
    rank_threshold: float | None = None
    singular_values: torch.Tensor | None = None
    condition_number: float | None = None
    method: Literal["svd", "estimate", "not_computed", "no_free_controls"] = "not_computed"


@dataclass(frozen=True)
class FitDiagnostics:
    """Compute provenance and observation/system diagnostics in original units.

    ``driver`` identifies the executed backend (``gelsd`` or ``lsmr``), or is
    ``None`` if no solve was executed. ``output_success`` and ``output_statuses``
    record termination for each output column, in prediction-column order.
    ``solution_available`` is false when a numerical failure left no finite
    iterate for at least one column; see :class:`FitResult` for its placeholder.
    Residual norms and condition estimates from LSMR concern the augmented
    weighted, constrained system, not the unpenalized prediction MSE.
    Iterative fields use ``None`` for a column whose RHS overflowed and was not
    solved, or empty tuples if no iterative solve ran at all. Overall success
    can still be false after successful solves if objective evaluation or
    optional diagnostics fail; these failures do not erase solver termination.
    """

    solver: str
    n_samples: int
    n_free_controls: int
    weight_sum: float
    data: MatrixDiagnostics
    system: MatrixDiagnostics
    compute_dtype: str = "float64"
    compute_device: str = "cpu"
    model_dtype: str = "float64"
    model_device: str = "cpu"
    iterations: tuple[int | None, ...] = ()
    stop_codes: tuple[int | None, ...] = ()
    residual_norms: tuple[float | None, ...] = ()
    normal_residual_norms: tuple[float | None, ...] = ()
    condition_estimates: tuple[float | None, ...] = ()
    driver: str | None = None
    output_success: tuple[bool, ...] = ()
    output_statuses: tuple[str, ...] = ()
    solution_available: bool = True


@dataclass(frozen=True)
class FitResult:
    """A model, objective metrics, diagnostics and an explicit termination state.

    Results are deliberately not callable. Predict with ``result.model(points)``.
    If ``diagnostics.solution_available`` is false, the model is only a finite
    placeholder preserving fixed controls, with free rows of unavailable output
    columns zero. It must not be used as a fitted solution. Finite unconverged
    LSMR iterates are retained.
    """

    model: BezierModel
    metrics: dict[str, float]
    diagnostics: FitDiagnostics
    success: bool
    status: str
    message: str

    def require_success(self) -> "FitResult":
        """Return this result, or raise a :class:`ConvergenceError` containing it.

        Returns
        -------
        FitResult
            This result when the solve succeeded.
        """
        if not self.success:
            raise ConvergenceError(self)
        return self

    def warn_if_failed(self) -> "FitResult":
        """Expose a failed numerical solve without hiding its partial result.

        Returns
        -------
        FitResult
            This result, also returned after emitting a failure warning.
        """
        if not self.success:
            warnings.warn(self.message, ConvergenceWarning, stacklevel=3)
        return self


@dataclass(frozen=True)
class SkeletonRegressionResult:
    """Assembled model and per-stage least-squares results.

    The vertex IDs of a simploid are its global coordinate indices, so a face
    lists the vertices of every factor. Each face result solves for that face's
    interior controls with every other control fixed. Faces are fitted in
    increasing dimension; a failed face stops assembly, leaving later faces
    absent and their controls zero. For spaces created with ``BezierSpace.on_set``
    and zero-dimensional spaces, keys are integer nondegenerate cell IDs,
    which distinguish local coordinate indices in different cells. The original
    single-cell simplex/simploid API retains its sorted coordinate-tuple keys.
    """

    model: BezierModel
    faces: dict[tuple[int, ...] | int, FitResult]
    success: bool
    status: str
    message: str

    def require_success(self) -> "SkeletonRegressionResult":
        """Raise :class:`ConvergenceError` with the failed face's result if a face fit failed.

        Returns
        -------
        SkeletonRegressionResult
            This result when every required face fit succeeded.
        """
        if not self.success:
            failed = list(self.faces.values())[-1]
            raise ConvergenceError(replace(failed, message=self.message))
        return self


@dataclass(frozen=True)
class SkeletonAllocation:
    """Reusable skeleton allocation and an explicit record of the selected strategy.

    ``ratio`` contains total fractions by dimension. ``cell_weights`` contains
    individual active-cell fractions on addressable spaces, and is empty for
    bare positive-dimensional simplex/simploid constructors. Pass this entire
    result as ``sample_skeleton(..., allocation=result)`` to preserve both the
    selected rounding strategy and unequal within-dimension cell fractions.

    ``strategy`` is the method actually used. ``fallback_reason`` is
    ``"single_cell_fast_path"`` when automatic selection retains the original
    unassembled algorithm, ``"zero_cell_coefficients"`` when equal dimension
    shares rescue zero-risk cells, and otherwise ``None``. The frozen space
    snapshot prevents accidental reuse on an incompatible domain or degree.

    Examples
    --------
    >>> from torch_bsf import BezierSpace, SkeletonAllocation
    >>> from torch_bsf.sampling import optimal_skeleton_ratio, sample_skeleton
    >>> space = BezierSpace.simplex(1, degree=2)
    >>> plan = optimal_skeleton_ratio(space)
    >>> isinstance(plan, SkeletonAllocation), plan.strategy, plan.fallback_reason
    (True, 'dimension', 'single_cell_fast_path')
    >>> len(sample_skeleton(space, 8, allocation=plan, seed=0))
    8
    """

    space: BezierSpace = field(repr=False)
    strategy: Literal["cell", "dimension"]
    ratio: tuple[float, ...]
    cell_weights: Mapping[int, float]
    fallback_reason: Literal["single_cell_fast_path", "zero_cell_coefficients"] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio", tuple(self.ratio))
        object.__setattr__(self, "cell_weights", MappingProxyType(dict(self.cell_weights)))
