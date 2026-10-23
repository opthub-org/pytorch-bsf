"""Explicit numerical results: a computed model is not a convergence certificate."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from torch_bsf.core import BezierModel


class ConvergenceWarning(UserWarning):
    """A numerical method stopped without meeting its convergence criteria."""


class ConvergenceError(RuntimeError):
    """Failure requested by ``require_success()``, retaining the numerical result."""

    def __init__(self, result: FitResult):
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
    """Compute provenance and observation/system diagnostics in original units."""

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
    iterations: tuple[int, ...] = ()
    stop_codes: tuple[int, ...] = ()
    residual_norms: tuple[float, ...] = ()
    normal_residual_norms: tuple[float, ...] = ()
    condition_estimates: tuple[float, ...] = ()


@dataclass(frozen=True)
class FitResult:
    """A model, objective metrics, diagnostics and an explicit termination state.

    Results are deliberately not callable. Predict with ``result.model(points)``.
    """

    model: BezierModel
    metrics: dict[str, float]
    diagnostics: FitDiagnostics
    success: bool
    status: str
    message: str

    def require_success(self) -> FitResult:
        """Return this result, or raise a :class:`ConvergenceError` containing it."""
        if not self.success:
            raise ConvergenceError(self)
        return self

    def warn_if_failed(self) -> FitResult:
        """Expose a failed numerical solve without hiding its partial result."""
        if not self.success:
            warnings.warn(self.message, ConvergenceWarning, stacklevel=3)
        return self
