"""Typed numerical solver configurations; importing this module does not load SciPy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from torch_bsf.core import _integer

DiagnosticLevel = Literal["basic", "full"]


@dataclass(frozen=True)
class DenseLeastSquares:
    """CPU float64 SVD solve of the augmented problem, without normal equations.

    Parameters
    ----------
    rcond
        Relative singular-value cutoff. ``None`` uses machine epsilon times
        the larger matrix dimension. The actual absolute threshold is recorded.
    diagnostics
        ``basic`` reuses the system SVD; ``full`` additionally diagnoses the
        weighted observation matrix when regularization changes the system.
    """

    rcond: float | None = None
    diagnostics: DiagnosticLevel = "basic"

    def __post_init__(self) -> None:
        if self.rcond is not None and (not math.isfinite(self.rcond) or self.rcond < 0):
            raise ValueError("rcond must be finite and nonnegative.")
        if self.diagnostics not in ("basic", "full"):
            raise ValueError("diagnostics must be 'basic' or 'full'.")


@dataclass(frozen=True)
class SparseLeastSquares:
    """Optional SciPy LSMR on CPU float64, with explicit per-output stop codes.

    Parameters
    ----------
    atol, btol
        Backward-error stopping tolerances.
    conlim
        Estimated condition-number limit; zero disables this limit.
    max_iter
        Maximum iterations per output; ``None`` uses the backend default.
        Condition or iteration limits are failures, not successful convergence.
    """

    atol: float = 1e-8
    btol: float = 1e-8
    conlim: float = 1e8
    max_iter: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in (self.atol, self.btol, self.conlim)):
            raise ValueError("solver tolerances must be finite and nonnegative.")
        if self.max_iter is not None:
            _integer(self.max_iter, "max_iter", 1)


LeastSquaresSolver = DenseLeastSquares | SparseLeastSquares
