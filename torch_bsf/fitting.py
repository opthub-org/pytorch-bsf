"""One supervised fitting entry point for every space and numerical solver."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TypeVar

import torch

from torch_bsf._problem import LeastSquaresProblem
from torch_bsf.core import BezierModel, BezierSpace, FixedControls, QuadraticPenalty, Samples
from torch_bsf.results import FitDiagnostics, FitResult, MatrixDiagnostics
from torch_bsf.solvers import DenseLeastSquares, LeastSquaresSolver, SparseLeastSquares


def _svd_info(shape: tuple[int, int], singular: torch.Tensor, rank: int, rcond: float | None) -> MatrixDiagnostics:
    threshold = (
        ((torch.finfo(torch.float64).eps * max(shape)) if rcond is None else rcond) * singular[0].item()
        if len(singular)
        else 0.0
    )
    condition = float("inf") if rank < shape[1] else (singular[0] / singular[-1]).item() if len(singular) else None
    return MatrixDiagnostics(shape, rank, threshold, singular.detach().clone(), condition, "svd")


def _data_info(problem: LeastSquaresProblem, rcond: float | None) -> MatrixDiagnostics:
    a = problem.basis.to_dense()[:, problem.free] * problem.scale[:, None]
    singular = torch.linalg.svdvals(a)
    cutoff = (
        ((torch.finfo(torch.float64).eps * max(a.shape)) if rcond is None else rcond) * singular[0] if len(singular) else 0.0
    )
    rank = int((singular > cutoff).sum())
    return _svd_info((a.shape[0], a.shape[1]), singular, rank, rcond)


def _dense_solve(
    problem: LeastSquaresProblem, rhs: torch.Tensor, solver: DenseLeastSquares, diagnostics: FitDiagnostics
) -> tuple[torch.Tensor, FitDiagnostics]:
    a = problem.dense()
    relative = torch.finfo(torch.float64).eps * max(a.shape) if solver.rcond is None else solver.rcond
    solved = torch.linalg.lstsq(a, rhs, rcond=relative, driver="gelsd")
    solution = solved.solution.clone()
    available = torch.isfinite(solution).all(dim=0)
    solution[:, ~available] = 0
    system = _svd_info(problem.shape, solved.singular_values, int(solved.rank), solver.rcond)
    data = diagnostics.data
    if not problem.n_regularizers:
        data = system
    return solution, replace(
        diagnostics,
        data=data,
        system=system,
        solution_available=bool(available.all()),
        output_success=tuple(bool(valid) for valid in available),
        output_statuses=tuple("solved" if valid else "numerical_failure" for valid in available),
    )


def _complete_diagnostics(
    problem: LeastSquaresProblem, solver: LeastSquaresSolver, diagnostics: FitDiagnostics
) -> tuple[FitDiagnostics, str | None]:
    if (
        isinstance(solver, DenseLeastSquares)
        and solver.diagnostics == "full"
        and problem.n_regularizers
        and diagnostics.system.method == "svd"
    ):
        try:
            return replace(diagnostics, data=_data_info(problem, solver.rcond)), None
        except (torch.linalg.LinAlgError, ArithmeticError) as error:
            return diagnostics, f"Additional data diagnostics failed: {error}"
    return diagnostics, None


_LSMR_STATUSES = {
    0: "zero_solution",
    1: "compatible_tolerance",
    2: "least_squares_tolerance",
    3: "condition_limit",
    4: "compatible_machine_precision",
    5: "least_squares_machine_precision",
    6: "machine_condition_limit",
    7: "iteration_limit",
}


def _sparse_solve(
    problem: LeastSquaresProblem, rhs: torch.Tensor, solver: SparseLeastSquares, diagnostics: FitDiagnostics
) -> tuple[torch.Tensor, FitDiagnostics]:
    matrix = problem.sparse()
    from scipy.sparse.linalg import lsmr

    answers = [
        lsmr(
            matrix,
            rhs[:, j].numpy(),
            atol=solver.atol,
            btol=solver.btol,
            conlim=solver.conlim,
            maxiter=solver.max_iter,
        )
        for j in range(rhs.shape[1])
    ]
    solution = torch.stack([torch.from_numpy(answer[0]) for answer in answers], dim=1)
    codes = tuple(int(a[1]) for a in answers)
    # A stop code alone cannot certify a result containing non-finite diagnostics.
    available = torch.isfinite(solution).all(dim=0)
    finite = tuple(
        bool(valid) and all(math.isfinite(float(v)) for v in a[3:]) for valid, a in zip(available, answers, strict=True)
    )

    solution[:, ~available] = 0
    conditions = tuple(float(a[6]) for a in answers)
    system = MatrixDiagnostics(problem.shape, condition_number=max(conditions) if all(finite) else None, method="estimate")
    return solution, replace(
        diagnostics,
        system=system,
        iterations=tuple(int(a[2]) for a in answers),
        stop_codes=codes,
        residual_norms=tuple(float(a[3]) for a in answers),
        normal_residual_norms=tuple(float(a[4]) for a in answers),
        condition_estimates=conditions,
        solution_available=bool(available.all()),
        output_success=tuple(code in (0, 1, 2, 4, 5) and valid for code, valid in zip(codes, finite, strict=True)),
        output_statuses=tuple(
            _LSMR_STATUSES.get(code, "unknown_stop_code") if valid else "numerical_failure"
            for code, valid in zip(codes, finite, strict=True)
        ),
    )


_DiagnosticValue = TypeVar("_DiagnosticValue")


def _expand_columns(values: tuple[_DiagnosticValue, ...], usable: torch.Tensor) -> tuple[_DiagnosticValue | None, ...]:
    if not values:
        return ()
    remaining = iter(values)
    return tuple(next(remaining) if valid else None for valid in usable)


def _solve_finite_rhs(
    problem: LeastSquaresProblem, solver: LeastSquaresSolver, diagnostics: FitDiagnostics
) -> tuple[torch.Tensor, FitDiagnostics]:
    usable = torch.isfinite(problem.rhs).all(dim=0)
    outputs = len(usable)
    solution = torch.zeros(problem.n_free, outputs, dtype=torch.float64)
    success, statuses = [False] * outputs, ["rhs_overflow"] * outputs
    if not usable.any():
        return solution, replace(
            diagnostics, driver=None, solution_available=False, output_success=tuple(success), output_statuses=tuple(statuses)
        )
    rhs = problem.rhs[:, usable]
    if isinstance(solver, DenseLeastSquares):
        computed, evidence = _dense_solve(problem, rhs, solver, diagnostics)
    else:
        computed, evidence = _sparse_solve(problem, rhs, solver, diagnostics)
    solution[:, usable] = computed
    for reduced, original in enumerate(torch.nonzero(usable).flatten().tolist()):
        success[original] = evidence.output_success[reduced]
        statuses[original] = evidence.output_statuses[reduced]
    return solution, replace(
        evidence,
        solution_available=evidence.solution_available and bool(usable.all()),
        output_success=tuple(success),
        output_statuses=tuple(statuses),
        iterations=_expand_columns(evidence.iterations, usable),
        stop_codes=_expand_columns(evidence.stop_codes, usable),
        residual_norms=_expand_columns(evidence.residual_norms, usable),
        normal_residual_norms=_expand_columns(evidence.normal_residual_norms, usable),
        condition_estimates=_expand_columns(evidence.condition_estimates, usable),
    )


def fit(
    space: BezierSpace,
    samples: Samples,
    *,
    solver: LeastSquaresSolver = DenseLeastSquares(),
    penalty: QuadraticPenalty = QuadraticPenalty(),
    fixed: FixedControls | None = None,
) -> FitResult:
    r"""Fit a space to observations, returning an explicit numerical result.

    Minimizes ``sum(w * squared_output_error) / sum(w)`` plus ``ridge * ||P||²``
    and ``smoothness * ||D P||²`` in the original output units. Data MSE divides
    the data term by the output dimension. Fixed coefficients are exact values,
    not initial guesses. Neither the space nor observations are modified.

    Parameters
    ----------
    space
        A compiled single-cell or glued-cell polynomial space.
    samples
        Coordinates, outputs and optional sample weights.
    solver
        Typed dense SVD or sparse LSMR configuration. Both compute on CPU
        float64 and return a float64 model; no implicit downcast is performed.
    penalty
        Quadratic weights, independent of the number of observations.
    fixed
        Explicit globally shared coefficient values.

    Returns
    -------
    FitResult
        Model, metrics, diagnostics and termination state. This type never
        changes with diagnostic settings. Predict with ``result.model(...)``.
        Numerical failures return ``success=False`` and emit ConvergenceWarning;
        ``require_success()`` raises with the result attached. If no finite
        iterate exists, ``diagnostics.solution_available`` is false and the
        model contains placeholder free coefficients rather than a fit.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import BezierSpace, Samples, fit
    >>> t = torch.eye(2)
    >>> result = fit(BezierSpace.simplex(1, degree=1), Samples(t, [[1.], [2.]]))
    >>> result.success, result.model(t).tolist()
    (True, [[1.0], [2.0]])
    """
    if not isinstance(space, BezierSpace) or not isinstance(samples, Samples):
        raise TypeError("fit requires a BezierSpace and Samples.")
    if not isinstance(solver, DenseLeastSquares | SparseLeastSquares):
        raise TypeError("solver must be a typed least-squares configuration.")
    problem = LeastSquaresProblem(space, samples, penalty, fixed)
    n = problem.n_free
    outputs = problem.values.shape[1]
    diagnostics = FitDiagnostics(
        type(solver).__name__,
        len(samples),
        n,
        samples.weights.sum().item(),
        MatrixDiagnostics((len(samples), n)),
        MatrixDiagnostics(problem.shape),
        output_success=(True,) * outputs,
        output_statuses=("solved",) * outputs,
    )
    success, status, message = True, "solved", "Least-squares optimum computed."
    solution = torch.zeros(n, outputs, dtype=torch.float64)
    try:
        if not n:
            diagnostics = replace(
                diagnostics,
                data=MatrixDiagnostics((len(samples), 0), rank=0, method="no_free_controls"),
                system=MatrixDiagnostics(problem.shape, rank=0, method="no_free_controls"),
                output_statuses=("all_fixed",) * outputs,
            )
            status, message = "all_fixed", "All coefficients are fixed; no factorization was needed."
        elif isinstance(solver, DenseLeastSquares):
            diagnostics = replace(diagnostics, driver="gelsd")
            solution, diagnostics = _solve_finite_rhs(problem, solver, diagnostics)
            success = all(diagnostics.output_success)
            status = "solved" if success else "numerical_failure"
            message = (
                "Least-squares optimum computed."
                if success
                else f"Dense solve unavailable for one or more outputs: {diagnostics.output_statuses}."
            )
        else:
            diagnostics = replace(diagnostics, driver="lsmr")
            solution, diagnostics = _solve_finite_rhs(problem, solver, diagnostics)
            success = all(diagnostics.output_success)
            status = "converged" if success else "not_converged"
            if any(s in ("numerical_failure", "rhs_overflow") for s in diagnostics.output_statuses):
                status = "numerical_failure"
            message = (
                "LSMR converged for every output."
                if success
                else f"LSMR failed for one or more outputs: {diagnostics.output_statuses}."
            )
    except (torch.linalg.LinAlgError, ArithmeticError) as error:
        success, status, message = False, "numerical_failure", str(error)
        diagnostics = replace(
            diagnostics,
            solution_available=False,
            output_success=(False,) * outputs,
            output_statuses=(status,) * outputs,
        )
    coefficients = problem.coefficients(solution)
    diagnostics, diagnostic_error = _complete_diagnostics(problem, solver, diagnostics)
    if diagnostic_error is not None and success:
        success, status, message = False, "diagnostic_failure", diagnostic_error
    metrics = problem.metrics(coefficients)
    if not all(math.isfinite(value) for value in metrics.values()):
        success, status, message = False, "numerical_failure", "Objective evaluation overflowed."
    return FitResult(BezierModel(space, coefficients), metrics, diagnostics, success, status, message).warn_if_failed()
