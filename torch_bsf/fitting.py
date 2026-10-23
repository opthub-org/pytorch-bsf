"""One supervised fitting entry point for every space and numerical solver."""

from __future__ import annotations

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
    data = MatrixDiagnostics((len(samples), n))
    system = MatrixDiagnostics(problem.shape)
    iterations: tuple[int, ...] = ()
    codes: tuple[int, ...] = ()
    residuals: tuple[float, ...] = ()
    normals: tuple[float, ...] = ()
    conditions: tuple[float, ...] = ()
    success, status, message = True, "solved", "Least-squares optimum computed."
    if not n:
        solution = torch.empty(0, problem.values.shape[1], dtype=torch.float64)
        data = MatrixDiagnostics((len(samples), 0), rank=0, method="no_free_controls")
        system = MatrixDiagnostics(problem.shape, rank=0, method="no_free_controls")
        status, message = "all_fixed", "All coefficients are fixed; no factorization was needed."
    elif isinstance(solver, DenseLeastSquares):
        a = problem.dense()
        relative = torch.finfo(torch.float64).eps * max(a.shape) if solver.rcond is None else solver.rcond
        solved = torch.linalg.lstsq(a, problem.rhs, rcond=relative, driver="gelsd")
        solution = solved.solution
        system = _svd_info(problem.shape, solved.singular_values, int(solved.rank), solver.rcond)
        if not problem.n_regularizers:
            data = system
        elif solver.diagnostics == "full":
            data = _data_info(problem, solver.rcond)
    else:
        matrix = problem.sparse()
        from scipy.sparse.linalg import lsmr

        answers = [
            lsmr(
                matrix,
                problem.rhs[:, j].numpy(),
                atol=solver.atol,
                btol=solver.btol,
                conlim=solver.conlim,
                maxiter=solver.max_iter,
            )
            for j in range(problem.values.shape[1])
        ]
        solution = torch.stack([torch.from_numpy(answer[0]) for answer in answers], dim=1)
        codes = tuple(int(a[1]) for a in answers)
        iterations = tuple(int(a[2]) for a in answers)
        residuals = tuple(float(a[3]) for a in answers)
        normals = tuple(float(a[4]) for a in answers)
        conditions = tuple(float(a[6]) for a in answers)
        success = all(code in (0, 1, 2, 4, 5) for code in codes)
        system = MatrixDiagnostics(problem.shape, condition_number=max(conditions), method="estimate")
        status = "converged" if success else "not_converged"
        message = (
            "LSMR converged for every output." if success else f"LSMR did not converge for every output; stop codes: {codes}."
        )
    if not torch.isfinite(solution).all():
        raise ArithmeticError("least-squares solver produced non-finite coefficients.")
    coefficients = problem.coefficients(solution)
    model = BezierModel(space, coefficients)
    diagnostics = FitDiagnostics(
        type(solver).__name__,
        len(samples),
        n,
        samples.weights.sum().item(),
        data,
        system,
        iterations=iterations,
        stop_codes=codes,
        residual_norms=residuals,
        normal_residual_norms=normals,
        condition_estimates=conditions,
    )
    return FitResult(model, problem.metrics(coefficients), diagnostics, success, status, message).warn_if_failed()
