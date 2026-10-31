"""Supervised fitting at known parameters: one entry point for every space and solver, and face-by-face simplex fitting."""

import math
from dataclasses import replace
from typing import TypeVar

import torch

from torch_bsf._problem import LeastSquaresProblem
from torch_bsf._skeleton import (
    _control_faces,
    _faces_with_interior_controls,
    _interior_and_boundary,
    _single_skeleton,
    _skeleton_cell,
    _skeleton_observations,
    _skeleton_stages,
)
from torch_bsf.core import BezierModel, BezierSpace, FixedControls, QuadraticPenalty, Samples
from torch_bsf.results import FitDiagnostics, FitResult, MatrixDiagnostics, SkeletonRegressionResult
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


def fit_skeleton_regression(
    space: BezierSpace,
    samples: Samples,
    *,
    solver: LeastSquaresSolver = DenseLeastSquares(),
) -> SkeletonRegressionResult:
    """Fit a simploidal space cell by cell from observations at known parameters.

    This is the inductive skeleton fitting analyzed by Tanaka et al. (AAAI 2020).
    Vertices are fitted first, then edges, then faces of higher dimension. On each face,
    ordinary least squares on the observations lying on that face estimates
    the controls interior to it, with the controls on its boundary fixed at
    their estimates. Contrast :func:`fit`, which uses every observation for
    every control, and :func:`~torch_bsf.point_cloud.fit_skeleton`, which
    infers unknown parameters.

    A row lies on the face spanned by its nonzero coordinates; zeros must be
    exact. A face of a simploid is a product of one face per simplex factor and
    is keyed by the sorted global coordinate indices of its vertices. Draw rows
    with :func:`~torch_bsf.sampling.sample_skeleton` and the
    :func:`~torch_bsf.sampling.optimal_skeleton_ratio` to minimize the
    asymptotic risk given by :func:`~torch_bsf.sampling.skeleton_risk_coefficients`.

    For an assembled set, exact-zero boundaries are mapped through the compiled
    face and degeneracy maps to their nondegenerate cell. Each row is used once.
    Shared cells are fitted once. ``faces`` uses integer cell IDs, while single
    simplex/simploid spaces retain their coordinate-tuple keys. Vertices, including
    isolated vertices, have one proper control and use their directly tagged rows.

    Parameters
    ----------
    space
        A simplex, simploid or assembled set, with positive degrees in positive-dimensional factors.
    samples
        Observations at known parameters. Every face with at most ``degree``
        vertices in each factor needs at least one row of positive weight. Rows
        on other faces, which have no interior controls, are ignored and
        counted in the result's message. Weights apply within each face.
    solver
        Least-squares configuration used for every face.

    Returns
    -------
    SkeletonRegressionResult
        The assembled model and each face's :class:`FitResult`. A face whose
        rows do not identify its interior controls is not a failure; inspect
        that face's ``diagnostics.data.rank``. If a face's solve fails, as
        reported by its :class:`FitResult` (with a
        :class:`~torch_bsf.ConvergenceWarning`), fitting stops there:
        ``success`` is ``False``, ``faces`` ends with the failed face, and
        controls of later faces stay zero.

    Raises
    ------
    TypeError
        If ``space`` or ``samples`` has the wrong type.
    ValueError
        If a positive-dimensional factor has degree zero, or an active face/cell has no row of positive weight.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import BezierSpace, Samples
    >>> from torch_bsf.fitting import fit_skeleton_regression
    >>> from torch_bsf.sampling import sample_skeleton
    >>> space = BezierSpace.simplex(1, degree=2)
    >>> points = sample_skeleton(space, 8, seed=0)
    >>> t = points.parameters
    >>> result = fit_skeleton_regression(space, Samples(points, (t[:, 1] ** 2)[:, None]))
    >>> sorted(result.faces), torch.allclose(result.model(t), (t[:, 1] ** 2)[:, None])
    ([(0,), (0, 1), (1,)], True)

    On a square of degree one, only the vertices have interior controls; each
    vertex key lists one coordinate index of each factor.

    >>> square = BezierSpace.simploid((1, 1), degree=1)
    >>> points = sample_skeleton(square, 8, seed=0)
    >>> result = fit_skeleton_regression(square, Samples(points, points.parameters[:, [1]] * points.parameters[:, [3]]))
    >>> sorted(result.faces)
    [(0, 2), (0, 3), (1, 2), (1, 3)]
    """
    if not _single_skeleton(space):
        return _fit_assembled_skeleton(space, samples, solver)
    cell = _skeleton_cell(space)
    if not isinstance(samples, Samples):
        raise TypeError("samples must be Samples.")
    space.validate(samples.points)
    support = samples.points.parameters.cpu() != 0
    faces = _faces_with_interior_controls(cell.dimensions, cell.degree)
    # Group rows by their distinct supports in one vectorized pass; row order within a face is kept.
    supports, inverse = torch.unique(support, dim=0, return_inverse=True)
    face_of_support = {tuple(row.nonzero().flatten().tolist()): k for k, row in enumerate(supports)}
    empty = torch.zeros(0, dtype=torch.long)
    rows = {
        face: (inverse == face_of_support[face]).nonzero().flatten() if face in face_of_support else empty for face in faces
    }
    # Rows on faces without interior controls belong to no face.
    ignored = len(support) - sum(len(r) for r in rows.values())
    reason = f"more than degree={cell.degree[0]} vertices" if len(cell.dimensions) == 1 else "no interior controls"
    ignored_message = f" Ignored {ignored} rows on faces with {reason}." if ignored else ""
    missing = [face for face in faces if not (samples.weights[rows[face]] > 0).any()]
    if missing:
        raise ValueError(f"faces without rows of positive weight: {missing}.")
    control_faces = _control_faces(cell.dimensions, cell.degree)
    coefficients = torch.zeros(space.n_controls, samples.values.shape[1], dtype=torch.float64)
    results: dict[tuple[int, ...] | int, FitResult] = {}
    for face in faces:
        free, _ = _interior_and_boundary(control_faces, face)
        fixed = FixedControls({k: coefficients[k] for k in range(space.n_controls) if k not in free})
        result = fit(space, samples.take(rows[face]), solver=solver, fixed=fixed)
        results[face] = result
        coefficients[free] = result.model.coefficients.detach()[free]
        if not result.success:
            message = f"Face {face}: {result.message}{ignored_message}"
            return SkeletonRegressionResult(BezierModel(space, coefficients), results, False, result.status, message)
    # Every face uses the same solver, so they share its success status ("solved" or "converged").
    status = results[faces[-1]].status
    message = f"Every face least-squares problem was solved.{ignored_message}"
    return SkeletonRegressionResult(BezierModel(space, coefficients), results, True, status, message)


def _fit_assembled_skeleton(space: BezierSpace, samples: Samples, solver: LeastSquaresSolver) -> SkeletonRegressionResult:
    """Estimate each nondegenerate cell's proper controls once, sharing all boundary estimates."""
    stages = _skeleton_stages(space)
    if not isinstance(samples, Samples):
        raise TypeError("samples must be Samples.")
    _, groups = _skeleton_observations(space, samples.points)
    empty = torch.empty(0, dtype=torch.long)
    missing = [cid for cid, _ in stages if not (samples.weights[groups.get(cid, empty)] > 0).any()]
    if missing:
        raise ValueError(f"cells without rows of positive weight: {missing}.")
    ignored = len(samples) - sum(len(groups[cid]) for cid, _ in stages)
    suffix = f" Ignored {ignored} rows on cells with no interior controls." if ignored else ""
    coefficients = torch.zeros(space.n_controls, samples.values.shape[1], dtype=torch.float64)
    results: dict[tuple[int, ...] | int, FitResult] = {}
    for cid, proper in stages:
        rows = groups[cid]
        # Canonicalization determines ownership. Preserve the validated original design:
        # dropping a nearly unit collapsed factor would otherwise change its basis values.
        selected = samples.take(rows)
        fixed = FixedControls({k: coefficients[k] for k in range(space.n_controls) if k not in proper})
        result = fit(space, selected, solver=solver, fixed=fixed)
        results[cid] = result
        coefficients[list(proper)] = result.model.coefficients.detach()[list(proper)]
        if not result.success:
            message = f"Cell {cid}: {result.message}{suffix}"
            return SkeletonRegressionResult(BezierModel(space, coefficients), results, False, result.status, message)
    message = f"Every cell least-squares problem was solved.{suffix}"
    return SkeletonRegressionResult(BezierModel(space, coefficients), results, True, result.status, message)
