"""Value-only Bézier simplex fitting (Kobayashi et al., AAAI 2019).

Algorithm 2 alternates constrained Newton parameter updates and linear
least squares. Algorithm 3 fits faces in increasing support size, preserving
the controls already estimated on their boundaries.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierModel, BezierSpace, FixedControls, Samples, _integer
from torch_bsf.fitting import fit
from torch_bsf.results import ConvergenceWarning, FitResult


@dataclass(frozen=True)
class PointCloudFitResult:
    """A fitted surface and inferred barycentric rows in observation order.

    ``objective_history`` contains the initial mean squared Euclidean distance
    and each accepted alternating update (no division by output dimension).
    Success describes local numerical convergence, not global optimality or
    identification of a unique parameterization. All tensors are CPU float64.
    """

    model: BezierModel
    parameters: torch.Tensor
    objective_history: tuple[float, ...]
    success: bool
    status: str
    message: str
    n_iterations: int
    parameter_converged: bool
    control_result: FitResult | None

    def require_success(self) -> PointCloudFitResult:
        """Raise with the termination message if convergence was not established."""
        if not self.success:
            raise RuntimeError(self.message)
        return self


@dataclass(frozen=True)
class SkeletonFitResult:
    """Assembled model and local results keyed by sorted zero-based vertex IDs.

    Each face result contains local barycentric coordinates for that face's
    observations. Previously fitted boundary controls remain exactly fixed.
    """

    model: BezierModel
    faces: dict[tuple[int, ...], PointCloudFitResult]
    success: bool
    status: str
    message: str

    def require_success(self) -> SkeletonFitResult:
        """Raise if any face failed to converge."""
        if not self.success:
            raise RuntimeError(self.message)
        return self


def _simplex(space: BezierSpace) -> tuple[int, int]:
    if not isinstance(space, BezierSpace):
        raise TypeError("space must be a BezierSpace.")
    if space.n_cells != 1 or len(space.cells[0].dimensions) > 1:
        raise ValueError("point-cloud fitting requires a single simplex.")
    cell = space.cells[0]
    dimension = cell.dimensions[0] if cell.dimensions else 0
    degree = cell.degree[0] if cell.degree else 0
    if dimension and degree < 1:
        raise ValueError("a positive-dimensional simplex requires degree >= 1.")
    return dimension + 1, degree


def _values(values: Any) -> torch.Tensor:
    raw = torch.as_tensor(values)
    if raw.is_complex():
        raise ValueError("values must be real.")
    y = torch.as_tensor(values, dtype=torch.float64).detach().cpu().clone()
    if y.ndim != 2 or min(y.shape) < 1 or not torch.isfinite(y).all():
        raise ValueError("values must be finite with shape (n_samples, n_outputs), both positive.")
    return y


def _project(t: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex, including its boundary."""
    ordered = t.sort(dim=1, descending=True).values
    offset = ordered.cumsum(1) - 1
    count = torch.arange(1, t.shape[1] + 1, dtype=t.dtype)
    support = (ordered > offset / count).sum(1).clamp_min(1)
    threshold = offset.gather(1, support[:, None] - 1) / support[:, None]
    return (t - threshold).clamp_min(0)


def _initial(space: BezierSpace, y: torch.Tensor, initial: BezierModel | None) -> BezierModel:
    width, degree = _simplex(space)
    if initial is not None:
        if not isinstance(initial, BezierModel) or not initial.space.compatible(space) or initial.n_outputs != y.shape[1]:
            raise ValueError("initial must be a compatible BezierModel with the observed output dimension.")
        return BezierModel(space, initial.coefficients.detach().to(device="cpu", dtype=torch.float64))
    if width == 1:
        return BezierModel(space, y.mean(0, keepdim=True))
    if y.shape[1] != width:
        raise ValueError("automatic initialization requires one output per vertex; supply initial for other embeddings.")
    vertices = y[y.argmin(0)]
    powers = torch.tensor([a[0] for a in _indices(space.cells[0].dimensions, (degree,))], dtype=torch.float64)
    return BezierModel(space, (powers / degree) @ vertices)


class _Polynomial:
    """Analytic Bernstein derivatives, well-defined even at zero coordinates."""

    def __init__(self, model: BezierModel):
        width, degree = _simplex(model.space)
        self.coefficients = model.coefficients.detach()
        self.powers = torch.tensor([a[0] for a in _indices((width - 1,), (degree,))], dtype=torch.long)
        self.factors = torch.tensor(
            [math.factorial(degree) / math.prod(math.factorial(int(v)) for v in a) for a in self.powers],
            dtype=torch.float64,
        )

    def derivative(self, t: torch.Tensor, axes: tuple[int, ...] = ()) -> torch.Tensor:
        powers = self.powers.clone()
        factors = self.factors.clone()
        for axis in axes:
            factors *= powers[:, axis].clamp_min(0)
            powers[:, axis] -= 1
        basis = t[:, None, :].pow(powers.clamp_min(0)[None, :, :]).prod(2) * factors
        return basis @ self.coefficients

    def geometry(self, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.derivative(t) - y
        width = t.shape[1]
        jacobian = torch.stack([self.derivative(t, (i,)) for i in range(width)], dim=2)
        gradient = torch.einsum("no,noi->ni", residual, jacobian)
        hessian = torch.einsum("noi,noj->nij", jacobian, jacobian)
        for i in range(width):
            for j in range(i, width):
                term = (residual * self.derivative(t, (i, j))).sum(1)
                hessian[:, i, j] += term
                if i != j:
                    hessian[:, j, i] += term
        return residual.square().sum(1), gradient, hessian


def _line_search(poly: _Polynomial, t: torch.Tensor, direction: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    best = t.clone()
    error = (poly.derivative(t) - y).square().sum(1)
    pending = torch.ones(len(t), dtype=torch.bool)
    step = 1.0
    for _ in range(25):
        candidate = _project(t + step * direction)
        candidate_error = (poly.derivative(candidate) - y).square().sum(1)
        improved = pending & torch.isfinite(candidate_error) & (candidate_error < error)
        best[improved], error[improved] = candidate[improved], candidate_error[improved]
        pending &= ~improved & ((candidate - t).abs().amax(1) > 1e-15)
        if not pending.any():
            break
        step *= 0.5
    return best


def _parameters(
    model: BezierModel, y: torch.Tensor, t: torch.Tensor, max_iter: int, tolerance: float
) -> tuple[torch.Tensor, bool]:
    poly = _Polynomial(model)
    width = t.shape[1]
    # Eliminate the final barycentric coordinate, enforcing sum(t) == 1.
    tangent = torch.cat((torch.eye(width - 1), -torch.ones(1, width - 1))).to(torch.float64)
    for _ in range(max_iter):
        _, gradient, hessian = poly.geometry(t, y)
        if not torch.isfinite(gradient).all() or not torch.isfinite(hessian).all():
            raise ArithmeticError("Non-finite parameter derivatives.")
        if (_project(t - gradient) - t).abs().max().item() <= tolerance:
            return t, True
        reduced = tangent.T @ hessian @ tangent
        rhs = gradient @ tangent
        delta = -(torch.linalg.pinv(reduced, hermitian=True) @ rhs[:, :, None]).squeeze(2) @ tangent.T
        newton = _line_search(poly, t, delta, y)
        # Indefinite/singular Hessians and active boundaries need a descent fallback.
        scale = hessian.abs().sum(2).amax(1).clamp_min(1e-12)
        descent = _line_search(poly, t, -gradient / scale[:, None], y)
        use_newton = (poly.derivative(newton) - y).square().sum(1) < (poly.derivative(descent) - y).square().sum(1)
        t = torch.where(use_newton[:, None], newton, descent)
    _, gradient, _ = poly.geometry(t, y)
    return t, bool((_project(t - gradient) - t).abs().max() <= tolerance)


def _settings(max_iter: int, parameter_max_iter: int, tolerance: float, parameter_tolerance: float) -> None:
    _integer(max_iter, "max_iter", 1)
    _integer(parameter_max_iter, "parameter_max_iter", 1)
    for name, value in (("tolerance", tolerance), ("parameter_tolerance", parameter_tolerance)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")


def _parameter_stationary(model: BezierModel, y: torch.Tensor, t: torch.Tensor, tolerance: float) -> bool:
    if not t.shape[1]:
        return True
    poly = _Polynomial(model)
    residual = poly.derivative(t) - y
    gradient = torch.stack([(residual * poly.derivative(t, (i,))).sum(1) for i in range(t.shape[1])], 1)
    return bool(torch.isfinite(gradient).all() and (_project(t - gradient) - t).abs().max() <= tolerance)


def _apply_fixed(model: BezierModel, y: torch.Tensor, fixed: FixedControls | None) -> None:
    if fixed is None:
        return
    if not isinstance(fixed, FixedControls):
        raise TypeError("fixed must be FixedControls.")
    # Reuse the supervised problem's shape and index validation.
    from torch_bsf._problem import LeastSquaresProblem
    from torch_bsf.core import QuadraticPenalty

    space = model.space
    coordinates = torch.full((len(y), space.width), 1 / max(space.width, 1), dtype=torch.float64)
    problem = LeastSquaresProblem(space, Samples(coordinates, y), QuadraticPenalty(), fixed)
    with torch.no_grad():
        model.coefficients[~problem.free] = problem.fixed[~problem.free]


def fit_point_cloud(
    space: BezierSpace,
    values: Any,
    *,
    initial: BezierModel | None = None,
    fixed: FixedControls | None = None,
    max_iter: int = 100,
    parameter_max_iter: int = 100,
    tolerance: float = 1e-5,
    parameter_tolerance: float = 1e-5,
) -> PointCloudFitResult:
    """Fit values without supplied domain coordinates (paper Algorithm 2).

    Minimize the mean squared Euclidean distance by alternating parameter and
    control updates. Parameters use analytic first/second derivatives with
    projected Newton steps, backtracking and a gradient fallback. Controls use
    CPU float64 SVD least squares, never normal equations. ``fixed`` specifies
    exact controls throughout, including before the first parameter update.

    Automatic initialization uses the observed minimum of each output as a
    vertex and the affine grid between those vertices, as in the paper. It
    assumes minimization objectives and one output per vertex. Supply an
    ``initial`` model for arbitrary embeddings or a different orientation.
    No input model or observation is modified. Only single simplices are
    supported. Exhausting either iteration budget does not certify convergence.

    ``tolerance`` bounds absolute change in root mean squared distance;
    ``parameter_tolerance`` bounds the projected gradient residual. The
    stopping rule uses the absolute change, correcting the signed-change
    expression printed in the paper. The objective uses the original units;
    scale observations explicitly when outputs have incompatible units.
    """
    width, _ = _simplex(space)
    _settings(max_iter, parameter_max_iter, tolerance, parameter_tolerance)
    y = _values(values)
    model = _initial(space, y, initial)
    _apply_fixed(model, y, fixed)
    t = torch.full((len(y), space.width), 1 / width, dtype=torch.float64)
    control_result = None
    history = [float((model(t).detach() - y).square().sum(1).mean())]
    if not math.isfinite(history[0]):
        message = "Initial point-cloud objective overflowed."
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
        return PointCloudFitResult(model, t, tuple(history), False, "numerical_failure", message, 0, False, None)
    success, status, message, parameter_converged = False, "iteration_limit", "Alternating iteration limit reached.", False
    for _ in range(max_iter):
        try:
            if width > 1:
                # Solve the inner problem more tightly than the final-model check:
                # the subsequent control update changes the parameter gradient.
                t, parameter_converged = _parameters(model, y, t, parameter_max_iter, parameter_tolerance * 0.1)
            else:
                parameter_converged = True
            solved = fit(space, Samples(t, y), fixed=fixed)
        except (torch.linalg.LinAlgError, ArithmeticError) as error:
            status, message = "numerical_failure", str(error)
            break
        control_result = solved
        if not solved.success:
            status, message = "control_failure", solved.message
            break
        objective = float((solved.model(t).detach() - y).square().sum(1).mean())
        if not math.isfinite(objective):
            status, message = "numerical_failure", "Point-cloud objective overflowed."
            break
        model = solved.model
        history.append(objective)
        parameter_converged = _parameter_stationary(model, y, t, parameter_tolerance)
        change = abs(math.sqrt(history[-2]) - math.sqrt(objective))
        if change <= tolerance and parameter_converged:
            success, status, message = True, "converged", "Alternating distance and parameter tolerances satisfied."
            break
    result = PointCloudFitResult(
        model, t, tuple(history), success, status, message, len(history) - 1, parameter_converged, control_result
    )
    if not success:
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    return result


def _nondominated(y: torch.Tensor, face: tuple[int, ...]) -> torch.Tensor:
    selected = y[:, face]
    keep = torch.ones(len(y), dtype=torch.bool)
    # Linear auxiliary memory rather than an N x N x M dominance tensor.
    for i, point in enumerate(selected):
        keep[i] = not bool(((selected <= point).all(1) & (selected < point).any(1)).any())
    return y[keep]


def fit_skeleton(
    space: BezierSpace,
    values: Any,
    *,
    face_values: Mapping[tuple[int, ...], Any] | None = None,
    initial: BezierModel | None = None,
    max_iter: int = 100,
    parameter_max_iter: int = 100,
    tolerance: float = 1e-5,
    parameter_tolerance: float = 1e-5,
) -> SkeletonFitResult:
    """Fit faces inductively while fixing their boundaries (paper Algorithm 3).

    Faces are sorted tuples of zero-based vertex IDs. Only support sizes up
    to ``min(degree, number_of_vertices)`` are needed. By default each face's
    samples are the nondominated points for its objective subset (all outputs
    are minimization objectives). This requires one output per vertex.
    ``face_values`` instead supplies every required face's value-only samples,
    allowing arbitrary embeddings with ``initial``. Full original output
    vectors must be retained on every face. No face memberships or coordinates
    are inferred from an arbitrary embedding.

    Vertex controls are sample means. Higher faces use ``fit_point_cloud``
    with all lower-face controls fixed exactly. Inspect ``faces`` for local
    stopping evidence; failure on a boundary stops assembly and returns the
    finite partial model with ``success=False``.
    """
    width, degree = _simplex(space)
    _settings(max_iter, parameter_max_iter, tolerance, parameter_tolerance)
    y = _values(values)
    model = _initial(space, y, initial)
    required = [face for size in range(1, min(degree, width) + 1) for face in combinations(range(width), size)]
    if width == 1:
        required = [(0,)]
    if face_values is None and width > 1 and y.shape[1] != width:
        raise ValueError("automatic face extraction requires one output per vertex; supply face_values.")
    if face_values is not None and set(face_values) != set(required):
        raise ValueError(f"face_values must contain exactly these sorted vertex tuples: {required}.")
    data = {
        face: _values(face_values[face]) if face_values is not None else y if width == 1 else _nondominated(y, face)
        for face in required
    }
    if any(v.shape[1] != y.shape[1] for v in data.values()):
        raise ValueError("every face must retain the full observed output dimension.")
    global_indices = _indices(space.cells[0].dimensions, space.cells[0].degree)
    faces: dict[tuple[int, ...], PointCloudFitResult] = {}
    for face in required:
        local_space = BezierSpace.simplex(len(face) - 1, degree=degree)
        local_indices = _indices(local_space.cells[0].dimensions, local_space.cells[0].degree)
        rows = []
        boundary = {}
        for k, index in enumerate(local_indices):
            powers = index[0] if index else (degree,)
            full = tuple(powers[face.index(i)] if i in face else 0 for i in range(width))
            row = global_indices.index((full,)) if width > 1 else 0
            rows.append(row)
            if len(face) > 1 and any(p == 0 for p in powers):
                boundary[k] = model.coefficients[row].detach()
        result = fit_point_cloud(
            local_space,
            data[face],
            initial=BezierModel(local_space, model.coefficients[rows]),
            fixed=FixedControls(boundary),
            max_iter=max_iter,
            parameter_max_iter=parameter_max_iter,
            tolerance=tolerance,
            parameter_tolerance=parameter_tolerance,
        )
        faces[face] = result
        with torch.no_grad():
            model.coefficients[rows] = result.model.coefficients
        if not result.success:
            return SkeletonFitResult(model, faces, False, result.status, f"Face {face}: {result.message}")
    return SkeletonFitResult(model, faces, True, "converged", "Every required face converged.")
