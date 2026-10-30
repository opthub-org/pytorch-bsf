"""Value-only Bézier simplex and simploid fitting (Kobayashi et al., AAAI 2019).

Algorithm 2 alternates constrained Newton parameter updates and linear
least squares. Algorithm 3 fits faces in increasing support size, preserving
the controls already estimated on their boundaries.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from torch_bsf._kernels.combinatorics import _indices, _multinomial, _positions, _recursive_basis, _simplex_indices
from torch_bsf._numeric import _finite_mean
from torch_bsf._skeleton import _blocks, _control_faces, _faces_with_interior_controls
from torch_bsf.core import BezierModel, BezierSpace, Cell, FixedControls, Samples, _integer
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

    def require_success(self) -> "PointCloudFitResult":
        """Raise with the termination message if convergence was not established.

        Returns
        -------
        PointCloudFitResult
            This result when convergence was established.
        """
        if not self.success:
            raise RuntimeError(self.message)
        return self


@dataclass(frozen=True)
class SkeletonFitResult:
    """Assembled model and local results keyed by sorted zero-based vertex IDs.

    Each face result contains local barycentric coordinates for that face's
    observations. On a simploid, factors in which the face is a single vertex
    have no local coordinates, so the columns are the face's vertices in the
    other factors, in key order. Previously fitted boundary controls remain
    exactly fixed.
    """

    model: BezierModel
    faces: dict[tuple[int, ...], PointCloudFitResult]
    success: bool
    status: str
    message: str

    def require_success(self) -> "SkeletonFitResult":
        """Raise if any face failed to converge.

        Returns
        -------
        SkeletonFitResult
            This result when every required face fit converged.
        """
        if not self.success:
            raise RuntimeError(self.message)
        return self


def _cell(space: BezierSpace) -> Cell:
    """The cell of a single simplex or simploid whose factors all have positive degree."""
    if not isinstance(space, BezierSpace):
        raise TypeError("space must be a BezierSpace.")
    if space.n_cells != 1:
        raise ValueError("point-cloud fitting requires a single simplex or simploid.")
    cell = space.cells[0]
    if any(d < 1 for d in cell.degree):
        raise ValueError("every positive-dimensional factor requires degree >= 1.")
    return cell


def _values(values: Any) -> torch.Tensor:
    raw = torch.as_tensor(values)
    if raw.is_complex():
        raise ValueError("values must be real.")
    # Preserve direct float64 parsing for Python inputs, but transfer tensors
    # before casting because MPS cannot represent float64.
    y = raw.detach().cpu().double() if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64).cpu()
    y = y.clone()
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


def _project_factors(t: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Euclidean projection onto a product of probability simplices, one per factor block."""
    if len(dimensions) == 1:
        return _project(t)
    return torch.cat([_project(t[:, block.start : block.stop]) for block in _blocks(dimensions)], 1)


def _barycenters(n: int, dimensions: tuple[int, ...]) -> torch.Tensor:
    """``n`` copies of the barycenter of every factor."""
    return torch.tensor([1 / (a + 1) for a in dimensions for _ in range(a + 1)], dtype=torch.float64).repeat(n, 1)


def _initial(space: BezierSpace, y: torch.Tensor, initial: BezierModel | None) -> BezierModel:
    cell = _cell(space)
    if initial is not None:
        if not isinstance(initial, BezierModel) or not initial.space.compatible(space) or initial.n_outputs != y.shape[1]:
            raise ValueError("initial must be a compatible BezierModel with the observed output dimension.")
        return BezierModel(space, initial.coefficients.detach().to(device="cpu").to(dtype=torch.float64))
    if not cell.dimensions:
        return BezierModel(space, _finite_mean(y, 0, keepdim=True))
    if len(cell.dimensions) > 1:
        raise ValueError("automatic initialization requires a single simplex; supply initial for a simploid.")
    if y.shape[1] != cell.width:
        raise ValueError("automatic initialization requires one output per vertex; supply initial for other embeddings.")
    vertices = y[y.argmin(0)]
    powers = torch.tensor([a[0] for a in _indices(cell.dimensions, cell.degree)], dtype=torch.float64)
    return BezierModel(space, (powers / cell.degree[0]) @ vertices)


class _Polynomial:
    """Analytic Bernstein derivatives, well-defined even at zero coordinates.

    A product basis function is a product of multinomial coefficients times a monomial in the concatenated
    coordinates, so ``powers`` holds the concatenated multi-indices and ``degrees`` the degree of each factor.
    """

    def __init__(self, model: BezierModel):
        cell = _cell(model.space)
        self.coefficients = model.coefficients.detach()
        self.dimensions, self.degrees = cell.dimensions, cell.degree
        indices = _indices(cell.dimensions, cell.degree)
        self.powers = torch.tensor([sum(a, ()) for a in indices], dtype=torch.long).reshape(len(indices), cell.width)
        self.high = max(cell.degree, default=0) >= 16
        self.factors = torch.tensor(
            [1.0] * len(self.powers)
            if self.high
            else [math.prod(_multinomial(d, factor) for d, factor in zip(cell.degree, a, strict=True)) for a in indices],
            dtype=torch.float64,
        )

    def derivative(self, t: torch.Tensor, axes: tuple[int, ...] = ()) -> torch.Tensor:
        if self.high:
            return self._high_derivative(t, axes)
        powers = self.powers.clone()
        factors = self.factors.clone()
        for axis in axes:
            factors *= powers[:, axis].clamp_min(0)
            powers[:, axis] -= 1
        basis = t[:, None, :].pow(powers.clamp_min(0)[None, :, :]).prod(2) * factors
        return basis @ self.coefficients

    def _high_derivative(self, t: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        # The derivative of a product basis function is the product of the derivatives of its factors.
        powers = self.powers.clone()
        for axis in axes:
            powers[:, axis] -= 1
        derivative = torch.ones((len(t), len(powers)), dtype=t.dtype)
        for block, full in zip(_blocks(self.dimensions), self.degrees, strict=True):
            degree = full - sum(axis in block for axis in axes)
            if degree < 0:
                return torch.zeros((len(t), self.coefficients.shape[1]), dtype=t.dtype)
            positions = {alpha: i + 1 for i, alpha in enumerate(_simplex_indices(len(block), degree))}
            indices = torch.tensor([positions.get(tuple(row), 0) for row in powers[:, block.start : block.stop].tolist()])
            basis = _recursive_basis(t[:, block.start : block.stop], degree)
            padded = torch.cat((torch.zeros_like(basis[:, :1]), basis), 1)
            derivative = derivative * (padded[:, indices] * math.prod(range(degree + 1, full + 1)))
        return derivative @ self.coefficients

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
        candidate = _project_factors(t + step * direction, poly.dimensions)
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
    # Eliminate the final barycentric coordinate of each factor, enforcing a unit sum per factor.
    tangent = torch.block_diag(*(torch.cat((torch.eye(a), -torch.ones(1, a))).to(torch.float64) for a in poly.dimensions))
    for _ in range(max_iter):
        _, gradient, hessian = poly.geometry(t, y)
        if not torch.isfinite(gradient).all() or not torch.isfinite(hessian).all():
            raise ArithmeticError("Non-finite parameter derivatives.")
        if (_project_factors(t - gradient, poly.dimensions) - t).abs().max().item() <= tolerance:
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
    return t, bool((_project_factors(t - gradient, poly.dimensions) - t).abs().max() <= tolerance)


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
    projected = _project_factors(t - gradient, poly.dimensions)
    return bool(torch.isfinite(gradient).all() and (projected - t).abs().max() <= tolerance)


def _apply_fixed(model: BezierModel, y: torch.Tensor, fixed: FixedControls | None) -> None:
    if fixed is None:
        return
    if not isinstance(fixed, FixedControls):
        raise TypeError("fixed must be FixedControls.")
    # Reuse the supervised problem's shape and index validation.
    from torch_bsf._problem import LeastSquaresProblem
    from torch_bsf.core import QuadraticPenalty

    space = model.space
    coordinates = _barycenters(len(y), space.cells[0].dimensions)
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
    No input model or observation is modified. Exhausting either iteration
    budget does not certify convergence.

    A simploid, a product of simplices, is fitted the same way: each row's
    coordinates stay on every factor's simplex, and Newton steps keep each
    factor's coordinate sum. Automatic initialization needs a single simplex,
    so supply ``initial`` for a simploid. Assembled spaces are not supported.

    ``tolerance`` bounds absolute change in root mean squared distance;
    ``parameter_tolerance`` bounds the projected gradient residual. The
    stopping rule uses the absolute change, correcting the signed-change
    expression printed in the paper. The objective uses the original units;
    scale observations explicitly when outputs have incompatible units.

    Parameters
    ----------
    space : BezierSpace
        Single simplex or simploid specifying the dimensions and polynomial degrees.
    values : array-like
        Finite observations of shape ``(n_samples, n_outputs)`` in the original output units.
    initial : BezierModel, optional
        Initial model with compatible space and output dimension; otherwise infer minimization
        vertices, which requires a single simplex.
    fixed : FixedControls, optional
        Coefficient rows kept exact throughout fitting, including initialization.
    max_iter : int, optional
        Positive budget of alternating parameter and control updates.
    parameter_max_iter : int, optional
        Positive iteration budget for each parameter-coordinate solve.
    tolerance : float, optional
        Finite nonnegative tolerance on the absolute change in root mean squared distance.
    parameter_tolerance : float, optional
        Finite nonnegative tolerance on the projected parameter-gradient residual.

    Returns
    -------
    PointCloudFitResult
        Model, inferred barycentric coordinates, objective history and convergence evidence.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace` or ``fixed`` is not
        :class:`~torch_bsf.FixedControls`.
    ValueError
        If ``space`` has several cells or a positive-dimensional factor of degree
        zero, ``values`` or a setting is invalid, ``initial`` is incompatible, or
        automatic initialization is impossible.

    Examples
    --------
    Recover a bilinear map of a square, a product of two segments, from values
    alone.

    >>> import torch
    >>> from torch_bsf import BezierModel, BezierSpace, fit_point_cloud
    >>> from torch_bsf.sampling import grid
    >>> space = BezierSpace.simploid((1, 1), degree=1)
    >>> truth = BezierModel(space, torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    >>> values = truth(grid(space, 4)).detach()
    >>> result = fit_point_cloud(space, values, initial=truth).require_success()
    >>> result.objective_history[-1] < 1e-12
    True
    """
    cell = _cell(space)
    _settings(max_iter, parameter_max_iter, tolerance, parameter_tolerance)
    y = _values(values)
    model = _initial(space, y, initial)
    _apply_fixed(model, y, fixed)
    t = _barycenters(len(y), cell.dimensions)
    control_result = None
    initial_residual = model(t).detach() - y
    history = [float(initial_residual.square().sum(1).mean())]
    if not math.isfinite(history[0]):
        message = "Initial point-cloud objective overflowed."
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
        return PointCloudFitResult(model, t, tuple(history), False, "numerical_failure", message, 0, False, None)
    if not cell.dimensions and (initial_residual == 0).all():
        return PointCloudFitResult(model, t, tuple(history), True, "converged", "Exact initial fit.", 0, True, None)
    success, status, message, parameter_converged = False, "iteration_limit", "Alternating iteration limit reached.", False
    for _ in range(max_iter):
        try:
            if cell.dimensions:
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


def _lift(
    index: tuple[tuple[int, ...], ...], parts: list[list[int]], dimensions: Sequence[int], degree: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """The global multi-index of a control of a face, given its multi-index on the face's local space.

    ``parts`` holds the face's vertex positions within each factor. ``index`` has one entry per factor with more
    than one vertex; a factor at one vertex puts its whole degree on that vertex.
    """
    local = iter(index)
    lifted = []
    for part, dimension, d in zip(parts, dimensions, degree, strict=True):
        factor = [0] * (dimension + 1)
        for v, power in zip(part, next(local) if len(part) > 1 else (d,), strict=True):
            factor[v] = power
        lifted.append(tuple(factor))
    return tuple(lifted)


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

    On a simplex, faces are sorted tuples of zero-based vertex IDs, and only
    support sizes up to ``min(degree, number_of_vertices)`` are needed. By default each face's
    samples are the nondominated points for its objective subset (all outputs
    are minimization objectives). This requires a single simplex and one
    output per vertex.
    ``face_values`` instead supplies every required face's value-only samples,
    allowing arbitrary embeddings with ``initial``. Full original output
    vectors must be retained on every face. No face memberships or coordinates
    are inferred from an arbitrary embedding.

    Vertex controls are sample means. Higher faces use ``fit_point_cloud``
    with all lower-face controls fixed exactly. Inspect ``faces`` for local
    stopping evidence; failure on a boundary stops assembly and returns the
    finite partial model with ``success=False``.

    A face of a simploid, a product of simplices, is a product of one face per
    factor, keyed by the sorted global coordinate indices of its vertices; the
    faces with at most ``degree`` vertices in every factor are needed. Each face
    is fitted on the product of its factors' faces. A simploid requires both
    ``face_values`` and ``initial``.

    Parameters
    ----------
    space : BezierSpace
        Single simplex or simploid specifying the dimensions and polynomial degrees.
    values : array-like
        Finite observations of shape ``(n_samples, n_outputs)`` in the original output units.
    face_values : mapping, optional
        Samples for every required sorted vertex tuple, retaining all output columns; otherwise extract nondominated faces.
    initial : BezierModel, optional
        Compatible initial model, required for embeddings that cannot use automatic minimization vertices.
    max_iter : int, optional
        Positive budget of alternating parameter and control updates.
    parameter_max_iter : int, optional
        Positive iteration budget for each parameter-coordinate solve.
    tolerance : float, optional
        Finite nonnegative tolerance on the absolute change in root mean squared distance.
    parameter_tolerance : float, optional
        Finite nonnegative tolerance on the projected parameter-gradient residual.

    Returns
    -------
    SkeletonFitResult
        Assembled model, per-face results and an explicit success or partial-failure state.

    Raises
    ------
    TypeError
        If ``space`` is not a :class:`~torch_bsf.BezierSpace`.
    ValueError
        If ``space`` has several cells or a positive-dimensional factor of degree
        zero, ``values``, ``face_values`` or a setting is invalid, ``initial`` is
        incompatible, or automatic initialization or face extraction is impossible.

    Examples
    --------
    Fit a bilinear square from values at its four corners.

    >>> import torch
    >>> from torch_bsf import BezierModel, BezierSpace, fit_skeleton
    >>> space = BezierSpace.simploid((1, 1), degree=1)
    >>> truth = BezierModel(space, torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    >>> corners = {(0, 2): [[0.0, 0.0]], (0, 3): [[1.0, 0.0]], (1, 2): [[0.0, 1.0]], (1, 3): [[1.0, 1.0]]}
    >>> values = torch.tensor([v[0] for v in corners.values()])
    >>> result = fit_skeleton(space, values, face_values=corners, initial=truth).require_success()
    >>> sorted(result.faces), torch.equal(result.model.coefficients, truth.coefficients.double())
    ([(0, 2), (0, 3), (1, 2), (1, 3)], True)
    """
    cell = _cell(space)
    _settings(max_iter, parameter_max_iter, tolerance, parameter_tolerance)
    y = _values(values)
    model = _initial(space, y, initial)
    required = _faces_with_interior_controls(cell.dimensions, cell.degree) if cell.dimensions else [(0,)]
    automatic = len(cell.dimensions) == 1 and y.shape[1] == cell.width
    if face_values is None and cell.dimensions and not automatic:
        raise ValueError("automatic face extraction requires a single simplex and one output per vertex; supply face_values.")
    if face_values is not None and set(face_values) != set(required):
        raise ValueError(f"face_values must contain exactly these sorted vertex tuples: {required}.")
    data = {
        face: _values(face_values[face]) if face_values is not None else _nondominated(y, face) if cell.dimensions else y
        for face in required
    }
    if any(v.shape[1] != y.shape[1] for v in data.values()):
        raise ValueError("every face must retain the full observed output dimension.")
    positions = _positions(cell.dimensions, cell.degree)
    control_faces = _control_faces(cell.dimensions, cell.degree)
    blocks = _blocks(cell.dimensions)
    faces: dict[tuple[int, ...], PointCloudFitResult] = {}
    for face in required:
        # Each factor's face as vertex positions within that factor; factors at one vertex leave the local space.
        parts = [[i - block.start for i in face if i in block] for block in blocks]
        wide = [j for j, part in enumerate(parts) if len(part) > 1]
        local_space = BezierSpace.simploid([len(parts[j]) - 1 for j in wide], degree=[cell.degree[j] for j in wide])
        rows = []
        boundary = {}
        for k, index in enumerate(_indices(local_space.cells[0].dimensions, local_space.cells[0].degree)):
            row = positions[_lift(index, parts, cell.dimensions, cell.degree)]
            rows.append(row)
            if wide and control_faces[row] != face:
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
