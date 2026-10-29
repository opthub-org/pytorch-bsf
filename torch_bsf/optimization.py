"""Objective minimization, separate from supervised least-squares regression."""

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import torch

from torch_bsf._kernels.sgd import _simploid_gram
from torch_bsf._numeric import _finite_mean
from torch_bsf._validation import _choice
from torch_bsf.core import BezierModel, BezierSpace, FixedControls, Points, QuadraticPenalty, _integer
from torch_bsf.geometry import Simploid
from torch_bsf.results import ConvergenceWarning
from torch_bsf.sampling import _generator, _probabilities, sample

Loss = Callable[[torch.Tensor, Points], torch.Tensor]


@dataclass(frozen=True)
class PreconditionedSGD:
    """Domain-metric SGD with a fixed sample set and uniform random minibatches."""

    max_iter: int = 100
    step_size: float = 0.05
    batch_size: int | None = None
    preconditioner: Literal["gram", "none"] = "gram"
    step_decay: Literal["constant", "sqrt"] = "constant"
    gradient_tolerance: float | None = None

    def __post_init__(self) -> None:
        _integer(self.max_iter, "max_iter", 1)
        if self.batch_size is not None:
            _integer(self.batch_size, "batch_size", 1)
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive.")
        _choice(self.preconditioner, "preconditioner", ("gram", "none"), message="unknown preconditioner or step_decay.")
        _choice(self.step_decay, "step_decay", ("constant", "sqrt"), message="unknown preconditioner or step_decay.")
        if self.gradient_tolerance is not None and (not math.isfinite(self.gradient_tolerance) or self.gradient_tolerance < 0):
            raise ValueError("gradient_tolerance must be finite and nonnegative.")


@dataclass(frozen=True)
class Iteration:
    """Pre-update minibatch loss and gradient norm at a zero-based iteration."""

    iteration: int
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class OptimizationResult:
    """An optimized copy, history and an honest stopping state.

    Budget exhaustion and a user callback are not convergence certificates.
    Only all-fixed or a checked full-sample gradient tolerance yield success.
    """

    model: BezierModel
    history: tuple[Iteration, ...]
    success: bool
    status: str
    message: str
    objective: float
    n_iterations: int
    seed: int | None

    def require_success(self) -> "OptimizationResult":
        """Raise if no convergence criterion was established."""
        if not self.success:
            raise RuntimeError(self.message)
        return self


def gram_matrix(space: BezierSpace, *, cell_weights: Mapping[int | Simploid, float] | None = None) -> torch.Tensor:
    """Compute the exact Bernstein Gram matrix for the sampling distribution."""
    weights = _probabilities(space, cell_weights)
    gram = torch.zeros(space.n_controls, space.n_controls, dtype=torch.float64)
    covered = torch.zeros(space.n_controls, dtype=torch.bool)
    for cid in torch.nonzero(weights).flatten().tolist():
        cell = space.cells[cid]
        columns = torch.tensor(cell.controls)
        local = _simploid_gram(cell.dimensions, cell.degree) * weights[cid]
        gram.index_put_((columns[:, None].expand_as(local), columns[None, :].expand_as(local)), local, accumulate=True)
        covered[columns] = True
    if not covered.all():
        raise ValueError("sampling distribution does not cover every coefficient.")
    return gram


def linear_scalarization(objective: Callable[[torch.Tensor], torch.Tensor]) -> Loss:
    """Turn differentiable objective values into a simplex-weighted loss."""

    def loss(values: torch.Tensor, points: Points) -> torch.Tensor:
        outputs = objective(values)
        t = points.parameters
        if outputs.shape != t.shape or not torch.allclose(
            t.sum(1), torch.ones(len(t), device=t.device, dtype=t.dtype), atol=1e-6
        ):
            raise ValueError("linear scalarization requires simplex coordinates matching the objective dimension.")
        return (outputs * t).sum(1)

    return loss


def _optimization_device(tensor: torch.Tensor) -> torch.device:
    # MPS has no float64 kernels; autograd supports transfers to/from CPU.
    return torch.device("cpu") if tensor.device.type == "mps" else tensor.device


def _wide(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=_optimization_device(tensor)).to(dtype=torch.float64)


def _objective(model: BezierModel, loss: Loss, points: Points, penalty: QuadraticPenalty) -> torch.Tensor:
    values = loss(model(points), points)
    if values.shape != (len(points),) or not torch.isfinite(values).all():
        raise ValueError("loss must return one finite value per sample.")
    # Keep Python-float penalty weights representable even with float32 controls.
    coefficients: torch.Tensor = model.coefficients
    if penalty.ridge or penalty.smoothness:
        coefficients = _wide(coefficients)
    objective = _finite_mean(values, 0)
    if penalty.ridge or penalty.smoothness:
        objective = _wide(objective)
    if penalty.ridge:
        objective = objective + ((coefficients * penalty.ridge) * coefficients).sum()
    if penalty.smoothness:
        edges = model.space.adjacency().to(coefficients.device)
        differences = coefficients[edges[:, 0]] - coefficients[edges[:, 1]]
        if torch.isfinite(differences).all():
            smoothness = ((differences * penalty.smoothness) * differences).sum()
        else:
            # Halve before subtracting: the true difference can exceed float64
            # while its weighted square remains representable.
            half = coefficients / 2
            differences = half[edges[:, 0]] - half[edges[:, 1]]
            # Fold the exact factor into the weight so backward never forms
            # an overflowing 4*difference before multiplying the tiny weight.
            smoothness = ((differences * (penalty.smoothness * 4)) * differences).sum()
        objective = objective + smoothness
    if not torch.isfinite(objective):
        raise ArithmeticError("loss and penalties produced a non-finite objective.")
    return objective


def _fix_controls(model: BezierModel, fixed: FixedControls | None) -> torch.Tensor:
    p = model.coefficients
    free = torch.ones(len(p), dtype=torch.bool, device=p.device)
    with torch.no_grad():
        for row, value in (fixed.values if fixed is not None else {}).items():
            if row >= len(p) or value.shape != (model.n_outputs,):
                raise ValueError("fixed coefficient does not match the model.")
            p[row] = value.to(p)
            free[row] = False
    return free


def _model_for_optimization(model: BezierModel) -> BezierModel:
    fitted = model.copy()
    if fitted.coefficients.dtype in (torch.float16, torch.bfloat16):
        fitted = fitted.to(dtype=torch.float32)
    return fitted


def _gradient(value: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    (gradient,) = torch.autograd.grad(value, coefficients)
    if not torch.isfinite(gradient).all():
        raise ArithmeticError("loss produced a non-finite gradient.")
    return gradient


def _finite_metric(matrix: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(matrix).all():
        raise ArithmeticError("optimization metric matrix is not representable in float64.")
    return matrix


def _scaled_gradient(gradient: torch.Tensor, alpha: float) -> tuple[torch.Tensor, float]:
    wide = _wide(gradient)
    scaled = wide * alpha
    if torch.isfinite(scaled).all() and ((scaled != 0) | (wide == 0)).all():
        return scaled, 1.0
    return wide, alpha


def _updated_values(coefficients: torch.Tensor, alpha: float, direction: torch.Tensor) -> torch.Tensor:
    source, step = _wide(coefficients), _wide(direction)
    updated = source - alpha * step
    invalid = ~torch.isfinite(updated)
    if invalid.any() and torch.isfinite(step).all():
        # Rare overflow cancellation needs a fused product/subtraction. Exact
        # rationals also support Python versions without math.fma.
        selected = torch.nonzero(invalid.flatten()).flatten().cpu().tolist()
        source_cpu, step_cpu = source.detach().cpu().flatten(), step.detach().cpu().flatten()
        exact_alpha = Fraction.from_float(alpha)
        for index in selected:
            exact = Fraction.from_float(source_cpu[index].item()) - exact_alpha * Fraction.from_float(step_cpu[index].item())
            try:
                updated.flatten()[index] = float(exact)
            except OverflowError as error:
                raise ArithmeticError("optimization update produced non-finite coefficients.") from error
    return updated.to(dtype=coefficients.dtype).to(device=coefficients.device)


def _update(coefficients: torch.Tensor, free: torch.Tensor, alpha: float, direction: torch.Tensor) -> None:
    with torch.no_grad():
        updated = _updated_values(coefficients[free], alpha, direction)
        if not torch.isfinite(updated).all():
            raise ArithmeticError("optimization update produced non-finite coefficients.")
        coefficients[free] = updated


def _regularizer(
    topology: torch.Tensor | None, penalty: QuadraticPenalty, alpha: float, size: int, device: torch.device
) -> torch.Tensor:
    # Multiply step-scaled weights by an unweighted topology, whose integer
    # degrees cannot overflow. Topology/device transfer happens only once.
    matrix = (
        torch.zeros(size, size, dtype=torch.float64, device=device)
        if topology is None
        else topology * ((penalty.smoothness * alpha) * 2)
    )
    if penalty.ridge:
        matrix.diagonal().add_((penalty.ridge * alpha) * 2)
    return matrix


def _prepare_regularizer(
    model: BezierModel, penalty: QuadraticPenalty, free: torch.Tensor
) -> Callable[[float], torch.Tensor | None]:
    if not penalty.ridge and not penalty.smoothness:
        return lambda alpha: None
    selected = free.cpu()
    device = _optimization_device(model.coefficients)
    topology = None
    if penalty.smoothness:
        topology = torch.zeros(len(free), len(free), dtype=torch.float64)
        for i, j in model.space.adjacency().tolist():
            topology[i, i] += 1
            topology[j, j] += 1
            topology[i, j] -= 1
            topology[j, i] -= 1
        topology = topology[selected][:, selected].to(device=device)
    size = int(selected.sum().item())
    return lambda alpha: _regularizer(topology, penalty, alpha, size, device)


def _prepare_step(
    model: BezierModel,
    method: PreconditionedSGD,
    penalty: QuadraticPenalty,
    free: torch.Tensor,
    cell_weights: Mapping[int | Simploid, float] | None,
) -> Callable[[torch.Tensor, float], tuple[torch.Tensor, float]]:
    if method.preconditioner == "none" and not penalty.smoothness:
        if not penalty.ridge:
            return lambda gradient, alpha: (gradient, alpha)

        def diagonal_step(gradient: torch.Tensor, alpha: float) -> tuple[torch.Tensor, float]:
            denominator = 1 + (penalty.ridge * alpha) * 2
            if not math.isfinite(denominator):
                raise ArithmeticError("optimization metric matrix is not representable in float64.")
            rhs, update_alpha = _scaled_gradient(gradient, alpha)
            return rhs / denominator, update_alpha

        return diagonal_step
    metric = (
        gram_matrix(model.space, cell_weights=cell_weights)
        if method.preconditioner == "gram"
        else torch.eye(len(free), dtype=torch.float64)
    )
    selected = free.cpu()
    metric = metric[selected][:, selected].to(device=_optimization_device(model.coefficients))
    regularizer_for = _prepare_regularizer(model, penalty, free)
    regularized = bool(penalty.ridge or penalty.smoothness)
    factor: tuple[torch.Tensor, torch.Tensor] | None = None

    def step(gradient: torch.Tensor, alpha: float) -> tuple[torch.Tensor, float]:
        nonlocal factor
        if factor is None or (regularized and method.step_decay == "sqrt"):
            regularizer = regularizer_for(alpha)
            matrix = metric if regularizer is None else metric + regularizer
            factor = torch.linalg.lu_factor(_finite_metric(matrix))
        rhs, update_alpha = _scaled_gradient(gradient, alpha)
        result: torch.Tensor = torch.linalg.lu_solve(factor[0], factor[1], rhs)
        return result, update_alpha

    return step


def minimize(
    model: BezierModel,
    loss: Loss,
    *,
    method: PreconditionedSGD = PreconditionedSGD(),
    points: Points | None = None,
    n_samples: int = 100,
    seed: int | None = None,
    cell_weights: Mapping[int | Simploid, float] | None = None,
    penalty: QuadraticPenalty = QuadraticPenalty(),
    fixed: FixedControls | None = None,
    callback: Callable[[Iteration, BezierModel], bool | None] | None = None,
) -> OptimizationResult:
    """Minimize a differentiable loss on a copied model with domain-aware SGD.

    Parameters
    ----------
    model
        Initial model; never mutated.
    loss
        ``loss(predicted_values, points)`` returning one loss per sample.
    method
        Typed preconditioned SGD settings, independent of Lightning. Float16
        and bfloat16 initial models are copied and promoted to float32 on the
        same device for basis evaluation and gradients. Metric solves and update
        arithmetic use float64 before casting back to the model dtype. On MPS,
        float64 penalties and solves run on CPU with differentiable transfers.
        The returned model then has float32 coefficients.
    points
        Fixed samples. Otherwise draw ``n_samples`` with the stated cell weights.
    n_samples, seed, cell_weights
        Sampling count, local random seed and explicit cell distribution.
    penalty, fixed
        Quadratic penalties and exact fixed coefficient values.
    callback
        Receives a pre-update Iteration and the current model. Returning True stops without claiming convergence.

    Returns
    -------
    OptimizationResult
        Copied model, pre-update history, final full-sample objective and stopping state.

    Raises
    ------
    ValueError
        The loss does not return one finite value per sample.
    ArithmeticError
        The combined objective, gradient or coefficient update is non-finite,
        or the step-scaled metric matrix cannot be represented in float64.
        Zero-weight penalties are not evaluated. Active penalties accumulate in
        float64 and apply weights before the final product, while gradients retain
        the model dtype. Truly unrepresentable results raise on every stopping path.
    """
    fitted = _model_for_optimization(model)
    p = fitted.coefficients
    batch = sample(model.space, n_samples, seed=seed, cell_weights=cell_weights) if points is None else points
    batch = batch.to(device=p.device, dtype=p.dtype)
    model.space.validate(batch)
    if not len(batch):
        raise ValueError("minimize requires at least one sample.")
    free = _fix_controls(fitted, fixed)
    if not free.any():
        return OptimizationResult(
            fitted,
            (),
            True,
            "all_fixed",
            "All coefficients are fixed.",
            _objective(fitted, loss, batch, penalty).item(),
            0,
            seed,
        )
    step_for = _prepare_step(fitted, method, penalty, free, cell_weights)
    generator = _generator(None if seed is None else seed + 1)
    history: list[Iteration] = []
    success, status, message = False, "iteration_limit", "Iteration budget exhausted; convergence was not established."
    for k in range(method.max_iter):
        if method.gradient_tolerance is not None:
            full = _objective(fitted, loss, batch, penalty)
            full_grad = _gradient(full, p)
            if full_grad[free].norm().item() <= method.gradient_tolerance:
                success, status, message = True, "gradient_tolerance", "Full-sample gradient tolerance reached."
                break
        rows = torch.randperm(len(batch), generator=generator)[: method.batch_size or len(batch)]
        current = batch.take(rows)
        value = _objective(fitted, loss, current, penalty)
        gradient = _gradient(value, p)
        record = Iteration(k, value.item(), gradient[free].norm().item())
        history.append(record)
        if callback is not None and callback(record, fitted):
            status, message = "callback_stop", "Stopped by callback; convergence was not established."
            break
        alpha = method.step_size / math.sqrt(k + 1) if method.step_decay == "sqrt" else method.step_size
        direction, update_alpha = step_for(gradient[free], alpha)
        _update(p, free, update_alpha, direction)
    final = _objective(fitted, loss, batch, penalty).item()
    result = OptimizationResult(fitted, tuple(history), success, status, message, final, len(history), seed)
    if not success and status == "iteration_limit":
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    return result
