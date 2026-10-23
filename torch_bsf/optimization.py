"""Objective minimization, separate from supervised least-squares regression."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from torch_bsf._kernels.sgd import _simploid_gram
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
        if self.preconditioner not in ("gram", "none") or self.step_decay not in ("constant", "sqrt"):
            raise ValueError("unknown preconditioner or step_decay.")
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

    def require_success(self) -> OptimizationResult:
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


def _objective(model: BezierModel, loss: Loss, points: Points, penalty: QuadraticPenalty) -> torch.Tensor:
    values = loss(model(points), points)
    if values.shape != (len(points),) or not torch.isfinite(values).all():
        raise ValueError("loss must return one finite value per sample.")
    coefficients = model.coefficients
    edges = model.space.adjacency().to(coefficients.device)
    return (
        values.mean()
        + penalty.ridge * coefficients.square().sum()
        + penalty.smoothness * (coefficients[edges[:, 0]] - coefficients[edges[:, 1]]).square().sum()
    )


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
        same device for basis evaluation, gradients and supported metric solves.
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
    metric = (
        gram_matrix(model.space, cell_weights=cell_weights)
        if method.preconditioner == "gram"
        else torch.eye(len(p), dtype=torch.float64)
    )
    edges = model.space.adjacency()
    regularizer = torch.eye(len(p), dtype=torch.float64) * penalty.ridge
    for i, j in edges.tolist():
        regularizer[i, i] += penalty.smoothness
        regularizer[j, j] += penalty.smoothness
        regularizer[i, j] -= penalty.smoothness
        regularizer[j, i] -= penalty.smoothness
    cpu_free = free.cpu()
    metric = metric[cpu_free][:, cpu_free].to(p)
    regularizer = regularizer[cpu_free][:, cpu_free].to(p)
    generator = _generator(None if seed is None else seed + 1)
    history: list[Iteration] = []
    success, status, message = False, "iteration_limit", "Iteration budget exhausted; convergence was not established."
    for k in range(method.max_iter):
        if method.gradient_tolerance is not None:
            full = _objective(fitted, loss, batch, penalty)
            (full_grad,) = torch.autograd.grad(full, p)
            if full_grad[free].norm().item() <= method.gradient_tolerance:
                success, status, message = True, "gradient_tolerance", "Full-sample gradient tolerance reached."
                break
        rows = torch.randperm(len(batch), generator=generator)[: method.batch_size or len(batch)]
        current = batch.take(rows)
        value = _objective(fitted, loss, current, penalty)
        (gradient,) = torch.autograd.grad(value, p)
        if not torch.isfinite(gradient).all():
            raise ArithmeticError("loss produced a non-finite gradient.")
        record = Iteration(k, value.item(), gradient[free].norm().item())
        history.append(record)
        if callback is not None and callback(record, fitted):
            status, message = "callback_stop", "Stopped by callback; convergence was not established."
            break
        alpha = method.step_size / math.sqrt(k + 1) if method.step_decay == "sqrt" else method.step_size
        direction = torch.linalg.solve(metric + 2 * alpha * regularizer, gradient[free])
        with torch.no_grad():
            p[free] -= alpha * direction
    final = _objective(fitted, loss, batch, penalty).item()
    result = OptimizationResult(fitted, tuple(history), success, status, message, final, len(history), seed)
    if not success and status == "iteration_limit":
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    return result
