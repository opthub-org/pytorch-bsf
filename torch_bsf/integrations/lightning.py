"""Opt-in Lightning execution for the common weighted regression objective."""

import math
from typing import Any

import torch

from torch_bsf._problem import LeastSquaresProblem
from torch_bsf.core import BezierModel, FixedControls, QuadraticPenalty, Samples


def _check_learning_rate(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("learning_rate must be finite and positive.")


def regression_module(
    model: BezierModel,
    samples: Samples,
    *,
    learning_rate: float = 0.01,
    penalty: QuadraticPenalty = QuadraticPenalty(),
    fixed: FixedControls | None = None,
) -> Any:
    """Create a LightningModule from an explicit initial model and observations.

    The caller owns Trainer settings and a one-item DataLoader (e.g. ``[0]``).
    Each step uses the complete normalized weighted objective, so batch and
    device partitioning cannot silently change regularization. This adapter
    supports one device; use core least squares for deterministic regression.
    Its ``bezier_model()`` method returns an independent CPU float64 model.

    Parameters
    ----------
    model : BezierModel
        Initial coefficients and compiled geometry; the original model is not modified.
    samples : Samples
        Complete weighted regression observations used on every training step.
    learning_rate : float, optional
        Finite positive Adam learning rate.
    penalty : QuadraticPenalty, optional
        Ridge and control-net smoothness penalties.
    fixed : FixedControls, optional
        Exact coefficient constraints excluded from optimization.

    Returns
    -------
    lightning.pytorch.LightningModule
        Module owning the free coefficients and exposing ``bezier_model()``.
    """
    _check_learning_rate(learning_rate)
    try:
        from lightning.pytorch import LightningModule
    except ModuleNotFoundError as error:
        if error.name != "lightning":
            raise
        raise ImportError('Install "pytorch-bsf[lightning]" for Lightning execution.') from error
    problem = LeastSquaresProblem(model.space, samples, penalty, fixed)
    if not problem.free.any():
        raise ValueError("all controls are fixed; no training is necessary.")

    class RegressionModule(LightningModule):
        """Train the free coefficients of the captured regression problem."""

        operator: torch.Tensor
        target: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.controls = torch.nn.Parameter(model.coefficients.detach().cpu().double()[problem.free].clone())
            self.register_buffer("operator", problem.dense())
            self.register_buffer("target", problem.rhs)

        def on_fit_start(self) -> None:
            """Check that training uses exactly one device."""
            if self.trainer.world_size != 1:
                raise ValueError("regression_module supports a single device.")

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            """Evaluate and log the complete regression objective.

            Parameters
            ----------
            batch : object
                Ignored DataLoader item; every step uses the complete regression problem.
            batch_idx : int
                Ignored batch index supplied by Lightning.

            Returns
            -------
            torch.Tensor
                Scalar weighted regression objective, including regularization.
            """
            loss = (self.operator @ self.controls - self.target).square().sum()
            self.log("objective", loss)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            """Create the optimizer for the free coefficients.

            Returns
            -------
            torch.optim.Optimizer
                Adam optimizer acting on the free coefficient rows.
            """
            return torch.optim.Adam([self.controls], lr=learning_rate)

        def bezier_model(self) -> BezierModel:
            """Export the current fitted Bézier model.

            Returns
            -------
            BezierModel
                Independent CPU float64 model with free and fixed coefficient rows assembled.
            """
            return BezierModel(model.space, problem.coefficients(self.controls.detach().cpu().double()))

    return RegressionModule()
