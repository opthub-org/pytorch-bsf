"""Explicit ensembling, independent of cross-validation and model selection."""

from collections.abc import Sequence

import torch
from torch import nn

from torch_bsf._numeric import _finite_mean
from torch_bsf.core import BezierModel, Points


class Ensemble(nn.Module):
    """Mean predictor over copied models with the same space and output dimension.

    Parameters
    ----------
    models : sequence of BezierModel
        Nonempty collection with matching space, output dimension, coefficient dtype and device.
    """

    def __init__(self, models: Sequence[BezierModel]):
        super().__init__()
        if not models:
            raise ValueError("an ensemble requires at least one model.")
        first = models[0]
        for model in models:
            if not first.space.compatible(model.space) or model.n_outputs != first.n_outputs:
                raise ValueError("ensemble models must share their space and output dimension.")
            if (model.coefficients.device, model.coefficients.dtype) != (first.coefficients.device, first.coefficients.dtype):
                raise ValueError("ensemble models must share dtype and device.")
        self.space = first.space
        self.models = nn.ModuleList([model.copy() for model in models])

    def members(self, points: Points | torch.Tensor) -> torch.Tensor:
        """Predict with every member, returning shape ``(n_models, n_points, n_outputs)``.

        Parameters
        ----------
        points : Points or torch.Tensor
            Evaluation coordinates accepted by every member model; multi-cell spaces require Points.

        Returns
        -------
        torch.Tensor
            Member predictions of shape ``(n_models, n_points, n_outputs)``.
        """
        return torch.stack([model(points) for model in self.models])

    def forward(self, points: Points | torch.Tensor) -> torch.Tensor:
        """Return the ensemble mean, without changing the member models.

        Finite groups near sum overflow use an exact CPU reduction to retain
        cancelling residuals. The overflow-risk check synchronizes accelerators;
        only affected groups transfer to CPU. Ordinary groups remain vectorized on their
        device, and finite-input gradients follow the ordinary mean.

        Parameters
        ----------
        points : Points or torch.Tensor
            Evaluation coordinates accepted by every member model; multi-cell spaces require Points.

        Returns
        -------
        torch.Tensor
            Mean predictions of shape ``(n_points, n_outputs)``.
        """
        return _finite_mean(self.members(points), 0)
