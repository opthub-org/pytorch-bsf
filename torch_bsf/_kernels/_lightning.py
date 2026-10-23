"""The Lightning module behind the ``"trainer"`` solver of :mod:`torch_bsf._kernels.fitting`.

This module imports :mod:`lightning.pytorch` at the top level, so the solver imports it only when it runs;
install Lightning with ``pip install "pytorch-bsf[lightning]"``.
"""

from collections.abc import Sequence
from typing import Any, cast

import lightning.pytorch as L
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_bsf._kernels.fitting import _LinearBezierModel, _parameter, _smoothness_penalty


class _LightningWrapper(L.LightningModule):
    """Trains a linear Bézier model with :class:`lightning.pytorch.Trainer`.

    The training loss is the mean squared error plus ``smoothness_weight`` times the smoothness penalty.
    A batch is ``(*inputs, values)``, where ``inputs`` are the arguments of the model's ``forward``.
    """

    def __init__(self, model: nn.Module, smoothness_weight: float = 0.0):
        super().__init__()
        self.model = model
        self.smoothness_weight = smoothness_weight
        if smoothness_weight > 0:
            self.register_buffer("adjacency", cast(_LinearBezierModel, model).adjacency(), persistent=False)

    def _predict_batch(self, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        *inputs, y = batch
        return self.model(*inputs), y

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        loss = mse
        if self.smoothness_weight > 0:
            parameter = _parameter(cast(_LinearBezierModel, self.model))
            loss = mse + self.smoothness_weight * _smoothness_penalty(parameter, cast(torch.Tensor, self.adjacency))
        self.log("train_mse", mse, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> None:
        y_hat, y = self._predict_batch(batch)
        self.log("val_mse", F.mse_loss(y_hat, y), sync_dist=True)
        self.log("val_mae", F.l1_loss(y_hat, y), sync_dist=True)

    def test_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", F.l1_loss(y_hat, y), sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.model.parameters())
