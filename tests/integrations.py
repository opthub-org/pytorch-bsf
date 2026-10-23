"""Optional adapters share the common objective and persistence format."""

from unittest.mock import patch

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, FixedControls, QuadraticPenalty, Samples, fit
from torch_bsf.integrations.lightning import regression_module
from torch_bsf.integrations.mlflow import log_result


def test_lightning_complete_objective(tmp_path):
    lightning = pytest.importorskip("lightning.pytorch")
    space = BezierSpace.simplex(1, degree=1)
    observations = Samples(torch.eye(2), [[1.0], [3.0]], weights=[1, 2])
    fixed = FixedControls({0: [1.0]})
    penalty = QuadraticPenalty(0.01, 0.02)
    initial = BezierModel(space, [[0.0], [0.0]])
    module = regression_module(initial, observations, learning_rate=0.05, penalty=penalty, fixed=fixed)
    trainer = lightning.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(1))))
    model = module.bezier_model()
    assert model.coefficients[0].item() == 1
    assert model.coefficients[1].item() > 0
    # Compare the adapter objective at this model with the explicit original-unit formula.
    with patch.object(module, "log"):
        loss = module.training_step(None, 0)
    p = model.coefficients
    residual = (model(observations.points) - observations.values).square().flatten()
    expected = (residual * observations.weights).sum() / 3 + 0.01 * p.square().sum() + 0.02 * (p[0] - p[1]).square().sum()
    assert loss.item() == pytest.approx(expected.item())
    assert torch.count_nonzero(initial.coefficients) == 0


def test_mlflow_explicit_recording(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    result = fit(BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), [[1.0], [2.0]]))
    previous = mlflow.get_tracking_uri()
    try:
        run_id = log_result(result, tracking_uri="sqlite:///" + str(tmp_path / "tracking.db"))
        run = mlflow.get_run(run_id)
        assert run.data.metrics["data_mse"] == result.metrics["data_mse"]
        assert run.data.tags["bsf.success"] == "True"
        assert {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)} == {"diagnostics.json", "model.bsf"}
        assert mlflow.active_run() is None
    finally:
        mlflow.set_tracking_uri(previous)
