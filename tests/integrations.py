"""Optional adapters share the common objective and persistence format."""

from unittest.mock import patch

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, FixedControls, QuadraticPenalty, Samples, fit
from torch_bsf.integrations.lightning import regression_module
from torch_bsf.integrations.mlflow import log_result


def test_mlflow_records_unavailable_failure_without_writing_a_model():
    from unittest.mock import Mock

    from torch_bsf.integrations.mlflow import _record
    from torch_bsf.results import ConvergenceWarning

    with patch("torch.linalg.lstsq", side_effect=torch.linalg.LinAlgError("failed")), pytest.warns(ConvergenceWarning):
        result = fit(BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), [[1.0], [2.0]]))
    recording = Mock()
    with patch("torch_bsf.integrations.mlflow.save_model", side_effect=AssertionError("placeholder saved")):
        assert _record(recording, result, "failure-run") == "failure-run"
    recording.log_metrics.assert_called_once_with(result.metrics)
    recording.log_dict.assert_called_once()
    recording.log_artifact.assert_not_called()
    assert recording.set_tags.call_args.args[0]["bsf.solution_available"] == "False"


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
        with mlflow.start_run() as active:
            assert log_result(result) == active.info.run_id
            assert mlflow.active_run() is active
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize("adapter,package", [("lightning", "lightning"), ("mlflow", "mlflow"), ("sparse", "scipy")])
@pytest.mark.parametrize("missing_dependency", [False, True])
def test_optional_execution_import_errors(monkeypatch, adapter, package, missing_dependency):
    import builtins

    from torch_bsf._problem import scipy_sparse

    original = builtins.__import__

    def intercept(name, *args, **kwargs):
        if name.split(".")[0] == package:
            raise ModuleNotFoundError("unavailable", name="broken_transitive_dependency" if missing_dependency else package)
        return original(name, *args, **kwargs)

    model = BezierModel(BezierSpace.simplex(1, degree=1), [[0.0], [0.0]])
    observations = Samples(torch.eye(2), [[1.0], [2.0]])
    result = fit(model.space, observations)
    monkeypatch.setattr(builtins, "__import__", intercept)
    with pytest.raises(ModuleNotFoundError if missing_dependency else ImportError) as error:
        if adapter == "lightning":
            regression_module(model, observations)
        elif adapter == "mlflow":
            log_result(result)
        else:
            scipy_sparse()
    if missing_dependency:
        assert error.value.name == "broken_transitive_dependency"
    else:
        assert "pytorch-bsf[" in str(error.value)


def test_lightning_rejects_invalid_execution_modes():
    from types import SimpleNamespace

    model = BezierModel(BezierSpace.simplex(1, degree=1), [[0.0], [0.0]])
    observations = Samples(torch.eye(2), [[1.0], [2.0]])
    with pytest.raises(ValueError):
        regression_module(model, observations, learning_rate=0)
    pytest.importorskip("lightning")
    with pytest.raises(ValueError):
        regression_module(model, observations, fixed=FixedControls({0: [0.0], 1: [0.0]}))
    module = regression_module(model, observations)
    module._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="single device"):
        module.on_fit_start()
