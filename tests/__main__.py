"""CLI integration against the public operations."""

import json
import subprocess
import sys

import numpy as np
import pytest

from torch_bsf.__main__ import main
from torch_bsf.io import load_model


def data(tmp_path):
    t = np.column_stack([np.linspace(0, 1, 12), np.linspace(1, 0, 12)])
    p, v = tmp_path / "params.csv", tmp_path / "values.csv"
    np.savetxt(p, t, delimiter=",")
    np.savetxt(v, t * 2, delimiter=",")
    return ["--params", str(p), "--values", str(v), "--degree", "1"]


def test_fit_predict(tmp_path, capsys):
    options = data(tmp_path)
    path = tmp_path / "model.bsf"
    main(["fit", *options, "--save", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert report["success"] and report["metrics"]["data_mse"] < 1e-25
    assert report["model_saved"]
    output = tmp_path / "predictions.csv"
    main(["predict", "--model", str(path), "--params", options[1], "--save", str(output)])
    assert np.allclose(np.loadtxt(output, delimiter=","), np.loadtxt(options[3], delimiter=","))
    assert load_model(path).n_outputs == 2


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("tracking", [False, True])
def test_unavailable_fit_never_creates_or_overwrites_artifacts(tmp_path, capsys, existing, tracking):
    from unittest.mock import patch

    import torch

    from torch_bsf import BezierModel, BezierSpace
    from torch_bsf.io import save_model
    from torch_bsf.results import ConvergenceWarning

    path = tmp_path / "model.bsf"
    previous = None
    if existing:
        save_model(path, BezierModel(BezierSpace.simplex(1, degree=1), [[3.0, 4.0], [5.0, 6.0]]))
        previous = path.read_bytes()
    with (
        patch("torch.linalg.lstsq", side_effect=torch.linalg.LinAlgError("decomposition failed")),
        patch("torch_bsf.integrations.mlflow.log_result") as record,
        pytest.warns(ConvergenceWarning),
        pytest.raises(SystemExit) as error,
    ):
        main(["fit", *data(tmp_path), "--save", str(path), *(["--track"] if tracking else [])])
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["success"] and not report["model_saved"]
    assert not report["diagnostics"]["solution_available"]
    assert (path.read_bytes() if path.exists() else None) == previous
    assert record.call_count == int(tracking)


def test_diagnostic_failure_keeps_available_cli_artifact(tmp_path, capsys):
    from unittest.mock import patch

    import torch

    from torch_bsf.results import ConvergenceWarning

    path = tmp_path / "model.bsf"
    with (
        patch("torch.linalg.svdvals", side_effect=torch.linalg.LinAlgError("data diagnostics failed")),
        pytest.warns(ConvergenceWarning),
        pytest.raises(SystemExit) as error,
    ):
        main(["fit", *data(tmp_path), "--save", str(path), "--ridge", ".1", "--diagnostics", "full"])
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["model_saved"] and report["status"] == "diagnostic_failure"
    assert load_model(path).n_outputs == 2


@pytest.mark.parametrize("command", ["cross-validate", "select-degree"])
def test_selection(tmp_path, capsys, command):
    options = ["--degrees", "1", "2"] if command == "select-degree" else []
    main([command, *data(tmp_path), "--folds", "3", *options])
    assert json.loads(capsys.readouterr().out)["success"]


def test_no_optional_framework_imported(tmp_path):
    script = """import sys
from torch_bsf.__main__ import main
main(sys.argv[1:])
assert not any(m.split('.')[0] in ('lightning','mlflow','scipy') for m in sys.modules)
"""
    run = subprocess.run(
        [sys.executable, "-c", script, "fit", *data(tmp_path), "--save", str(tmp_path / "m")], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr


def test_legacy_flags_rejected():
    with pytest.raises(SystemExit) as error:
        main(["--max_epochs", "1"])
    assert error.value.code == 2


def test_minimal_install_supports_full_core(tmp_path):
    script = """
import importlib.abc
import sys
class ExcludeExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lightning','mlflow','scipy','sklearn','matplotlib'}:
            raise ModuleNotFoundError('extra unavailable', name=fullname.split('.')[0])
sys.meta_path.insert(0,ExcludeExtras())
import torch
from torch_bsf import BezierSpace, Samples, fit, minimize, PreconditionedSGD
from torch_bsf.sampling import grid
from torch_bsf.io import save_model,load_model
space=BezierSpace.simplex(1,degree=1)
points=grid(space,4)
result=fit(space,Samples(points,points.parameters[:,:1]))
assert result.success
optimized=minimize(result.model,lambda y,t: (y[:,0]-t.parameters[:,0]).square(),
                   points=points,method=PreconditionedSGD(gradient_tolerance=1e-7))
assert optimized.success
save_model(sys.argv[1],optimized.model)
assert torch.allclose(load_model(sys.argv[1])(points),result.model(points))
"""
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path / "model.bsf")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_predict_multi_cell_model_with_explicit_ids(tmp_path):
    import torch

    from torch_bsf import BezierModel, BezierSpace, Points, SimploidalSet
    from torch_bsf.io import save_model

    domain = SimploidalSet()
    a, b = domain.add_simplex(2), domain.add_simploid((1, 1))
    space = BezierSpace.on_set(domain, degree=2)
    model = BezierModel(space, torch.arange(space.n_controls, dtype=torch.float64)[:, None])
    points = Points.from_cells(space, [(a, [[0.2, 0.3, 0.5]]), (b, [[0.1, 0.9, 0.4, 0.6]])])
    saved, parameters, ids, output = [tmp_path / name for name in ("model.bsf", "params.csv", "ids.csv", "predicted.csv")]
    save_model(saved, model)
    np.savetxt(parameters, points.parameters.numpy(), delimiter=",")
    np.savetxt(ids, points.cell_ids.numpy(), delimiter=",", fmt="%d")
    main(["predict", "--model", str(saved), "--params", str(parameters), "--cell-ids", str(ids), "--save", str(output)])
    assert np.allclose(np.loadtxt(output), model(points).detach().numpy().flatten())


@pytest.mark.parametrize(
    "options",
    [["--dimensions", "2"], ["--solver", "sparse", "--diagnostics", "full"], ["--tracking-uri", "sqlite:///unrequested"]],
)
def test_fit_invalid_options(tmp_path, options):
    with pytest.raises(SystemExit) as error:
        main(["fit", *data(tmp_path), "--save", str(tmp_path / "out"), *options])
    assert error.value.code == 2


def test_sparse_cli_and_explicit_weights(tmp_path, capsys):
    pytest.importorskip("scipy")
    weights = tmp_path / "weights.csv"
    np.savetxt(weights, np.arange(1, 13), delimiter=",")
    main(["fit", *data(tmp_path), "--solver", "sparse", "--weights", str(weights), "--save", str(tmp_path / "model")])
    assert json.loads(capsys.readouterr().out)["success"]


def test_minimize_cli_and_failure_status(tmp_path, capsys):
    from torch_bsf import BezierModel, BezierSpace
    from torch_bsf.io import save_model

    model = BezierModel(BezierSpace.simplex(1, degree=1), [[0.0], [0.0]])
    saved = tmp_path / "initial"
    save_model(saved, model)
    objective = tmp_path / "objective_for_cli.py"
    objective.write_text("import torch\ndef objective(x):\n return torch.cat([(x-1).square(),(x+1).square()],1)\n")
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.warns(UserWarning), pytest.raises(SystemExit) as error:
            main(
                [
                    "minimize",
                    "--init",
                    str(saved),
                    "--objective",
                    "objective_for_cli:objective",
                    "--max-iter",
                    "1",
                    "--save",
                    str(tmp_path / "result"),
                ]
            )
        assert error.value.code == 1
        assert json.loads(capsys.readouterr().out)["status"] == "iteration_limit"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("objective_for_cli", None)


def test_explicit_tracking_adapter_is_invoked(tmp_path):
    from unittest.mock import patch

    with patch("torch_bsf.integrations.mlflow.log_result") as record:
        main(["fit", *data(tmp_path), "--track", "--save", str(tmp_path / "model")])
    assert record.call_count == 1


def test_invalid_prediction_cell_identifiers(tmp_path):
    from torch_bsf import BezierModel, BezierSpace
    from torch_bsf.io import save_model

    options = data(tmp_path)
    model_path = tmp_path / "model"
    save_model(model_path, BezierModel(BezierSpace.simplex(1, degree=1), [[0.0], [1.0]]))
    ids = tmp_path / "ids"
    np.savetxt(ids, np.full(12, 0.5))
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--model",
                str(model_path),
                "--params",
                options[1],
                "--cell-ids",
                str(ids),
                "--save",
                str(tmp_path / "predicted"),
            ]
        )


def test_python_module_entrypoint_help(monkeypatch, capsys):
    import runpy

    import torch_bsf.__main__ as cli

    assert cli.__file__ is not None
    monkeypatch.setattr(sys, "argv", ["torch_bsf", "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert error.value.code == 0
    assert "cross-validate" in capsys.readouterr().out


def test_constant_cli_fit_and_prediction(tmp_path, capsys):
    parameters = tmp_path / "dummy.csv"
    values = tmp_path / "values.csv"
    saved = tmp_path / "model.bsf"
    predicted = tmp_path / "predicted.csv"
    np.savetxt(parameters, np.arange(5), delimiter=",")
    np.savetxt(values, np.arange(5), delimiter=",")
    main(["fit", "--params", str(parameters), "--values", str(values), "--degree", "3", "--save", str(saved)])
    assert json.loads(capsys.readouterr().out)["success"]
    assert load_model(saved).space.width == 0
    main(["predict", "--model", str(saved), "--params", str(parameters), "--save", str(predicted)])
    assert np.allclose(np.loadtxt(predicted), 2)
