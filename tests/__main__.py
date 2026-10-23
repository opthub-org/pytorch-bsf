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
    output = tmp_path / "predictions.csv"
    main(["predict", "--model", str(path), "--params", options[1], "--save", str(output)])
    assert np.allclose(np.loadtxt(output, delimiter=","), np.loadtxt(options[3], delimiter=","))
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
