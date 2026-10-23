"""Tests for torch_bsf.__main__ (the CLI entry point)."""

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests import requires_lightning

_REPO_ROOT = Path(__file__).parent.parent
_PARAMS_CSV = _REPO_ROOT / "params.csv"
_VALUES_CSV = _REPO_ROOT / "values.csv"
_CLI_TIMEOUT = 300  # seconds


def _init_file(tmp_path: Path) -> Path:
    """Saves a random Bézier simplex fitting params.csv and values.csv to use as --init."""
    import numpy as np

    from torch_bsf.simploidal import BezierSimploid, save

    n_params = np.loadtxt(_PARAMS_CSV, delimiter=",", ndmin=2).shape[1]
    n_values = np.loadtxt(_VALUES_CSV, delimiter=",", ndmin=2).shape[1]
    init_file = tmp_path / "model.pt"
    save(init_file, BezierSimploid.randn((n_params - 1,), 1, n_values))
    return init_file


def _run_cli(*args, cwd=None):
    """Run `python -m torch_bsf` with the given arguments."""
    workdir = cwd or _REPO_ROOT
    env = os.environ.copy()
    # Force MLflow to use a local file-based backend under the working directory
    env["MLFLOW_TRACKING_URI"] = f"file://{workdir}"
    # Remove any external MLflow credentials to keep tests hermetic
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        env.pop(var, None)
    # Test the torch_bsf of this checkout, as the in-process tests do, even when cwd is elsewhere
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-m", "torch_bsf"] + list(args),
        capture_output=True,
        text=True,
        cwd=workdir,
        env=env,
        timeout=_CLI_TIMEOUT,
    )


def test_cli_basic_run(tmp_path):
    """CLI should complete successfully with minimal required arguments."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        "--max_epochs=1",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_cli_missing_params_fails():
    """CLI should exit non-zero when --params is missing."""
    result = _run_cli(
        f"--values={_VALUES_CSV}",
        "--degree=1",
    )
    assert result.returncode != 0


def test_cli_missing_values_fails():
    """CLI should exit non-zero when --values is missing."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        "--degree=1",
    )
    assert result.returncode != 0


def test_cli_missing_degree_and_init_fails():
    """CLI should fail with an argparse error when neither --degree nor --init is given."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--loglevel=0",
    )
    assert result.returncode != 0


def test_cli_missing_degree_and_init_is_argparse_error():
    """CLI should emit a clean argparse error (not a Python traceback) when --degree and --init are both omitted."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_both_degree_and_init_fails(tmp_path):
    """CLI should fail with an argparse error when both --degree and --init are specified."""
    init_file = _init_file(tmp_path)

    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        f"--init={init_file}",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_cli_both_degree_and_init_is_argparse_error(tmp_path):
    """CLI should emit a clean argparse error (not a Python traceback) when both --degree and --init are given."""
    init_file = _init_file(tmp_path)

    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        f"--init={init_file}",
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_help_flag():
    """CLI --help should exit 0 and include expected metavar/description text."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "--degree N" in result.stdout
    assert "--init FILE" in result.stdout
    assert "--devices N|auto" in result.stdout


def test_cli_init_flag(tmp_path):
    """CLI should accept --init to load an existing model."""
    init_file = _init_file(tmp_path)

    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        f"--init={init_file}",
        "--max_epochs=1",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_cli_normalize_option(tmp_path):
    """CLI should accept --normalize with valid choices."""
    for norm in ("none", "max", "std", "quantile"):
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--degree=1",
            f"--normalize={norm}",
            "--max_epochs=1",
            "--loglevel=0",
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"normalize={norm} failed: {result.stderr}"


def test_cli_meshgrid_saved(tmp_path):
    """CLI should create a meshgrid output file."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        "--max_epochs=1",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert any(tmp_path.iterdir()), "Expected at least one output file in tmp_path"


# ---------------------------------------------------------------------------
# In-process tests for main() to get coverage of the module
# ---------------------------------------------------------------------------


class TestMainInProcess:
    """Call torch_bsf.__main__.main() directly with patched sys.argv."""

    def test_main_basic(self, tmp_path, monkeypatch):
        """main() should run training and save a meshgrid CSV."""
        from torch_bsf.__main__ import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
        for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "torch_bsf",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--degree=1",
                "--max_epochs=1",
                "--loglevel=0",
                "--accelerator=cpu",
                "--devices=1",
            ],
        )
        main()
        assert any(tmp_path.iterdir()), "Expected at least one output file"

    def test_main_with_init(self, tmp_path, monkeypatch):
        """main() should accept --init to load an existing model."""
        from torch_bsf.__main__ import main

        init_file = _init_file(tmp_path)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
        for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "torch_bsf",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                f"--init={init_file}",
                "--max_epochs=1",
                "--loglevel=0",
                "--accelerator=cpu",
                "--devices=1",
            ],
        )
        main()
        assert any(tmp_path.iterdir()), "Expected at least one output file"


# ---------------------------------------------------------------------------
# Bezier simploid fitting (--simploid-type) and the options shared with simplices
# ---------------------------------------------------------------------------


def _write_prism_data(directory: Path, n: int = 60) -> tuple[Path, Path]:
    """Write samples of a degree-2 function on the prism (simploid type (2, 1)) to CSV files."""
    import numpy as np

    from torch_bsf.simploidal import sample_simploid

    t = sample_simploid((2, 1), n, seed=0).double().numpy()
    x = np.concatenate([t[:, :3] * t[:, 3:4], 3.0 + 10.0 * t[:, 4:] ** 2], axis=1)
    params, values = directory / "prism_params.csv", directory / "prism_values.csv"
    np.savetxt(params, t, delimiter=",")
    np.savetxt(values, x, delimiter=",")
    return params, values


def _run_main(monkeypatch, tmp_path, *args):
    """Run torch_bsf.__main__.main() in-process in ``tmp_path`` with the given arguments."""
    from torch_bsf.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("sys.argv", ["torch_bsf", *args])
    try:
        main()
    finally:
        mlflow = sys.modules.get("mlflow")  # imported by main() only if installed and logging
        if mlflow is not None:
            mlflow.end_run()  # do not leak a run started on demand into other tests


def test_cli_simploid_fits_and_saves_control_points(tmp_path):
    """--simploid-type=2,1 --degree=2 should fit a Bezier simploid and save its control points and predictions."""
    import numpy as np
    import torch

    import torch_bsf.simploidal as ts

    params, values = _write_prism_data(tmp_path)
    result = _run_cli(
        f"--params={params}", f"--values={values}", "--simploid-type=2,1", "--degree=2", "--loglevel=0", cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr

    saved = tmp_path / "prism_params.csv,prism_values.csv,control_points,t_2-1,d_2.csv"
    assert f"Control points saved: {saved.name}" in result.stdout
    model = ts.load(saved)
    assert model.simploid_type == (2, 1)
    assert model.degree == (2, 2)
    t = torch.from_numpy(np.loadtxt(params, delimiter=",", ndmin=2)).float()
    x = torch.from_numpy(np.loadtxt(values, delimiter=",", ndmin=2)).float()
    assert torch.allclose(model(t), x, atol=1e-4)

    meshgrid = np.loadtxt(tmp_path / "prism_params.csv,prism_values.csv,meshgrid,t_2-1,d_2.csv", delimiter=",", ndmin=2)
    assert np.allclose(meshgrid, model(t).detach().numpy(), atol=1e-5)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".yml", ".pt"])
def test_main_simploid_save_option(tmp_path, monkeypatch, suffix):
    """--save should choose the path and the format of the saved control points."""
    import torch

    import torch_bsf.simploidal as ts

    params, values = _write_prism_data(tmp_path)
    save = tmp_path / f"model{suffix}"
    _run_main(
        monkeypatch,
        tmp_path,
        f"--params={params}",
        f"--values={values}",
        "--simploid_type=(2, 1)",
        "--degree=2",
        f"--save={save}",
        "--loglevel=0",
    )
    model = ts.load(save)
    t = ts.sample_simploid((2, 1), 10, seed=1)
    x = torch.cat([t[:, :3] * t[:, 3:4], 3.0 + 10.0 * t[:, 4:] ** 2], dim=1)
    assert torch.allclose(model(t), x, atol=1e-4)


@pytest.mark.parametrize("normalize", ["max", "std", "quantile"])
def test_main_simploid_normalize_predicts_in_original_scale(tmp_path, monkeypatch, normalize):
    """With --normalize, the saved control points and predictions should be in the original scale."""
    import numpy as np

    params, values = _write_prism_data(tmp_path)
    _run_main(
        monkeypatch,
        tmp_path,
        f"--params={params}",
        f"--values={values}",
        "--simploid-type=2,1",
        "--degree=2",
        f"--normalize={normalize}",
        "--ridge=1e-9",
        "--loglevel=0",
    )
    meshgrid = np.loadtxt(tmp_path / "prism_params.csv,prism_values.csv,meshgrid,t_2-1,d_2.csv", delimiter=",", ndmin=2)
    assert np.allclose(meshgrid, np.loadtxt(values, delimiter=",", ndmin=2), atol=1e-3)


@requires_lightning
def test_main_simploid_trainer_solver_with_logging(tmp_path, monkeypatch):
    """--solver=trainer should train with the Lightning options, and loglevel 2 should log to MLflow."""
    # The trainer imports Lightning lazily, i.e., after MLflow autologging has installed its import hook for
    # Lightning.  Under pytest, which rewrites the assertions of every module (python_files = "*.py"), that hook
    # fails to load Lightning, so import it beforehand, as importing torch_bsf did before Lightning became optional.
    import lightning.pytorch  # noqa: F401

    import torch_bsf.simploidal as ts

    params, values = _write_prism_data(tmp_path)
    _run_main(
        monkeypatch,
        tmp_path,
        f"--params={params}",
        f"--values={values}",
        "--simploid-type=2,1",
        "--degree=1",
        "--solver=trainer",
        "--max_epochs=1",
        "--accelerator=cpu",
        "--devices=1",
        "--loglevel=2",
    )
    model = ts.load(tmp_path / "prism_params.csv,prism_values.csv,control_points,t_2-1,d_1.csv")
    assert model.simploid_type == (2, 1)
    assert model.degree == (1, 1)
    assert (tmp_path / "prism_params.csv,prism_values.csv,meshgrid,t_2-1,d_1.csv").exists()


def test_main_simploid_lstsq_logs_metrics(tmp_path, monkeypatch):
    """The lstsq solver should log its parameters and training error to MLflow."""
    mlflow = pytest.importorskip("mlflow")

    params, values = _write_prism_data(tmp_path)
    _run_main(
        monkeypatch, tmp_path, f"--params={params}", f"--values={values}", "--simploid-type=2,1", "--degree=2", "--loglevel=1"
    )
    run = mlflow.last_active_run()
    assert run is not None
    assert run.data.params["simploid_type"] == "2,1"
    assert run.data.params["solver"] == "lstsq"
    assert run.data.metrics["train_mse"] < 1e-6


def test_main_simploid_in_mlflow_project_run(tmp_path, monkeypatch):
    """Inside an MLflow project run, the parameters the project already logged should be kept."""
    mlflow = pytest.importorskip("mlflow")

    params, values = _write_prism_data(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        with mlflow.start_run() as run:  # what `mlflow run -e simploid -P ridge=0 ...` does
            mlflow.log_params({"simploid_type": "2, 1", "degree": "2", "ridge": "0"})
        monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)
        _run_main(
            monkeypatch,
            tmp_path,
            f"--params={params}",
            f"--values={values}",
            "--simploid-type=2, 1",
            "--degree=2",
            "--ridge=0",
            "--loglevel=2",
        )
        logged = mlflow.get_run(run.info.run_id).data
    finally:
        mlflow.set_tracking_uri(None)  # None restores the default URI at runtime
    assert logged.params == {
        "simploid_type": "2, 1",
        "degree": "2",
        "ridge": "0",
        "solver": "lstsq",
        "smoothness_weight": "0.0",
        "normalize": "none",
        "split_ratio": "1.0",
    }
    assert "train_mse" in logged.metrics


def test_cli_simploid_help():
    """--help should document the simploid options."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "--simploid-type A1,...,An" in result.stdout
    assert "--solver {lstsq,trainer}" in result.stdout
    assert "--ridge W" in result.stdout
    assert "--save FILE" in result.stdout


def test_cli_simploid_error_is_clean(tmp_path):
    """A column mismatch should be a clean argparse error, not a traceback."""
    params, values = _write_prism_data(tmp_path)
    result = _run_cli(
        f"--params={params}", f"--values={values}", "--simploid-type=1,1", "--degree=2", "--loglevel=0", cwd=tmp_path
    )
    assert result.returncode == 2
    assert "--params has 5 columns, but the type 1,1 requires sum(Ai + 1) = 4 columns" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (["--simploid-type=1,1", "--degree=2"], "--params has 5 columns, but the type 1,1 requires sum(Ai + 1) = 4"),
        (["--simploid-type=2,0", "--degree=2"], "invalid simploid type '2,0'"),
        (["--simploid-type=a", "--degree=2"], "invalid simploid type 'a'"),
        (["--simploid-type=2,1", "--init=model.pt"], "cannot load --init"),
        (["--simploid-type=2,1", "--degree=2", "--freeze=[[[2,0,0],[2,0]]]"], "--freeze requires --init"),
        (["--simploid-type=2,1", "--degree=2", "--smoothness_weight=-1"], "smoothness_weight"),
        (["--simploid-type=2,1", "--degree=2", "--split_ratio=0"], "--split_ratio must satisfy 0 < R <= 1"),
        (["--simploid-type=2,1", "--degree=2", "--split_ratio=0.01"], "leaves no training samples out of 60"),
        (["--simploid-type=2,1", "--degree=-1"], "--degree must be non-negative"),
        (["--simploid-type=2,1", "--degree=2", "--ridge=-1"], "--ridge must be non-negative"),
        (["--simploid-type=2,1", "--degree=2", "--solver=trainer", "--ridge=0.1"], "--ridge is supported only with"),
        (["--simploid-type=2,1", "--degree=2", "--meshgrid=MESH"], "--meshgrid has 2 columns, but the type 2,1"),
        (["--simploid-type=2,1", "--degree=2", "--values=SHORT"], "--params has 60 rows, but --values has 3 rows"),
    ],
)
def test_main_simploid_errors(tmp_path, monkeypatch, capsys, args, message):
    """Invalid simploid options and data should exit with a clear argparse error."""
    import numpy as np

    params, values = _write_prism_data(tmp_path)
    np.savetxt(tmp_path / "mesh.csv", np.ones((3, 2)), delimiter=",")
    np.savetxt(tmp_path / "short.csv", np.ones((3, 4)), delimiter=",")
    args = [a.replace("MESH", str(tmp_path / "mesh.csv")).replace("SHORT", str(tmp_path / "short.csv")) for a in args]
    if not any(a.startswith("--values=") for a in args):
        args.append(f"--values={values}")
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, tmp_path, f"--params={params}", "--loglevel=0", *args)
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_main_simploid_init_freeze_and_smoothness(tmp_path, monkeypatch):
    """--init, --freeze and --smoothness_weight work together with --simploid-type."""
    import torch_bsf.simploidal as ts

    params, values = _write_prism_data(tmp_path)
    init = ts.BezierSimploid.randn((2, 1), 2, 4)
    ts.save(tmp_path / "init.json", init)
    out = tmp_path / "fitted.json"
    _run_main(
        monkeypatch,
        tmp_path,
        f"--params={params}",
        f"--values={values}",
        "--simploid-type=2,1",
        f"--init={tmp_path / 'init.json'}",
        "--freeze=[[[2,0,0],[2,0]], [[0,0,2],[0,2]]]",
        "--smoothness_weight=0.01",
        f"--save={out}",
        "--loglevel=0",
    )
    model = ts.load(out)
    assert model.simploid_type == (2, 1)
    for index in (((2, 0, 0), (2, 0)), ((0, 0, 2), (0, 2))):
        assert model[index].tolist() == init[index].tolist()
    assert model[((1, 1, 0), (1, 1))].tolist() != init[((1, 1, 0), (1, 1))].tolist()


def test_main_split_ratio_logs_validation_error(tmp_path, monkeypatch):
    """--split_ratio holds out samples for validation, and --seed makes the split reproducible."""
    mlflow = pytest.importorskip("mlflow")

    params, values = _write_prism_data(tmp_path)
    args = [f"--params={params}", f"--values={values}", "--simploid-type=2,1", "--degree=2", "--split_ratio=0.5"]
    _run_main(monkeypatch, tmp_path, *args, "--seed=0", "--loglevel=1")
    run = mlflow.last_active_run()
    assert run is not None
    assert run.data.params["split_ratio"] == "0.5"
    assert run.data.metrics["train_mse"] < 1e-6
    assert run.data.metrics["val_mse"] < 1e-6  # the data is a degree-2 polynomial


def test_main_split_ratio_without_seed(tmp_path, monkeypatch):
    """Without --seed, --split_ratio splits the samples at random."""
    mlflow = pytest.importorskip("mlflow")

    params, values = _write_prism_data(tmp_path)
    args = [f"--params={params}", f"--values={values}", "--simploid-type=2,1", "--degree=2", "--split_ratio=0.5"]
    _run_main(monkeypatch, tmp_path, *args, "--loglevel=1")
    run = mlflow.last_active_run()
    assert run is not None
    assert run.data.metrics["val_mse"] < 1e-6


def test_main_simplex_is_the_default_type(tmp_path, monkeypatch):
    """Without --simploid-type, a Bézier simplex of dimension n_params - 1 is fitted."""
    import numpy as np

    import torch_bsf.simploidal as ts
    from torch_bsf.sampling import simplex_grid

    t = simplex_grid(3, 4).double().numpy()
    np.savetxt(tmp_path / "p.csv", t, delimiter=",")
    np.savetxt(tmp_path / "v.csv", 5.0 + t[:, :2] ** 2, delimiter=",")
    _run_main(monkeypatch, tmp_path, "--params=p.csv", "--values=v.csv", "--degree=2", "--normalize=std", "--loglevel=0")
    model = ts.load(tmp_path / "p.csv,v.csv,control_points,t_2,d_2.csv")
    assert model.simploid_type == (2,)
    meshgrid = np.loadtxt(tmp_path / "p.csv,v.csv,meshgrid,t_2,d_2.csv", delimiter=",", ndmin=2)
    np.testing.assert_allclose(meshgrid, 5.0 + t[:, :2] ** 2, atol=1e-4)


def test_main_help_mentions_sgd():
    """The help of the data fitting command should point to the sgd subcommand."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "python -m torch_bsf sgd" in result.stdout


# ---------------------------------------------------------------------------
# Pareto set learning by stochastic gradient descent (python -m torch_bsf sgd)
# ---------------------------------------------------------------------------


def sgd_objective(x: torch.Tensor) -> torch.Tensor:
    """f_m(x) = ||x - e_m||^2 (m = 1, 2, 3), whose Pareto set is the triangle spanned by e_1, e_2, e_3, reached at x = t."""
    return ((x.unsqueeze(1) - torch.eye(3, dtype=x.dtype)) ** 2).sum(dim=-1)


def sgd_objective_raises(x: torch.Tensor) -> torch.Tensor:
    raise RuntimeError("cannot evaluate")


def sgd_objective_not_differentiable(x: torch.Tensor) -> torch.Tensor:
    return sgd_objective(x).detach()


_SGD_WEIGHT = torch.zeros(3, requires_grad=True)


def sgd_objective_disconnected(x: torch.Tensor) -> torch.Tensor:
    """Values that require grad through a parameter of their own, but do not depend on ``x``."""
    return _SGD_WEIGHT.expand(len(x), 3) * 1.0


def sgd_objective_list(x: torch.Tensor) -> list[float]:
    return [0.0, 0.0, 0.0]


SGD_NOT_CALLABLE = 1

_SGD_OBJECTIVE = "tests.__main__:sgd_objective"
_SGD_ARGS = (f"--objective={_SGD_OBJECTIVE}", "--n-objectives=3", "--n-variables=3")


def _write_simplex_grid(path: Path, n_params: int = 3) -> torch.Tensor:
    """Writes a grid on the simplex of ``n_params`` parameters to ``path`` and returns it."""
    import numpy as np

    from torch_bsf.sampling import simplex_grid

    t = simplex_grid(n_params, 4)
    np.savetxt(path, t.double().numpy(), delimiter=",")
    return t


def test_cli_sgd_fits_and_saves_control_points_and_meshgrid(tmp_path):
    """sgd should fit a Bezier simplex to the Pareto set and save its control points and the meshgrid."""
    import numpy as np

    import torch_bsf.simploidal as ts

    t = _write_simplex_grid(tmp_path / "mesh.csv")
    result = _run_cli(
        "sgd",
        *_SGD_ARGS,
        "--degree=1",
        "--batch-size=20",
        "--step-size=0.2",
        "--seed=0",
        "--meshgrid=mesh.csv",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    saved = tmp_path / "tests.__main__.sgd_objective,control_points,t_2,d_1.csv"
    assert f"Control points saved: {saved.name}" in result.stdout
    model = ts.load(saved)
    assert model.simploid_type == (2,)
    assert model.degree == (1,)
    assert torch.allclose(model.control_points, torch.eye(3), atol=1e-3)
    meshgrid = tmp_path / "tests.__main__.sgd_objective,meshgrid,t_2,d_1.csv"
    assert f"Meshgrid saved: {meshgrid.name}" in result.stdout
    np.testing.assert_allclose(np.loadtxt(meshgrid, delimiter=",", ndmin=2), t.numpy(), atol=1e-3)


@pytest.mark.parametrize("meshgrid", [[], ["--meshgrid=."]])
def test_main_sgd_matches_fit_sgd(tmp_path, monkeypatch, meshgrid):
    """The options should be passed to fit_sgd, and no meshgrid is saved unless a file is given."""
    import torch_bsf
    import torch_bsf.simploidal as ts

    options = dict(n_samples=50, batch_size=10, max_iter=20, step_size=0.1, step_decay="sqrt", preconditioner="none", seed=3)
    args = [f"--{k.replace('_', '-')}={v}" for k, v in options.items()]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, "--degree=2", *args, "--save=model.pt", "--loglevel=0", *meshgrid)
    expected = torch_bsf.fit_sgd(
        ts.BezierSimploid.zeros((2,), 2, n_values=3),
        ts.linear_scalarization(sgd_objective),
        **options,  # type: ignore[arg-type]
    )
    assert torch.equal(ts.load(tmp_path / "model.pt").control_points, expected.control_points)
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix != ".db") == ["model.pt"]


def test_main_sgd_saves_meshgrid(tmp_path, monkeypatch):
    """With a meshgrid file, sgd should save the fitted model evaluated at its rows."""
    import numpy as np

    import torch_bsf.simploidal as ts

    t = _write_simplex_grid(tmp_path / "mesh.csv")
    args = ["--degree=1", "--batch-size=20", "--step-size=0.2", "--seed=0", "--meshgrid=mesh.csv", "--save=model.pt"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, *args, "--loglevel=0")
    model = ts.load(tmp_path / "model.pt")
    meshgrid = np.loadtxt(tmp_path / "tests.__main__.sgd_objective,meshgrid,t_2,d_1.csv", delimiter=",", ndmin=2)
    np.testing.assert_allclose(meshgrid, model(t).detach().numpy(), atol=1e-6)


def test_main_sgd_accepts_underscored_options(tmp_path, monkeypatch):
    """The options should also be accepted with underscores, as the other options of the CLI."""
    import torch_bsf.simploidal as ts

    args = ["--n_samples=10", "--batch_size=5", "--max_iter=1", "--step_size=0.1", "--step_decay=sqrt"]
    _run_main(
        monkeypatch,
        tmp_path,
        "sgd",
        f"--objective={_SGD_OBJECTIVE}",
        "--n_objectives=3",
        "--n_variables=3",
        "--degree=1",
        *args,
        "--save=model.json",
        "--loglevel=0",
    )
    assert ts.load(tmp_path / "model.json").degree == (1,)


@pytest.mark.parametrize("loglevel", [1, 2])
def test_main_sgd_logs_to_mlflow(tmp_path, monkeypatch, loglevel):
    """sgd should log its options and the final loss, and at loglevel 2 the control points and the model."""
    import torch_bsf.__main__ as cli
    import torch_bsf.simploidal as ts

    mlflow = pytest.importorskip("mlflow")

    logged_models = []
    log_model = cli._log_model

    def spy(model, n_params, n_values):  # MLflow 3 logs the model outside the artifacts of the run
        logged_models.append((n_params, n_values))
        log_model(model, n_params, n_values)

    monkeypatch.setattr(cli, "_log_model", spy)
    args = ["--degree=1", "--step-size=0.2", "--seed=0", f"--loglevel={loglevel}"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, *args)
    assert logged_models == ([(3, 3)] if loglevel == 2 else [])
    run = mlflow.last_active_run()
    assert run is not None
    assert run.data.params["objective"] == _SGD_OBJECTIVE
    assert run.data.params["n_objectives"] == "3"
    assert run.data.params["preconditioner"] == "gram"
    assert run.data.params["batch_size"] == "None"
    # The minimum of t^T f(x) is sum_m t_m ||t - e_m||^2 = 1 - ||t||^2 at x = t, which the linear model reaches.
    (t,) = ts.sample_domain(ts.BezierSimploid.zeros((2,), 1, n_values=3), 100, seed=0)
    assert abs(run.data.metrics["train_loss"] - (1 - (t**2).sum(dim=1)).mean().item()) < 1e-4
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    expected = {"tests.__main__.sgd_objective,control_points,t_2,d_1.csv"} if loglevel == 2 else set()
    assert expected <= artifacts and (loglevel == 2 or not artifacts)  # MLflow < 3 also logs the model as an artifact


@pytest.mark.parametrize("init", [False, True])
def test_main_sgd_mlproject_entry_point(tmp_path, monkeypatch, init):
    """The command of the sgd entry point of MLproject should run in an MLflow project run, keeping its parameters."""
    import shlex

    import yaml

    import torch_bsf.simploidal as ts

    mlflow = pytest.importorskip("mlflow")

    entry = yaml.safe_load((_REPO_ROOT / "MLproject").read_text())["entry_points"]["sgd"]
    given = dict(objective=_SGD_OBJECTIVE, n_objectives="3", n_variables="3", degree="1", batch_size="20", step_size="0.2")
    if init:  # start from the solution with a vertex frozen
        ts.save(tmp_path / "init.csv", ts.BezierSimploid((2,), 1, torch.eye(3)))
        given |= dict(init=str(tmp_path / "init.csv"), freeze="[[1, 0, 0]]", smoothness_weight="1e-4", log_every="10")
    parameters = {k: v.get("default") for k, v in entry["parameters"].items()} | given
    assert all(v is not None for v in parameters.values()), parameters
    command = shlex.split(entry["command"].format(**{k: shlex.quote(str(v)) for k, v in parameters.items()}))
    assert command[:4] == ["python", "-m", "torch_bsf", "sgd"]

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        with mlflow.start_run() as run:  # what `mlflow run -e sgd -P objective=... ...` does
            mlflow.log_params(parameters)
        monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)
        _run_main(monkeypatch, tmp_path, *command[3:])
        logged = mlflow.get_run(run.info.run_id).data
    finally:
        mlflow.set_tracking_uri(None)  # None restores the default URI at runtime
    assert logged.params == {k: str(v) for k, v in parameters.items()}
    assert 0 < logged.metrics["train_loss"] < 1  # the loss of the initial model x = 0 is 1
    model = ts.load(tmp_path / "tests.__main__.sgd_objective,control_points,t_2,d_1.csv")
    assert torch.allclose(model.control_points, torch.eye(3), atol=1e-2 if init else 1e-3)
    history = mlflow.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}").get_metric_history(run.info.run_id, "batch_loss")
    assert sorted(m.step for m in history) == list(range(0, 100, 10 if init else 1))
    if init:  # the vertex stays, and the smoothing moves the others off the solution slightly
        assert model[[1, 0, 0]].tolist() == [1.0, 0.0, 0.0]
        assert not torch.allclose(model.control_points, torch.eye(3), atol=1e-4)


def test_cli_sgd_help():
    """sgd --help should document the options of fit_sgd."""
    result = _run_cli("sgd", "--help")
    assert result.returncode == 0
    # Python < 3.13 repeats the metavar after every alias ("--n-objectives M, --n_objectives M"), so check them apart.
    for option in (
        "--objective MODULE:FUNCTION",
        "--n-objectives",
        "--n_objectives M",
        "--n-variables",
        "--n_variables L",
        "--degree N",
        "--n-samples",
        "--n_samples N",
        "--batch-size",
        "--batch_size N",
        "--max-iter",
        "--max_iter N",
        "--step-size",
        "--step_size ALPHA",
        "--step-decay",
        "--step_decay {constant,sqrt}",
        "--preconditioner {gram,none}",
        "--seed N",
        "--meshgrid CSV|DIR",
        "--save FILE",
        "--init FILE",
        "--freeze INDICES",
        "--smoothness-weight",
        "--smoothness_weight W",
        "--log-every",
        "--log_every N",
    ):
        assert option in result.stdout, option
    assert result.stdout.isascii()


def test_cli_sgd_error_is_clean(tmp_path):
    """An objective that cannot be imported should be a clean argparse error, not a traceback."""
    result = _run_cli("sgd", "--objective=no_such_module_for_torch_bsf:f", "--n-objectives=3", "--n-variables=3", "--degree=1")
    assert result.returncode == 2
    assert "cannot load --objective: cannot import module 'no_such_module_for_torch_bsf'" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (["--objective=nocolon"], "cannot load --objective: expected MODULE:FUNCTION"),
        (["--objective=tests.__main__:"], "cannot load --objective: expected MODULE:FUNCTION"),
        (["--objective=no_such_module_for_torch_bsf:f"], "cannot import module 'no_such_module_for_torch_bsf'"),
        (["--objective=tests.__main__:missing"], "module 'tests.__main__' has no attribute 'missing'"),
        (["--objective=tests.__main__:SGD_NOT_CALLABLE"], "'tests.__main__:SGD_NOT_CALLABLE' is not callable"),
        (
            ["--n-objectives=2"],
            "--objective must map points of shape (2, 3) (--n-variables) to a tensor of shape (2, 2) (--n-objectives), "
            "but it returned shape (2, 3)",
        ),
        (["--n-variables=2"], "--objective failed on points of shape (2, 2): RuntimeError"),
        (["--objective=tests.__main__:sgd_objective_raises"], "--objective failed on points of shape (2, 3): RuntimeError"),
        (["--objective=tests.__main__:sgd_objective_list"], "but it returned a list"),
        (["--objective=tests.__main__:sgd_objective_not_differentiable"], "--objective must be differentiable"),
        (["--objective=tests.__main__:sgd_objective_disconnected"], "do not depend on the points through autograd"),
        (["--n-objectives=1"], "--n-objectives must be at least 2, but 1 is given"),
        (["--n-variables=0"], "--n-variables must be at least 1, but 0 is given"),
        (["--degree=-1"], "--degree must be at least 0, but -1 is given"),
        (["--n-samples=0"], "--n-samples must be at least 1, but 0 is given"),
        (["--batch-size=0"], "--batch-size must be at least 1, but 0 is given"),
        (["--max-iter=-1"], "--max-iter must be at least 0, but -1 is given"),
        (["--step-size=0"], "--step-size must be positive, but 0.0 is given"),
        (["--step-decay=linear"], "invalid choice: 'linear'"),
        (["--meshgrid=MESH"], "--meshgrid has 2 columns, but the type 2 requires sum(Ai + 1) = 3 columns"),
        (["--smoothness-weight=-0.1"], "--smoothness-weight must be non-negative, but -0.1 is given"),
        (["--smoothness_weight=nan"], "--smoothness-weight must be non-negative, but nan is given"),
        (["--log-every=-1"], "--log-every must be at least 0, but -1 is given"),
        (["--freeze=[[1,0,0]]"], "--freeze requires --init"),
        (["--freeze=[[1,0"], "argument --freeze: invalid _optional_index_list value"),
    ],
)
def test_main_sgd_errors(tmp_path, monkeypatch, capsys, args, message):
    """Invalid objectives and options should exit with a clear argparse error in ASCII."""
    _write_simplex_grid(tmp_path / "mesh.csv", n_params=2)
    args = [a.replace("MESH", str(tmp_path / "mesh.csv")) for a in args]
    defaults = dict(a.split("=", 1) for a in (*_SGD_ARGS, "--degree=1"))
    defaults.update(a.split("=", 1) for a in args)
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, tmp_path, "sgd", *(f"{k}={v}" for k, v in defaults.items()), "--loglevel=0")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert err.isascii()


@pytest.mark.parametrize(
    "init, args, message",
    [
        ("missing", [], "cannot load --init:"),
        (((2, 1), 1, 3), [], "--init is a Bezier simploid of type 2,1, but --n-objectives=3 requires a Bezier simplex"),
        (((1,), 1, 3), [], "--init is a Bezier simploid of type 1, but --n-objectives=3 requires a Bezier simplex of type 2"),
        (((2,), 2, 3), [], "--init has degree 2, but --degree=1 is given"),
        (((2,), 1, 2), [], "--init has 2 values, but --n-variables=3 is given"),
        (((2,), 1, 3), ["--freeze=[[2,0,0]]"], "--freeze has [2, 0, 0], which is not a control point index of a Bezier "),
        (((2,), 1, 3), ["--freeze=[[1,0]]"], "--freeze has [1, 0], which is not a control point index"),
        (((2,), 1, 3), ["--freeze=3"], "--freeze must be a list of indices, but 3 is given"),
    ],
)
def test_main_sgd_init_errors(tmp_path, monkeypatch, capsys, init, args, message):
    """An --init that does not match the options, or an invalid --freeze, should exit with a clear argparse error in ASCII."""
    import torch_bsf.simploidal as ts

    if init != "missing":
        ts.save(tmp_path / "init.csv", ts.BezierSimploid.randn(*init))
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, "--degree=1", "--init=init.csv", *args, "--loglevel=0")
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert err.isascii()


def test_main_sgd_init_freeze_and_smoothness(tmp_path, monkeypatch):
    """--init, --freeze and --smoothness-weight should be passed to fit_sgd, keeping the frozen control points bitwise."""
    import torch_bsf
    import torch_bsf.simploidal as ts

    init = ts.BezierSimploid.randn((2,), 3, 3)
    ts.save(tmp_path / "init.csv", init)
    init = ts.load(tmp_path / "init.csv")  # as read by the command
    vertices = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    args = ["--degree=3", "--init=init.csv", f"--freeze={vertices}", "--smoothness-weight=0.01", "--seed=0"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, *args, "--save=model.csv", "--loglevel=0")
    model = ts.load(tmp_path / "model.csv")
    for index in vertices:
        assert model[index].tolist() == init[index].tolist()
    assert model[[1, 1, 1]].tolist() != init[[1, 1, 1]].tolist()
    expected = torch_bsf.fit_sgd(init, ts.linear_scalarization(sgd_objective), freeze=vertices, smoothness_weight=0.01, seed=0)
    ts.save(tmp_path / "expected.csv", expected)
    assert (tmp_path / "model.csv").read_text() == (tmp_path / "expected.csv").read_text()


@pytest.mark.parametrize("log_every", [None, 3, 0])
def test_main_sgd_logs_loss_history(tmp_path, monkeypatch, log_every):
    """sgd should log batch_loss of every --log-every-th iteration with its step, without changing the fitted model."""
    import torch_bsf
    import torch_bsf.simploidal as ts

    mlflow = pytest.importorskip("mlflow")

    options = dict(batch_size=20, max_iter=10, step_size=0.2, seed=0)
    args = [f"--{k.replace('_', '-')}={v}" for k, v in options.items()]
    args += [] if log_every is None else [f"--log-every={log_every}"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, "--degree=1", *args, "--save=model.pt", "--loglevel=1")
    history: list[float] = []
    expected = torch_bsf.fit_sgd(
        ts.BezierSimploid.zeros((2,), 1, n_values=3),
        ts.linear_scalarization(sgd_objective),
        callback=lambda k, model, batch_loss: history.append(batch_loss),
        **options,  # type: ignore[arg-type]
    )
    assert torch.equal(ts.load(tmp_path / "model.pt").control_points, expected.control_points)
    run = mlflow.last_active_run()
    assert run is not None
    logged = mlflow.MlflowClient().get_metric_history(run.info.run_id, "batch_loss")
    every = 1 if log_every is None else log_every
    steps = list(range(0, 10, every)) if every else []
    assert [(m.step, m.value) for m in sorted(logged, key=lambda m: m.step)] == [(k, history[k]) for k in steps]
    assert run.data.params["log_every"] == str(every)
    assert run.data.params["init"] == "None" and run.data.params["freeze"] == "None"
    assert "train_loss" in run.data.metrics


def test_main_sgd_logs_long_loss_history(tmp_path, monkeypatch):
    """A history longer than the 1,000 metrics of an MLflow batch should be logged in full."""
    mlflow = pytest.importorskip("mlflow")

    args = ["--degree=1", "--n-samples=10", "--max-iter=1001", "--step-size=0.2", "--seed=0", "--loglevel=1"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, *args)
    run = mlflow.last_active_run()
    assert run is not None
    logged = mlflow.MlflowClient().get_metric_history(run.info.run_id, "batch_loss")
    assert sorted(m.step for m in logged) == list(range(1001))


def test_main_sgd_starts_from_init(tmp_path, monkeypatch):
    """--init alone should give the initial model: no iterations save it unchanged, and iterations start from it."""
    import torch_bsf
    import torch_bsf.simploidal as ts

    ts.save(tmp_path / "init.csv", ts.BezierSimploid.randn((2,), 2, 3))
    args = (*_SGD_ARGS, "--degree=2", "--init=init.csv", "--seed=0", "--loglevel=0")
    _run_main(monkeypatch, tmp_path, "sgd", *args, "--max-iter=0", "--save=model0.csv")
    assert (tmp_path / "model0.csv").read_text() == (tmp_path / "init.csv").read_text()
    _run_main(monkeypatch, tmp_path, "sgd", *args, "--max-iter=5", "--save=model5.csv")
    init = ts.load(tmp_path / "init.csv")
    expected = torch_bsf.fit_sgd(init, ts.linear_scalarization(sgd_objective), max_iter=5, seed=0)
    assert torch.equal(ts.load(tmp_path / "model5.csv").control_points, expected.control_points)


# ---------------------------------------------------------------------------
# Without MLflow, an optional dependency (the extra pytorch-bsf[mlflow])
# ---------------------------------------------------------------------------

_NO_MLFLOW_WARNING = (
    "warning: MLflow is not installed, so nothing is logged (--loglevel=2 is ignored); "
    'install it with: pip install "pytorch-bsf[mlflow]"'
)


def _python(code: str, cwd: Path, mlflow: bool = True) -> subprocess.CompletedProcess[str]:
    """Runs Python ``code`` in a fresh interpreter of this checkout, where ``import mlflow`` fails unless ``mlflow``."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    block = "" if mlflow else "sys.modules['mlflow'] = None; "
    return subprocess.run(
        [sys.executable, "-c", f"import sys; {block}{code}"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=_CLI_TIMEOUT,
    )


def test_import_does_not_load_mlflow(tmp_path):
    """Importing the package and its commands should not import MLflow, and should work without it."""
    modules = "torch_bsf, torch_bsf.__main__, torch_bsf.model_selection.kfold"
    result = _python(f"import {modules}; print('mlflow' in sys.modules)", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    result = _python(f"import {modules}", tmp_path, mlflow=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("helper", ["_check_loglevel", "_autolog"])
@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'sqlalchemy'", name="sqlalchemy"),
        ModuleNotFoundError("No module named 'mlflow.tracking'", name="mlflow.tracking"),
        ImportError("cannot import name 'incompatible_api'"),
    ],
)
def test_broken_mlflow_is_not_reported_as_uninstalled(monkeypatch, capsys, helper, error):
    """Only absence of MLflow itself may disable requested tracking; broken installations must fail visibly."""
    import torch_bsf.__main__ as cli

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "mlflow":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(type(error)) as caught:
        getattr(cli, helper)(2)
    assert caught.value is error
    assert "MLflow is not installed" not in capsys.readouterr().err


@pytest.mark.parametrize("module", ["torch_bsf", "torch_bsf.model_selection.kfold"])
def test_cli_logging_disabled_does_not_import_mlflow(tmp_path, module):
    """A cold CLI with --loglevel=0 must work even if importing the unused MLflow would fail."""
    argv = [module, f"--params={_PARAMS_CSV}", f"--values={_VALUES_CSV}", "--degree=1", "--loglevel=0"]
    if module.endswith("kfold"):
        argv.append("--num_folds=2")
    code = f"""
import builtins
import runpy
real_import = builtins.__import__
def reject_mlflow(name, *args, **kwargs):
    if name == 'mlflow' or name.startswith('mlflow.'):
        raise AssertionError('MLflow must not be imported with logging disabled')
    return real_import(name, *args, **kwargs)
builtins.__import__ = reject_mlflow
sys.argv = {argv!r}
runpy.run_module({module!r}, run_name='__main__')
assert 'mlflow' not in sys.modules
"""
    result = _python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "saved:" in result.stdout


def test_disabled_logging_disables_existing_autolog(monkeypatch):
    """An already imported MLflow still needs its autologging disabled for repeated in-process CLI calls."""
    from types import SimpleNamespace

    from torch_bsf.__main__ import _autolog

    calls = []
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(autolog=lambda **kwargs: calls.append(kwargs)))
    _autolog(0)
    assert len(calls) == 1
    assert calls[0]["disable"] is True


@pytest.mark.parametrize(
    "module, args, saved",
    [
        ("torch_bsf", [f"--params={_PARAMS_CSV}", f"--values={_VALUES_CSV}", "--degree=1"], "control_points"),
        ("torch_bsf", ["sgd", *_SGD_ARGS, "--degree=1", "--max-iter=5"], "control_points"),
        (
            "torch_bsf.model_selection.kfold",
            [f"--params={_PARAMS_CSV}", f"--values={_VALUES_CSV}", "--degree=1", "--num_folds=2"],
            "2fold,meshgrid",
        ),
    ],
)
def test_cli_without_mlflow(tmp_path, module, args, saved):
    """Without MLflow, the commands should fit and save their files at the default loglevel, warning that nothing is logged."""
    argv = [module, *args]
    code = f"import runpy; sys.argv = {argv!r}; runpy.run_module({module!r}, run_name='__main__')"
    result = _python(code, tmp_path, mlflow=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count(_NO_MLFLOW_WARNING) == 1
    assert any(saved in p.name for p in tmp_path.iterdir()), list(tmp_path.iterdir())
    assert not (tmp_path / "mlruns").exists() and not (tmp_path / "mlflow.db").exists()


@pytest.mark.parametrize("solver", ["lstsq", pytest.param("trainer", marks=requires_lightning)])
def test_main_without_mlflow(tmp_path, monkeypatch, capsys, no_mlflow, solver):
    """Without MLflow, the fitting command should save the control points and the meshgrid, warning once."""
    import torch_bsf.simploidal as ts

    params, values = _write_prism_data(tmp_path)
    args = [f"--params={params}", f"--values={values}", "--simploid-type=2,1", "--degree=1", f"--solver={solver}"]
    _run_main(monkeypatch, tmp_path, *args, "--max_epochs=1", "--accelerator=cpu", "--devices=1")
    assert ts.load(tmp_path / "prism_params.csv,prism_values.csv,control_points,t_2-1,d_1.csv").degree == (1, 1)
    assert (tmp_path / "prism_params.csv,prism_values.csv,meshgrid,t_2-1,d_1.csv").exists()
    assert capsys.readouterr().err.count(_NO_MLFLOW_WARNING) == 1
    assert sys.modules["mlflow"] is None


@pytest.mark.parametrize("loglevel", [0, 1, 2])
def test_main_sgd_without_mlflow(tmp_path, monkeypatch, capsys, no_mlflow, loglevel):
    """Without MLflow, sgd should save the control points, warning once unless --loglevel=0."""
    import torch_bsf.simploidal as ts

    args = ["--degree=1", "--max-iter=5", "--log-every=1", "--save=model.csv", f"--loglevel={loglevel}"]
    _run_main(monkeypatch, tmp_path, "sgd", *_SGD_ARGS, *args)
    assert ts.load(tmp_path / "model.csv").degree == (1,)
    err = capsys.readouterr().err
    assert err.count("MLflow is not installed") == (loglevel > 0)
    assert sys.modules["mlflow"] is None
