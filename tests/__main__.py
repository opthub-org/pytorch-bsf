"""Tests for torch_bsf.__main__ (the CLI entry point)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    import mlflow

    from torch_bsf.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("sys.argv", ["torch_bsf", *args])
    try:
        main()
    finally:
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


def test_main_simploid_trainer_solver_with_logging(tmp_path, monkeypatch):
    """--solver=trainer should train with the Lightning options, and loglevel 2 should log to MLflow."""
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
    import mlflow

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
    import mlflow

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
        mlflow.set_tracking_uri(None)  # type: ignore[arg-type]  # None restores the default URI at runtime
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
    import mlflow

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
    import mlflow

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
