"""Tests for torch_bsf.model_selection.kfold (the k-fold CLI entry point)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
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
    """Run `python -m torch_bsf.model_selection.kfold` with the given arguments."""
    workdir = cwd or _REPO_ROOT
    env = os.environ.copy()
    # Force MLflow to use a local file-based backend under the working directory
    env["MLFLOW_TRACKING_URI"] = f"file://{workdir}"
    # Remove any external MLflow credentials to keep tests hermetic
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, "-m", "torch_bsf.model_selection.kfold"] + list(args),
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
        "--num_folds=2",
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
    assert "--simploid-type A1,...,An" in result.stdout
    assert "--stratified" not in result.stdout
    assert "--devices N|auto" in result.stdout


def test_cli_init_flag(tmp_path):
    """CLI should accept --init to load an existing model."""
    init_file = _init_file(tmp_path)

    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        f"--init={init_file}",
        "--num_folds=2",
        "--max_epochs=1",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_cli_output_files_created(tmp_path):
    """CLI should create per-fold and averaged meshgrid output files."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        "--num_folds=2",
        "--max_epochs=1",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    output_files = list(tmp_path.iterdir())
    assert len(output_files) > 0, "Expected output files to be created"


# ---------------------------------------------------------------------------
# In-process tests for main() to get coverage of the module
# ---------------------------------------------------------------------------


class TestKFoldMainInProcess:
    """Call torch_bsf.model_selection.kfold.main() directly with patched sys.argv."""

    def test_main_basic(self, tmp_path, monkeypatch):
        """main() should run k-fold training and save output CSV files."""
        from torch_bsf.model_selection.kfold import main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
        for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "kfold",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--degree=1",
                "--num_folds=2",
                "--max_epochs=1",
                "--loglevel=0",
                "--accelerator=cpu",
                "--devices=1",
            ],
        )
        main()
        output_files = list(tmp_path.iterdir())
        assert len(output_files) > 0, "Expected output files to be created"

    def test_main_with_init(self, tmp_path, monkeypatch):
        """main() should accept --init to load an existing model."""
        from torch_bsf.model_selection.kfold import main

        init_file = _init_file(tmp_path)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
        for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "kfold",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                f"--init={init_file}",
                "--num_folds=2",
                "--max_epochs=1",
                "--loglevel=0",
                "--accelerator=cpu",
                "--devices=1",
            ],
        )
        main()
        output_files = list(tmp_path.iterdir())
        assert len(output_files) > 0, "Expected output files to be created"


def test_main_writes_meshgrids_of_every_fold(tmp_path, monkeypatch):
    """main() writes the meshgrid of every fold and their mean, tagged with the type and degree."""
    import numpy as np

    from torch_bsf.model_selection.kfold import main

    params = np.array([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.2, 0.8], [0.0, 1.0, 0.6, 0.4], [0.3, 0.7, 0.0, 1.0]])
    np.savetxt(tmp_path / "p.csv", params, delimiter=",")
    np.savetxt(tmp_path / "v.csv", params[:, :1] * params[:, 2:3], delimiter=",")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["kfold", "--params=p.csv", "--values=v.csv", "--degree=1", "--simploid-type=1,1", "--num_folds=2", "--loglevel=0"],
    )
    main()
    prefix = "p.csv,v.csv,2fold,meshgrid,t_1-1,d_1"
    folds = [np.loadtxt(f"{prefix},{k}.csv", delimiter=",", ndmin=2) for k in range(2)]
    np.testing.assert_allclose(np.loadtxt(f"{prefix}.csv", delimiter=",", ndmin=2), (folds[0] + folds[1]) / 2)


def test_main_rejects_split_ratio(tmp_path, monkeypatch, capsys):
    from torch_bsf.model_selection.kfold import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["kfold", f"--params={_PARAMS_CSV}", f"--values={_VALUES_CSV}", "--degree=1", "--split_ratio=0.5"]
    )
    with pytest.raises(SystemExit):
        main()
    assert "--split_ratio is not supported" in capsys.readouterr().err
