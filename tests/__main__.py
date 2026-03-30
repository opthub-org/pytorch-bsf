"""Tests for torch_bsf.__main__ (the CLI entry point)."""

import subprocess
import sys
from pathlib import Path
import os

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_PARAMS_CSV = _REPO_ROOT / "params.csv"
_VALUES_CSV = _REPO_ROOT / "values.csv"
_CLI_TIMEOUT = 300  # seconds


def _run_cli(*args, cwd=None):
    """Run `python -m torch_bsf` with the given arguments."""
    workdir = cwd or _REPO_ROOT
    env = os.environ.copy()
    # Force MLflow to use a local file-based backend under the working directory
    env["MLFLOW_TRACKING_URI"] = f"file://{workdir}"
    # Remove any external MLflow credentials to keep tests hermetic
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN"):
        env.pop(var, None)
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
    """CLI should raise when neither --degree nor --init is given."""
    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--loglevel=0",
    )
    assert result.returncode != 0


def test_cli_both_degree_and_init_fails(tmp_path):
    """CLI should raise when both --degree and --init are specified."""
    import torch_bsf.bezier_simplex as tbbs

    init_file = tmp_path / "model.pt"
    tbbs.save(str(init_file), tbbs.randn(n_params=2, n_values=2, degree=1))

    result = _run_cli(
        f"--params={_PARAMS_CSV}",
        f"--values={_VALUES_CSV}",
        "--degree=1",
        f"--init={init_file}",
        "--loglevel=0",
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_cli_init_flag(tmp_path):
    """CLI should accept --init to load an existing model."""
    import torch_bsf.bezier_simplex as tbbs

    init_file = tmp_path / "model.pt"
    tbbs.save(str(init_file), tbbs.randn(n_params=2, n_values=2, degree=1))

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
