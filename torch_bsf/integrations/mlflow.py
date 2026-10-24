"""Explicit MLflow recording, independent of core numerical execution."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch_bsf.io import save_model
from torch_bsf.results import FitResult


def log_result(result: FitResult, *, tracking_uri: str | None = None) -> str:
    """Record metrics, diagnostics and the versioned model artifact in a run.

    Reuse an active run; otherwise start and finish one. Requires the mlflow extra.
    Returns the run ID. No autologging or pickled Python models are used.
    Unavailable placeholder models are never written; failure metrics and
    diagnostics are still recorded with ``bsf.solution_available=False``.
    """
    try:
        import mlflow
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        raise ImportError('Install "pytorch-bsf[mlflow]" to record results.') from error
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    active = mlflow.active_run()
    if active is not None:
        return _record(mlflow, result, active.info.run_id)
    with mlflow.start_run() as run:
        return _record(mlflow, result, run.info.run_id)


def _record(mlflow: Any, result: FitResult, run_id: str) -> str:
    mlflow.log_metrics(result.metrics)
    mlflow.set_tags(
        {
            "bsf.success": str(result.success),
            "bsf.status": result.status,
            "bsf.solution_available": str(result.diagnostics.solution_available),
        }
    )
    mlflow.log_dict(
        json.loads(json.dumps(asdict(result.diagnostics), default=lambda value: value.tolist())), "diagnostics.json"
    )
    if not result.diagnostics.solution_available:
        return run_id
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.bsf"
        save_model(path, result.model)
        mlflow.log_artifact(str(path))
    return run_id
