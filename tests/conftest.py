"""Fixtures shared by the tests."""

import sys

import pytest


@pytest.fixture
def no_mlflow(monkeypatch):
    """Makes ``import mlflow`` (and its submodules) fail as if MLflow, an optional dependency, were not installed."""
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:  # undo the patches of Lightning by the autologging of earlier tests, which import mlflow
        mlflow.autolog(disable=True, silent=True)
    for name in [m for m in sys.modules if m == "mlflow" or m.startswith("mlflow.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mlflow", None)
