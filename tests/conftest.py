"""Fixtures shared by the tests, and the Hypothesis profiles of the property-based tests."""

import os
import sys

import pytest
from hypothesis import settings

# Property-based tests run with the profile named by the HYPOTHESIS_PROFILE environment variable:
# - "ci" (default): few examples, derandomized so that every run (and CI) tests the same inputs, no deadline
#   because CI machines vary in speed, and no example database, which would make runs depend on earlier ones.
# - "thorough": many random examples for local bug hunting, e.g. `HYPOTHESIS_PROFILE=thorough pytest tests/splitting.py`.
#   Failing examples are saved in .hypothesis/ and replayed first next time; pin them with @example.
settings.register_profile("ci", max_examples=50, derandomize=True, deadline=None, database=None, print_blob=True)
settings.register_profile("thorough", max_examples=2000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture
def no_mlflow(monkeypatch):
    """Makes ``import mlflow`` (and its submodules) fail as if MLflow, an optional dependency, were not installed."""
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:  # undo the patches of Lightning by the autologging of earlier tests, which import mlflow
        mlflow.autolog(disable=True, silent=True)
    for name in [m for m in sys.modules if m == "mlflow" or m.startswith("mlflow.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mlflow", None)
