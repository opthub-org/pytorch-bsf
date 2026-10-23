"""Tests for torch_bsf.sklearn when scikit-learn is unavailable."""

import subprocess
import sys

import pytest


def test_sklearn_unavailable_check_sklearn_raises():
    """_check_sklearn() should raise ImportError when sklearn is not available."""
    import torch_bsf.sklearn as sklearn_mod

    # Save original state.
    original_available = sklearn_mod._sklearn_available
    try:
        # Simulate sklearn being unavailable.
        sklearn_mod._sklearn_available = False
        with pytest.raises(ImportError, match="scikit-learn"):
            sklearn_mod._check_sklearn()
        with pytest.raises(ImportError, match="scikit-learn"):
            sklearn_mod.BezierSimploidRegressor()
    finally:
        # Restore original state.
        sklearn_mod._sklearn_available = original_available


def test_sklearn_unavailable_import_succeeds():
    """torch_bsf.sklearn imports without scikit-learn, and the regressor asks for it (#409)."""
    code = """
import sys
sys.modules["sklearn"] = None  # make `import sklearn` raise ImportError
import torch_bsf.sklearn as sklearn_mod
assert not sklearn_mod._sklearn_available
try:
    sklearn_mod.BezierSimploidRegressor()
except ImportError as e:
    assert "scikit-learn" in str(e), e
else:
    raise AssertionError("BezierSimploidRegressor() did not raise ImportError")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
