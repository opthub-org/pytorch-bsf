"""Tests for torch_bsf.sklearn when scikit-learn is unavailable."""

import importlib.util
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


def test_sklearn_unavailable_import_succeeds(monkeypatch):
    """torch_bsf.sklearn imports without scikit-learn, and the regressor asks for it (#409)."""
    import torch_bsf.sklearn as sklearn_mod

    # Load a separate copy of the module while `import sklearn` raises ImportError,
    # so that the already imported torch_bsf.sklearn is left untouched.
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.base", None)
    spec = importlib.util.spec_from_file_location("_torch_bsf_sklearn_without_sklearn", sklearn_mod.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not module._sklearn_available
    with pytest.raises(ImportError, match="scikit-learn"):
        module.BezierSimploidRegressor()
