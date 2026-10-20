"""Tests for torch_bsf.sklearn when scikit-learn is unavailable."""
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
    finally:
        # Restore original state.
        sklearn_mod._sklearn_available = original_available
