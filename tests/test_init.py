"""Tests for torch_bsf.__init__ (version fallback and public API)."""

import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest


def test_version_attribute_exists():
    """torch_bsf.__version__ should be a non-empty string."""
    import torch_bsf

    assert isinstance(torch_bsf.__version__, str)
    assert len(torch_bsf.__version__) > 0


def test_version_fallback_on_package_not_found():
    """When the package metadata is missing, __version__ falls back to 'unknown'."""
    import torch_bsf

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("pytorch-bsf")):
        importlib.reload(torch_bsf)

    assert torch_bsf.__version__ == "unknown"

    # Restore by reloading without the mock so subsequent tests are unaffected.
    importlib.reload(torch_bsf)


def test_public_api_exports():
    """torch_bsf should export the documented public symbols."""
    import torch_bsf

    for name in ("BezierSimploid", "fit", "fit_kfold", "split", "SimploidalSet", "fit_simploidal_set"):
        assert hasattr(torch_bsf, name)
    assert not hasattr(torch_bsf, "fit_simploid")


@pytest.mark.parametrize("module", ["torch_bsf.bezier_simplex", "torch_bsf.control_points"])
def test_legacy_modules_are_removed(module):
    """The Bézier simplex is a Bézier simploid of one factor; the separate implementation is gone."""
    import importlib.util

    assert importlib.util.find_spec(module) is None
