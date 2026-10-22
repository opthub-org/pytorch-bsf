import importlib.util

import pytest

#: Runs a test only if Lightning, the optional dependency of ``solver="trainer"``, is installed.
requires_lightning = pytest.mark.skipif(
    importlib.util.find_spec("lightning") is None, reason='needs Lightning (pip install "pytorch-bsf[lightning]")'
)
