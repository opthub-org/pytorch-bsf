"""Keep documentation source links tied to their build's revision and tree."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configuration():
    # Loading conf.py does not require Sphinx or optional integrations.
    return runpy.run_path(str(ROOT / "docs" / "conf.py"))


@pytest.mark.parametrize("ref", ["master", "v0.23.0", "docs/version-links"])
def test_source_links_use_documented_tree(configuration, tmp_path, ref):
    package = tmp_path / "torch_bsf"
    package.mkdir()
    (package / "fitting.py").write_text("\n" * 40 + "def fit():\n    pass\n", encoding="utf-8")
    configuration["_configure_source_links"](
        SimpleNamespace(srcdir=tmp_path / "docs"), SimpleNamespace(smv_current_version=ref)
    )
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert url == f"https://github.com/opthub-org/pytorch-bsf/blob/{ref}/torch_bsf/fitting.py#L41-L42"


def test_source_links_use_master_without_multiversion(configuration):
    configuration["_configure_source_links"](SimpleNamespace(srcdir=ROOT / "docs"), SimpleNamespace(smv_current_version=""))
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert "/blob/master/torch_bsf/fitting.py#L" in url


def test_source_links_include_property_decorator(configuration, tmp_path):
    package = tmp_path / "torch_bsf"
    package.mkdir()
    (package / "core.py").write_text(
        "class BezierModel:\n    @property\n    def n_outputs(self):\n        return 1\n", encoding="utf-8"
    )
    configuration["_configure_source_links"](
        SimpleNamespace(srcdir=tmp_path / "docs"), SimpleNamespace(smv_current_version="v0.23.0")
    )
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.core", "fullname": "BezierModel.n_outputs"})
    assert url.endswith("/torch_bsf/core.py#L2-L4")


def test_source_links_fall_back_without_guessing_lines(configuration, tmp_path):
    configuration["_configure_source_links"](
        SimpleNamespace(srcdir=tmp_path / "docs"), SimpleNamespace(smv_current_version="v0.23.0")
    )
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert url == "https://github.com/opthub-org/pytorch-bsf/blob/v0.23.0/torch_bsf/fitting.py"
    assert configuration["linkcode_resolve"]("js", {"module": "torch_bsf.fitting"}) is None
    assert configuration["linkcode_resolve"]("py", {"module": ""}) is None


def test_source_links_register_config_hook(configuration):
    events = []
    app = SimpleNamespace(connect=lambda event, callback: events.append((event, callback)))
    configuration["setup"](app)
    assert events == [("config-inited", configuration["_configure_source_links"])]
