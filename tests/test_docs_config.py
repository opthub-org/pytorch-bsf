"""Keep documentation source links and version links tied to their build's ref and tree."""

import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = "PYTORCH_BSF_DOCS_VERSION"


@pytest.fixture
def configuration(monkeypatch):
    # Loading conf.py does not require Sphinx or optional integrations.
    monkeypatch.delenv(VARIABLE, raising=False)
    return runpy.run_path(str(ROOT / "docs" / "conf.py"))


def _configure(configuration, monkeypatch, srcdir, ref):
    if ref:
        monkeypatch.setenv(VARIABLE, ref)
    else:
        monkeypatch.delenv(VARIABLE, raising=False)
    configuration["_configure_source_links"](SimpleNamespace(srcdir=srcdir), SimpleNamespace())


@pytest.mark.parametrize("ref", ["master", "v0.23.0", "docs/version-links"])
def test_source_links_use_documented_tree(configuration, monkeypatch, tmp_path, ref):
    package = tmp_path / "torch_bsf"
    package.mkdir()
    (package / "fitting.py").write_text("\n" * 40 + "def fit():\n    pass\n", encoding="utf-8")
    _configure(configuration, monkeypatch, tmp_path / "docs", ref)
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert url == f"https://github.com/opthub-org/pytorch-bsf/blob/{ref}/torch_bsf/fitting.py#L41-L42"


def test_local_build_links_to_master_without_version_menu(configuration, monkeypatch):
    _configure(configuration, monkeypatch, ROOT / "docs", "")
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert "/blob/master/torch_bsf/fitting.py#L" in url
    context: dict[str, Any] = {}
    configuration["_add_version_links"](None, "index", "page.html", context, None)
    assert context == {}


def test_source_links_include_property_decorator(configuration, monkeypatch, tmp_path):
    package = tmp_path / "torch_bsf"
    package.mkdir()
    (package / "core.py").write_text(
        "class BezierModel:\n    @property\n    def n_outputs(self):\n        return 1\n", encoding="utf-8"
    )
    _configure(configuration, monkeypatch, tmp_path / "docs", "v0.23.0")
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.core", "fullname": "BezierModel.n_outputs"})
    assert url.endswith("/torch_bsf/core.py#L2-L4")


def test_source_links_fall_back_without_guessing_lines(configuration, monkeypatch, tmp_path):
    _configure(configuration, monkeypatch, tmp_path / "docs", "v0.23.0")
    url = configuration["linkcode_resolve"]("py", {"module": "torch_bsf.fitting", "fullname": "fit"})
    assert url == "https://github.com/opthub-org/pytorch-bsf/blob/v0.23.0/torch_bsf/fitting.py"
    assert configuration["linkcode_resolve"]("js", {"module": "torch_bsf.fitting"}) is None
    assert configuration["linkcode_resolve"]("py", {"module": ""}) is None


def test_setup_registers_source_and_version_hooks(configuration):
    events = []
    app = SimpleNamespace(connect=lambda event, callback: events.append((event, callback)))
    configuration["setup"](app)
    assert events == [
        ("config-inited", configuration["_configure_source_links"]),
        ("html-page-context", configuration["_add_version_links"]),
    ]


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_release_tags_are_newest_first_by_version_number(configuration, tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", "c")
    for tag in ["v0.2.0", "v0.10.0", "v0.9.1", "v1.0.0-rc.1", "v1.0", "vx", "release-1", "v0.10.1"]:
        _git(tmp_path, "tag", tag)
    assert configuration["_release_tags"](tmp_path) == ["v0.10.1", "v0.10.0", "v0.9.1", "v0.2.0"]


def test_release_tags_are_empty_outside_git(configuration, tmp_path):
    assert configuration["_release_tags"](tmp_path / "missing") == []


def test_release_tags_are_empty_without_git(configuration, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert configuration["_release_tags"](ROOT) == []


@pytest.mark.parametrize(
    ("current", "branches"),
    [("master", ["master"]), ("v0.2.0", ["master"]), ("fix/topic", ["master", "fix/topic"])],
)
def test_version_menu_lists_master_and_the_built_branch(configuration, current, branches):
    switcher = configuration["_version_switcher"](current, ["v0.2.0", "v0.1.0"])
    assert switcher == {"current": current, "tags": ["v0.2.0", "v0.1.0"], "branches": branches}


def test_version_menu_is_absent_without_a_ref(configuration):
    assert configuration["_version_switcher"]("", ["v0.1.0"]) is None


@pytest.mark.parametrize(
    ("current", "pagename", "up"),
    [
        ("master", "index", "../"),
        ("master", "applications/index", "../../"),
        ("v0.2.0", "faq", "../"),
        ("fix/topic", "advanced/guides/x", "../../../../"),
    ],
)
def test_version_links_open_the_same_page_or_the_index(configuration, current, pagename, up):
    links = configuration["_version_links"](configuration["_version_switcher"](current, ["v0.1.0"]), pagename)
    assert links["current_version"] == {"name": current}
    assert links["versions"]["tags"] == [
        {"name": "v0.1.0", "url": f"{up}v0.1.0/{pagename}.html", "fallback": f"{up}v0.1.0/index.html"}
    ]
    assert links["versions"]["branches"][0] == {
        "name": "master",
        "url": f"{up}master/{pagename}.html",
        "fallback": f"{up}master/index.html",
    }


def test_version_links_quote_unusual_ref_names(configuration):
    links = configuration["_version_links"](configuration["_version_switcher"]("fix/a#b", []), "index")
    branch = links["versions"]["branches"][1]
    assert branch == {"name": "fix/a#b", "url": "../../fix/a%23b/index.html", "fallback": "../../fix/a%23b/index.html"}


def test_version_links_are_added_to_each_page(configuration, monkeypatch):
    _configure(configuration, monkeypatch, ROOT / "docs", "master")
    context: dict[str, Any] = {"other": 1}
    configuration["_add_version_links"](None, "faq", "page.html", context, None)
    assert context["other"] == 1
    assert context["current_version"] == {"name": "master"}
    assert context["versions"]["branches"] == [
        {"name": "master", "url": "../master/faq.html", "fallback": "../master/index.html"}
    ]
