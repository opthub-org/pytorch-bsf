# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import ast
import importlib.util
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import torch_bsf

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "PyTorch-BSF"
copyright = "2021, OptHub Inc. and FUJITSU LIMITED"
author = "OptHub Inc. and FUJITSU LIMITED"

# The full version, including alpha/beta/rc tags
release = version = torch_bsf.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_rtd_theme",
    "sphinx_proof",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]

# -- autodoc -----------------------------------------------------------------
autodoc_inherit_docstrings = False
# Optional dependencies that may not be present in the docs build environment.
# These modules are imported lazily (inside functions) but autodoc may still
# try to resolve type annotations that reference them.
# Keep installed integrations real: mocking sklearn's base classes while
# importing our estimator would otherwise break its later doctest examples.
autodoc_mock_imports = [name for name in ("matplotlib", "sklearn", "scipy") if importlib.util.find_spec(name) is None]

# -- doctest -----------------------------------------------------------------
# https://www.sphinx-doc.org/ja/master/usage/extensions/doctest.html#confval-doctest_global_setup
import doctest  # noqa: E402

doctest_default_flags = doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | getattr(doctest, "DONT_ACCEPT_TRUE_FOR_1", 0)
doctest_global_setup = """
import torch
import torch_bsf
import contextlib
import os

if not hasattr(torch_bsf,"_original_fit"):
    torch_bsf._original_fit = torch_bsf.fit
    # 1>/dev/null 2>&1
    def quiet_fit(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return torch_bsf._original_fit(*args, **kwargs)
    torch_bsf.fit = quiet_fit
"""
doctest_test_doctest_blocks = "default"

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "pl": ("https://lightning.ai/docs/pytorch/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The RTD sidebar has a colored background; use the reversed compact wordmark.
html_logo = "_static/logo-compact-white.svg"
html_favicon = "_static/favicon.ico"


# -- Version switcher and source links --------------------------------------

# The GHPages producer builds only the pushed ref and names it in this variable:
# a branch such as ``master`` or a release tag such as ``v1.2.3``. The name is also
# the ref's URL root on GitHub Pages. A local build leaves it unset, so its source
# links document master and it shows no version switcher.
DOCS_VERSION_VARIABLE = "PYTORCH_BSF_DOCS_VERSION"
_RELEASE_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")

# https://www.sphinx-doc.org/ja/master/usage/extensions/linkcode.html
# Source links follow the documented ref and tree, independently of the
# installed package's version.
_source_ref = "master"
_source_root = Path(__file__).resolve().parent.parent
_switcher = None


def _release_tags(root):
    """Return the release tags of ``root``, newest first, or none outside Git."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "tag", "--list", "v*"], check=True, capture_output=True, text=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []
    releases = [(tuple(map(int, match.groups())), tag) for tag in listed if (match := _RELEASE_TAG.fullmatch(tag))]
    return [tag for _, tag in sorted(releases, reverse=True)]


def _version_switcher(current, tags):
    """Return the refs listed under Other Versions, or ``None`` for a local build.

    Tags are the release tags known when this ref is built. Branches are master and
    the branch being built, because other branch previews come and go.
    """
    if not current:
        return None
    branches = ["master"]
    if current != "master" and current not in tags:
        branches.append(current)
    return {"current": current, "tags": tags, "branches": branches}


def _configure_source_links(app, config):
    global _source_ref, _source_root, _switcher
    current = os.environ.get(DOCS_VERSION_VARIABLE, "")
    _source_ref = current or "master"
    _source_root = Path(app.srcdir).resolve().parent
    _switcher = _version_switcher(current, _release_tags(_source_root))


def _version_links(switcher, pagename):
    """Link each listed ref to this page, and to its index for a page it lacks.

    Every ref is published under its own root, so a link climbs out of this page's
    directory and this ref's root before entering the other ref's root.
    """
    up = "../" * (pagename.count("/") + len(PurePosixPath(switcher["current"]).parts))

    def link(name):
        root = up + quote(name, safe="/")
        return {"name": name, "url": f"{root}/{quote(pagename, safe='/')}.html", "fallback": f"{root}/index.html"}

    return {
        "current_version": {"name": switcher["current"]},
        "versions": {kind: [link(name) for name in switcher[kind]] for kind in ("tags", "branches")},
    }


def _add_version_links(app, pagename, templatename, context, doctree):
    if _switcher is not None:
        context.update(_version_links(_switcher, pagename))


def _definition_lines(path, fullname):
    """Find lines in the documented tree, not an imported newer checkout."""
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    for name in fullname.split("."):
        node = next(
            item
            for item in nodes
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
        )
        nodes = node.body
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    return start, node.end_lineno


def _source_file(obj):
    """Return the imported file that defines ``obj``.

    A class re-exported under its public module (``__module__`` rewritten) would
    resolve to that module, so look for the loaded submodule that defines it.
    """
    if inspect.isclass(obj):
        for name, module in list(sys.modules.items()):
            if not name.startswith("torch_bsf.") or vars(module).get(obj.__name__) is not obj:
                continue
            path = Path(inspect.getsourcefile(module)).resolve()
            try:
                _definition_lines(path, obj.__qualname__)
            except (StopIteration, SyntaxError, OSError):
                continue
            return path
    return Path(inspect.getsourcefile(obj)).resolve()


def linkcode_resolve(domain, info):
    """Link a public Python object to the same ref and source lines as its docs."""
    if domain != "py" or not info.get("module"):
        return None
    filename = info["module"].replace(".", "/") + ".py"
    fragment = ""
    try:
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        if isinstance(obj, property):
            obj = obj.fget
        obj = inspect.unwrap(obj)
        imported_file = _source_file(obj)
        relative = imported_file.relative_to(Path(torch_bsf.__file__).resolve().parent)
        defining = (Path("torch_bsf") / relative).as_posix()
        # Keep the public file unless the documented tree has the defining one
        # (an older tag predates the private modules).
        start, end = _definition_lines(_source_root / defining, obj.__qualname__)
        filename, fragment = defining, f"#L{start}-L{end}"
    except (AttributeError, KeyError, OSError, TypeError, ValueError, StopIteration, SyntaxError):
        # Some generated members have no standalone definition. The file link
        # still points to the correct ref, without guessing newer line numbers.
        pass
    return f"https://github.com/opthub-org/pytorch-bsf/blob/{quote(_source_ref, safe='/')}/{filename}{fragment}"


def setup(app):
    app.connect("config-inited", _configure_source_links)
    app.connect("html-page-context", _add_version_links)
