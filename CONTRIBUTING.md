# Contributing Guide

This document is a guide for contributing to pytorch-bsf.

## Table of Contents

1. [Development Policy](#development-policy)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [First Things to Try](#first-things-to-try)
4. [Development Workflow](#development-workflow)
5. [Commit Rules](#commit-rules)
6. [Coding Conventions](#coding-conventions)
7. [Writing Tests](#writing-tests)
8. [Repository Structure](#repository-structure)

---

## Development Policy

pytorch-bsf is a Python library for Bézier Simploid Fitting (formerly Bézier Simplex Fitting). Development follows these principles:

- **Simple API**: Keep the API minimal so users can get started with a single call to `torch_bsf.fit()`.
- **Backward compatibility**: Do not break existing public APIs in minor releases. Reserve breaking changes for major version bumps.
- **Type safety**: Add type hints to all public APIs (PEP 561 compliant; `py.typed` marker is included).
- **Test coverage**: Every new feature or bug fix must be accompanied by tests.
- **Documentation sync**: Changes to the public API must include updates to the Sphinx docs and docstrings.

### Dependency versions

pytorch-bsf is a library installed alongside other packages, so `[project].dependencies` in `pyproject.toml` declares the widest ranges that work, not the latest versions:

- **Lower bound**: the oldest version that passes the tests. The `lowest` job of `.github/workflows/python-package.yml` installs every lower bound on Python 3.10 (`uv pip install --resolution lowest-direct '.[lightning,mlflow]'`, so the bounds of the optional `lightning` and `mlflow` are checked too) and runs pytest, so a bound that does not work fails CI. Raise a lower bound only when the code needs a newer feature or a bug fix, and say why in the PR.
- **Upper bound**: below the next major version (`<3`). Widen it after checking that the new major version works.
- **Dependabot** uses `versioning-strategy: widen`: it opens a PR only when a new release falls outside a range, and never raises a lower bound.
- Declare only packages that `torch_bsf` imports. Do not pin transitive dependencies to work around a bug in another package; if one is unavoidable, add a comment with the upstream issue.
- Keep `environment.yml` in sync with `pyproject.toml`.

---

## Setting Up the Development Environment

### Prerequisites

- Python 3.10 or later
- [Conda](https://docs.conda.io/) or pip

### With Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate pytorch-bsf
```

### With pip

```bash
# Clone the repository
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# Install in editable mode with development dependencies
pip install -e ".[develop]"
```

The `[develop]` extras include:

| Package | Purpose |
|---------|---------|
| `pytest`, `pytest-cov`, `pytest-randomly` | Running tests |
| `mypy` + type stubs | Static type checking |
| `ruff` | Linting, import sorting and code formatting |
| `pre-commit` | Running Ruff automatically on `git commit` |

With Conda, install them into the environment with `pip install -e ".[develop]"` as well.

MLflow is an optional extra (`pip install -e ".[develop,mlflow]"`, also included in `all`). Without it, the CLI logs nothing and the MLflow tests are skipped; the `no-mlflow` CI job runs the tests that way.

### Install the git hook

Install the [pre-commit](https://pre-commit.com/) hook so that Ruff checks and formats your changes on every commit:

```bash
pre-commit install
```

Run this once per clone. The hook is installed in the shared `.git/hooks` directory, so it also runs in every `git worktree` of the clone (including `.claude/worktrees/`). If a hook fixes or reformats files, the commit is aborted; review the changes, `git add` them and commit again.

---

## First Things to Try

After setting up the environment, run the following to verify everything works.

### 1. Run the tests

```bash
pytest tests/
```

All tests should pass.

### 2. Try the CLI

The repository includes sample data (`params.csv`, `values.csv`):

```bash
python -m torch_bsf --params=params.csv --values=values.csv --degree=3
```

### 3. Run the quickstart script

```bash
bash examples/quickstart/run.sh
```

### 4. Run with MLflow

```bash
mlflow run . --entry-point main \
  -P params=params.csv \
  -P values=values.csv \
  -P degree=3
```

---

## Development Workflow

### Filing issues

Issues are written so that a developer or an AI agent can implement them and open a PR without asking for more context. Use one of the issue forms:

| Form | Use for | Label |
|------|---------|-------|
| **Bug report** | Incorrect behavior | `bug` |
| **Feature / improvement** | New features, API changes, refactoring, tests, performance, CI, tooling, dependencies | `enhancement` |
| **Documentation** | Sphinx docs, README, docstrings, examples | `documentation` |

- Fill in **Acceptance criteria** with concrete, checkable items; project-wide checks (tests, lint, type check, doctest, docs build) are implied.
- Name the files, modules or APIs involved when you know them, and state what is out of scope.
- Record decisions already made (backward compatibility, design choices) so the implementer does not have to guess.
- Link related issues or PRs to keep discussions focused.
- When filing from the command line (`gh issue create --body-file`), keep the same `### <Field>` headings as the form.

### Creating branches

Branch off from `master`:

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

Use a branch name that combines the commit type, issue number, and a short description:

| Type | Example |
|------|---------|
| New feature | `feat/123-add-new-sampler` |
| Bug fix | `fix/456-fix-nan-in-output` |
| Documentation | `docs/789-update-contributing` |
| Refactoring | `refactor/101-simplify-validator` |
| CI / build | `ci/102-update-workflow` |

### PR review and merge

1. Push your branch and open a Pull Request targeting `master`.
2. In the PR description, include what changed, the related issue (`Closes #XXX`), and how to test it.
3. Ensure all CI checks (GitHub Actions) pass.
4. Address reviewer comments and merge once you have an approval.
5. Delete the branch after merging.

---

## Commit Rules

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. `release-please` uses this format to automate CHANGELOG generation and version management.

### Format

```
<type>(<scope>): <subject>
```

- `<scope>` is optional.
- Write `<subject>` in English using the imperative present tense (e.g., "add feature" ✓, "added feature" ✗).
- Keep the first line under 72 characters.

### Commit types

| Type | Description | Version impact |
|------|-------------|---------------|
| `feat` | Add a new feature | Minor version bump |
| `fix` | Fix a bug | Patch version bump |
| `docs` | Documentation changes only | None |
| `style` | Changes that do not affect behavior (whitespace, formatting) | None |
| `refactor` | Code change that is neither a fix nor a feature | None |
| `test` | Add or update tests | None |
| `chore` | Changes to the build process or auxiliary tools | None |
| `ci` | Changes to CI configuration | None |
| `deps` | Add or update dependencies | None |
| `perf` | Performance improvement | None |

### Breaking changes

For changes that break backward compatibility, add `BREAKING CHANGE:` to the commit body or footer. This triggers a major version bump automatically.

```
feat!: remove deprecated normalize parameter

BREAKING CHANGE: The `normalize` parameter has been removed.
Use `preprocessing` instead.
```

### Examples

```
feat(model_selection): add elastic net grid search
fix: resolve UnpicklingError in PyTorch 2.6+
docs: add CONTRIBUTING.md
deps: update starlette constraint for mlflow compatibility
test: add parametrized tests for BezierSimploid
ci: add Python 3.14 to test matrix
```

---

## Coding Conventions

### Linter and formatter

[Ruff](https://docs.astral.sh/ruff/) lints, sorts imports and formats code. The pre-commit hook (see [Install the git hook](#install-the-git-hook)) runs `ruff check --fix` and `ruff format` on the files you commit. CI runs the same hooks on all files and fails if they report a problem or change a file. To run them manually:

```bash
# Run the hooks on all files, as CI does
pre-commit run --all-files

# Or run Ruff directly
ruff check .           # lint and sort imports (add --fix to apply automatic fixes)
ruff format --check .  # check formatting (drop --check to rewrite files)
```

The Ruff version is pinned by `rev` in `.pre-commit-config.yaml`, which is the single source of truth: the git hook and CI both use it, and Dependabot opens a weekly PR to bump it. Run `pre-commit autoupdate` only if you need a newer Ruff before Dependabot does. The `ruff` in the `[develop]` extras is unpinned (for editors and direct use); if its results differ from the hook's, trust `pre-commit run --all-files`.

Configuration in `pyproject.toml` (`[tool.ruff]` section):

- `line-length = 127`
- `target-version = "py310"`
- `select = ["E", "F", "W", "I", "UP", "B", "SIM", "C90"]` (pycodestyle errors and warnings, Pyflakes, isort, pyupgrade, flake8-bugbear, flake8-simplify, McCabe complexity)
- `max-complexity = 10` (`[tool.ruff.lint.mccabe]` section): split a function whose cyclomatic complexity exceeds 10 into smaller helpers

`B905` requires an explicit `strict=` on every `zip()`: pass `strict=True` when the iterables must have the same length, and `strict=False` only when truncation is intended.

### Ignoring bulk formatting commits in git blame

Commits that only reformat code (for example, the switch to Ruff in #364) are listed in `.git-blame-ignore-revs` at the repository root, so that `git blame` shows the commit that actually wrote each line. GitHub's blame view reads this file automatically. To use it locally, run this once in your clone:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

After a PR whose main purpose is a repository-wide reformat (for example, a formatter upgrade or a new rule that rewrites many files) is merged, open a follow-up PR that appends its commit to `.git-blame-ignore-revs`: a comment line with the commit subject, then the full 40-character SHA. Use the squash-merged SHA on `master`, not a SHA from the PR branch. Keep reformatting PRs free of unrelated changes, because every line they touch is hidden from blame.

### Type checking

Add type hints to all functions in `torch_bsf`, public or private. Use mypy for static analysis of both the package and the tests (as CI does):

```bash
mypy torch_bsf tests
```

Configuration in `pyproject.toml` (`[tool.mypy]` section):

- `python_version = 3.10` (type checking targets the minimum supported version)
- `warn_return_any`, `warn_unused_configs`, `warn_unused_ignores`, `warn_redundant_casts`
- `disallow_untyped_defs`, `disallow_incomplete_defs`: every function in `torch_bsf` must be fully annotated
- `check_untyped_defs`: the bodies of unannotated functions (e.g., tests) are type-checked too
- `disallow_any_generics`: write `dict[str, Any]`, not a bare `dict`, and `npt.NDArray[Any]` (`import numpy.typing as npt`), not a bare `np.ndarray`, which older NumPy versions reject
- `[[tool.mypy.overrides]]` for `tests.*` turns off `disallow_untyped_defs` and `disallow_incomplete_defs`, so test functions and fixtures need no annotations
- Optional dependencies without type information (`mlflow`, `scipy`, `sklearn`, `matplotlib`, `mpl_toolkits`) use `ignore_missing_imports`

Every `# type: ignore` must name its error code, e.g. `# type: ignore[arg-type]` for a test that passes an invalid value on purpose.

### Style guidelines

- **Indentation**: 4 spaces
- **Maximum line length**: 127 characters (`line-length` in `[tool.ruff]`)
- **Strings**: Double quotes (`ruff format` default)
- **Docstrings**: NumPy style (use ``Parameters\n----------``, ``Returns\n-------``, ``Raises\n------`` sections)
- **Type hints**: Use Python 3.10 built-in type hint syntax without `from __future__ import annotations`

### "Bezier" vs "Bézier" spelling

Use the spelling consistently depending on the context:

| Spelling | Where to use |
|----------|-------------|
| `Bezier` (no accent) | Python identifiers (`BezierSimploid`, `fit_simploidal_set`, `plot_bezier_simplex`, …) and any string that may appear in a console (CLI `description`/`help` strings, `print()` output, log/warning/error messages) |
| `Bézier` (with accent) | Everything else: docstrings, RST/Markdown documentation, code comments, `pyproject.toml` metadata, example script module docstrings |

The rationale for keeping `Bezier` in console-facing text is that this library is used on HPC clusters where the locale may be `LANG=C`, causing non-ASCII characters to be displayed incorrectly or to raise encoding errors.

---

## Writing Tests

Tests live in the `tests/` directory.

### Running tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=torch_bsf --cov-report=html

# Run a specific file
pytest tests/simploidal.py
```

### Naming conventions

- Test file names mirror the module under test (e.g., `torch_bsf/splitting.py` → `tests/splitting.py`).
- Test function names start with `test_` or `_test_`.

### Writing parametrized tests

Use `@pytest.mark.parametrize` to cover multiple input combinations:

```python
import pytest
import torch_bsf as tbsf


@pytest.mark.parametrize(
    "dim, n_values, degree",
    [(dim, n_values, degree) for dim in range(1, 4) for n_values in range(1, 3) for degree in range(3)],
)
def test_zeros(dim: int, n_values: int, degree: int) -> None:
    bs = tbsf.BezierSimploid.zeros((dim,), degree, n_values)
    assert bs.n_params == dim + 1
    assert bs.n_values == n_values
    assert bs.degree == (degree,)
```

### Docstring tests

Write runnable examples in docstrings for all public APIs. They are verified automatically in CI:

```python
def fit(params, values, degree):
    """Fit a Bézier simplex to the given data.

    Parameters
    ----------
    params : torch.Tensor
        The parameter data on the simplex.
    values : torch.Tensor
        The label data.
    degree : int
        The degree of the Bézier simplex.

    Returns
    -------
    BezierSimploid
        A fitted Bézier simplex.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf
    >>> params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> values = torch.tensor([[0.0], [0.25], [1.0]])
    >>> bs = torch_bsf.fit(params=params, values=values, degree=2)
    """
```

### Sphinx doctest

Code blocks in RST files under `docs/` are also tested via Sphinx doctest:

```bash
sphinx-build -b doctest docs/ docs/_build/doctest
```

---

## Repository Structure

```
pytorch-bsf/
├── .github/
│   ├── workflows/                    # GitHub Actions workflows
│   │   ├── python-package.yml        # ruff (via pre-commit) + pytest + mypy + doctest; pytest on lowest deps
│   │   ├── release-please-action.yml # Automated releases (CHANGELOG + PyPI publish)
│   │   ├── sphinx-pages.yml          # Build and publish Sphinx docs to GitHub Pages
│   │   ├── codeql-analysis.yml       # Security scanning
│   │   └── python-package-conda.yml  # End-to-end tests in a Conda environment
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml            # Bug report form
│   │   ├── feature_request.yml       # Feature / improvement form
│   │   ├── documentation.yml         # Documentation form
│   │   └── config.yml                # Disables blank issues
│   ├── CODEOWNERS                    # Code ownership settings
│   └── dependabot.yml                # Automated dependency updates (pip and pre-commit hooks)
├── torch_bsf/                        # Main package
│   ├── __init__.py                   # Public API exports (BezierSimploid, fit, fit_kfold, split, …)
│   ├── __main__.py                   # CLI entry point
│   ├── _cli_args.py                  # Shared CLI arguments and data loading
│   ├── simploidal/                   # Bézier simploids (a Bézier simplex is the case of one factor)
│   │   ├── bezier.py                 # BezierSimploid model
│   │   ├── combinatorics.py          # Control point indices and Bernstein basis
│   │   ├── fitting.py                # fit, fit_kfold and fit_simploidal_set
│   │   ├── io.py                     # save and load
│   │   ├── sets.py                   # Simploidal sets
│   │   └── space.py                  # Bézier spaces on simploidal sets
│   ├── active_learning.py            # Query-by-committee point suggestion
│   ├── plotting.py                   # Visualization
│   ├── preprocessing.py              # Data scaling (MinMax, Std, Quantile, None)
│   ├── sampling.py                   # Sampling utilities
│   ├── sklearn.py                    # scikit-learn regressor
│   ├── splitting.py                  # Splitting Bézier simploids
│   ├── validator.py                  # Parsers of CLI arguments
│   ├── py.typed                      # PEP 561 type hint marker
│   └── model_selection/              # Model selection sub-package
│       ├── __init__.py
│       ├── degree_selection.py       # Degree selection by cross-validation
│       ├── kfold.py                  # K-fold cross-validation CLI
│       └── elastic_net_grid.py       # Elastic net grid search CLI
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── simploidal.py                 # Unit tests for torch_bsf.simploidal
│   ├── validator.py                  # Unit tests for Validator
│   ├── …                             # One file per module (see Naming conventions)
│   └── data/                         # Test data files
├── docs/                             # Sphinx documentation
│   ├── conf.py                       # Sphinx configuration
│   ├── index.rst                     # Documentation home page
│   ├── requirements.txt              # Documentation build dependencies
│   └── applications/                 # Application examples (RST)
├── examples/                         # Example scripts
│   └── quickstart/
│       └── run.sh                    # Quickstart shell script
├── CHANGELOG.md                      # Version history (auto-generated by release-please)
├── CONTRIBUTING.md                   # This file
├── LICENSE                           # MIT License
├── README.md                         # Project overview and usage
├── .pre-commit-config.yaml           # Git hooks (Ruff) run on commit and in CI
├── MLproject                         # MLflow project definition
├── environment.yml                   # Conda environment definition
├── pyproject.toml                    # Package configuration, dependencies, tool settings
└── examples/                         # Example scripts
    └── quickstart/
        └── run.sh                    # Quickstart shell script
```
