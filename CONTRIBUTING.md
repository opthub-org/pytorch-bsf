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

pytorch-bsf is a Python library for Bézier Simplex Fitting. Development follows these principles:

- **Simple API**: Keep the API minimal so users can get started with a single call to `torch_bsf.fit()`.
- **Backward compatibility**: Do not break existing public APIs in minor releases. Reserve breaking changes for major version bumps.
- **Type safety**: Add type hints to all public APIs (PEP 561 compliant; `py.typed` marker is included).
- **Test coverage**: Every new feature or bug fix must be accompanied by tests.
- **Documentation sync**: Changes to the public API must include updates to the Sphinx docs and docstrings.

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
| `black` | Code formatting |
| `isort` | Import sorting |

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
test: add parametrized tests for BezierSimplex
ci: add Python 3.14 to test matrix
```

---

## Coding Conventions

### Formatters

Format your code before committing:

```bash
# Sort imports
isort torch_bsf/ tests/

# Format code (use 127 characters to match the flake8 line-length setting)
black --line-length 127 torch_bsf/ tests/
```

### Linter

CI runs flake8. You can run the same checks locally:

```bash
# Check for syntax errors and undefined names (same settings as CI)
flake8 --select=E9,F63,F7,F82 --show-source --statistics torch_bsf/ tests/

# General code style check
flake8 --max-complexity=10 --max-line-length=127 torch_bsf/ tests/
```

### Type checking

Add type hints to all public APIs. Use mypy for static analysis:

```bash
mypy torch_bsf/
```

Configuration in `pyproject.toml` (`[tool.mypy]` section):

- `python_version = 3.10` (type checking targets the minimum supported version)
- `warn_return_any = True`
- `warn_unused_configs = True`

### Style guidelines

- **Indentation**: 4 spaces
- **Maximum line length**: 127 characters (matches `flake8 --max-line-length=127`; run `black` with `--line-length 127` accordingly)
- **Strings**: Double quotes (black default)
- **Docstrings**: NumPy style (use ``Parameters\n----------``, ``Returns\n-------``, ``Raises\n------`` sections)
- **Type hints**: Use Python 3.10 built-in type hint syntax without `from __future__ import annotations`

### "Bezier" vs "Bézier" spelling

Use the spelling consistently depending on the context:

| Spelling | Where to use |
|----------|-------------|
| `Bezier` (no accent) | Python identifiers (`BezierSimplex`, `bezier_simplex`, `plot_bezier_simplex`, …) and any string that may appear in a console (CLI `description`/`help` strings, `print()` output, log/warning/error messages) |
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
pytest tests/bezier_simplex.py
```

### Naming conventions

- Test file names mirror the module under test (e.g., `torch_bsf/bezier_simplex.py` → `tests/bezier_simplex.py`).
- Test function names start with `test_` or `_test_`.

### Writing parametrized tests

Use `@pytest.mark.parametrize` to cover multiple input combinations:

```python
import pytest
import torch_bsf as tbsf


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    [
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ],
)
def test_zeros(n_params: int, n_values: int, degree: int) -> None:
    bs = tbsf.BezierSimplex.zeros(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    assert bs.degree == degree
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
    BezierSimplex
        A trained Bézier simplex.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf
    >>> params = torch.tensor([[0.0], [0.5], [1.0]])
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
│   │   ├── python-package.yml        # pytest + flake8 + doctest
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
│   └── dependabot.yml                # Automated dependency updates
├── torch_bsf/                        # Main package
│   ├── __init__.py                   # Public API exports (BezierSimplex, fit)
│   ├── __main__.py                   # CLI entry point
│   ├── bezier_simplex.py             # BezierSimplex class, fit function, DataModule
│   ├── control_points.py             # Control point management and index calculation
│   ├── preprocessing.py              # Data scaling (MinMax, Std, Quantile, None)
│   ├── sampling.py                   # Sampling utilities
│   ├── validator.py                  # Input validation
│   ├── py.typed                      # PEP 561 type hint marker
│   └── model_selection/              # Model selection sub-package
│       ├── __init__.py
│       ├── kfold.py                  # K-fold cross-validation CLI
│       └── elastic_net_grid.py       # Elastic net grid search CLI
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── bezier_simplex.py             # Unit tests for BezierSimplex
│   ├── control_points.py             # Unit tests for ControlPoints
│   ├── validator.py                  # Unit tests for Validator
│   ├── test_control_point_index_format.py
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
├── MLproject                         # MLflow project definition
├── environment.yml                   # Conda environment definition
├── pyproject.toml                    # Package configuration, dependencies, tool settings
└── examples/                         # Example scripts
    └── quickstart/
        └── run.sh                    # Quickstart shell script
```
