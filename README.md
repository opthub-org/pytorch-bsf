# PyTorch-BSF

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opthub-org/pytorch-bsf/master/docs/_static/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/opthub-org/pytorch-bsf/master/docs/_static/logo.svg" width="448" alt="PyTorch-BSF — Bézier Simploid Fitting">
</picture>

[![PyPI](https://img.shields.io/pypi/v/pytorch-bsf.svg)](https://pypi.org/project/pytorch-bsf/)
[![Python versions](https://img.shields.io/pypi/pyversions/pytorch-bsf.svg)](https://pypi.org/project/pytorch-bsf/)
[![PyTest](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package.yml)
[![Documentation](https://github.com/opthub-org/pytorch-bsf/actions/workflows/sphinx-pages.yml/badge.svg)](https://opthub-org.github.io/pytorch-bsf/)
[![codecov](https://codecov.io/gh/opthub-org/pytorch-bsf/graph/badge.svg?token=0BNDP4QUVT)](https://codecov.io/gh/opthub-org/pytorch-bsf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/opthub-org/pytorch-bsf/blob/master/LICENSE)

**Bézier Simploid Fitting in PyTorch.**

PyTorch-BSF is a Python library for fitting parametric curves, surfaces and
higher-dimensional models to data. It is useful for approximating Pareto fronts
in multi-objective optimization, building surrogate response models and
interpolating observations on meshes.

A Bézier simplex generalizes a Bézier curve to a simplex, such as a triangle
or tetrahedron. A **simploid** is a product of simplices, allowing domains such
as squares and prisms. Cells can also be joined along shared faces. These
geometries use the same `BezierSpace`, `BezierModel` and regression API.

[Documentation](https://opthub-org.github.io/pytorch-bsf/) ·
[PyPI](https://pypi.org/project/pytorch-bsf/) ·
[Changelog](https://github.com/opthub-org/pytorch-bsf/blob/master/CHANGELOG.md) ·
[Issue tracker](https://github.com/opthub-org/pytorch-bsf/issues)

## Features

- **Regression and point cloud fitting:** fit known parameter-response pairs,
  or estimate unknown coordinates for points on a single Bézier simplex.
- **Controlled geometry:** use simplex, product and joined-cell domains, with
  shared control points, exact fixed controls and quadratic regularization.
- **Interpolation and optimization:** construct shape-preserving interpolants
  on simplicial meshes or optimize coefficients against a differentiable cost.
- **Numerical diagnostics:** solve least-squares problems with dense SVD or
  optional sparse LSMR, and inspect errors, conditioning and termination state.
- **Model evaluation and reuse:** cross-validation, degree selection,
  ensembles, random/Sobol/grid sampling, plotting and versioned model archives.
- **Optional integrations:** scikit-learn estimators, Lightning execution and
  explicit MLflow experiment recording.

## Installation

Requires **Python 3.11 or later** and **PyTorch 2.3 or later (below 3.0)**.
<!-- python-support:start -->

CI covers Python 3.11 through 3.15.
The optional `mlflow` and `all` extras are tested
on Python 3.11 through 3.14.

For Python 3.15, install the other integrations with
`python -m pip install "pytorch-bsf[lightning,sparse,sklearn,plot]"`.

<!-- python-support:end -->

Install the package and its core dependencies with:

```bash
python -m pip install pytorch-bsf
```

Add integrations as needed, for example:

```bash
python -m pip install 'pytorch-bsf[sparse,plot]'
```

| Extra | Provides |
| --- | --- |
| `sparse` | SciPy LSMR least-squares solver |
| `plot` | Matplotlib visualization |
| `sklearn` | scikit-learn estimator |
| `lightning` | Lightning execution adapter |
| `mlflow` | Opt-in experiment recording |
| `all` | All of the integrations above |

Dense fitting and random, Sobol and grid sampling work with the core
installation. See the [installation guide](https://opthub-org.github.io/pytorch-bsf/master/installation.html)
for details and alternative environments.

## Quickstart

Fit the quadratic response `y = 4x²` from three observations on `0 ≤ x ≤ 1`,
then predict at two new locations. A segment uses barycentric coordinates
`[1-x, x]`: each row contains two nonnegative weights that sum to one.

```python
import torch
from torch_bsf import BezierSpace, Samples, fit

space = BezierSpace.simplex(dimension=1, degree=2)
parameters = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float64)
result = fit(space, Samples(parameters, [[0.0], [1.0], [4.0]]))
result.require_success()

# x = 0.25 and x = 0.75
query = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64)
with torch.no_grad():
    print(result.model(query))
# tensor([[0.2500],
#         [2.2500]], dtype=torch.float64)
```

`fit` returns a `FitResult` containing a model, metrics, diagnostics and
termination state. `require_success()` raises if the numerical solve failed.
Predictions have shape `(n_points, n_outputs)`; `result.model` is a PyTorch
module supporting automatic differentiation. Core least-squares fitting runs
on **CPU in float64**, with both dense and sparse solvers.

Choose a different entry point when your inputs differ:

| Your inputs | Entry point | Guide |
| --- | --- | --- |
| Known parameter coordinates and measured responses | `fit(space, Samples(...))` | [Regression](https://opthub-org.github.io/pytorch-bsf/master/quickstart.html) |
| Points with unknown parameter coordinates on a single simplex | `fit_point_cloud(space, values)` | [Point cloud fitting](https://opthub-org.github.io/pytorch-bsf/master/advanced/point_cloud.html) |
| An initial model and a differentiable cost function | `minimize(initial_model, loss, ...)` | [Objective minimization](https://opthub-org.github.io/pytorch-bsf/master/advanced/objective_minimization.html) |

### Command line

The same fitting and prediction operations are available from the CLI. First
create `params.csv` and `values.csv` using the [Quickstart CSV example](https://opthub-org.github.io/pytorch-bsf/master/quickstart.html#prepare-your-own-observations),
then run in the directory containing those files:

```bash
python -m torch_bsf fit --params params.csv --values values.csv --degree 2 --save model.bsf
python -m torch_bsf predict --model model.bsf --params params.csv --save predictions.csv
```

Fitting saves a reusable model archive to `model.bsf`; prediction writes one
output row per parameter row to `predictions.csv`. Use `python -m torch_bsf --help`
or see the [CLI guide](https://opthub-org.github.io/pytorch-bsf/master/cli.html)
for more operations and options.

## Documentation

Start at the [documentation home](https://opthub-org.github.io/pytorch-bsf/)
and choose the tag matching your installed version under **Other Versions**.
Guide links in this README target **development (`master`)** and may describe
unreleased changes. Before 1.0, minor releases may include breaking API changes;
consult the migration guide when upgrading.

| Resource | What you will find |
| --- | --- |
| [Quickstart](https://opthub-org.github.io/pytorch-bsf/master/quickstart.html) | Python examples, CSV inputs and saved-model prediction |
| [Geometry](https://opthub-org.github.io/pytorch-bsf/master/whatis.html) | Bézier simplices and the mathematical background |
| [User guide](https://opthub-org.github.io/pytorch-bsf/master/advanced.html) | Domains, solvers, regularization, interpolation, sampling and model selection |
| [Applications](https://opthub-org.github.io/pytorch-bsf/master/applications.html) | Worked examples with reproducible figures |
| [API reference](https://opthub-org.github.io/pytorch-bsf/master/api.html) | Public classes, functions and arguments |
| [Migration guide](https://opthub-org.github.io/pytorch-bsf/master/migration.html) | Replacements for earlier APIs |

Runnable scripts are available in [examples](https://github.com/opthub-org/pytorch-bsf/tree/master/examples).
With a repository checkout and Bash, run `bash examples/quickstart/run.sh`
from the repository root for the complete CLI walkthrough.

## Contributing and support

Bug reports, documentation improvements and code contributions are welcome.
Use the [issue tracker](https://github.com/opthub-org/pytorch-bsf/issues) for
bugs and feature requests; include a minimal reproducer and environment details
when reporting a bug. The [FAQ](https://opthub-org.github.io/pytorch-bsf/master/faq.html)
covers common usage questions.

See the [contributing guide](https://github.com/opthub-org/pytorch-bsf/blob/master/CONTRIBUTING.md)
for development setup, running tests and pull request requirements.

## Research references

For research using Bézier simplex fitting, cite the relevant methodological
papers:

- Kobayashi et al. (AAAI 2019), [Bézier Simplex Fitting: Describing Pareto Fronts of Simplicial Problems with Small Samples in Multi-Objective Optimization](https://doi.org/10.1609/aaai.v33i01.33012304).
- Tanaka et al. (AAAI 2020), [Asymptotic Risk of Bézier Simplex Fitting](https://doi.org/10.1609/aaai.v34i03.5622).

See the [bibliography](https://opthub-org.github.io/pytorch-bsf/master/references.html)
for further background and references for individual methods.

## License

PyTorch-BSF is distributed under the [MIT License](https://github.com/opthub-org/pytorch-bsf/blob/master/LICENSE).
