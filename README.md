# PyTorch-BSF

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opthub-org/pytorch-bsf/master/docs/_static/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/opthub-org/pytorch-bsf/master/docs/_static/logo.svg" width="448" alt="PyTorch-BSF — Bézier Simploid Fitting">
</picture>

[![CodeQL](https://github.com/opthub-org/pytorch-bsf/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/codeql-analysis.yml)
[![PyTest](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package.yml)
[![Release](https://github.com/opthub-org/pytorch-bsf/actions/workflows/release-please-action.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/release-please-action.yml)
[![Conda](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/python-package-conda.yml)
[![GHPages](https://github.com/opthub-org/pytorch-bsf/actions/workflows/sphinx-pages.yml/badge.svg)](https://github.com/opthub-org/pytorch-bsf/actions/workflows/sphinx-pages.yml)
[![codecov](https://codecov.io/gh/opthub-org/pytorch-bsf/graph/badge.svg?token=0BNDP4QUVT)](https://codecov.io/gh/opthub-org/pytorch-bsf)
[![PyPI Downloads](https://static.pepy.tech/badge/pytorch-bsf)](https://pepy.tech/projects/pytorch-bsf)

**Fit smooth manifolds with explicit geometry, observations and numerical evidence.**

PyTorch-BSF fits Bézier simplices, products of simplices, and cells glued along
shared faces. A single model type and regression API cover all three.

Choose an entry point from the data you have:

| Your inputs | Operation and result | Guide |
| --- | --- | --- |
| Known parameter coordinates and measured responses | `fit(space, Samples(...))` learns a model predicting responses at new coordinates. | [Regression quickstart](https://opthub-org.github.io/pytorch-bsf/master/quickstart.html) |
| Observed points on a curve or surface, with unknown parameter coordinates | `fit_point_cloud(space, values)` estimates a model and coordinates for each observation on a single simplex. | [Point cloud fitting](https://opthub-org.github.io/pytorch-bsf/master/advanced/point_cloud.html) |
| A differentiable cost function, with no measured target responses | `minimize(initial_model, loss, ...)` optimizes model coefficients against the cost. | [Objective minimization](https://opthub-org.github.io/pytorch-bsf/master/advanced/objective_minimization.html) |

Install with Python 3.10 or later:

```bash
pip install pytorch-bsf
# Optional: sparse solving, plotting, scikit-learn, execution or recording
pip install 'pytorch-bsf[sparse,plot,sklearn,lightning,mlflow]'
```

Conda is an optional environment manager. Lightning and MLflow are optional
adapters and are never imported by core fitting. Random/Sobol sampling needs
no extra dependency. Tracking is opt-in.

Copy this complete example to fit a quadratic response on a segment:

```python
import torch
from torch_bsf import BezierSpace, Samples, fit

space = BezierSpace.simplex(dimension=1, degree=2)
t = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
result = fit(space, Samples(t, [[0.0], [1.0], [4.0]]))
result.require_success()
predictions = result.model(t)
print(predictions)
print(predictions.shape)
```

A segment position `x` is represented by barycentric coordinates `[1-x, x]`:
the two columns sum to one and describe one independent input dimension.
The example evaluates `x = 0, 0.5, 1`. Predictions are approximately
`[[0.0], [1.0], [4.0]]`, with shape `torch.Size([3, 1])`: three locations and
one response per location. Floating-point roundoff can leave tiny values near
zero. See [the geometry guide](https://opthub-org.github.io/pytorch-bsf/master/whatis.html)
for simplices, products and glued cells.

- `BezierSpace` holds immutable geometry and shared coefficient ordering.
- `BezierModel` is a differentiable PyTorch module containing coefficients.
- `fit(space, samples)` always returns `FitResult`: model, metrics, diagnostics and stopping state.
- Dense SVD and optional sparse LSMR solve the same weighted, penalized objective in CPU float64.
- Cross-validation, degree search and ensembles retain explicit evidence and ownership.
- Sampling, plotting, subdivision and versioned persistence use the same model types.

For the CLI, first create `params.csv` and `values.csv` using the
[Quickstart CSV example](https://opthub-org.github.io/pytorch-bsf/master/quickstart.html#prepare-your-own-observations).
Then run these commands in the directory containing those files:

```bash
python -m torch_bsf fit --params params.csv --values values.csv --degree 2 --save model.bsf
python -m torch_bsf predict --model model.bsf --params params.csv --save predictions.csv
```

Fitting saves `model.bsf`; prediction saves one output row per input row to
`predictions.csv`. With the Quickstart data, these are approximately `0`, `1`
and `4`. The commands above work with a pip installation. The complete
[quickstart script](https://github.com/opthub-org/pytorch-bsf/blob/master/examples/quickstart/run.sh)
requires a repository checkout and Bash; run it from the repository root:

```bash
bash examples/quickstart/run.sh
```

Save and load any model, including glued geometry, with
`torch_bsf.io.save_model` / `load_model`. The format is versioned and does not
pickle Python classes. CSV coefficient export is a separate operation.

The unified pre-1.0 API consolidates previous APIs without compatibility
aliases. See the [development migration guide (master)](https://opthub-org.github.io/pytorch-bsf/master/migration.html),
[documentation](https://opthub-org.github.io/pytorch-bsf/) and
[contributing guide](https://github.com/opthub-org/pytorch-bsf/blob/master/CONTRIBUTING.md).
Choose the tag matching your installed version in the documentation's
**Other Versions** menu; `master` can include unreleased changes.

Research foundations:

- [Kobayashi et al. (AAAI 2019)](https://doi.org/10.1609/aaai.v33i01.33012304), Bézier simplex fitting.
- [Tanaka et al. (AAAI 2020)](https://doi.org/10.1609/aaai.v34i03.5622), asymptotic risk.
