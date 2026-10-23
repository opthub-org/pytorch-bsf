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

```python
import torch
from torch_bsf import BezierSpace, Samples, fit

space = BezierSpace.simplex(dimension=1, degree=2)
t = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
result = fit(space, Samples(t, [[0.0], [1.0], [4.0]]))
result.require_success()
predictions = result.model(t)
print(result.metrics, result.diagnostics)
```

- `BezierSpace` holds immutable geometry and shared coefficient ordering.
- `BezierModel` is a differentiable PyTorch module containing coefficients.
- `fit(space, samples)` always returns `FitResult`: model, metrics, diagnostics and stopping state.
- Dense SVD and optional sparse LSMR solve the same weighted, penalized objective in CPU float64.
- Objective minimization uses `minimize` with explicit `PreconditionedSGD` settings.
- Cross-validation, degree search and ensembles retain explicit evidence and ownership.
- Sampling, plotting, subdivision and versioned persistence use the same model types.

Install with Python 3.10 or later:

```bash
pip install pytorch-bsf
# Optional: sparse solving, plotting, scikit-learn, execution or recording
pip install 'pytorch-bsf[sparse,plot,sklearn,lightning,mlflow]'
```

Conda is an optional environment manager. Lightning and MLflow are optional
adapters and are never imported by core fitting. Random/Sobol sampling needs
no extra dependency. Tracking is opt-in.

```bash
python -m torch_bsf fit --params params.csv --values values.csv --degree 3 --save model.bsf
python -m torch_bsf predict --model model.bsf --params params.csv --save predictions.csv
bash examples/quickstart/run.sh
```

Save and load any model, including glued geometry, with
`torch_bsf.io.save_model` / `load_model`. The format is versioned and does not
pickle Python classes. CSV coefficient export is a separate operation.

This v0 release deliberately consolidates previous APIs without compatibility
aliases. See the [migration guide](https://opthub-org.github.io/pytorch-bsf/master/migration.html),
[documentation](https://opthub-org.github.io/pytorch-bsf/) and
[contributing guide](CONTRIBUTING.md).

Research foundations:

- [Kobayashi et al. (AAAI 2019)](https://doi.org/10.1609/aaai.v33i01.33012304), Bézier simplex fitting.
- [Tanaka et al. (AAAI 2020)](https://doi.org/10.1609/aaai.v34i03.5622), asymptotic risk.
