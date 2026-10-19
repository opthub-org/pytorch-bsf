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

**Fit smooth, high-dimensional manifolds to your data — from a single GPU to a multi-node cluster.**

`pytorch-bsf` brings **B**ézier **S**imploid **F**itting (BSF) to PyTorch. A Bézier simplex is a high-dimensional generalization of the Bézier curve: where a curve models a 1-D path, a Bézier simplex can model an arbitrarily complex point cloud as a smooth parametric hyper-surface in any number of dimensions. Bézier simploids go further: they are tensor products of Bézier simplices defined on products of simplices such as prisms and cubes, and simploidal sets glue such cells along common faces. This makes them a natural tool for representing **Pareto fronts** in multi-objective optimization, interpolating scattered observations, and fitting geometric structures in high-dimensional spaces.

Key features:

- **Simple, Pythonic API** — train a model in one line with `torch_bsf.fit()`, call it like any PyTorch module, and persist control points in `.pt`, `.csv`, `.tsv`, `.json`, or `.yaml` formats.
- **Scalable Training** — fully vectorized `forward` pass backed by [PyTorch Lightning](https://www.pytorchlightning.ai/) for seamless scaling from a single CPU to multi-GPU, multi-node clusters.
- **Robust & Automatic Fitting** — built-in smoothness regularization tames noisy data, and automatic degree selection via k-fold cross-validation removes guesswork.
- **Rich ML Ecosystem** — scikit-learn-compatible `BezierSimplexRegressor` for use in `Pipeline` and `GridSearchCV`; [MLflow](https://mlflow.org/) experiment tracking; active learning; and advanced sampling strategies (Dirichlet, Sobol).
- **Beyond Simplices** — `torch_bsf.simploidal` fits Bézier simploids and simploidal sets when a single simplex is not the right parameter domain.
- **Ready to Run** — CLI entry points, a pre-built Docker image, and a Conda/MLflow project so you can run `pytorch-bsf` without installing it into your local environment.

<img src="https://opthub-org.github.io/pytorch-bsf/master/_images/bezier-simplex.png" width="49%" alt="A Bézier simplex and its control points"><img src="https://opthub-org.github.io/pytorch-bsf/master/_images/bezier-simplex-fitting.png" width="49%" alt="A Bézier simplex that fits to a dataset">

> BSF formerly stood for *Bézier Simplex Fitting*, which remains the special case where the parameter domain is a single simplex.

See [Full Documentation](https://opthub-org.github.io/pytorch-bsf/) and the following papers for technical details.

- Kobayashi, K., Hamada, N., Sannai, A., Tanaka, A., Bannai, K., & Sugiyama, M. (2019). Bézier Simplex Fitting: Describing Pareto Fronts of Simplicial Problems with Small Samples in Multi-Objective Optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 2304-2313. <https://doi.org/10.1609/aaai.v33i01.33012304>
- Tanaka, A., Sannai, A., Kobayashi, K., & Hamada, N. (2020). Asymptotic Risk of Bézier Simplex Fitting. Proceedings of the AAAI Conference on Artificial Intelligence, 34(03), 2416-2424. <https://doi.org/10.1609/aaai.v34i03.5622>

## Quickstart

**Requirements:** For local installation and CLI/library usage, Python >=3.10. For Docker-only usage, only Docker is required.

First, prepare your data:

```bash
cat <<EOS > params.csv
1.00, 0.00
0.75, 0.25
0.50, 0.50
0.25, 0.75
0.00, 1.00
EOS
cat <<EOS > values.csv
0.00, 1.00
3.00, 2.00
4.00, 5.00
7.00, 6.00
8.00, 9.00
EOS
```

### 1. via Docker (No Installation Required)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/).

A pre-built image is available on GHCR, built on `continuumio/miniconda3` with PyTorch installed via the `pytorch` conda channel, providing Intel MKL as the BLAS backend:

```bash
docker run --rm \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/workspace \
  ghcr.io/opthub-org/pytorch-bsf \
  python -m torch_bsf \
  --params=params.csv \
  --values=values.csv \
  --degree=3
```

### 2. via MLflow (No Installation Required)

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda, and `pip install mlflow`.

MLflow creates a conda environment from the project's `environment.yml`, which uses the `pytorch` conda channel and Intel MKL:

```bash
mlflow run https://github.com/opthub-org/pytorch-bsf \
  -P params=params.csv \
  -P values=values.csv \
  -P degree=3
```

### 3. via CLI (After Installation)

This method is suitable when you do not need experiment tracking. If you prefer to install the package, you can run it via the command line interface.

Installation:

```bash
pip install pytorch-bsf
```

Execution:

```bash
python -m torch_bsf \
  --params=params.csv \
  --values=values.csv \
  --degree=3
```

<details>
<summary>Click to see all MLflow / CLI parameters</summary>

| Parameter            | Type                                                         | Default      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| params               | path                                                         | required     | The parameter data file, which contains input observations for training a Bézier simplex. The file must be a CSV (`.csv`) or TSV (`.tsv`) file. Each line in the file represents an input observation, corresponding to an output observation in the same line in the value data file.                                                                                                                                                                |
| values               | path                                                         | required     | The value data file, which contains output observations for training a Bézier simplex. The file must be a CSV (`.csv`) or TSV (`.tsv`) file. Each line in the file represents an output observation, corresponding to an input observation in the same line in the parameter data file.                                                                                                                                                               |
| meshgrid             | path                                                         | `None`       | The meshgrid data file used for prediction after training. The file format is the same as `params`. If omitted, `params` is used as the meshgrid.                                                                                                                                                                                                                                                                                                 |
| init                 | path                                                         | `None`       | Load initial control points from a file. The file must be of pickled PyTorch (`.pt`), CSV (`.csv`), TSV (`.tsv`), JSON (`.json`), or YAML (`.yml` or `.yaml`). Either this option or `--degree` must be specified, but not both.                                                                                                                                                                                                                  |
| degree               | int $(x \ge 1)$                                              | `None`       | Generate initial control points at random with specified degree. Either this option or `--init` must be specified, but not both.                                                                                                                                                                                                                                                                                                                  |
| freeze               | list[list[int]]                                              | `None`       | Indices of control points to exclude from training. By default, all control points are trained.                                                                                                                                                                                                                                                                                                                                                   |
| header               | int $(x \ge 0)$                                              | `0`          | The number of header lines in params/values files.                                                                                                                                                                                                                                                                                                                                                                                                |
| normalize            | `"none"`, `"max"`, `"std"`, `"quantile"`                     | `"none"`     | The data normalization: `"max"` scales each feature such that the minimum is 0 and the maximum is 1, suitable for uniformly distributed data; `"std"` does so such that the mean is 0 and the standard deviation is 1, suitable for non-uniformly distributed data; `"quantile"` does so such that the 5-percentile is 0 and the 95-percentile is 1, suitable for data containing outliers; `"none"` does not perform any scaling, suitable for pre-normalized data.                  |
| split_ratio          | float $(0 < x \le 1)$                                        | `1.0`        | The ratio of training data against validation data. When set to `1.0` (the default), all data is used for training and the validation step is skipped.                                                                                                                                                                                                                                                                                            |
| batch_size           | int $(x \ge 1)$                                              | `None`       | The size of minibatch. The default (`None`) uses all records in a single batch.                                                                                                                                                                                                                                                                                                                                                                   |
| max_epochs           | int $(x \ge 1)$                                              | `2`          | The number of epochs to stop training.                                                                                                                                                                                                                                                                                                                                                                                                            |
| smoothness_weight    | float $(x \ge 0)$                                            | `0.0`        | The weight of smoothness penalty. Adding a small weight (e.g., `0.1`) helps produce a smooth, stable manifold when fitting noisy datasets.                                                                                                                                                                                                                                                                                                        |
| accelerator          | `"auto"`, `"cpu"`, `"gpu"`, etc.                             | `"auto"`     | Accelerator to use. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerator.html).                                                                                                                                                                                                                                                                                                        |
| strategy             | `"auto"`, `"dp"`, `"ddp"`, etc.                              | `"auto"`     | Distributed strategy. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html).                                                                                                                                                                                                                                                                                                         |
| devices              | int $(x \ge -1)$                                             | `"auto"`     | The number of accelerators to use. By default, use all available devices. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html).                                                                                                                                                                                                                                                  |
| num_nodes            | int $(x \ge 1)$                                              | `1`          | The number of compute nodes to use. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html).                                                                                                                                                                                                                                                                                                  |
| precision            | `"64-true"`, `"32-true"`, `"16-mixed"`, `"bf16-mixed"`, etc. | `"32-true"`  | The precision of floating point numbers.                                                                                                                                                                                                                                                                                                                                                                                                          |
| loglevel             | int $(0 \le x \le 2)$                                        | `2`          | What objects to be logged. `0`: nothing; `1`: metrics; `2`: metrics and models.                                                                                                                                                                                                                                                                                                                                                                   |
| enable_checkpointing | flag                                                         | `False`      | With this flag, model files will be stored every epoch during training.                                                                                                                                                                                                                                                                                                                                                                           |
| log_every_n_steps    | int $(x \ge 1)$                                              | `1`          | The interval of training steps when training loss is logged.                                                                                                                                                                                                                                                                                                                                                                                      |
| simploid-type        | comma-separated ints $(a_i \ge 1)$                           | `None`       | Fit a Bézier simploid of type $(a_1, \ldots, a_n)$, i.e., a tensor product of Bézier simplices, instead of a Bézier simplex (e.g., `2,1` for a prism). The `params` file must have $\sum_i (a_i + 1)$ columns. `--degree` applies to every factor; `--init`, `--freeze`, `--smoothness_weight` and `--split_ratio` are not supported. MLflow: use the `simploid` entry point (`-e simploid -P simploid_type=2,1`). See [Simploidal Sets](https://opthub-org.github.io/pytorch-bsf/master/advanced/simploidal.html). |
| solver               | `"lstsq"`, `"trainer"`                                       | `"lstsq"`    | How to fit a Bézier simploid: `"lstsq"` solves the least-squares problem exactly; `"trainer"` trains with the trainer options above. Requires `--simploid-type`.                                                                                                                                                                                                                                                                                  |
| ridge                | float $(x \ge 0)$                                            | `0.0`        | The ridge penalty weight of the `"lstsq"` solver. Requires `--simploid-type`.                                                                                                                                                                                                                                                                                                                                                                    |
| save                 | path                                                         | see desc.    | The file to save the control points of the fitted Bézier simploid to (`.csv`, `.tsv`, `.json`, `.yml`, `.yaml` or `.pt`). Defaults to `<params>,<values>,control_points,t_<a_1-...-a_n>,d_<degree>.csv`. Requires `--simploid-type`.                                                                                                                                                                                                         |

</details>

### 4. via Python Script

This is the most flexible method, allowing you to use custom data loaders to read large-scale data and finely control logging. Train a model using the Python API `fit()`, and call the resulting model to predict.

```python
import torch
import torch_bsf

# Prepare training data
ts = torch.tensor(  # parameters on a simplex
    [
        [8 / 8, 0 / 8],
        [7 / 8, 1 / 8],
        [6 / 8, 2 / 8],
        [5 / 8, 3 / 8],
        [4 / 8, 4 / 8],
        [3 / 8, 5 / 8],
        [2 / 8, 6 / 8],
        [1 / 8, 7 / 8],
        [0 / 8, 8 / 8],
    ]
)
xs = 1 - ts * ts  # values corresponding to the parameters

# Train a model
bs = torch_bsf.fit(params=ts, values=xs, degree=3)

# Predict with the trained model
t = [
    [0.2, 0.8],
    [0.7, 0.3],
]
x = bs(t)
print(f"{t} -> {x}")
```

## Saving and Loading Models

Save a trained model and reload it later:

```python
import torch_bsf
from torch_bsf.bezier_simplex import save, load

# Train
bs = torch_bsf.fit(params=ts, values=xs, degree=3)

# Save (supported formats: .pt, .csv, .tsv, .json, .yml/.yaml)
save("model.pt", bs)

# Load
bs = load("model.pt")
```

## K-Fold Cross-Validation

Run k-fold cross-validation via the CLI:

```bash
python -m torch_bsf.model_selection.kfold \
  --params=params.csv \
  --values=values.csv \
  --degree=3 \
  --num_folds=5
```

Additional parameters for k-fold (all standard parameters are also accepted):

| Parameter   | Type  | Default | Description                                         |
| ----------- | ----- | ------- | --------------------------------------------------- |
| num_folds   | int   | `5`     | Number of folds.                                    |
| shuffle     | bool  | `True`  | Whether to shuffle data before splitting.           |
| stratified  | bool  | `True`  | Whether to use stratified splitting.                |

The command saves per-fold meshgrid predictions as well as an ensemble mean:

- `{params},{values},{num_folds}fold,meshgrid,d_{degree},r_{split_ratio},{k}.csv` (per fold)
- `{params},{values},{num_folds}fold,meshgrid,d_{degree},r_{split_ratio}.csv` (mean over folds)

## Elastic Net Grid Search

Generate a grid of 3D parameter points on the standard 2-simplex for elastic net hyperparameter search:

```bash
python -m torch_bsf.model_selection.elastic_net_grid \
  --n_lambdas=102 \
  --n_alphas=12 \
  --n_vertex_copies=10 \
  --base=10
```

| Parameter        | Type  | Default | Description                                                                       |
| ---------------- | ----- | ------- | --------------------------------------------------------------------------------- |
| n_lambdas        | int   | `102`   | Number of samples along the lambda axis (log scale).                              |
| n_alphas         | int   | `12`    | Number of samples along the alpha axis (linear scale).                            |
| n_vertex_copies  | int   | `10`    | Number of duplicated samples at each vertex (useful for k-fold cross-validation). |
| base             | float | `10`    | Base of the log space.                                                            |

The output is printed to stdout as CSV with three columns (one row per grid point).

## Advanced Topics

The library includes several capabilities beyond basic fitting.
Full documentation for each topic is linked below.

| Topic | Summary |
| ----- | ------- |
| [Visualization](https://opthub-org.github.io/pytorch-bsf/master/advanced/visualization.html) | Plot fitted 2D curves and 3D surfaces with a single call. |
| [Advanced Sampling](https://opthub-org.github.io/pytorch-bsf/master/advanced/sampling.html) | Generate simplex parameter points via Dirichlet random sampling or quasi-random Sobol sequences, in addition to the default uniform grid. |
| [Automatic Degree Selection](https://opthub-org.github.io/pytorch-bsf/master/advanced/auto_degree.html) | Automatically pick the best polynomial degree using k-fold cross-validation. |
| [Scikit-learn Integration](https://opthub-org.github.io/pytorch-bsf/master/advanced/sklearn.html) | `BezierSimplexRegressor` exposes a standard `fit` / `predict` / `score` API compatible with `Pipeline` and `GridSearchCV`. |
| [Active Learning](https://opthub-org.github.io/pytorch-bsf/master/advanced/active_learning.html) | Suggest the most informative next sampling points using Query-By-Committee or density-based strategies. |

## Author

OptHub Inc. and FUJITSU LIMITED

## License

MIT
