Installation
============

Python 3.10 or later is required. Install the numerical core with::

    pip install pytorch-bsf

The core depends on PyTorch, NumPy and PyYAML (used by internal geometry tools).
Conda is an optional environment manager. The repository's ``environment.yml``
and Docker image remain available for reproducibility.

Optional extras:

* ``pytorch-bsf[sparse]``: SciPy LSMR least squares.
* ``pytorch-bsf[sklearn]``: scikit-learn estimator.
* ``pytorch-bsf[plot]``: Matplotlib plotting.
* ``pytorch-bsf[lightning]``: explicit Lightning execution adapter.
* ``pytorch-bsf[mlflow]``: explicit experiment recording.
* ``pytorch-bsf[all]``: every integration.

Random, Sobol and grid sampling use PyTorch and need no extra.
Importing the package loads neither Lightning nor MLflow nor SciPy.

This v0 redesign intentionally removes old names and import paths. See
:doc:`migration` for replacement patterns.
