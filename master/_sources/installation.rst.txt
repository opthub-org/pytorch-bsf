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

Choose dependencies for your workflow
-------------------------------------

The minimal installation is enough for dense regression, random and Sobol
sampling, grids, fixed controls, degree selection, objective minimization and
model archives. Add extras only for the integrations your application calls.
This makes a small CPU fitting environment easier to reproduce and avoids
importing experiment services during numerical computation.

.. list-table:: Optional dependencies and their entry points
   :header-rows: 1
   :widths: 18 28 54

   * - Extra
     - Adds
     - Needed when
   * - ``sparse``
     - SciPy
     - Executing ``fit(..., solver=SparseLeastSquares(...))``.
   * - ``plot``
     - Matplotlib and SciPy
     - Drawing fitted images and running the guide figure generator.
   * - ``sklearn``
     - scikit-learn
     - Using ``BezierRegressor`` or scikit-learn searches.
   * - ``lightning``
     - Lightning
     - Creating the explicit ``regression_module`` execution adapter.
   * - ``mlflow``
     - MLflow
     - Calling ``log_result`` or opting into CLI tracking with ``--track``.
   * - ``develop``
     - Ruff, mypy, pytest and development tools
     - Editing and verifying this repository; see ``CONTRIBUTING.md``.

Combine extras in one quoted requirement:

.. code-block:: bash

    pip install "pytorch-bsf[plot,sparse]"
    pip install "pytorch-bsf[sklearn,mlflow]"

Quoting prevents shells from treating square brackets as filename patterns.
``all`` includes every integration, but documentation-build extensions are
listed separately in ``docs/requirements.txt``. In a development checkout,
install the package editable before executing example scripts so imports
resolve to the checked-out implementation rather than an older wheel.

Check the environment you actually run
--------------------------------------

Use the same interpreter for installation and execution. This is particularly
important when a notebook, Conda environment and system Python coexist.

.. code-block:: bash

    python -m pip install pytorch-bsf
    python -c "import torch_bsf; print(torch_bsf.__version__); print(torch_bsf.__file__)"
    python -m torch_bsf --help

If an optional operation reports a missing dependency, install its extra into
that interpreter. Constructing a sparse solver configuration does not execute
a sparse solve, so import success alone does not verify that SciPy is available.
Similarly, importing the numerical core does not prove that Lightning or MLflow
is installed or usable. Test the particular integration your application needs.

The repository's Conda environment and Docker image provide alternative
packaging workflows, while the public numerical operations stay the same.
Inside Docker, mount input files and an output directory, then use the current
``fit`` or ``predict`` subcommand. Model paths inside the container must point
to that mount if outputs should survive container removal. See :doc:`quickstart`
for data files, shell commands and saved-model prediction.

Upgrading from the earlier API
------------------------------

Installing an extra does not restore removed public names or CLI flags.
``solver="trainer"`` is replaced by an explicit Lightning adapter, legacy
model constructors by ``BezierSpace`` plus ``BezierModel``, and automatic
tracking by explicit recording. Migrate calls using :doc:`migration` before
assuming an installation failure explains an import error for an old symbol.

Core least-squares models now use CPU float64 and normalized data loss with
explicit quadratic penalties. Existing penalty values may therefore require
validation under the new objective. Keep the training observations and old
evaluation results so the upgrade can be compared on the same data. Model
archives have a versioned format; an arbitrary old pickle or coefficient CSV
is not a geometry-preserving archive for the redesigned API.
