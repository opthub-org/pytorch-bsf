Installation
============

PyTorch-BSF requires Python 3.10 or later.
Install it from PyPI:

.. code-block:: bash

   pip install pytorch-bsf

This installs only the required dependencies: PyTorch, NumPy and PyYAML.
They are enough to fit a model with the default solver (``solver="lstsq"``) or with :func:`torch_bsf.fit_sgd`,
to predict, to save and load models, and to run the CLI (``python -m torch_bsf``) without experiment tracking.

The :doc:`Docker image and the MLflow project <quickstart>` install PyTorch-BSF with Lightning and MLflow, so they
need neither the ``lightning`` nor the ``mlflow`` extra below; they do not include the packages of the other extras
(SciPy, scikit-learn and matplotlib).


Optional extras
---------------

Other features need more packages, which the following extras install.
Pick the extras for the features you use, or ``all`` for every feature:

.. code-block:: bash

   pip install "pytorch-bsf[lightning]"
   pip install "pytorch-bsf[lightning,mlflow]"
   pip install "pytorch-bsf[all]"

.. list-table::
   :header-rows: 1
   :widths: 12 22 66

   * - Extra
     - Packages
     - Features
   * - ``lightning``
     - ``lightning``
     - Training with the Lightning trainer: ``solver="trainer"`` in :func:`torch_bsf.fit` and
       :class:`~torch_bsf.sklearn.BezierSimploidRegressor`, and ``--solver=trainer`` on the command line,
       including GPUs and distributed training (see :doc:`advanced/acceleration`).
       Without it, ``solver="trainer"`` raises :exc:`ImportError`.
   * - ``mlflow``
     - ``mlflow``
     - Logging the CLI runs to MLflow (``--loglevel``): ``python -m torch_bsf`` logs the parameters, metrics and
       models, and ``python -m torch_bsf.model_selection.kfold`` logs the errors of each fold and their means.
       Without it, the CLI still fits and saves its outputs (the control points or the predictions), but prints a
       warning and logs nothing.
   * - ``sampling``
     - ``scipy``
     - Quasi-random Sobol sampling with :func:`torch_bsf.sampling.simplex_sobol` (see :doc:`advanced/sampling`).
   * - ``sklearn``
     - ``scikit-learn``
     - The scikit-learn estimator :class:`~torch_bsf.sklearn.BezierSimploidRegressor`
       (see :doc:`advanced/sklearn`).
   * - ``plot``
     - ``matplotlib``, ``scipy``
     - Plotting fitted models with :mod:`torch_bsf.plotting` (see :doc:`advanced/visualization`).
   * - ``all``
     - All of the above
     - Every feature above.
   * - ``develop``
     - ``ruff``, ``pytest``, ``mypy``, etc.
     - Tools to develop PyTorch-BSF itself (linting, type checking and testing); see ``CONTRIBUTING.md``.

For example, to train :class:`~torch_bsf.sklearn.BezierSimploidRegressor` with ``solver="trainer"``, install
``pip install "pytorch-bsf[sklearn,lightning]"``.


.. _upgrading-optional-extras:

Upgrading from a version that required Lightning and MLflow
-----------------------------------------------------------

Up to version 0.19.0, installing PyTorch-BSF also installed Lightning and MLflow.
In later versions they are optional, so upgrading with ``pip install -U pytorch-bsf`` alone may change what your
code does:

* Code that uses ``solver="trainer"`` (in :func:`torch_bsf.fit`, :func:`torch_bsf.fit_kfold`,
  :class:`~torch_bsf.sklearn.BezierSimploidRegressor` or the :doc:`automatic degree selection
  <advanced/auto_degree>`) or ``--solver=trainer`` on the command line raises :exc:`ImportError`.
  Install the ``lightning`` extra.
* The CLI (``python -m torch_bsf`` and ``python -m torch_bsf.model_selection.kfold``) stops logging to MLflow and
  prints a warning instead.
  Install the ``mlflow`` extra.

To keep the behavior of the earlier versions, upgrade with both extras:

.. code-block:: bash

   pip install -U "pytorch-bsf[lightning,mlflow]"

If Lightning and MLflow are already in your environment (e.g., left by the earlier version), PyTorch-BSF keeps using
them, but listing the extras records that your project needs them.
Code that uses only the default ``solver="lstsq"`` or :func:`torch_bsf.fit_sgd` and does not log to MLflow needs no
extra.
