Migrating to the unified v0 API
===============================

This guide describes the unified API in documentation version |release|. There are no
legacy aliases or alternate result types. Later pre-1.0 releases can add
further changes: use the version selector to read the guide matching your
installed release, and consult its changelog. The ``master`` copy documents
development toward the next release; it can include unreleased additions.

.. list-table::
   :header-rows: 1

   * - Previous API
     - Replacement
   * - BezierSimploid / SimploidalBezierSpace
     - BezierSpace + BezierModel
   * - fit(params, values, degree, ...)
     - fit(space, Samples(params, values), ...).model
   * - fit_simploidal_set
     - BezierSpace.on_set + Samples.from_cells + fit
   * - fit_sgd
     - minimize(model, loss, method=PreconditionedSGD(...))
   * - fit_kfold / automatic ensemble
     - cross_validate / explicit Ensemble(result.models)
   * - solver="trainer" and Trainer keyword arguments
     - integrations.lightning.regression_module
   * - init / freeze for regression
     - FixedControls with explicit global coefficient values
   * - ridge / smoothness_weight
     - QuadraticPenalty(ridge=..., smoothness=...)
   * - implicit normalization
     - explicit output transforms before fitting
   * - save/load chosen by extension
     - io.save_model / io.load_model
   * - separate simplex and simploid samplers and plotters
     - sampling.sample / sampling.grid / plotting.plot_model
   * - BezierSimploidRegressor
     - sklearn.BezierRegressor

Regression now solves in CPU float64 and returns a float64 model. Penalties
are defined relative to weighted mean squared output-vector error, not a
sample-count-dependent sum. ``data_mse`` additionally divides by output size.
Existing numerical penalty values may therefore need retuning.

The internal ``_kernels`` package retains tested geometry algorithms. It is
private and is not a compatibility API.

Python 3.11 minimum (unreleased)
--------------------------------

The next release supports Python 3.11 through 3.15. Recreate Python 3.10
environments with a supported interpreter before upgrading pytorch-bsf.
The Python 3.10 dependency resolution selected vulnerable Pillow 12.3.0
and filelock 4.1.0. Their security fixes require Python 3.11 or later:
Pillow 13.0.0 and filelock 4.2.1. The refreshed ``uv.lock`` keeps Pillow
13.0.0 and filelock 4.2.4 for supported interpreters.
The NumPy lower bound is now 1.23.2, which provides Python 3.11 wheels.

Python 3.15 is tested with the core and the Lightning, SciPy, scikit-learn
and plotting integrations. The ``mlflow`` and ``all`` extras currently
require Python 3.11-3.14: MLflow 3.17.2 requires PyArrow below 26, while
Python 3.15 wheels start with PyArrow 26. Use the other extras on Python
3.15 until MLflow supports that dependency combination.

See the `Pillow security advisory
<https://github.com/advisories/GHSA-5x8w-8jj5-q52h>`_ and
`filelock security advisory
<https://github.com/advisories/GHSA-v4x8-3wr5-3wq8>`_.

Separate geometry, observations and fitted state
------------------------------------------------

Construct ``BezierSpace`` from your input domain, then pair parameter rows
and outputs in ``Samples``. Regression returns a ``FitResult`` containing a
model, numerical status, metrics and diagnostics. Call the model to predict;
the result object itself is not callable.

.. testcode:: migration

    import torch
    from torch_bsf import BezierSpace, Samples, fit, QuadraticPenalty

    x = torch.linspace(0, 1, 20, dtype=torch.float64)
    parameters = torch.stack([1 - x, x], dim=1)
    values = (x.square() + .2 * x)[:, None]
    space = BezierSpace.simplex(1, degree=2)
    result = fit(space, Samples(parameters, values)).require_success()
    predictions = result.model(parameters)
    assert predictions.shape == (20, 1)
    assert result.metrics["data_mse"] < 1e-24
    assert result.diagnostics.compute_dtype == "float64"

For ``simplex``, the dimension is one less than the barycentric column
count: two columns describe a segment, three describe a triangle. For
``simploid``, pass factor dimensions and a matching degree tuple. For an
assembled domain, retain per-row cell IDs with ``Points`` or
``Samples.from_cells``. Equal local coordinates in different cells need
not describe the same location. See :doc:`advanced/simploidal` for the
geometric picture and coordinate conventions.

Translate penalties and constraints deliberately
-------------------------------------------------

Replace numeric penalty keywords with a ``QuadraticPenalty`` configuration:

.. testcode:: migration

    regularized = fit(
        space, Samples(parameters, values),
        penalty=QuadraticPenalty(ridge=1e-5, smoothness=1e-4),
    ).require_success()
    assert regularized.model(parameters).shape == values.shape

Retune these strengths under the new normalized objective rather than
assuming that the old numbers have the same effect. The regression data
term is weighted mean squared vector error; the reported ``data_mse`` is
also divided by output count. Compare predictions in original output units,
not just an old scalar loss value.

Use ``FixedControls`` for coefficient values known exactly. An old initial
guess is not an exact constraint and should not automatically become a
fixed row. Conversely, a boundary that must remain exact should not be
represented only by a large sample weight. Resolve rows with
``space.control_index`` because coefficient ordering is a property of the
compiled space. See :doc:`advanced/fixed_controls`.

For an original differentiable cost, construct a ``BezierModel`` with
explicit coefficients and pass it to ``minimize`` with a typed method.
Regression ``fit`` has no iterative initial model. Lightning is an optional
adapter around the common model rather than a regression solver selector.
See :doc:`advanced/initial_models` and :doc:`advanced/objective_minimization`.

Replace implicit preprocessing and model storage
------------------------------------------------

Fit output transforms on training data, apply the same transform to exact
fixed values, and invert predictions before reporting physical errors.
Within cross-validation, learn preprocessing separately inside each fold.
Generic feature standardization is inappropriate for barycentric inputs
because it destroys their per-factor sum-to-one constraints.

Use the explicit archive interface for new models:

.. testcode:: migration

    from pathlib import Path
    from tempfile import TemporaryDirectory
    from torch_bsf.io import save_model, load_model

    with TemporaryDirectory() as directory:
        path = Path(directory) / "curve.bsf"
        save_model(path, result.model)
        restored = load_model(path)
        assert torch.equal(restored(parameters), predictions)

The archive stores compiled geometry and coefficients regardless of the
filename extension. It does not imply conversion of a legacy artifact.
For old models, reconstruct their space and coefficient correspondence
explicitly, then compare predictions at reference points before saving in
the new format. A control-point CSV alone does not reconstruct glued topology.

Validation and integration changes
----------------------------------

``cross_validate`` returns fold results, and an ensemble must be constructed
explicitly when that is the intended predictor. Selecting a degree does
not by itself refit on all observations; perform that final fit with the
selected configuration. The scikit-learn ``GridSearchCV`` adapter follows
its own default refit policy. See :doc:`advanced/model_selection` and
:doc:`advanced/sklearn`.

Before replacing an existing workflow, compare a fixed batch of predictions,
boundary values, transformed outputs and save/load results. Inspect
``success`` and solver diagnostics rather than treating a returned model
as sufficient evidence of convergence. Numerical differences may reflect
the new precision, objective normalization or deliberately changed penalties;
identify which configuration is being compared.
