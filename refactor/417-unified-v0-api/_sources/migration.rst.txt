Migrating to the unified v0 API
===============================

This release makes breaking API changes before 1.0. There are no legacy
aliases or alternate result types.

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
