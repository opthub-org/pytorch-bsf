Advanced Topics
===============

The common space, model and observation types connect regression, optimization
and optional integrations. The guides follow a fitting workflow: prepare
observations, inspect predictions, then adjust the model or acquire more data.
Continue with domain construction when you need products or connected cells.
Objective minimization has its own sequence for differentiable costs without
regression targets, followed by optional integrations and larger-scale solves.

Each practical guide explains when the feature is needed, provides runnable
examples, and describes how settings affect results. Before/after figures
state what changed and retain reproduction scripts and numerical evidence.

Choose a guide by the question you need to answer
-------------------------------------------------

.. list-table:: Practical entry points
   :header-rows: 1
   :widths: 40 30 30

   * - Question
     - Start here
     - What to inspect afterward
   * - Only points on a curve or surface are known; parameter coordinates are unknown.
     - :doc:`advanced/point_cloud`
     - Local convergence, inferred coordinates and reconstruction distances
   * - Outputs have very different scales.
     - :doc:`advanced/normalization`
     - Errors in original units
   * - A fit oscillates between measurements.
     - :doc:`advanced/model_selection`, :doc:`advanced/regularization`
     - Independent prediction errors
   * - Endpoint or boundary values are exact.
     - :doc:`advanced/fixed_controls`
     - Full boundary coverage of the constraints
   * - Reference evaluations are expensive.
     - :doc:`advanced/active_learning`
     - Errors on a fixed external validation set
   * - A measurement budget must be split among vertices, edges and faces.
     - :doc:`advanced/skeleton_sampling`
     - Rows per face and each face's least-squares rank
   * - A plot is coarse or misses a feature.
     - :doc:`advanced/visualization`, :doc:`advanced/grid_sampling`
     - Numerical residuals and render resolution
   * - Independent parameter groups vary together.
     - :doc:`advanced/simploidal`
     - Per-factor sums and coordinate order
   * - Neighboring local regions must agree.
     - :doc:`advanced/simploidal`
     - Predictions along identified faces
   * - Exact data on a mesh must not overshoot.
     - :doc:`advanced/interpolation`
     - Prediction range in each cell
   * - A large fit exhausts memory.
     - :doc:`advanced/solvers_and_execution`
     - Sparse termination and conditioning
   * - There are costs but no target responses.
     - :doc:`advanced/initial_models`, :doc:`advanced/objective_minimization`
     - Full-sample objective and gradient stopping

Guide chapters
--------------

Start by recording the domain, output units and measurement process. Use
sampling and visualization to examine coverage before changing the model.
Then choose capacity with held-out data. A larger degree can reduce bias,
but increases controls and can worsen conditioning; regularization imposes
an additional assumption rather than adding measurements.

Keep the evaluation set fixed while comparing settings. If you change both
degree and training locations, a better result cannot be attributed to
either change alone. Boundary constraints, coefficient penalties and output
transforms should be recorded with every comparison because they change
what problem is being solved.

``FitResult.success`` describes numerical termination. Observation rank
describes whether data identify the free controls, while held-out errors
describe predictive performance. None replaces the others. An accurate
interior fit can still miss an important boundary, and a regularized
full-rank system can still have an underdetermined data matrix.

For regression, follow the first two groups from data preparation through
validation and refinement. The remaining groups are independent paths:
choose them when your domain, objective or execution environment requires it.

.. toctree::
   :maxdepth: 1
   :caption: Prepare data and fit

   advanced/normalization
   advanced/sampling
   advanced/skeleton_sampling
   advanced/skeleton_risk_theory
   advanced/simploidal_risk
   advanced/skeleton_risk_verification
   advanced/point_cloud

.. toctree::
   :maxdepth: 1
   :caption: Inspect and improve a fit

   advanced/visualization
   advanced/grid_sampling
   advanced/model_selection
   advanced/regularization
   advanced/fixed_controls
   advanced/active_learning

Start with the geometric simploid guide before interpolation or subdivision.
Exact subdivision changes charts while preserving the represented function;
assess it differently from refitting. The category-theory chapter is an
optional mathematical continuation about faces, degeneracies and gluing.

.. toctree::
   :maxdepth: 1
   :caption: Build and refine domains

   advanced/simploidal
   advanced/interpolation
   advanced/splitting
   advanced/simploidal_category

.. toctree::
   :maxdepth: 1
   :caption: Optimize differentiable objectives

   advanced/initial_models
   advanced/objective_minimization

.. toctree::
   :maxdepth: 1
   :caption: Integrations and performance

   advanced/sklearn
   advanced/solvers_and_execution
