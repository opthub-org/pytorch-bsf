Advanced Topics
===============

The common space, model and observation types connect regression, optimization
and optional integrations. Start with normalization and visualization to
understand your data and fitted image. Use degree selection, regularization
and fixed controls to improve a model under explicit assumptions. Sampling
and active learning help decide where to obtain additional observations;
assembled domains describe how local charts connect. Objective minimization
is a separate path for differentiable costs without regression targets.

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
   * - Outputs have very different scales.
     - :doc:`advanced/normalization`
     - Errors in original units
   * - A fit oscillates between measurements.
     - :doc:`advanced/auto_degree`, :doc:`advanced/regularization`
     - Independent prediction errors
   * - Endpoint or boundary values are exact.
     - :doc:`advanced/partial`
     - Full boundary coverage of the constraints
   * - Reference evaluations are expensive.
     - :doc:`advanced/active_learning`
     - Errors on a fixed external validation set
   * - A plot is coarse or misses a feature.
     - :doc:`advanced/visualization`, :doc:`advanced/grid_sampling`
     - Numerical residuals and render resolution
   * - Independent parameter groups vary together.
     - :doc:`advanced/simploidal`
     - Per-factor sums and coordinate order
   * - Neighboring local regions must agree.
     - :doc:`advanced/simploidal`
     - Predictions along identified faces
   * - A large fit exhausts memory.
     - :doc:`advanced/acceleration`
     - Sparse termination and conditioning
   * - There are costs but no target responses.
     - :doc:`advanced/initialization`, :doc:`advanced/sgd`
     - Full-sample objective and gradient stopping

Follow a fit through the guides
-------------------------------

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

The category-theory chapter is a mathematical continuation of the geometric
simploid guide. It is useful for understanding faces, degeneracies and
gluing abstractly; it is not a prerequisite for fitting a square or prism.
The scikit-learn guide uses the same core numerical model in estimator
workflows. Exact subdivision in the splitting guide changes charts while
preserving the represented function, so assess it differently from refitting.

Guide chapters
--------------

.. toctree::
   :maxdepth: 1

   advanced/normalization
   advanced/sklearn
   advanced/initialization
   advanced/visualization
   advanced/splitting
   advanced/auto_degree
   advanced/regularization
   advanced/grid_sampling
   advanced/partial
   advanced/sampling
   advanced/acceleration
   advanced/active_learning
   advanced/simploidal
   advanced/simploidal_category
   advanced/sgd
