Applications
============

This section explores the practical applications of Bézier simplex fitting. By exploiting the continuous relationships between optimal solutions and competing objectives, Bézier simplices provide a powerful framework for multi-objective optimization and parametric modeling.

The most prominent and elegant application of this technique is the continuous approximation of the multi-parameter regularization path in Elastic Net and other sparse modeling methods. We detail how a Bézier simplex can efficiently capture these paths, enabling comprehensive model selection without repeatedly re-training models over a discrete grid.

The other examples use small quadratic reference problems to explain the
same workflow in different settings. The assumptions belong to the stated
reference formulation: a realistic network, controller or inverse problem
can introduce constraints, nonconvexity or discontinuities absent from the
example. Strong convexity can give a unique weighted-sum optimizer, but a
one-to-one simplex topology requires further conditions. See :doc:`whatis`
for the precise unconstrained theorem.

What is fitted in each application?
-----------------------------------

First distinguish the optimization decision from the surrogate input. A
reference solver takes priority weights and returns an optimal decision.
The Bézier model learns that map, its objective values, or both. It does
not learn directly from an untagged cloud of Pareto points: every output
row needs its generating parameter row.

.. list-table:: Reference problems and surrogate outputs
   :header-rows: 1
   :widths: 25 35 40

   * - Application
     - Input parameter
     - Decision-map output
   * - Elastic Net
     - Loss and penalty weights on a triangle
     - Regression coefficients
   * - Portfolio
     - Two regularized objective weights
     - Three unconstrained allocation variables
   * - Smart grids
     - Cost versus emissions weights
     - Three generator outputs
   * - Federated learning
     - Two task weights
     - Two model parameters in the toy problem
   * - Model predictive control
     - Priority weights and independent initial state
     - Five future inputs
   * - Communications
     - Two or three priority weights
     - Two allocation variables in the toy problem
   * - Supply chain
     - Inventory versus ordering-cost weights
     - Five future orders
   * - Medical imaging
     - Logarithmic regularization coordinate
     - 256 image pixels
   * - Facility location
     - Weights for two, three or four centers
     - Two spatial coordinates

The objective-map output instead contains one column per objective.
Its low dimensionality makes it convenient to plot, but it does not contain
enough information to recover decisions in general. Evaluate original
objectives on predicted decisions when checking consistency between two
independently fitted maps.

A common reproduction and validation workflow
---------------------------------------------

1. Define objectives, feasible decisions and parameter bounds. Record all
   fixed problem data such as demand, initial state and measurements.
2. Solve at training parameters and retain convergence evidence. Numerical
   errors in these reference solutions become errors in the training data.
3. Choose a simplex, product or assembled domain and fit aligned ``Samples``.
   Inspect numerical success, observation rank and residuals.
4. Obtain new reference responses at held-out parameters, including important
   boundaries. Compare each output and check feasibility of decisions.
5. Change degree, sampling or penalties according to those errors. Retain
   the parameter transform and problem definition with the saved model.

Figure scripts are named on each application page and run from the
repository root. Install the optional dependencies imported by the selected
script, typically ``pytorch-bsf[plot,sklearn,sparse]``. Reported numerical
values describe those examples and their stated evaluation sets; small
training error or a smooth picture alone does not establish generalization.

Choosing a domain and a next guide
----------------------------------

Two priorities give a segment and three give a triangle. An independent
operating variable requires another factor: two priorities plus initial
state give a square, as in the MPC example. This geometric distinction is
illustrated in the :ref:`simploid overview <simploidal-space-overview>`.
More output variables do not require more domain factors.

Use :doc:`quickstart` for the fitting API, :doc:`advanced/normalization` for
units and transforms, and :doc:`advanced/auto_degree` for model selection.
For an original differentiable objective without reference target values,
use :doc:`advanced/sgd`; that is a different operation from regression on
solver-generated observations.

.. toctree::
   :maxdepth: 1

   applications/elastic_net
   applications/portfolio
   applications/smart_grids
   applications/federated_learning
   applications/model_predictive_control
   applications/communications
   applications/supply_chain
   applications/medical_imaging
   applications/facility_location
