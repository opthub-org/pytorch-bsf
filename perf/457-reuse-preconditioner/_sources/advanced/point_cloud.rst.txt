Fitting from values alone
=========================

Use ``fit_point_cloud`` when each observation is a point on a curve or
surface and its domain coordinate is unknown. The existing ``fit`` solves
a different problem: regression from known parameter/value pairs.

The value-only methods implement Algorithms 2 and 3 of
:cite:t:`kobayashi2019bezier`. Both currently support a single simplex.
They return ordinary ``BezierModel`` objects, so prediction, plotting and
model serialization use the existing interfaces.

All-at-once fitting
-------------------

The unknowns are both the control points and a barycentric coordinate
for every observation. ``fit_point_cloud`` alternates two steps:

1. Hold the model fixed and minimize each point's squared distance to
   the surface using analytic first and second derivatives and Newton steps.
2. Hold the inferred coordinates fixed and solve for the controls using
   the existing float64 SVD least-squares implementation.

For example, fit a two-objective Schaffer front without providing its
parameter coordinates to the fitting routine:

.. code-block:: python

   import torch
   from torch_bsf import BezierSpace, fit_point_cloud

   u = torch.linspace(0, 1, 21, dtype=torch.float64)
   values = torch.stack((u**2, (1 - u)**2), dim=1)
   space = BezierSpace.simplex(dimension=1, degree=2)
   result = fit_point_cloud(space, values).require_success()

   fitted_values = result.model(result.parameters)
   error = (fitted_values - values).square().sum(1).mean()
   assert error < 1e-12

``result.parameters`` contains the inferred barycentric rows in the
original observation order. A zero-dimensional simplex uses empty
coordinate rows, consistent with ``BezierSpace``. ``objective_history``
records mean squared Euclidean distances, starting with the initial model
and coordinates; ``control_result`` retains the last linear solve's
diagnostics. Controls supplied through ``fixed=FixedControls(...)`` remain
exact throughout fitting.

Automatic initialization places vertex ``i`` at the sample minimizing
output ``i``, then initializes the remaining controls on the affine grid
between vertices. This follows the paper and assumes minimization
objectives, with one output per vertex. Use ``initial=BezierModel(...)``
to specify another orientation or fit a simplex embedded in a different
output dimension. The initial model is copied.

Inductive skeleton fitting
--------------------------

``fit_skeleton`` fits vertices first, then edges, then higher faces. Each
face keeps every previously estimated boundary control fixed exactly.
Only faces with at most ``min(degree, number_of_vertices)`` vertices are
needed: a degree-three simplex has no controls supported on four or more
vertices, even when its domain dimension is larger.

.. code-block:: python

   from torch_bsf import fit_skeleton

   result = fit_skeleton(space, values).require_success()
   model = result.model
   edge_result = result.faces[(0, 1)]

By default, a face's samples are the points nondominated under the
objectives corresponding to its vertices, using the paper's definition
of ``X_J``. All output columns are retained. This extraction assumes that
output columns are minimization objectives and that the Pareto front has
the stated simplex structure. It takes quadratic time in sample count
per face and linear auxiliary memory. For larger data or observations
already separated by subproblem, supply the samples explicitly:

.. code-block:: python

   faces = {
       (0,): values[:1],
       (1,): values[-1:],
       (0, 1): values,
   }
   result = fit_skeleton(space, values, face_values=faces).require_success()

Face keys are sorted tuples of zero-based vertex IDs. Supply exactly all
required faces and full output vectors for each sample. No parameter
coordinates are needed. Explicit ``face_values`` and an ``initial`` model
also allow fitting arbitrary embeddings. A vertex with repeated or noisy
observations is fitted to their mean. Each face result contains coordinates
local to that face, in its supplied or extracted sample order. A failed
face stops assembly and leaves an explicitly unsuccessful partial result.

Numerical interpretation
------------------------

The parameter updates enforce nonnegative coordinates summing to one.
Projection, backtracking and a gradient fallback safeguard Newton updates
at boundaries and singular or indefinite Hessians. These safeguards extend
the paper's plain Newton description. The algorithm is a local nonlinear
method: different initializations can give different surfaces and domain
parameterizations are generally not identifiable from values alone.

``tolerance`` bounds the absolute change in root mean squared distance;
``parameter_tolerance`` bounds the projected gradient residual. The code
uses an absolute change rather than the signed stopping expression printed
in the paper. Success requires both stopping checks. Iteration exhaustion
returns ``success=False`` and emits ``ConvergenceWarning``, preserving the
finite model. Use ``require_success()`` when a failed fit must halt a
workflow. Convergence is a numerical stopping statement, not a certificate
of global optimality or reconstruction accuracy.

Distances and tolerances use the original output units. Normalize outputs
explicitly when their scales should contribute equally. These routines
do not automatically filter the input cloud to a Pareto front or infer
face memberships for arbitrary embeddings. Choose degree and validate
surface coverage separately from numerical convergence.
