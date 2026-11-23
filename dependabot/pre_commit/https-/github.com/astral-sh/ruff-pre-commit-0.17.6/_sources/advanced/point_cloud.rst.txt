Fitting from values alone
=========================

Use ``fit_point_cloud`` when each observation is a point on a curve or
surface and its domain coordinate is unknown. The existing ``fit`` solves
a different problem: regression from known parameter/value pairs.

The value-only methods implement Algorithms 2 and 3 of
:cite:t:`kobayashi2019bezier`. Both support a single simplex or simploid,
but not assembled spaces; see :ref:`point-cloud-simploids`.
``fit_point_cloud`` returns a ``PointCloudFitResult``; ``fit_skeleton``
returns a ``SkeletonFitResult``. Each result holds an ordinary ``BezierModel``
in ``result.model`` for prediction, plotting and serialization. Inspect
``result.success``, ``result.status`` and ``result.message`` for numerical
termination, or call ``result.require_success()`` to raise on failure.

All-at-once fitting
-------------------

The unknowns are both the control points and a barycentric coordinate
for every observation. ``fit_point_cloud`` alternates two steps:

1. Hold the model fixed and minimize each point's squared distance to
   the surface using analytic first and second derivatives and Newton steps.
2. Hold the inferred coordinates fixed and solve for the controls using
   the existing float64 SVD least-squares implementation.

For example, fit a two-objective Schaffer front without providing its
parameter coordinates to the fitting routine. The examples on this
page share the named ``point-cloud-demo`` session: run them in page order;
the later examples reuse ``torch``, ``space`` and ``values`` from this one.

.. testcode:: point-cloud-demo

   import torch
   from torch_bsf import BezierSpace, fit_point_cloud

   u = torch.linspace(0, 1, 21, dtype=torch.float64)
   values = torch.stack((u**2, (1 - u)**2), dim=1)
   space = BezierSpace.simplex(dimension=1, degree=2)
   result = fit_point_cloud(space, values).require_success()

   assert result.success and result.status == "converged"
   assert result.parameters.shape == (len(values), 2)
   assert torch.all(result.parameters >= 0)
   torch.testing.assert_close(
       result.parameters.sum(1), torch.ones(len(values), dtype=torch.float64),
       rtol=0, atol=8 * torch.finfo(torch.float64).eps,
   )
   fitted_values = result.model(result.parameters)
   # Row i reconstructs values[i], rather than a reordered cloud.
   row_errors = (fitted_values - values).square().sum(1)
   assert row_errors.max() < 1e-12

``result.parameters`` contains the inferred barycentric rows in the
original observation order, with shape ``(n_observations, dimension + 1)``
for a positive-dimensional simplex. Here the 21 observations have two
barycentric coordinates each; ``result.model(result.parameters)[i]``
approximates ``values[i]``. A zero-dimensional simplex uses empty
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

.. testcode:: point-cloud-demo

   from torch_bsf import fit_skeleton

   skeleton = fit_skeleton(space, values).require_success()
   model = skeleton.model
   assert skeleton.success and skeleton.status == "converged"
   assert set(skeleton.faces) == {(0,), (1,), (0, 1)}
   edge_result = skeleton.faces[(0, 1)]
   assert edge_result.success
   assert edge_result.parameters.shape == (len(values), 2)
   # The assembled edge retains the fitted vertex controls exactly.
   for vertex, powers in [(0, (2, 0)), (1, (0, 2))]:
       vertex_result = skeleton.faces[(vertex,)]
       assert vertex_result.success
       control = model.coefficients[space.control_index(powers)]
       assert torch.equal(control, vertex_result.model.coefficients[0])

``skeleton.faces`` maps sorted vertex-ID tuples to ``PointCloudFitResult``
objects. In this segment example, ``(0,)`` and ``(1,)`` identify the vertices
and ``(0, 1)`` identifies the whole edge. Each face has its own numerical
status and locally inferred coordinates; the assembled predictor is
``skeleton.model``.

By default, a face's samples are the points nondominated under the
objectives corresponding to its vertices, using the paper's definition
of ``X_J``. All output columns are retained. This extraction assumes that
output columns are minimization objectives and that the Pareto front has
the stated simplex structure. It takes quadratic time in sample count
per face and linear auxiliary memory. For larger data or observations
already separated by subproblem, supply the samples explicitly:

.. testcode:: point-cloud-demo

   faces = {
       (0,): values[:1],
       (1,): values[-1:],
       (0, 1): values,
   }
   explicit = fit_skeleton(space, values, face_values=faces).require_success()
   assert all(face.success for face in explicit.faces.values())
   for vertex, powers in [(0, (2, 0)), (1, (0, 2))]:
       control = explicit.model.coefficients[space.control_index(powers)]
       assert torch.equal(control, faces[(vertex,)][0])
   edge = explicit.faces[(0, 1)]
   assert edge.parameters.shape == (len(values), 2)
   row_errors = (edge.model(edge.parameters) - faces[(0, 1)]).square().sum(1)
   assert row_errors.max() < 1e-12

Face keys are sorted tuples of zero-based vertex IDs. Supply exactly all
required faces and full output vectors for each sample. No parameter
coordinates are needed. Explicit ``face_values`` and an ``initial`` model
also allow fitting arbitrary embeddings. A vertex with repeated or noisy
observations is fitted to their mean. Each face result contains coordinates
local to that face, in its supplied or extracted sample order. A failed
face stops assembly and leaves an explicitly unsuccessful partial result.

.. _point-cloud-simploids:

Products of simplices
---------------------

A simploid, a product of simplices, is fitted the same way. Every row's
coordinates stay on each factor's simplex, and Newton steps keep each
factor's coordinate sum. Automatic initialization and automatic face
extraction rely on the Pareto-front structure of a single simplex, so a
simploid needs an ``initial`` model for ``fit_point_cloud``, and both
``initial`` and ``face_values`` for ``fit_skeleton``. Face keys are the sorted
global coordinate indices of a face's vertices, as in
:func:`~torch_bsf.fitting.fit_skeleton_regression`. The required faces are
those in which no factor has more vertices than its degree, and each face is
fitted on the product of its factors' faces. A factor in which the face is a
single vertex contributes no local coordinates, so a face's ``parameters``
hold the face's vertices in the other factors, in key order.

.. testcode:: point-cloud-demo

   from torch_bsf import BezierModel
   from torch_bsf.sampling import sample

   # A bilinear surface over the square, with one control per corner.
   square = BezierSpace.simploid((1, 1), degree=1)
   corners = torch.tensor(
       [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.float64
   )
   surface = BezierModel(square, corners)
   observed = surface(sample(square, 40, seed=0)).detach()

   square_fit = fit_point_cloud(square, observed, initial=surface).require_success()
   assert square_fit.objective_history[-1] < 1e-12
   assert square_fit.parameters.shape == (40, 4)

   corner_values = {(0, 2): corners[:1], (0, 3): corners[1:2], (1, 2): corners[2:3], (1, 3): corners[3:]}
   square_skeleton = fit_skeleton(square, corners, face_values=corner_values, initial=surface)
   print(sorted(square_skeleton.require_success().faces))

.. testoutput:: point-cloud-demo

   [(0, 2), (0, 3), (1, 2), (1, 3)]

Numerical interpretation
------------------------

The parameter updates enforce nonnegative coordinates summing to one in
every simplex factor.
Projection, backtracking and a gradient fallback safeguard Newton updates
at boundaries and singular or indefinite Hessians. These safeguards extend
the paper's plain Newton description. The algorithm is a local nonlinear
method: different initializations can give different surfaces and domain
parameterizations are generally not identifiable from values alone.

``tolerance`` bounds the absolute change in root mean squared distance;
``parameter_tolerance`` bounds the projected gradient residual. The code
uses an absolute change rather than the signed stopping expression printed
in the paper. Success requires both stopping checks. Iteration exhaustion
returns ``success=False``, ``status="iteration_limit"`` and emits
``ConvergenceWarning``, preserving the finite model. Use ``require_success()`` when a failed fit must halt a
workflow. Convergence is a numerical stopping statement, not a certificate
of global optimality or reconstruction accuracy.

Distances and tolerances use the original output units. Normalize outputs
explicitly when their scales should contribute equally. These routines
do not automatically filter the input cloud to a Pareto front or infer
face memberships for arbitrary embeddings. Choose degree and validate
surface coverage separately from numerical convergence.
