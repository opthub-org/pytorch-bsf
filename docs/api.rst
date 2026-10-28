Python API reference
====================

The Python reference describes the public objects and their arguments.
Begin with :doc:`quickstart` for a complete fit, or :doc:`advanced` for
practical choices and interpretation. The sections below supply exact signatures
once you know which operation you need.

Most workflows start with ``BezierSpace``, ``Samples`` and ``fit``. Use
``Points`` when cell identity must travel with coordinates, and use the
returned ``FitResult.model`` for prediction. The common types live in
``torch_bsf.core`` and are also available from the package root.

For numerical methods, consult ``fitting`` and ``solvers`` for regression,
or ``optimization`` for differentiable costs. Sampling, validation and
active learning build on the same space and model. Plotting, scikit-learn,
Lightning and MLflow integrations are optional; install the relevant extra
before executing their examples.

``geometry`` describes mutable topology construction, while a compiled
``BezierSpace`` records immutable geometry and coefficient layout. The
archive interface in ``io`` preserves that layout with model coefficients.
Private modules beginning with an underscore are implementation details.

core
----

``BezierSpace`` defines the domain and global control layout. ``Points``
carries coordinates and cell IDs, ``Samples`` adds observed values and
weights, and ``BezierModel`` adds output coefficients to a space. Geometry
does not depend on output count. Use these types to make data alignment
and model state explicit.

``Points``, ``Samples`` and ``BezierModel`` preserve floating tensor and
NumPy array dtypes. Python float lists use ``torch.get_default_dtype()``
(normally float32). To retain double-precision coordinates, observations
and coefficients, construct float64 tensors or NumPy arrays before passing
them in; converting an already rounded float32 tensor cannot recover precision.
Integer coordinates and coefficients become float64; integer observations
remain integers. ``Samples`` weights are stored as CPU float64 tensors.

The constructor contracts and executable examples below use ``n_samples``
for the number of aligned rows, ``coordinate_width`` for the number of
coordinate columns, and ``n_outputs`` for the number of response columns.
See :class:`~torch_bsf.core.Points` and :class:`~torch_bsf.core.Samples` for
shapes, defaults and validation. Their ``from_cells`` methods pack local
rows of different widths; :meth:`~torch_bsf.core.Samples.from_cells` shows
the shared order of coordinates, observations and weights. These helpers
convert nonempty coordinate batches and observations to CPU float64.
:meth:`~torch_bsf.core.BezierModel.forward` describes single-cell tensor
inputs, explicit multi-cell Points and the prediction shape.

``FixedControls`` stores CPU float64 values and reads non-tensor floating
inputs directly as float64, so ``FixedControls({0: [0.2]})`` preserves the
Python float value independently of the default dtype. Existing tensor
values are converted with their original precision. Complex fixed values
are rejected.

.. automodule:: torch_bsf.core
   :members:
   :show-inheritance:

geometry
--------

Build an assembled domain with ``SimploidalSet`` and identify the faces
that must share coefficients. Compile it into ``BezierSpace`` before
fitting. Builder mutations do not alter an existing compiled space.
See :doc:`advanced/simploidal` for chart orientation and continuity.

.. automodule:: torch_bsf.geometry
   :members:
   :show-inheritance:

preprocessing
-------------

Explicit output transforms make scaling visible in training and inference.
Fit statistics on training responses, transform exact fixed values in the
same way, and invert predictions before reporting original-unit errors.
See :doc:`advanced/normalization` for penalty and validation implications.

.. automodule:: torch_bsf.preprocessing
   :members:
   :show-inheritance:

sampling
--------

``sample`` returns seeded parameter draws and ``grid`` returns deterministic
lattices. Neither obtains response measurements. Product factors retain
independent barycentric normalization, and assembled samples retain cell
IDs. See :doc:`advanced/sampling` and :doc:`advanced/grid_sampling`.
``sample_skeleton`` splits rows among a simplex's vertices, edges and larger
faces, by default in the ratio that minimizes the asymptotic risk of
inductive skeleton fitting; see :doc:`advanced/skeleton_sampling`.

.. automodule:: torch_bsf.sampling
   :members:
   :show-inheritance:

fitting
-------

``fit`` solves regression on known parameter-response pairs and returns
``FitResult``. Fixed controls and quadratic penalties participate in the
same solve. Evaluate ``result.model(points)`` after checking the numerical
status; there is no initial guess for this linear least-squares operation.
``fit_skeleton_regression`` instead fits a simplex face by face from
observations on its faces; see :doc:`advanced/skeleton_sampling`.

.. automodule:: torch_bsf.fitting
   :members:
   :show-inheritance:

point_cloud
-----------

``fit_point_cloud`` fits observations whose parameter coordinates are
unknown, alternating inferred coordinates and controls. ``fit_skeleton``
fits faces inductively while preserving their boundaries. These nonlinear
methods return explicit local convergence states. See
:doc:`advanced/point_cloud` for assumptions, initialization and examples.

.. automodule:: torch_bsf.point_cloud
   :members:
   :show-inheritance:

results
-------

Results separate a computed model from evidence about the computation.
``require_success`` raises on unsuccessful termination. Data diagnostics
describe observation identifiability, whereas system diagnostics can also
include penalty rows. Successful termination is distinct from held-out
prediction accuracy.

.. automodule:: torch_bsf.results
   :members:
   :show-inheritance:

solvers
-------

Choose a typed dense or sparse least-squares configuration. Both compute
CPU float64 models; sparse solving uses the optional SciPy dependency.
Tolerances and iteration limits concern the numerical solve rather than
polynomial capacity. See :doc:`advanced/solvers_and_execution`.

.. automodule:: torch_bsf.solvers
   :members:
   :show-inheritance:

model_selection
---------------

Validation operations compare predictions on withheld observations and
retain fold results. Degree selection changes capacity and requires an
explicit final refit when using the core selection API. Learn output
preprocessing inside each fold to prevent validation leakage.

.. automodule:: torch_bsf.model_selection
   :members:
   :show-inheritance:

ensemble
--------

An ensemble combines explicitly selected compatible models for prediction.
It is a deliberate modeling choice, not an automatic consequence of
cross-validation. Compare its validation behavior with a single model
refitted on all available training observations.

.. automodule:: torch_bsf.ensemble
   :members:
   :show-inheritance:

active_learning
---------------

Active learning selects additional parameter locations from a candidate
pool. The application still obtains the responses and appends observations.
Assess progress on an external validation set rather than only the
adaptively selected training rows. See :doc:`advanced/active_learning`.

.. automodule:: torch_bsf.active_learning
   :members:
   :show-inheritance:

optimization
------------

Use ``minimize`` when evaluating a differentiable cost instead of observing
target responses. Supply an initial model, loss and typed method. Check
objective and stopping evidence in the optimization result. Sampling and
minibatch settings are explained in :doc:`advanced/objective_minimization`.

.. automodule:: torch_bsf.optimization
   :members:
   :show-inheritance:

interpolation
-------------

``SimplicialMesh`` compiles user-supplied simplices into a cubic
``BezierSpace`` whose shared faces share controls, and locates points in it.
``hermite_interpolant`` returns a continuous piecewise cubic ``BezierModel``
through vertex data that, when clamped, stays within the data range of each
simplex. ``lattice_mesh`` triangulates a simplex lattice conformingly, and
``refine`` bisects simplices while keeping the mesh conforming. See
:doc:`advanced/interpolation` for the construction and its error bound.

.. automodule:: torch_bsf.interpolation
   :members:
   :show-inheritance:

splitting
---------

Subdivision reparameterizes a model on smaller cells while preserving its
represented function. Keep the child-to-parent coordinate maps when
comparing values. Refitting child models is a separate operation that can
change the function; see :doc:`advanced/splitting`.

.. automodule:: torch_bsf.splitting
   :members:
   :show-inheritance:

io
--

``save_model`` and ``load_model`` preserve compiled geometry and coefficients
in a versioned archive independent of filename extension. Control-point
export is a data export rather than a replacement for topology-aware
persistence. Store application parameter transforms alongside the archive.

.. automodule:: torch_bsf.io
   :members:
   :show-inheritance:

plotting
--------

``plot_model`` displays selected output coordinates and optional control
nets. Rendering density changes the evaluated mesh, not the fitted
coefficients. Higher-dimensional cells are shown through two-dimensional
faces; inspect interior residuals separately.

.. automodule:: torch_bsf.plotting
   :members:
   :show-inheritance:

sklearn
-------

``BezierRegressor`` adapts the common fit to NumPy arrays and estimator
workflows. Features must encode valid barycentric blocks. Generic feature
standardization generally violates those constraints. See
:doc:`advanced/sklearn` for degree searches and output shapes.

.. automodule:: torch_bsf.sklearn
   :members:
   :exclude-members: set_fit_request, set_score_request
   :show-inheritance:

integrations.lightning
----------------------

The optional Lightning adapter's ``regression_module`` trains the common
model on supervised samples through a Lightning Trainer. See
:doc:`advanced/solvers_and_execution` for a complete single-device Trainer example
and its training settings. Use the core fit for a direct linear least-squares solve.
Objective-only optimization uses :func:`torch_bsf.minimize` with plain
PyTorch and does not require Lightning; see :doc:`advanced/objective_minimization`.

.. automodule:: torch_bsf.integrations.lightning
   :members:
   :show-inheritance:

integrations.mlflow
-------------------

The optional MLflow integration records experiments without making the
numerical core depend on a tracking service. Retain the parameter schema,
units and reference-problem definition with recorded metrics so that
artifacts can be interpreted outside their original run.

.. automodule:: torch_bsf.integrations.mlflow
   :members:
   :show-inheritance:
