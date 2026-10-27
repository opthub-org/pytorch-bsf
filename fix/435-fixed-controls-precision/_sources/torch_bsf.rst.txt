Python API
==========

This reference follows a model through geometry construction, numerical
fitting, evaluation and optional integrations. The public types are shared
across these operations. For runnable end-to-end examples and numerical
interpretation, start with :doc:`quickstart` and the :doc:`advanced` guides.

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

interpolation
-------------

``SimplicialMesh`` compiles user-supplied simplices into a cubic
``BezierSpace`` whose shared faces share controls, and locates points in it.
``hermite_interpolant`` returns a continuous piecewise cubic ``BezierModel``
through vertex data that, when clamped, stays within the data range of each
simplex. See :doc:`advanced/interpolation` for the construction and its
error bound.

.. automodule:: torch_bsf.interpolation
   :members:
   :show-inheritance:

fitting
-------

``fit`` solves regression on known parameter-response pairs and returns
``FitResult``. Fixed controls and quadratic penalties participate in the
same solve. Evaluate ``result.model(points)`` after checking the numerical
status; there is no initial guess for this linear least-squares operation.

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
polynomial capacity. See :doc:`advanced/acceleration`.

.. automodule:: torch_bsf.solvers
   :members:
   :show-inheritance:

optimization
------------

Use ``minimize`` when evaluating a differentiable cost instead of observing
target responses. Supply an initial model, loss and typed method. Check
objective and stopping evidence in the optimization result. Sampling and
minibatch settings are explained in :doc:`advanced/sgd`.

.. automodule:: torch_bsf.optimization
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

io
--

``save_model`` and ``load_model`` preserve compiled geometry and coefficients
in a versioned archive independent of filename extension. Control-point
export is a data export rather than a replacement for topology-aware
persistence. Store application parameter transforms alongside the archive.

.. automodule:: torch_bsf.io
   :members:
   :show-inheritance:

sampling
--------

``sample`` returns seeded parameter draws and ``grid`` returns deterministic
lattices. Neither obtains response measurements. Product factors retain
independent barycentric normalization, and assembled samples retain cell
IDs. See :doc:`advanced/sampling` and :doc:`advanced/grid_sampling`.

.. automodule:: torch_bsf.sampling
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

active_learning
---------------

Active learning selects additional parameter locations from a candidate
pool. The application still obtains the responses and appends observations.
Assess progress on an external validation set rather than only the
adaptively selected training rows. See :doc:`advanced/active_learning`.

.. automodule:: torch_bsf.active_learning
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

integrations.lightning
----------------------

The optional Lightning adapter exposes the common model to a training
framework. It does not select a different core regression algorithm.
Use the core fit when solving the stated linear least-squares problem;
see :doc:`advanced/sgd` for framework-based objective optimization.

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

preprocessing
-------------

Explicit output transforms make scaling visible in training and inference.
Fit statistics on training responses, transform exact fixed values in the
same way, and invert predictions before reporting original-unit errors.
See :doc:`advanced/normalization` for penalty and validation implications.

.. automodule:: torch_bsf.preprocessing
   :members:
   :show-inheritance:
