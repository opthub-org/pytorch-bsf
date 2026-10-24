Frequently asked questions
==========================

Do I need Conda, Lightning or MLflow?
-------------------------------------
No. Install the core with pip. Integrations are optional and explicit.

How do I predict?
-----------------
Call ``result.model(points)``. ``FitResult`` itself is not callable.

Does solver success imply identifiable coefficients?
----------------------------------------------------
No. Inspect data rank and condition diagnostics. Regularization can stabilize
a system without supplying missing observational information. See
:doc:`advanced/regularization`.

How do I choose a degree?
-------------------------
Use ``select_degree`` with explicit candidates and held-out folds, then
explicitly refit the chosen space. See :doc:`advanced/auto_degree`.

Where did the old APIs go?
--------------------------
See :doc:`migration`. The project is pre-1.0 and this release intentionally
consolidates model classes, solvers, sampling, plotting and persistence.

What are the objects I need for an ordinary fit?
------------------------------------------------

Create ``BezierSpace`` for geometry and degree, wrap parameter locations and
measured outputs in ``Samples``, then call ``fit(space, samples)``. The result
has a ``model``, metrics, numerical diagnostics and termination status. Use
``result.require_success()`` when the next operation should run only after a
successful solve. The space and observations are not modified by fitting.

A model is a PyTorch module with learned coefficients. A space contains no
learned output values and can be reused for several responses on the same
geometry. ``Points`` describes locations without targets, while ``Samples``
adds targets and optional row weights. Keeping those roles separate avoids
confusing a prediction request with new evidence for training.

Why does a one-column input produce a constant model?
-----------------------------------------------------

When dimensions are inferred, a single input column follows the constant-domain
convention. Its values are dummy features. An interval needs two barycentric
columns ``[1-x, x]``; a triangle needs three nonnegative coordinates summing
to one. A product needs one such block per factor. Check input representation
before tuning degree or penalties to fix an unexpectedly flat prediction.

In Python, construct a constant space explicitly with
``BezierSpace.simplex(0, degree=0)`` and zero-width parameter rows. The CLI and
scikit-learn adapter use one dummy feature column because tabular tools commonly
require at least one column. For unregularized constant regression, the fitted
output is the weighted observation mean.

How do I choose between simplex, product and assembled spaces?
--------------------------------------------------------------

Use a simplex for one set of trade-off weights, a product when independent
weight blocks vary together, and an assembled space when local charts must
share faces or represent a nontrivial topology. Increasing degree changes
flexibility within the chosen domain; it does not change connectivity or add
independent parameter blocks. See :doc:`advanced/simploidal` for complete
examples, coordinate layouts and shared-boundary checks.

A square product ``(1, 1)`` has four columns arranged in two sum-to-one blocks.
A tetrahedron has four columns in one sum-to-one block. Equal input widths
do not make their geometry interchangeable. Record dimensions and factor order
with raw data rather than trying to infer the domain from column count alone.

Why can a successful model still be poor?
-----------------------------------------

Solver success means the numerical termination criterion was met for the
specified problem. It does not mean the observations identify every coefficient,
the degree is suitable, or held-out predictions are accurate. A rank-deficient
minimum-norm solution can be computed successfully. Ridge can make the
augmented system full rank without adding any missing observational evidence.

Inspect ``diagnostics.data`` separately from ``diagnostics.system``, then
check independent errors and plot residuals across the domain. ``None`` in a
diagnostic field means uncomputed, not a perfect condition number or zero
rank. Sparse condition estimates are not exact singular-value diagnostics.
See :doc:`advanced/regularization` for interpreting the fields and failure states.

What should I adjust when the fit oscillates?
---------------------------------------------

First inspect observation coverage and verify coordinates and units. Then
compare lower degrees on identical held-out folds. A smaller degree may remove
unsupported variation without any penalty. If necessary, try logarithmically
spaced ridge or smoothness weights and compare prediction error, not only the
penalized training objective. More iterations are not a tuning parameter for
the direct dense regression solve.

Ridge pulls controls toward zero in output coordinates. Smoothness draws
adjacent controls together and can flatten genuine variation. Neither changes
degree or domain. A very strong penalty may make the curve look stable while
biasing it far from the true response. The before/after comparison in
:doc:`advanced/regularization` shows these different effects using the same data.

When should I normalize outputs?
--------------------------------

Use explicit transforms when output units make aggregate errors, plots or
acquisition scores misleading. Fit scaler statistics on training data, transform
targets and exact fixed values consistently, then invert predictions before
reporting physical errors. Generic feature standardization is generally
inappropriate for barycentric input blocks because it violates their constraints.

Without penalties, affine output scaling leaves the exact independent-column
least-squares predictions unchanged after inversion. It is not a way to add
capacity. Centering can change ridge's preferred origin, and relative output
scales can change model-selection or committee-query rankings. See
:doc:`advanced/normalization` for the distinction and the before/after scale plot.

Are control points predicted values on a grid?
----------------------------------------------

Vertex controls equal vertex values. Interior controls generally do not equal
the function values at corresponding barycentric lattice locations. The
coefficient net is a representation of the polynomial, not a table of observed
responses. Call ``model(points)`` to evaluate the polynomial at any desired
lattice or random point set.

Polynomial degree chooses controls; grid resolution chooses evaluation rows.
Increasing resolution makes a display denser without changing the model or
adding training information. See :doc:`advanced/grid_sampling` for point-count
formulas and examples that distinguish a denser plot from a better fit.

How can I preserve a known endpoint exactly?
--------------------------------------------

Use ``FixedControls`` with the endpoint's Bernstein multi-index resolved through
the space. Increasing an endpoint observation's weight only makes agreement a
soft preference. An exact fixed value is eliminated from the unknowns and
cannot move during fitting. On an assembled domain, a shared fixed row affects
every incident cell.

There is no strength parameter for a fixed row. If the value is uncertain,
keep it as a weighted observation. Fixing all rows returns the prescribed
model without factorization; a large resulting error means the prescribed
model conflicts with the observations, not that the constraints were ignored.
See :doc:`advanced/partial` for examples and a before/after endpoint plot.

Will splitting reduce my prediction error?
------------------------------------------

Exact subdivision preserves the parent function. It can isolate a region
for later refinement, but its two child models initially have the same error
as the parent at corresponding points. Compare child and parent values using
the returned coordinate maps. Passing the same raw coordinates to both charts
is usually an invalid comparison.

To improve a piecewise approximation, obtain suitable local observations and
refit under explicit boundary requirements. Independent child refits can break
continuity; subdividing one assembled patch does not automatically remesh its
neighbors. ``ResidualSplit`` groups existing residuals and does not promise
lower approximation error. See :doc:`advanced/splitting` for a checked exactness
comparison and the extra steps needed for refinement.

Does active learning guarantee better accuracy each round?
----------------------------------------------------------

No. Committee disagreement is a heuristic, and a batch can cluster near one
endpoint while neglecting the rest of the domain. Members can also agree
because their shared model class cannot represent the true response. Compare
with coverage-based ``farthest_points`` and maintain independent evaluation
locations. Append actual new measurements, never surrogate predictions
presented as measurements.

The guide's synthetic committee example deliberately reports a round where
whole-domain MSE increases slightly after acquisition. Its plotted local errors
explain why a high disagreement score is not a guarantee of global improvement.
See :doc:`advanced/active_learning` for candidate-pool, batch-size and output-unit
choices, with exact queries and errors saved in the generated metrics JSON.

Should I switch to the sparse solver or a GPU?
----------------------------------------------

Start with dense regression for small and medium problems. Sparse LSMR is
useful when many local cells make the global matrix mostly zero. A single
large product cell may still have a dense basis and gain little. Both core
solvers compute on CPU float64 and solve the same augmented objective. Sparse
stopping tolerances and iteration budgets can affect how closely their
predictions agree.

Lightning is an explicit optional execution adapter, not a ``fit`` solver flag.
It uses an application-owned single-device Trainer and the complete regression
objective at each step. GPU execution is not automatically faster for small
problems, and sparse-regression benchmarks do not measure this adapter. See
:doc:`advanced/acceleration` for measured scaling and an adapter example.

Why does optimization report failure after using every iteration?
-----------------------------------------------------------------

Iteration-budget exhaustion means no configured convergence criterion was
established. ``minimize`` still returns the model and history for inspection,
but it does not claim success because a curve looks flat. Set a justified
full-sample ``gradient_tolerance``, adjust step size or budget, and check the
loss and gradient path. Callback stops likewise do not certify convergence.

Histories describe pre-update minibatch values. The final objective describes
the complete fixed sample set after stopping. They need not match. More
iterations improve optimization of that fixed empirical set, not its coverage
of the continuous domain. See :doc:`advanced/sgd` for parameter effects and
:doc:`advanced/initialization` for sensitivity to the starting model.

What happens after a numerical solve fails?
-------------------------------------------

Inspect ``success``, ``status``, ``message`` and per-output diagnostics before
using the model. A finite iterative approximation may remain available even
when a condition or iteration limit stopped it. If no usable solution exists,
``diagnostics.solution_available`` is false and unavailable free coefficients
are placeholders. Do not treat their finite-looking metrics as a fitted optimum.

``require_success`` raises while retaining the regression result in the
exception. Invalid inputs raise immediately instead of creating a failed
result. A failed optional diagnostic decomposition can preserve an otherwise
computed model, so read the status rather than assuming every failure means
the same thing. The CLI reports nonzero exit status and whether a model was
saved; see :doc:`advanced/regularization` for the precise persistence policy.

How should I save a model and reproduce its figures?
----------------------------------------------------

Use ``io.save_model`` and ``io.load_model`` for the versioned non-pickled
archive. It preserves geometry and coefficients; the filename extension does
not select a different format. CSV coefficient export loses the geometry
needed to interpret shared rows and multi-cell domains. Store preprocessing
statistics, raw-data schema and experimental settings separately.

Run ``python examples/generate_guide_figures.py`` with the plot extra for the
current tutorial comparisons. Seeds, counts and measured errors are written
to ``docs/_static/guides/metrics.json``. Run the separate regularization
comparison script with plot and sparse extras for its dense/sparse-checked
figure. Historical SGD figures are explicitly labeled; they are not fresh
performance evidence for the redesigned API.

Can I compare before/after runs with different evaluation points?
-----------------------------------------------------------------

Use the same independent points, output units and summary weights whenever
you attribute a difference to degree, penalties or an acquisition round.
Changing the evaluation distribution can improve an average without improving
predictions at any particular physical location. If the intended deployment
distribution changes, state that change explicitly and report both old and
new evaluation metrics so that readers can distinguish the effects.
