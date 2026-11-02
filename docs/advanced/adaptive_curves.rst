Adaptive parameterized curves
=============================

Use :func:`~torch_bsf.adaptive_curve.fit_adaptive_curve` when a scalar parameter
can be queried and the response may change sharply. For example, the parameter
may be a temperature rendered into a text prompt, with a vector embedding as
the response. The parameter bounds are known; the locations of possible gaps
are unknown. This interface obtains responses itself. For selection from an
existing candidate pool, see :doc:`active_learning`.

Install SciPy 1.9.2 or later with ``pip install 'pytorch-bsf[adaptive]'``.
Ordinary imports and prediction from an existing result do not require SciPy.
Fitting computes on CPU in float64 and does not retain gradients. The callback
receives an independent tensor of shape ``(n, 1)`` and returns finite responses
of shape ``(n, d)``, including ``(n, 1)`` for scalar responses.

A connected guess can become two charts
---------------------------------------

The policy starts with nine uniformly spaced parameter anchors, unless a
smaller budget limits that initialization. It reserves global exploration
and revisits retained cut proposals. Scalar queries use local interpolation
errors; vector queries combine coverage with explicit boundary localization.
Sided continuation and genuinely nested
observations supply empirical gap evidence. Vector responses additionally use
directions estimated without either candidate-bracket endpoint and a private
central query jitter. These checks are not confidence intervals.

.. doctest::

   >>> import torch
   >>> from torch_bsf import fit_adaptive_curve
   >>> def evaluate(x):
   ...     return x + (x >= 0.413).double()
   >>> result = fit_adaptive_curve(evaluate, (0.0, 1.0), budget=64)
   >>> result.cost, len(result.gaps)
   (64, 1)
   >>> result.domain.betti_numbers()[0]
   2
   >>> gap = result.gaps[0]
   >>> bool(gap.left <= 0.413 <= gap.right)
   True
   >>> midpoint = [[(gap.left + gap.right) / 2]]
   >>> bool(torch.isnan(result.predict(midpoint)).all())
   True

Each chart contains connected polynomial pieces with shared endpoint controls.
The returned ``model`` is an actual :class:`~torch_bsf.core.BezierModel` on the
returned :class:`~torch_bsf.geometry.SimploidalSet`. Distinct charts have
separate controls even if their output curves touch or intersect. This records
the proposed topology of the parameter charts; it does not recover or certify
the topology of the embedding image. This API covers one parameter, not
unparameterized point clouds or multi-parameter surfaces.

To plot the initial single-simplex fit, final charts and acquisition order,
run :download:`the standalone example <../../examples/adaptive_curve.py>`::

    pip install 'pytorch-bsf[adaptive,plot]'
    python examples/adaptive_curve.py --output build/adaptive_curve.html

The generated HTML embeds its figure and reports learning and display-only
response rows separately. The example uses a known synthetic vector curve;
it does not establish comparative benchmark performance.

Costs and initialization
------------------------

``cost`` counts actual response rows requested from the evaluator. All initial
measurements, exploration, localization, audits and stochastic repeats share
the same budget. A deterministic repeated canonical input uses a cache and
costs no additional response. ``proposed_rows`` also counts those cache hits.
``counts`` records independent repeats at each unique input; ``values`` stores
their means. A callback failure or invalid response still incurs its requested
rows; the propagated exception includes a note with the charged count.

An application can provide ``initial_inputs`` of shape ``(n, 1)``. Their
responses are obtained and charged before the policy adds any missing uniform
anchors. An initial design that already includes the nine anchors does not
query them twice. This makes a declared comparison design reproducible without
accepting uncharged responses. The default nine-anchor run is a different
protocol from a comparison using sixteen charged initial points.

With ``deterministic=False``, initial repeat measurements and later replicates
estimate pooled vector noise. The caller owns the callback's noise generator.
``seed`` controls only the policy's private vector-query jitter. Scalar query
locations use the scalar evidence policy without jitter. Neither path seeds
the global random generator. Supported seeds are integers from zero through
``2**64 - 1``; an invalid seed is rejected before obtaining initial responses.

Resolution and prediction support
---------------------------------

For a known formatted-string protocol, pass its query-grid step explicitly,
for example ``resolution=1e-6`` for six decimal places. Queries snap to the
zero-aligned grid inside the bounds. The step is supplied by the application;
the learner does not infer the formatting rule. Distinct strings can have
different embeddings even when their numeric parameters are close.

``location_tolerance`` is a maximum proposed bracket width, enlarged to at
least one declared grid step. Its default is twice the domain width divided
by 256, with a positive float64 lower limit. Smaller targets require more
localization work. They do not guarantee
that every feature will be found. ``status`` distinguishes budget exhaustion,
an exhausted declared grid, floating-point exhaustion and insufficient
observations; budget exhaustion is not convergence.

``result.predict`` accepts original coordinates of shape ``(n, 1)``. It returns
NaNs strictly inside excluded brackets and finite values at their observed
endpoints. It does not silently fill gaps with nearest observations. A declared
grid affects query coordinates, not surrogate prediction coordinates. When
the first or last grid key lies inside the raw bounds, its endpoint value
extends to that raw bound. Check support as well as prediction error on an
independent validation sample, and retain all methods' excluded regions when
comparing their errors.

The compiled model itself accepts barycentric :class:`~torch_bsf.core.Points`.
``intervals`` supplies the original parameter bounds and compiled cell IDs.
Preserve that routing metadata together with any saved model; a model archive
alone does not retain the application's original scalar coordinates.

Model selection and limits
--------------------------

Predictor selection uses observed responses only. It compares conditioned
polynomial regression, continuous piecewise polynomials, interpolation and a
shared-weight rational candidate using input-volume-weighted validation
errors. Scalar shape-preserving interpolation is excluded from vector fits
because its coordinatewise decisions do not preserve rotations.

For extreme response magnitudes, predictor losses and operator conversions
use a common power-of-two response scale. Diagnostics retain normalized
scores and their scale alongside physical squared-error scores, which can
underflow or overflow. Gap, acquisition and noise diagnostics use physical
response units and remain subject to the range of float64. An unrepresentable
pooled noise variance raises a numerical error rather than supplying an
infinite noise estimate to a solver.

When the empirical noise estimate is zero or unavailable, scalar responses
can additionally use stationary RBF and Matérn kernel predictors selected by
observed cross-validation. A jointly
estimated constant or linear mean is included in the deleted-row calculation.
Cached Cholesky solves avoid cancellation in high-condition mean predictions.
These are conditional diagnostics for the acquired design, not unbiased
end-to-end generalization estimates. The original chart predictor remains
available when its score is better or kernel conversion fails.

For vector responses with proposed gaps and a zero or unavailable empirical
noise estimate, shared continuous piecewise polynomials and component offsets
can reduce the variance of
independent chart fits. The input charts remain disconnected. Diagnostics in
``spline_pooling`` distinguish selected and fallback models and compare actual
BSF endpoint contrasts with observed endpoint contrasts; a component count
alone does not establish faithful output geometry.

These vector charts can also use a shared stationary kernel with a random
intercept for each inferred chart. A finite input-volume-weighted
cross-validation search chooses the kernel, mean and regularization. If the
best regularization reaches the upper end of its initial grid, a bounded
expansion checks larger values. One convex weight combines the selected
kernel predictor with the shared C0 predictor, using their signed deleted-row
residuals at the same interior observations. This is a conditional selection
score; it does not validate the complete adaptive pipeline.

The mixture must retain an endpoint change in the observed direction beyond
the empirical detector threshold. Otherwise, or if numerical conversion
fails, the existing predictor remains available. ``stacked_gap_prediction``
records the selected model, weight, conditional scores and conversion checks.
This extra step obtains no evaluator responses and does not change the
query locations or proposed gap brackets.

The rational candidate follows the `AAA algorithm
<https://doi.org/10.1137/16M1106122>`_. It rejects detected real poles in the
chart. Rational and kernel operators convert to polynomial pieces without
querying the evaluator. Finite checks at observed, support and additional model-only
points can reject that conversion and retain the polynomial fallback. These
checks are empirical, not a certified uniform approximation bound. Export
from the resulting polynomial pieces to Bernstein controls preserves the
polynomial, subject to floating-point rounding. Conversion diagnostics remain
in ``diagnostics['capacity']`` and ``diagnostics['kernel_prediction']``.

For vector Matérn mixtures, conversion starts at the fitted kernel's known
support coordinates. This avoids fitting a single polynomial across kernel
breakpoints. Subsequent subdivision has a finite depth and a limit of 1024
pieces. These bounds differ from starting subdivision on the whole interval;
the scalar conversion policy is separate. Even with fewer pieces, the
resulting dense Bernstein coefficients can occupy more storage than the
observations. Inspect model size as well as query cost and predictive error.

A narrow unobserved pulse, a grid alias, a continuous cusp or deterministic
roughness can be mistaken for a gap or remain undiscovered. Proposed cuts can
be revoked as measurements arrive. Independent prediction error, support,
query cost and bracket width answer different questions; none by itself
establishes true topology. This interface makes no general superiority claim
over other adaptive approximation methods.
