Optimal skeleton sampling
=========================

When responses can be measured at chosen parameters, a simplex can be fitted
face by face: vertices first, then edges, then larger faces, each time
estimating only the controls interior to the face from observations on that
face. This is the inductive skeleton fitting of :cite:t:`tanaka2020asymptotic`.
Its accuracy depends on how a budget of :math:`N` observations is split among
the skeletons. This chapter gives the asymptotic risk for every dimension and
degree, the split that minimizes it, and the functions that compute, sample and
fit with it.

Sample, fit and compare
-----------------------

.. testcode:: skeleton

    import torch
    from torch_bsf import BezierModel, BezierSpace, Samples, fit_skeleton_regression
    from torch_bsf.sampling import optimal_skeleton_ratio, sample_skeleton, skeleton_risk_coefficients

    space = BezierSpace.simplex(2, degree=3)
    print([round(a, 4) for a in skeleton_risk_coefficients(space)])
    print([round(r, 3) for r in optimal_skeleton_ratio(space)])

    truth = BezierModel(space, torch.linspace(-1, 1, space.n_controls, dtype=torch.float64)[:, None])
    points = sample_skeleton(space, 300, seed=0)
    noise = 0.01 * torch.randn(300, 1, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    result = fit_skeleton_regression(space, Samples(points, truth(points).detach() + noise)).require_success()
    assert sorted(result.faces)[:3] == [(0,), (0, 1), (0, 1, 2)]
    assert (result.model.coefficients - truth.coefficients).abs().max() < 0.1

.. testoutput:: skeleton

    [0.1464, 3.5357, 1.0]
    [0.117, 0.576, 0.306]

:func:`~torch_bsf.sampling.sample_skeleton` places 11.7% of the rows on
vertices, 57.6% on edges and 30.6% inside the triangle by default, splits each
skeleton's share equally among its faces, and draws uniformly on each face.
Pass ``ratio`` to choose other shares. :func:`~torch_bsf.fitting.fit_skeleton_regression`
assigns each row to the face spanned by its nonzero coordinates, so zeros must
be exact; it reports one :class:`~torch_bsf.FitResult` per face. Rows on
faces with more than :math:`D` vertices carry no information for this
estimator and are ignored, and rows of zero weight do not count towards the
row a face needs.

Use :func:`~torch_bsf.fit` instead when observations are not on faces or when
all of them should inform every control. Fitting stratified rows with ``fit``
is a different estimator, and the ratio below is not optimal for it.
:func:`~torch_bsf.fit_skeleton` is the value-only variant for unknown
parameters, discussed in :doc:`point_cloud`.

The risk and its coefficients
-----------------------------

Let the simplex have :math:`M` vertices and degree :math:`D`, let responses
:math:`x = b(t) + \varepsilon` have :math:`L` outputs with independent noise of
variance :math:`\sigma^2`, and let :math:`N^{(m)}` rows lie on the
:math:`m`-skeleton (the union of faces with :math:`m + 1` vertices), split
equally among its :math:`\binom{M}{m+1}` faces and uniform on each. Measure the
error by

.. math::

   R = E \int_{\Delta^{M-1}} \lVert \hat b(t) - b(t) \rVert^2 \, dU(t),

with :math:`U` uniform on the simplex. Then

.. math::

   R = \sigma^2 L \sum_{m=0}^{M-1} \frac{a_m}{N^{(m)}} + o\!\left(\frac{1}{N}\right).

:func:`~torch_bsf.sampling.skeleton_risk_coefficients` returns
:math:`a_0, \dots, a_{M-1}`. They follow from three facts.

1. On a face :math:`I`, let :math:`T` be the controls whose multi-index has
   support exactly :math:`I` and :math:`B` those with smaller support. Only
   these controls are nonzero on the face. With :math:`n_I` rows, least squares
   for :math:`T` with :math:`B` fixed gives, to first order, the error

   .. math:: e_T = \eta_I - G^I_{TT}{}^{-1} G^I_{TB}\, e_B,

   where :math:`G^I` is the Gram matrix of the Bernstein basis under the
   uniform distribution on the face and :math:`\eta_I` is independent of
   everything else with covariance :math:`\sigma^2 G^I_{TT}{}^{-1} / n_I`.
   Fluctuations of the empirical Gram matrices contribute only
   :math:`o(1/N)`.
2. Starting from the vertices, the errors of all controls are therefore a
   linear map :math:`e = \sum_I E_I \eta_I` of the independent face noises.
3. The risk is :math:`E\,\mathrm{tr}(e^\top G e)` with :math:`G` the Gram
   matrix on the whole simplex. With :math:`n_I = N^{(m)} / \binom{M}{m+1}`,

   .. math::

      a_m = \binom{M}{m+1} \sum_{|I| = m+1} \mathrm{tr}\!\left(G^I_{TT}{}^{-1} E_I^\top G E_I\right).

The Gram entries are exact moments of the uniform distribution on a simplex,
:math:`E[t^\alpha] = k!\,\alpha! / (|\alpha| + k)!` on a :math:`k`-simplex.

Some coefficients have closed forms:

- :math:`a_m = 0` for :math:`m \geq D`: faces with more than :math:`D`
  vertices have no interior controls, and the optimal ratio gives them no rows.
- For :math:`D \geq M`, :math:`a_{M-1} = \binom{D-1}{M-1}`, the number of
  controls interior to the whole simplex, because their estimate is measured
  with the same Gram matrix it was fitted with.
- For :math:`D = 1`, :math:`a_0 = 2M / (M + 1)`: each vertex control is a mean
  of :math:`N^{(0)} / M` rows, and :math:`E[t_i^2] = 2 / (M (M + 1))`.

The optimal ratio
-----------------

Minimizing :math:`\sum_m a_m / N^{(m)}` subject to :math:`\sum_m N^{(m)} = N`
gives, by the Cauchy-Schwarz inequality,

.. math::

   \frac{N^{(m)}}{N} = \frac{\sqrt{a_m}}{\sum_k \sqrt{a_k}}, \qquad
   R^\ast = \frac{\sigma^2 L}{N} \Bigl(\sum_m \sqrt{a_m}\Bigr)^2.

:func:`~torch_bsf.sampling.optimal_skeleton_ratio` returns these fractions.
For comparison, ordinary least squares on :math:`N` uniform rows on the whole
simplex (all-at-once fitting) has risk
:math:`\sigma^2 L \binom{D+M-1}{D} / N`, the number of controls over
:math:`N`. The table lists :math:`N R^\ast / (\sigma^2 L)` and the all-at-once
value; the smaller one is in bold.

.. list-table:: Optimal inductive skeleton risk versus all-at-once risk
   :header-rows: 1
   :stub-columns: 1

   * - Degree
     - :math:`M = 2`
     - :math:`M = 3`
     - :math:`M = 4`
     - :math:`M = 5`
     - :math:`M = 6`
   * - 1
     - **1.333** / 2
     - **1.5** / 3
     - **1.6** / 4
     - **1.667** / 5
     - **1.714** / 6
   * - 2
     - **2.914** / 3
     - **5.496** / 6
     - **8.667** / 10
     - **11.99** / 15
     - **15.17** / 21
   * - 3
     - **3.727** / 4
     - 10.65 / **10**
     - 23.88 / **20**
     - 44.75 / **35**
     - 73.14 / **56**
   * - 4
     - **4.581** / 5
     - 15.07 / **15**
     - 43.04 / **35**
     - 104.9 / **70**
     - 220.2 / **126**
   * - 5
     - **5.467** / 6
     - **20.20** / 21
     - 64.12 / **56**
     - 184.3 / **126**
     - 471.8 / **252**
   * - 6
     - **6.374** / 7
     - **26.17** / 28
     - 90.36 / **84**
     - 283.9 / **210**
     - 824.0 / **462**

The degree-two and degree-three rows agree with Tables 1 and 2 of
:cite:t:`tanaka2020asymptotic`, which give these coefficients numerically for
:math:`D \in \{2, 3\}` and :math:`2 \leq M \leq 8`. The other rows are new.
They show that the paper's observation that the inductive skeleton fitting wins
for :math:`D < 3` does not extend to a rule in the degree: it wins for every
degree shown on curves (:math:`M = 2`), and again on triangles from degree five.

Notes on the original derivation
--------------------------------

The numerical results of the paper are correct; the statements around them
need three qualifications.

- The risk is an expectation, so the limit concerns :math:`N R_N`; writing
  convergence in probability of :math:`E[(Z^\top Z)^{-1}]` mixes two notions.
- If rows are assigned to faces at random, some face receives no rows with
  positive probability for every finite :math:`N`, and the estimator and
  :math:`R_N` are undefined. The analysis needs a deterministic split, which
  :func:`~torch_bsf.sampling.sample_skeleton` uses (up to one row per face).
- The matrices :math:`\Lambda^{(m)[k]}` of the paper's recursion are indexed
  inconsistently. The derivation above avoids them by propagating the control
  errors face by face.

Limits of the result
--------------------

The coefficients are asymptotic: with few rows per face, finite-sample terms
of relative size about (controls per face) / (rows per face) remain. They
assume the uniform distribution on each face, an equal split among faces of
the same size, and risk averaged uniformly over the simplex; a different
weighting of the error changes them. Bernstein Gram matrices become
ill-conditioned as the degree grows, and
:func:`~torch_bsf.sampling.skeleton_risk_coefficients` raises ``ValueError``
once the float64 result can no longer be guaranteed to six significant digits
(above degree 14 for :math:`M = 2`, 12 for :math:`M = 3`, 11 for
:math:`M = 4`, 9 for :math:`M = 5` and 8 for :math:`M = 6`). In that case, pass
``ratio`` to :func:`~torch_bsf.sampling.sample_skeleton` explicitly. Only a single simplex is supported, not products or
assembled spaces.
