Optimal skeleton sampling
=========================

When responses can be measured at chosen parameters, a simplex, or a product
of simplices, can be fitted face by face: vertices first, then edges, then larger faces, each time
estimating only the controls interior to the face from observations on that
face. This is the inductive skeleton fitting of :cite:t:`tanaka2020asymptotic`.
Its accuracy depends on how a budget of :math:`N` observations is split among
the skeletons. This chapter gives the asymptotic risk for every dimension and
degree, the split that minimizes it, and the functions that compute, sample and
fit with it. The same functions accept simploids; see
:ref:`skeleton-sampling-simploids`.
For the theoretical extension to assembled domains, with shared or
degenerate faces and a chosen evaluation measure, see :doc:`simploidal_risk`.

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
:math:`a_0, \dots, a_{M-1}`. They follow from three facts, proved in
:doc:`skeleton_risk_theory`.

1. On a face :math:`I`, let :math:`T` be the controls whose multi-index has
   support exactly :math:`I` and :math:`B` those with smaller support. Only
   these controls are nonzero on the face. With :math:`n_I` rows, least squares
   for :math:`T` with :math:`B` fixed gives, to first order, the error

   .. math:: e_T = \eta_I - G^I_{TT}{}^{-1} G^I_{TB}\, e_B,

   where :math:`G^I` is the Gram matrix of the Bernstein basis under the
   uniform distribution on the face. The innovations :math:`\eta_I` are
   independent across faces, with covariance
   :math:`\sigma^2 G^I_{TT}{}^{-1} / n_I` to first order.
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

The available formulas are summarized below. Their proofs and exact ranges
are in :doc:`skeleton_risk_theory`; independent checks and reference
polynomials are in :doc:`skeleton_risk_verification`.

- :math:`a_m = 0` for :math:`m \geq D`: faces with more than :math:`D`
  vertices have no interior controls, and the optimal ratio gives them no rows.
- For :math:`D \geq M`, :math:`a_{M-1} = \binom{D-1}{M-1}`, the number of
  controls interior to the whole simplex. Their estimate is measured with the
  same Gram matrix it was fitted with.
- For :math:`D \leq M`,
  :math:`a_{D-1} = \binom{M}{D}^2 (3D-1)! / ((D-1)!\,(M)_{2D})`, where
  :math:`(M)_k = M (M+1) \cdots (M+k-1)`. For :math:`D = 1` this is
  :math:`a_0 = 2M / (M + 1)`. :math:`a_{D-2}` and :math:`a_{D-3}`
  also have closed forms in both :math:`D` and :math:`M`; see
  :prf:ref:`prop-skeleton-second-faces` and
  :prf:ref:`thm-skeleton-third-faces`.
- On curves (:math:`M = 2`), :math:`a_0 = 4 / (D (D + 2))` and
  :math:`a_1 = D - 1`.
- On triangles (:math:`M=3`), all coefficients are closed for every degree:
  :prf:ref:`thm-skeleton-triangle-vertex-risk` gives
  :math:`a_0(3,D)` in harmonic numbers and :math:`\binom{2D}{D}` for
  :math:`D\geq1`;
  :math:`a_1=18(H_{D-1}+H_{2D+2}-H_{D+3})/(2D+3)` for
  :math:`D\geq2`; and :math:`a_2=\binom{D-1}{2}` for :math:`D\geq3`.
  Here :math:`H_j=\sum_{i=1}^j1/i`; inactive skeletons have coefficient zero. Hence
  :math:`(\sqrt{a_0}+\sqrt{a_1}+\sqrt{a_2})^2` is an explicit formula
  for the optimal normalized triangle risk at every degree.
- Every codimension-one coefficient has a single finite rational sum:
  see :prf:ref:`thm-skeleton-codimension-one`.
- For a fixed degree, every :math:`a_m` is a rational function of :math:`M`
  and converges as :math:`M` grows.
  :prf:ref:`cor-skeleton-large-dimension` gives every limit; for example,
  :math:`a_0 \to 2 D^2 (D+1)^2 / 4^D`.

These closed forms rest on a structure theorem: the errors that a face
passes to larger faces are determined by the errors it passes to the faces
with one more vertex. More generally, :prf:ref:`thm-skeleton-scalar-sum`
gives every coefficient as a finite scalar sum, without Gram-matrix
inversion or propagation recursion. Simplification to a bounded number of
functions remains open in the ranges listed in the theory chapter.

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
degree shown on curves (:math:`M = 2`), and on triangles at degrees five and six.
:doc:`skeleton_risk_theory` proves that it wins:

- on curves at every degree;
- for :math:`D \leq 2` in every dimension;
- for every degree once the dimension is large enough, for example from
  :math:`M = 11` at degree three, from :math:`M = 20` at degree four and from
  :math:`M = 29` at degree five, :math:`M = 39` at degree six,
  :math:`M = 49` at degree seven and :math:`M = 60` at degree eight.

.. _skeleton-sampling-simploids:

Products of simplices
---------------------

The same functions accept a single simploid, such as the prism
:math:`\Delta^2 \times \Delta^1` of :doc:`simploidal`. A face of a product
is a product of one face per factor, and its dimension is the sum of the
factor dimensions. The :math:`m`-skeleton consists of the faces of dimension
:math:`m`, so ``ratio`` has one entry per dimension from 0 to the dimension
of the product. Face keys are the sorted global coordinate indices of the
face's vertices, which list the vertices of every factor. On the prism,
coordinates 0 to 2 belong to the triangle and 3 to 4 to the interval, so
``(0, 3)`` is a corner and ``(0, 1, 3)`` a triangle edge at one end.

.. testcode:: skeleton

    prism = BezierSpace.simploid((2, 1), degree=(2, 1))
    print([round(a, 4) for a in skeleton_risk_coefficients(prism)])
    print([round(r, 3) for r in optimal_skeleton_ratio(prism)])

    truth = BezierModel(prism, torch.linspace(-1, 1, prism.n_controls, dtype=torch.float64)[:, None])
    points = sample_skeleton(prism, 300, seed=0)
    result = fit_skeleton_regression(prism, Samples(points, truth(points).detach() + noise)).require_success()
    print(list(result.faces)[:2], list(result.faces)[6:8])

.. testoutput:: skeleton

    [0.5, 4.0, 0.0, 0.0]
    [0.261, 0.739, 0.0, 0.0]
    [(0, 3), (0, 4)] [(0, 1, 3), (0, 1, 4)]

A face has interior controls when no factor has more vertices than its
degree. Faces of one dimension can differ in shape. The edges of the prism
are six triangle edges at the two ends of the interval and three intervals at
the triangle vertices. With degree one along the interval, only the triangle
edges have interior controls.
:func:`~torch_bsf.sampling.sample_skeleton` therefore splits each skeleton's
rows equally among its faces with interior controls. It uses all faces of a
skeleton only when none has interior controls, which happens only when
``ratio`` asks for such a skeleton explicitly. The risk coefficients assume
the same split, a uniform distribution on each factor of a face, and error
integrated uniformly over the product. Here
:math:`(\sum_m \sqrt{a_m})^2 \approx 7.33` is below the 12 controls of the
prism, so the optimal skeleton design beats all-at-once least squares.

:prf:ref:`thm-skeleton-product-risk` gives these coefficients directly in
terms of the simplex factors. If :math:`a^{(i)}_r` is factor :math:`i`'s
coefficient and :math:`f_{i,r}=\binom{M_i}{r+1}` its active face count, then

.. math::

   F_m=\sum_{r_1+\cdots+r_k=m}\prod_i f_{i,r_i},\qquad
   A_m=F_m\sum_{r_1+\cdots+r_k=m}\prod_i\frac{a^{(i)}_{r_i}}{f_{i,r_i}}.

The sums use :math:`0\leq r_i<\min(M_i,D_i)` for factor vertex counts
:math:`M_i` and degrees :math:`D_i`; an empty sum gives :math:`A_m=0`.
The optimal dimension-based ratio remains proportional to
:math:`\sqrt{A_m}`. If sampling shares can instead be chosen separately
for every vector of factor face dimensions, the optimal normalized risk is
:math:`\prod_i(\sum_r\sqrt{a^{(i)}_r})^2`, by
:prf:ref:`cor-skeleton-product-shape-ratio`. This can improve on the
dimension-based optimum. The current ``ratio`` argument only chooses
shares by total dimension; implementing independent shares by face shape
requires a separate sampling design.

For a square with degrees :math:`(2,3)`, the two curve factors have
coefficients :math:`(1/2,1)` and :math:`(4/15,2)`. The following checks the
product coefficients and the two allocation optima against the control
count of all-at-once fitting:

.. testcode:: skeleton

    from math import prod, sqrt

    square = BezierSpace.simploid((1, 1), degree=(2, 3))
    coefficients = skeleton_risk_coefficients(square)
    dimension_risk = sum(sqrt(a) for a in coefficients) ** 2
    shape_risk = prod(sum(sqrt(a) for a in factor) ** 2 for factor in [(1/2, 1), (4/15, 2)])
    print([round(a, 6) for a in coefficients])
    print(round(dimension_risk, 6), round(shape_risk, 6), square.n_controls)

.. testoutput:: skeleton

    [0.133333, 2.533333, 2.0]
    11.363687 10.862032 12

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

The coefficients describe the limit as every active face receives more rows;
they are not finite-sample error bounds. With few rows per face, a design can
be rank deficient or have large inverse-Gram moments. The formulas
assume the uniform distribution on each face, an equal split among faces of
the same size, and risk averaged uniformly over the simplex or simploid; a different
weighting of the error changes them. On a simploid, the equal split among the
faces of one dimension with interior controls is a convention, not an
optimum: faces of different shapes can deserve different shares. Bernstein Gram matrices become
ill-conditioned as the degree grows, and
:func:`~torch_bsf.sampling.skeleton_risk_coefficients` raises ``ValueError``
once the float64 result can no longer be guaranteed to six significant digits
(above degree 14 for :math:`M = 2`, 12 for :math:`M = 3`, 11 for
:math:`M = 4`, 9 for :math:`M = 5` and 8 for :math:`M = 6`). The Gram
matrix of degree :math:`D` on a simplex with :math:`M` vertices has the
condition number :math:`\binom{2D + M - 1}{D}`, which follows from its
eigenvalues :cite:p:`kirby2017fast`. On a simploid, the condition number of
the Gram matrix on the whole simploid is the product of those of its factors'
Gram matrices, so products of many factors are rejected even at low degree
(for example, 12 segments of degree 1). In that case, pass ``ratio`` to
:func:`~torch_bsf.sampling.sample_skeleton` explicitly. A single
simplex or simploid is supported by these functions. The general
assembled-space risk and optimal allocation are proved in
:doc:`simploidal_risk`; its estimator requires a separate cell-by-cell
design and constrained fits. The product
formulas and proofs in :ref:`skeleton-risk-products` reduce a simploid's
coefficients to its simplex factors. Thus products of curves and triangles
inherit their closed forms; other factors can use the general scalar sum.

The coefficients are computed factor by factor. On a face of a simploid, the
Gram matrices and the errors that the face passes on are Kronecker products
of those of its factor faces, so the trace of the face is the product of
theirs, and each :math:`a_m` is the number of faces of dimension :math:`m` with
interior controls times a sum of such products of traces computed on the
factors' simplices. Time and memory therefore grow with the cube and the
square of the number of controls of the largest factor, not of the product:
the product of eight triangles of degree one, with 6561 controls, takes a
fraction of a second. A simplex is its own only factor, so time grows with the
cube and memory with the square of its :math:`\binom{D + M - 1}{D}` controls.
Because the condition number of the Gram matrix has a closed form, a rejected
space is rejected before anything of the size of a factor is allocated. The
widest accepted simplices, such as :math:`M = 22` at degree 4 with 12650
controls, still need several dense float64 matrices of 1.3 GB each.
