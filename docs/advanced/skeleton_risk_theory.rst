Theory of the skeleton risk
===========================

This chapter develops the asymptotic risk of inductive skeleton fitting
on simplices and their products, from its assumptions to the general scalar
formula, then derives the closed forms and comparisons used in
:doc:`skeleton_sampling`.
The extension to general finite simploidal sets, including shared and
degenerate faces, is proved in :doc:`simploidal_risk`.
Further degree recurrences, codimension-two evaluations and vertex
reductions are proved in :doc:`skeleton_risk_extensions`.
Explicit tetrahedron vertex and six-vertex tetrahedral-face recurrences
are proved in :doc:`skeleton_risk_vertex_recurrence` and
:doc:`codimension_two_states`.
The five-vertex vertex recurrence is proved in
:doc:`skeleton_risk_five_vertex_recurrence`.
:doc:`codimension_two_family` gives a uniform fifteen-state construction
for every nonvertex codimension-two face, with both parity cases.
The tetrahedron vertex single sum and its large-degree constant are proved
in :doc:`skeleton_risk_vertex`. Its next asymptotic term and explicit error
bound are in :doc:`skeleton_risk_vertex_asymptotics`; the five-vertex vertex
single sum and its constant are in :doc:`skeleton_risk_five_vertex`.
:doc:`skeleton_risk_vertex_general` constructs every vertex coefficient by
rational univariate convolutions and gives a positive leading constant at
every fixed vertex count at least four. Explicit five-vertex error bounds
are in :doc:`skeleton_risk_five_vertex_asymptotics`.
Complete fixed-degree comparisons through
degree thirty are in :doc:`skeleton_risk_comparisons`.
:doc:`skeleton_risk_tetrahedron_comparison` classifies the tetrahedron
comparison at every degree.
The results are organized by mathematical dependence; finite verification
and reference tables are collected in :doc:`skeleton_risk_verification`.

:cite:t:`tanaka2020asymptotic` derived a Gram recursion and evaluated degrees
two and three. The results here give the asymptotic risk and optimal
allocation for every dimension and degree, as well as the following
explicit evaluations. The conditions in each row are part of the result.

.. list-table:: Results and their scope
   :header-rows: 1

   * - Range
     - Result
   * - Every :math:`M,D` and active :math:`m`
     - :prf:ref:`thm-skeleton-scalar-sum`: finite scalar sum, with no Gram inverse.
   * - :math:`D\geq M`
     - :prf:ref:`prop-skeleton-top`: :math:`a_{M-1}=\binom{D-1}{M-1}`.
   * - :math:`m=D-1,D-2,D-3`, where the skeleton exists
     - :prf:ref:`prop-skeleton-degree-faces`, :prf:ref:`prop-skeleton-second-faces`
       and :prf:ref:`thm-skeleton-third-faces`: formulas uniform in :math:`M,D`.
   * - :math:`M=2` or :math:`M=3`, every degree
     - :prf:ref:`thm-skeleton-curves`, :prf:ref:`cor-skeleton-triangle-edge-risk`
       and :prf:ref:`thm-skeleton-triangle-vertex-risk`: all coefficients in closed form.
   * - :math:`m=M-2` and :math:`D\geq M-1`
     - :prf:ref:`thm-skeleton-codimension-one`: one rational sum;
       :prf:ref:`thm-skeleton-facet-harmonics`: harmonic numbers with
       :math:`M-3` correction terms for :math:`M\geq3`, independent of the degree.
   * - Fixed :math:`D`, growing :math:`M`
     - :prf:ref:`thm-skeleton-rational`: dimension polynomials;
       :prf:ref:`cor-skeleton-large-dimension`: the limit of every coefficient.
   * - Fixed :math:`M,m`, growing :math:`D`
     - :prf:ref:`thm-risk-fixed-dimension-holonomic`: a polynomial-coefficient
       recurrence exists; explicit tetrahedron edge operators and a
       codimension-two harmonic reduction are in :doc:`skeleton_risk_extensions`.
       :prf:ref:`thm-risk-tetrahedron-vertex-recurrence` gives an order-ten
       vertex operator, and :doc:`codimension_two_states` gives an order-eight
       six-vertex tetrahedral-face operator.
       :prf:ref:`thm-risk-five-vertex-recurrence` supplies an order-seventeen
       five-vertex vertex operator, with initial values and no exceptional integers.
   * - :math:`M=4` or :math:`M=5`, :math:`m=0`, every degree
     - :prf:ref:`thm-risk-tetrahedron-vertex-single` and
       :prf:ref:`thm-risk-five-vertex-single`: one positive harmonic sum;
       :prf:ref:`cor-risk-tetrahedron-vertex-asymptotic` and
       :prf:ref:`cor-risk-five-vertex-asymptotic`: the leading constants.
       :prf:ref:`thm-risk-vertex-two-term-error` also gives the tetrahedron
       next term and an explicit error bound for :math:`D\geq256`.
       :prf:ref:`thm-risk-five-vertex-explicit-error` gives a five-vertex
       leading-term error bound on the same domain.
   * - Every :math:`M\geq2`, :math:`m=0`, every degree
     - :prf:ref:`thm-risk-general-vertex-construction`: rational univariate
       convolutions. For each fixed :math:`M\geq4`,
       :prf:ref:`thm-risk-general-vertex-asymptotic` gives a positive
       constant :math:`C_M` with :math:`a_0(M,D)\sim C_M/D^{M+1}`;
       the constant has certified rational truncation bounds.
   * - Every :math:`s\geq2`, :math:`M=s+2`, :math:`m=s-1`
     - :prf:ref:`thm-risk-uniform-codimension-two-state`: a common fifteen-state
       rational construction for every degree, including the inactive prefix.
       The output-row construction depends on :math:`s` but has no
       degree-length angular sum.
   * - Risk comparison
     - Curves win at every degree; :prf:ref:`prop-skeleton-wins` classifies
       all vertex counts for :math:`D\leq8`, and
       :prf:ref:`thm-risk-comparisons-nine-twenty` extends this through
       :math:`D=20`. :prf:ref:`thm-risk-comparisons-twentyone-thirty` covers
       degrees 21 through 30; :prf:ref:`thm-risk-tetrahedron-comparison` classifies
       every degree at :math:`M=4`. Every fixed degree wins for sufficiently
       large :math:`M`.
   * - A single product of simplices, with any positive factor degrees
     - :prf:ref:`thm-skeleton-product-risk`: coefficients from the simplex factors;
       :prf:ref:`cor-skeleton-product-shape-ratio`: optimal allocation by face shape.
   * - Any finite simploidal set, with compatible positive factor degrees
     - :prf:ref:`thm-skeleton-assembled-risk`: global propagation and weighted risk;
       :prf:ref:`cor-skeleton-assembled-allocation`: cell and grouped optima.
   * - Regular glued complexes, with any compatible positive factor degrees
     - :prf:ref:`thm-skeleton-regular-incidence`: scalar weighted face incidences;
       :prf:ref:`cor-skeleton-pure-incidence`: pure simplicial face-vector formula.
   * - Any finite simploidal set, with all factor degrees one
     - :prf:ref:`thm-skeleton-linear-assembled`: explicit corner-collision formula
       and comparison with matching all-at-once fitting.
   * - Every one-dimensional set, with any positive edge degrees
     - :prf:ref:`thm-skeleton-graph-risk`: closed coefficients including loops;
       :prf:ref:`cor-skeleton-loop-comparison`: complete single-loop comparison.

The proofs are self-contained apart from the cited small-ball inequality.
Statements certified by a finite computation are labeled computer-assisted.
The final section lists the ranges where further simplification or a
complete risk comparison is still open.

Setting
-------

Let the simplex have :math:`M\geq2` vertices
:math:`[M]=\{1,\dots,M\}` and degree :math:`D\geq1`.

**Controls and faces.** Controls are indexed by
:math:`\alpha \in \mathbb N^M` with :math:`|\alpha| = D`. The Bernstein
polynomials are :math:`B_\alpha(t) = \frac{D!}{\alpha!} t^\alpha`, and
:math:`\operatorname{supp}\alpha = \{i : \alpha_i > 0\}`. For a nonempty face
:math:`I \subseteq [M]`, let

.. math::

   T_I = \{\alpha : \operatorname{supp}\alpha = I\}, \qquad
   B_I = \{\alpha : \operatorname{supp}\alpha \subsetneq I\}

be the controls interior to :math:`I` and those on its boundary.
:math:`T_I` has :math:`\binom{D-1}{|I|-1}` elements, so it is empty when
:math:`|I| > D`. On the face :math:`I`, only the polynomials with support in
:math:`I` are nonzero.

**Gram matrices.** Let :math:`U_I` be the uniform probability measure on
the face :math:`I`, let :math:`G^I_{\alpha\beta} = \int B_\alpha B_\beta \, dU_I`,
and let :math:`G = G^{[M]}`. The Dirichlet moments

.. math::
   :label: skeleton-moments

   \int t^\gamma \, dU_I = \frac{k!\,\gamma!}{(|\gamma| + k)!}, \qquad
   k = |I| - 1, \quad \operatorname{supp}\gamma \subseteq I,

give the entries exactly. Write :math:`(M)_k = M (M+1) \cdots (M+k-1)` for
the rising factorial. The block :math:`G^I_{T_I T_I}` is positive definite,
because the Bernstein polynomials are linearly independent on the face.

**Model and design.** Each response is :math:`x = b(t) + \varepsilon \in \mathbb R^L`
with :math:`b(t) = \sum_\alpha B_\alpha(t) b_\alpha`. For every face
:math:`I` with :math:`1 \leq |I| \leq \min(D, M)`, :math:`n_I` rows are drawn
independently from :math:`U_I`. The noise vectors are independent of the rows
and of each other, with mean zero and covariance :math:`\sigma^2 I_L`.
Gaussian noise is not needed.

**Estimator.** Visit the faces in order of increasing size. On each face
:math:`I`, with :math:`T = T_I`, estimate the interior controls by least
squares on the rows of :math:`I`, holding the boundary controls at their
earlier estimates:

.. math::

   \hat b_T = \operatorname*{arg\,min}_c \sum_r \Bigl\lVert x_r
   - \sum_{\beta \in B_I} B_\beta(t_r) \hat b_\beta
   - \sum_{\alpha \in T} B_\alpha(t_r) c_\alpha \Bigr\rVert^2.

This is :func:`~torch_bsf.fitting.fit_skeleton_regression` with unit
weights.

**Risk and allocation.** The risk is
:math:`R_N = E \int \lVert \hat b(t) - b(t) \rVert^2 \, dU_{[M]}(t)` with
:math:`N = \sum_I n_I`. Let :math:`N^{(m)}` be the number of rows on faces with
:math:`m + 1` vertices. Assume that each such face receives
:math:`N^{(m)} / \binom{M}{m+1}` rows, up to one row, and that
:math:`N^{(m)} / N \to \rho_m > 0` for every :math:`m < \min(D, M)`.
:func:`~torch_bsf.sampling.sample_skeleton` with ``method="random"``
produces exactly this design.

**Propagation matrices.** For a face :math:`I` with :math:`T_I \neq \emptyset`,
define :math:`E_I` with rows indexed by all controls and columns by
:math:`T_I`:

- :math:`E_I[T_I]` is the identity;
- the row of every control whose support does not contain :math:`I` is zero;
- for the faces :math:`J \supsetneq I` with :math:`T_J \neq \emptyset`, in
  order of increasing size,

  .. math::
     :label: skeleton-propagation

     E_I[T_J] = -\bigl(G^J_{T_J T_J}\bigr)^{-1} G^J_{T_J B_J} E_I[B_J].

The risk coefficients are

.. math::

   a_m = \binom{M}{m+1} \sum_{|I| = m+1}
   \operatorname{tr}\Bigl(\bigl(G^I_{T_I T_I}\bigr)^{-1} E_I^\top G E_I\Bigr),
   \qquad m = 0, \dots, M - 1.

The sum is empty, and :math:`a_m = 0`, when :math:`m \geq D`.
:func:`~torch_bsf.sampling.skeleton_risk_coefficients` evaluates this
definition in floating point.

The asymptotic risk
-------------------

.. prf:theorem:: Asymptotic risk
   :label: thm-skeleton-risk

   Under the assumptions above, :math:`R_N` is finite for all sufficiently
   large :math:`N`, and

   .. math::

      \lim_{N \to \infty} N R_N = \sigma^2 L \sum_{m < \min(D, M)} \frac{a_m}{\rho_m}.

The proof needs to control the inverse of an empirical Gram matrix. Two
lemmas do this.

.. prf:lemma:: Small balls of sums
   :label: lem-skeleton-small-ball

   Let :math:`X_1, \dots, X_n \geq 0` be independent copies of :math:`X`, and
   suppose :math:`P(X \leq x) \leq (A x)^a` for all :math:`x > 0` and some
   :math:`A, a > 0`. Then for every :math:`\delta > 0`,

   .. math::

      P\Bigl(\sum_{i=1}^n X_i \leq n \delta\Bigr) \leq \bigl(e\,\Gamma(a+1) (A \delta)^a\bigr)^n.

.. prf:proof::

   For :math:`\lambda > 0`,
   :math:`E e^{-\lambda X} = \int_0^\infty \lambda e^{-\lambda x} P(X \leq x)\,dx
   \leq \Gamma(a+1) (A / \lambda)^a`. Markov's inequality applied to
   :math:`e^{-\lambda \sum X_i}` gives
   :math:`P(\sum X_i \leq n\delta) \leq e^{\lambda n \delta} (E e^{-\lambda X})^n`.
   Take :math:`\lambda = 1/\delta`.

.. prf:lemma:: Inverse moments of empirical Gram matrices
   :label: lem-skeleton-inverse-moments

   Let :math:`F` be a face and :math:`\chi = (B_\alpha)_{\alpha \in T}` for a
   set :math:`T` of :math:`k` controls with supports in :math:`F`, such that
   :math:`H = \int \chi \chi^\top dU_F` is positive definite. Let
   :math:`\hat H_n = \frac1n \sum_{r=1}^n \chi(t_r) \chi(t_r)^\top`,
   where the :math:`t_r` are drawn independently from :math:`U_F`. Then for every
   :math:`q > 0` there is an :math:`n_0` such that
   :math:`\sup_{n \geq n_0} E \lVert \hat H_n^{-1} \rVert^q < \infty`. In
   particular, :math:`\hat H_n` is almost surely invertible for
   :math:`n \geq n_0`.

.. prf:proof::

   Let :math:`\psi = H^{-1/2} \chi` and
   :math:`Q(u) = \frac1n \sum_r (u^\top \psi(t_r))^2`. Let :math:`\lambda`
   be the minimum of :math:`Q` on the unit sphere. Then
   :math:`\lVert \hat H_n^{-1} \rVert \leq \lVert H^{-1} \rVert / \lambda`,
   so it suffices to bound :math:`E \lambda^{-q}`.

   *Bound for one direction.* For a unit vector :math:`u`,
   :math:`f = u^\top \psi` is a polynomial of degree at most :math:`D` with
   :math:`\int f^2 \, dU_F = 1`. The uniform distribution on a convex set is
   log-concave, so the inequality of :cite:t:`carbery2001distributional` gives
   :math:`P(|f(t)| \leq s) \leq C D s^{1/D}`, with an absolute constant
   :math:`C`. Hence :math:`X = f(t)^2` satisfies :prf:ref:`lem-skeleton-small-ball` with
   :math:`a = 1/(2D)` and :math:`A = (CD)^{2D}`, and

   .. math::

      P\bigl(Q(u) \leq 4 \delta\bigr) \leq (c_1 \delta^a)^n, \qquad
      c_1 = e\,\Gamma(a+1) (4A)^a.

   *All directions at once.* Bernstein polynomials take values in
   :math:`[0, 1]`, so :math:`|\psi| \leq K = \lVert H^{-1/2} \rVert \sqrt k` and
   :math:`|\sqrt{Q(u)} - \sqrt{Q(v)}| \leq \sqrt{Q(u - v)} \leq K |u - v|`.
   Take a :math:`(\sqrt\delta / K)`-net of the unit sphere with at most
   :math:`(3K/\sqrt\delta)^k` points; :math:`\sqrt\delta \leq 1 \leq K` for
   :math:`\delta \leq 1`, because :math:`\int |\psi|^2 \, dU_F = k`. If
   :math:`\lambda \leq \delta`, some unit vector :math:`u` has
   :math:`Q(u) \leq \delta`, and the net point :math:`v` nearest to it has
   :math:`\sqrt{Q(v)} \leq 2\sqrt\delta`. A union bound over the net gives

   .. math::

      P(\lambda \leq \delta) \leq (3K)^k \delta^{-k/2} (c_1 \delta^a)^n.

   *Moment bound.* Choose :math:`\delta_1 \leq 1` with
   :math:`c_1 \delta_1^a \leq 1` and an integer :math:`n_0 > (q + k/2)/a`.
   For :math:`n \geq n_0` and :math:`\delta \leq \delta_1`,
   :math:`c_1 \delta^a \leq (\delta / \delta_1)^a \leq 1`, so the right-hand
   side is at most :math:`(3K)^k \delta_1^{-a n_0} \delta^{a n_0 - k/2}`.
   Therefore

   .. math::

      E \lambda^{-q} = \int_0^\infty q \delta^{-q-1} P(\lambda \leq \delta) \, d\delta
      \leq \delta_1^{-q} + q (3K)^k \delta_1^{-a n_0} \int_0^{\delta_1} \delta^{a n_0 - k/2 - q - 1} \, d\delta,

   which is finite and does not depend on :math:`n`.

.. prf:proof:: of :prf:ref:`thm-skeleton-risk`

   *Exact errors.* On a face :math:`I` with :math:`T = T_I` and :math:`B = B_I`,
   each row is
   :math:`x_r = \sum_{\beta \in B} B_\beta(t_r) b_\beta + \sum_{\alpha \in T} B_\alpha(t_r) b_\alpha + \varepsilon_r`.
   Let :math:`Z_T` and :math:`Z_B` be the design matrices of the rows of
   :math:`I`, and let :math:`\hat G^I = Z^\top Z / n_I` denote the corresponding
   blocks of the empirical Gram matrix. When :math:`\hat G^I_{TT}` is
   invertible, the normal equations give, exactly, the error matrix (one row
   per control, one column per output)

   .. math::

      e_T = \hat b_T - b_T = \eta_I - \bigl(\hat G^I_{TT}\bigr)^{-1} \hat G^I_{TB} e_B, \qquad
      \eta_I = (Z_T^\top Z_T)^{-1} Z_T^\top \varepsilon_I.

   Starting from the vertices, this gives :math:`e = \sum_I \hat E_I \eta_I`,
   where :math:`\hat E_I` is defined by :eq:`skeleton-propagation` with the
   empirical blocks :math:`\hat G^J` in place of :math:`G^J`. Condition on all
   rows. Then the :math:`\eta_I` are independent across faces with mean zero,
   and each column of :math:`\eta_I` has covariance
   :math:`\sigma^2 (Z_T^\top Z_T)^{-1} = \sigma^2 (\hat G^I_{TT})^{-1} / n_I`.
   Since :math:`\int \lVert \hat b - b \rVert^2 dU = \operatorname{tr}(e^\top G e)`,

   .. math::

      N R_N = \sigma^2 L\, E X_N, \qquad
      X_N = \sum_I \frac{N}{n_I} \operatorname{tr}\Bigl(\bigl(\hat G^I_{TT}\bigr)^{-1} \hat E_I^\top G \hat E_I\Bigr).

   *Almost sure limit.* :math:`R_N` depends only on the distribution of the
   design and the noise, so we may take the rows of face :math:`I` to be the
   first :math:`n_I` terms of one independent sequence per face. Every
   :math:`n_I` grows without bound. By the strong law of large numbers,
   :math:`\hat G^J \to G^J` almost surely for every face :math:`J`. The
   blocks :math:`G^J_{T_J T_J}` are invertible, so :math:`\hat E_I \to E_I`.
   Since :math:`N / n_I \to \binom{M}{m+1} / \rho_m`, it follows that
   :math:`X_N \to \sum_m a_m / \rho_m` almost surely.

   *Uniform integrability.* The entries of :math:`\hat G^J` lie in
   :math:`[0, 1]`, and :math:`N / n_I` is bounded. Unrolling
   :eq:`skeleton-propagation` therefore bounds :math:`X_N` by a constant
   times :math:`\prod_J (1 + \lVert (\hat G^J_{T_J T_J})^{-1} \rVert)^r`, for a
   fixed :math:`r` and the product over the :math:`F` faces with interior
   controls. By Hölder's inequality and :prf:ref:`lem-skeleton-inverse-moments` with
   :math:`q = 2rF`, :math:`\sup_{N \geq N_0} E X_N^2 < \infty`. Hence
   :math:`R_N` is finite for :math:`N \geq N_0`, the family :math:`X_N` is
   uniformly integrable, and :math:`E X_N` converges to the almost sure
   limit.

The same argument with a single face, the whole simplex, and all controls
gives the all-at-once risk.

.. prf:corollary:: All-at-once fitting
   :label: cor-skeleton-aao-risk

   For ordinary least squares with all controls on :math:`N` rows drawn
   independently from :math:`U_{[M]}`,
   :math:`N R_N \to \sigma^2 L \operatorname{tr}(G^{-1} G) = \sigma^2 L \binom{D+M-1}{D}`,
   the number of controls.

.. prf:theorem:: Optimal ratio
   :label: thm-skeleton-ratio

   Over fractions :math:`\rho_m \geq 0` with :math:`\sum_m \rho_m = 1`,
   :math:`\sum_m a_m / \rho_m` is minimized by
   :math:`\rho_m = \sqrt{a_m} / \sum_k \sqrt{a_k}`, and the minimum is
   :math:`(\sum_m \sqrt{a_m})^2`. A zero coefficient receives no rows;
   the corresponding term is defined as :math:`0/0=0`, while
   :math:`a_m/0=+\infty` for :math:`a_m>0`.

.. prf:proof::

   An allocation with :math:`a_m>0` and :math:`\rho_m=0` has infinite
   risk. For the remaining allocations, apply Cauchy-Schwarz on the
   indices with positive coefficients:
   :math:`(\sum_m \sqrt{a_m})^2 = (\sum_m \sqrt{a_m / \rho_m} \sqrt{\rho_m})^2
   \leq \sum_m (a_m / \rho_m) \sum_{m:a_m>0} \rho_m
   \leq \sum_m a_m/\rho_m`. Equality requires zero share on inactive
   indices and :math:`\sqrt{a_m/\rho_m}` proportional to
   :math:`\sqrt{\rho_m}` on the active ones, giving the stated ratio.

.. _skeleton-risk-products:

Products of simplices
---------------------

Let :math:`P=\prod_{i=1}^k\Delta^{M_i-1}` have :math:`M_i\geq2` vertices
and degree :math:`D_i\geq1` in factor :math:`i`. Zero-dimensional factors
can be omitted. A face :math:`F=\prod_i I_i` has dimension
:math:`\sum_i(\lvert I_i\rvert-1)` and interior controls exactly when
:math:`\lvert I_i\rvert\leq D_i` for every factor. Call these faces
*active*. On every face and on :math:`P`, use the product of the uniform
probability measures on its factors. Retain the correctly specified model,
independent noise with covariance :math:`\sigma^2I_L` for :math:`\sigma>0`,
and the estimator from the setting above, visiting faces in increasing
dimension. Here :math:`R_N` integrates the expected squared prediction
error over the uniform probability measure on :math:`P`.

Write :math:`a^{(i)}_r=a_r(M_i,D_i)` for the simplex risk coefficients and set

.. math::

   \mathcal R_i=\{0,\ldots,\min(M_i,D_i)-1\},\qquad
   f_{i,r}=\binom{M_i}{r+1}\quad(r\in\mathcal R_i).

Thus :math:`f_{i,r}` counts active :math:`r`-faces in factor :math:`i`.
The vector :math:`\mathbf r=(r_1,\ldots,r_k)` is the *shape* of a product
face; the factors retain their order, even when they have equal dimensions.
All sums over shapes below use :math:`r_i\in\mathcal R_i`.

.. prf:lemma:: Product propagation
   :label: lem-skeleton-product-propagation

   For an active face :math:`F=\prod_i I_i`, let
   :math:`H_F=G^F_{T_FT_F}`. Up to the common ordering of controls,

   .. math::

      G=\bigotimes_iG_i,\qquad H_F=\bigotimes_iH_{I_i},\qquad
      E_F=\bigotimes_iE_{I_i}.

   Consequently, the trace contributed by one face of shape :math:`\mathbf r` is

   .. math::
      :label: skeleton-product-face-trace

      \tau_{\mathbf r}
      =\operatorname{tr}(H_F^{-1}E_F^\top G E_F)
      =\prod_i\frac{a^{(i)}_{r_i}}{f_{i,r_i}^2}.

.. prf:proof::

   The Bernstein basis, the measure and the set of interior controls are
   products, which proves the two Gram identities. To prove the propagation
   identity, take the tensor product of the factor propagation matrices.
   Its block on :math:`T_F` is the identity, and its row at a control whose
   support does not contain :math:`F` is zero.

   For any strict active superface :math:`J=\prod_iJ_i\supsetneq F`, at
   least one :math:`J_i` strictly contains :math:`I_i`. On that factor, the
   error polynomial propagated from :math:`I_i` is orthogonal to every
   interior Bernstein polynomial of :math:`J_i`, by
   :eq:`skeleton-propagation`. Integrating a product interior polynomial
   against a column of the tensor-product error therefore gives zero on
   :math:`J`. These are exactly the normal equations
   :math:`H_J E_F[T_J]+G^J_{T_JB_J}E_F[B_J]=0`.
   Their unique solution on each successive face is
   :eq:`skeleton-propagation`, so the tensor product is :math:`E_F`.

   Inverses, matrix products and traces factor over Kronecker products.
   On factor :math:`i`, symmetry makes the trace equal on its
   :math:`f_{i,r}` faces, and the allocation convention gives
   :math:`a^{(i)}_r=f_{i,r}^2\operatorname{tr}(H_{I_i}^{-1}E_{I_i}^\top G_iE_{I_i})`.
   This proves :eq:`skeleton-product-face-trace`.

The same propagation identity has a useful polynomial formulation. It
shows how a closed form for a simplex error polynomial gives one on a
product, before computing any Gram trace.

.. prf:remark:: Tensor products of error polynomials
   :label: rem-skeleton-product-polynomials

   For source-face controls :math:`c_i\in\mathbb R^{T_{I_i}}`, let
   :math:`p^{(i)}_{c_i}` be the factor's error polynomial. On the product
   face :math:`F=\prod_i I_i`, put :math:`c=\bigotimes_i c_i`. Then

   .. math::
      :label: skeleton-product-polynomial

      p_{F,c}(t_1,\ldots,t_k)=\prod_i p^{(i)}_{c_i}(t_i).

   Here :math:`p_{F,c}=\sum_\alpha B_\alpha(E_Fc)_\alpha`.

.. prf:proof::

   The characterization in
   :prf:ref:`lem-skeleton-propagation-characterization` holds in the
   product Bernstein span, whose factor polynomials have degree at most
   :math:`D_i`: prescribe
   the controls on :math:`T_F`, set controls whose support does not contain
   :math:`F` to zero, and impose orthogonality to the interior Bernstein
   span on each strict active superface. Positive definite interior Gram
   blocks determine successive controls uniquely. Inactive faces have no
   interior controls and impose no condition.

   The product on the right of :eq:`skeleton-product-polynomial` has
   controls :math:`\prod_i(E_{I_i}c_i)_{\alpha_i}`. These agree with
   :math:`c` on :math:`T_F` and vanish when the support does not contain
   :math:`F`. The interior Bernstein span, or bubble space, on a superface
   :math:`J=\prod_iJ_i` is spanned by products :math:`\prod_i g_i` of factor
   bubbles. Since the face measure is a product,

   .. math::

      \int_J\prod_i p^{(i)}_{c_i}g_i\,dU_J
      =\prod_i\int_{J_i}p^{(i)}_{c_i}g_i\,dU_{J_i}=0.

   At least one :math:`J_i` strictly contains :math:`I_i`; its factor
   vanishes by the simplex characterization. Uniqueness proves the
   polynomial identity. Elementary tensors :math:`\bigotimes_i c_i` span
   all source-face controls, so linearity also recovers the full matrix
   identity :math:`E_F=\bigotimes_iE_{I_i}`.

Tensor products of orthogonal factor boundary modes are orthogonal on the
product face. Both their squared boundary norms and their propagated
squared error norms factor. Consequently, the Jacobi extensions and mode
norm ratios in :prf:ref:`thm-skeleton-scalar-sum` can be combined factor
by factor to obtain product formulas. New closed forms for a factor
therefore transfer through the polynomial identity as well as through
:eq:`skeleton-product-coefficients`. Shortening an unevaluated factor sum
remains a separate problem.

.. prf:theorem:: Product risk and allocation by dimension
   :label: thm-skeleton-product-risk

   Assign a deterministic number :math:`n_F` of independently drawn rows
   to every active face, with independent draws across faces, and let
   :math:`N=\sum_F n_F`. If :math:`n_F/N\to q_F>0`, then

   .. math::
      :label: skeleton-product-face-risk

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
      =\sum_{F\text{ active}}\frac{\tau_{\operatorname{shape}(F)}}{q_F}.

   In particular, let :math:`N^{(m)}` rows be split equally, up to one row,
   among the :math:`F_m` active faces of dimension :math:`m`, with
   :math:`N^{(m)}/N\to\rho_m>0` whenever :math:`F_m>0`. Then

   .. math::
      :label: skeleton-product-coefficients

      F_m&=\sum_{r_1+\cdots+r_k=m}\prod_i f_{i,r_i},\\
      A_m&=F_m\sum_{r_1+\cdots+r_k=m}
                      \prod_i\frac{a^{(i)}_{r_i}}{f_{i,r_i}},\\
      \lim_{N\to\infty}NR_N&=\sigma^2L\sum_{m:F_m>0}\frac{A_m}{\rho_m}.

   Empty sums give :math:`F_m=A_m=0`. Among these dimension-based
   allocations, the optimum is
   :math:`\rho_m^*=\sqrt{A_m}/\sum_j\sqrt{A_j}`, with normalized risk
   :math:`(\sum_m\sqrt{A_m})^2`.

.. prf:proof::

   The exact error recursion and conditional covariance argument of
   :prf:ref:`thm-skeleton-risk` apply to the finite collection of product
   faces. For the expectation limit, each face is convex in its affine
   span, its uniform measure is log-concave, and the product Bernstein
   basis has total polynomial degree at most :math:`\sum_iD_i`.
   Thus :prf:ref:`lem-skeleton-inverse-moments` and the same uniform
   integrability argument apply. At a vertex, the Gram block is the
   deterministic scalar one. This proves :eq:`skeleton-product-face-risk`.

   There are :math:`\prod_i f_{i,r_i}` faces of shape :math:`\mathbf r`.
   An equal split within dimension :math:`m` gives
   :math:`q_F=\rho_m/F_m`. Substitute
   :eq:`skeleton-product-face-trace` and count the faces to obtain
   :eq:`skeleton-product-coefficients`. Cauchy-Schwarz, as in
   :prf:ref:`thm-skeleton-ratio`, gives the optimum.

Equivalently, :math:`F_m` and :math:`A_m/F_m` are the coefficients of
:math:`z^m` in
:math:`\prod_i\sum_{r\in\mathcal R_i}f_{i,r}z^r` and
:math:`\prod_i\sum_{r\in\mathcal R_i}(a^{(i)}_r/f_{i,r})z^r`, respectively,
for active dimensions. This expresses the product coefficients using only
the simplex coefficients. Substitution of
:prf:ref:`thm-skeleton-scalar-sum` gives a finite scalar expression at
every factor dimension and degree. Closed forms for curves and triangles
therefore give closed forms for all products of those factors.

:func:`~torch_bsf.sampling.skeleton_risk_coefficients` computes
:math:`A_m`, and :func:`~torch_bsf.sampling.optimal_skeleton_ratio` returns
the optimal dimension-based allocation. The implementation combines
the factors' numerical trace sums; the formula also permits exact
arithmetic using the simplex scalar sums.

.. prf:corollary:: Allocation by face shape
   :label: cor-skeleton-product-shape-ratio

   Allow the share :math:`\rho_{\mathbf r}` of every active shape to be
   chosen separately, splitting it equally among faces of that shape.
   Then

   .. math::
      :label: skeleton-product-shape-risk

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
      &=\sum_{\mathbf r}\frac{\prod_i a^{(i)}_{r_i}}{\rho_{\mathbf r}},\\
      \rho_{\mathbf r}^*
      &=\prod_i\frac{\sqrt{a^{(i)}_{r_i}}}
                        {\sum_{s\in\mathcal R_i}\sqrt{a^{(i)}_s}},\\
      C_{\mathrm{shape}}^*
      &=\prod_i\left(\sum_{r\in\mathcal R_i}\sqrt{a^{(i)}_r}\right)^2
        \leq\left(\sum_m\sqrt{A_m}\right)^2.

   This is also the optimum over all individual-face allocations. The
   dimension-based optimum equals it if and only if all active faces of
   each total dimension have the same trace. In particular, equality holds
   when at most one factor has degree above one.

.. prf:proof::

   For shape :math:`\mathbf r`, the face count
   :math:`K_{\mathbf r}=\prod_i f_{i,r_i}` gives
   :math:`q_F=\rho_{\mathbf r}/K_{\mathbf r}`. Its total coefficient in
   :eq:`skeleton-product-face-risk` is
   :math:`K_{\mathbf r}^2\tau_{\mathbf r}=\prod_i a^{(i)}_{r_i}`.
   Cauchy-Schwarz minimizes the resulting sum, and
   :math:`\sum_{\mathbf r}\sqrt{\prod_i a^{(i)}_{r_i}}
   =\prod_i\sum_r\sqrt{a^{(i)}_r}` proves the factorization.
   All faces of one shape have the same positive trace, so their optimal
   individual shares are equal. Finally, every dimension-based allocation
   is a permissible shape-based allocation, proving the inequality.

   For the equality condition, let :math:`\tau_F` denote the trace of
   face :math:`F`. Cauchy-Schwarz within dimension :math:`m` gives

   .. math::

      \sum_{\dim F=m}\sqrt{\tau_F}
      \leq\sqrt{F_m\sum_{\dim F=m}\tau_F}=\sqrt{A_m}.

   Equality holds precisely when those positive traces agree. Summing
   over dimensions gives the square roots of the face and dimension
   optima, respectively, so their equality requires equality at every
   active dimension. If only factor :math:`i` has degree above one, an
   active face of dimension :math:`m` has :math:`r_i=m` and
   :math:`r_j=0` for :math:`j\neq i`. All faces of that dimension therefore
   have the same shape and trace. If all factors are linear, only vertices
   are active and their traces agree as well.

This corollary describes a larger design class than the current sampling
API: ``ratio`` has one entry per total dimension, and
:func:`~torch_bsf.sampling.sample_skeleton` splits it equally among that
dimension's active faces. It cannot specify independent shape shares.
The shape optimum requires a sampling design that implements those shares;
it does not change the face-by-face fitting rule.

.. prf:corollary:: All-at-once fitting on a product
   :label: cor-skeleton-product-aao-risk

   For ordinary least squares on :math:`N` independent rows from the
   uniform product measure,

   .. math::

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
      =\prod_i\binom{D_i+M_i-1}{D_i}.

.. prf:proof::

   The inverse-moment argument above applies to the full basis, and
   :math:`\operatorname{tr}(G^{-1}G)` is the product of the factors'
   control counts. Thus the ratio of the shape-optimal risk to the
   all-at-once risk is the product of the corresponding simplex ratios.

.. prf:corollary:: Closed forms on products
   :label: cor-skeleton-product-closed-forms

   The product coefficients :math:`A_m` have the following closed forms.

   - If :math:`D_i\geq M_i` for every factor, then
     :math:`A_{\dim P}=\prod_i\binom{D_i-1}{M_i-1}`, the number of controls
     interior to :math:`P`.
   - If :math:`D_i=1` for every factor, then
     :math:`A_0=\prod_i 2M_i/(M_i+1)` and the other coefficients vanish.
   - For :math:`k` segments of the same degree :math:`D\geq1`, put
     :math:`a_0=4/(D(D+2))` and :math:`a_1=D-1`. Then

     .. math::

        A_m=\binom km^2a_1^ma_0^{k-m}\qquad(0\leq m\leq k).

.. prf:proof::

   Substitute the simplex formulas into :eq:`skeleton-product-coefficients`.
   The top dimension has a single face, and :prf:ref:`prop-skeleton-top`
   gives the first claim. At degree one only vertices are active;
   :prf:ref:`prop-skeleton-degree-faces` gives the second. For segments,
   :prf:ref:`thm-skeleton-curves` gives :math:`(a_0,a_1)` and the two
   generating polynomials are :math:`(2+z)^k` and
   :math:`(a_0/2+a_1z)^k` when :math:`D\geq2`. Their coefficients give the
   third claim. At :math:`D=1` only :math:`A_0=a_0^k` is nonzero, giving
   the same formula.

For these equal-degree segments, every active face of dimension :math:`m`
has trace :math:`a_1^m(a_0/4)^{k-m}`. Thus allocation by dimension attains
the individual-face optimum of :prf:ref:`cor-skeleton-product-shape-ratio`.
More explicitly, with

.. math::

   p=\frac{\sqrt{a_1}}{\sqrt{a_0}+\sqrt{a_1}},

the optimal dimension fractions form the binomial distribution
:math:`\rho_m^*=\binom kmp^m(1-p)^{k-m}` and the normalized optimal risk is
:math:`(\sqrt{a_0}+\sqrt{a_1})^{2k}`. At :math:`D=1`, :math:`p=0` means all
rows go to vertices. Taking square roots of the closed form for
:math:`A_m` and applying the binomial theorem proves these statements.
By :prf:ref:`cor-skeleton-curves-win`, this risk is strictly below the
:math:`(D+1)^k` of all-at-once fitting at every :math:`k\geq1` and degree
:math:`D\geq1`.

For example, on a square with degrees :math:`(2,3)`, the two curve factors
have coefficients :math:`(1/2,1)` and :math:`(4/15,2)`. Both have face
counts :math:`(2,1)`. Therefore
:math:`(A_0,A_1,A_2)=(2/15,38/15,2)`. The normalized optimal risks are
approximately :math:`11.363687` with dimension-based allocation,
:math:`10.862032` with shape-based allocation, and :math:`12` for
all-at-once fitting. For a prism of degrees :math:`(2,1)`, the triangle
and segment coefficients :math:`(3/8,3,0)` and :math:`(4/3,0)` give
:math:`(A_0,A_1,A_2,A_3)=(1/2,4,0,0)`.

These product factorizations concern a single product cell. In an assembled
simploidal set, identified faces share controls; :doc:`simploidal_risk`
incorporates their multiplicities and propagation into the global risk.

How errors propagate
--------------------

Fix a face :math:`I` with :math:`|I| = m + 1 \leq D` and a vector
:math:`c \in \mathbb R^{T_I}`. Its *error polynomial* is
:math:`p_c(t) = \sum_\alpha B_\alpha(t) (E_I c)_\alpha`. For a face
:math:`J`, the *bubble functions* of :math:`J` are the products
:math:`t^{\mathbf 1_J} g` with :math:`g` a polynomial of degree at most
:math:`D - |J|`. These are exactly the span of
:math:`\{B_\alpha : \alpha \in T_J\}` on :math:`J`.

.. prf:lemma:: Characterization
   :label: lem-skeleton-propagation-characterization

   :math:`p_c` is the unique polynomial of degree at most :math:`D` such that

   (i) its controls in :math:`T_I` are :math:`c`;
   (ii) its controls whose support does not contain :math:`I` vanish;
   (iii) for every face :math:`J \supsetneq I` with :math:`|J| \leq D`, its
         restriction to :math:`J` is orthogonal in :math:`L^2(U_J)` to the bubble
         functions of :math:`J`.

.. prf:proof::

   On :math:`J`, :math:`p_c = \sum_{\operatorname{supp}\beta \subseteq J} B_\beta (E_I c)_\beta`,
   and the supports contained in :math:`J` are exactly :math:`T_J \cup B_J`.
   Hence :eq:`skeleton-propagation` reads
   :math:`\int B_\alpha p_c \, dU_J = 0` for every :math:`\alpha \in T_J`. That
   is condition (iii). Because :math:`G^J_{T_J T_J}` is invertible, (iii)
   determines the controls in :math:`T_J` from those in :math:`B_J`.
   Induction on :math:`|J|` gives uniqueness. Faces with more than :math:`D`
   vertices have no interior controls and impose no condition.

The propagation from :math:`I` to every larger face is determined by the
propagation to the faces with one more vertex.

.. prf:theorem:: Reduction to one more vertex
   :label: thm-skeleton-propagation-reduction

   Fix :math:`j \notin I`. Let :math:`e(\beta, n)` be the control of
   :math:`p_c` on the face :math:`I \cup \{j\}` with entries :math:`\beta` on
   :math:`I` and :math:`n` at :math:`j`. By symmetry it does not depend on
   :math:`j`. Then for every :math:`\alpha` whose support contains :math:`I`,

   .. math::
      :label: skeleton-reduction

      (E_I c)_\alpha = (n + 1)\, e(\alpha|_I, n) \prod_{l \notin I} \frac{1}{\alpha_l + 1},
      \qquad n = \sum_{l \notin I} \alpha_l.

   Equivalently, let
   :math:`\phi(x, y) = \sum_{|\beta| + n = D} e(\beta, n) \frac{D!}{\beta!\,n!} x^\beta y^n`
   be the error polynomial on :math:`I \cup \{j\}` in homogeneous form, and let
   :math:`\Phi = \partial_y (y \phi)`. Then, with :math:`s_l` independent and
   uniform on :math:`[0, 1]`,

   .. math::
      :label: skeleton-reduction-integral

      p_c(t) = E_s\, \Phi\Bigl(t_I, \sum_{l \notin I} s_l t_l\Bigr).

.. prf:proof::

   *The two forms agree.* :math:`\Phi(x, y) = \sum (n+1) e(\beta, n) \frac{D!}{\beta!\,n!} x^\beta y^n`
   and :math:`E_s (\sum_l s_l t_l)^n = \sum_{|r| = n} \frac{n!}{r!} t^r \prod_l \frac{1}{r_l + 1}`,
   because :math:`E s^r = 1/(r+1)`. Multiplying these gives the Bernstein
   coefficients :eq:`skeleton-reduction`.

   *Conditions (i) and (ii).* Let :math:`p` be the polynomial defined by
   :eq:`skeleton-reduction`. If :math:`n = 0`, then :math:`(E_I c)_\alpha = e(\alpha, 0) = c_\alpha`.
   If :math:`\alpha_i = 0` for some :math:`i \in I`, then
   :math:`e(\alpha|_I, n)` is a control of :math:`I \cup \{j\}` whose
   support does not contain :math:`I`, so it vanishes. It remains to check
   condition (iii) of :prf:ref:`lem-skeleton-propagation-characterization`.

   *Reducing (iii) to an integral over a cone.* Let :math:`J = I \cup K` with
   :math:`k = |K| \geq 1` and :math:`|J| \leq D`, and let :math:`g` be a
   polynomial of degree at most :math:`D - |J|`. Write :math:`x = t_I`,
   :math:`|x| = \sum_i x_i`, and parametrize :math:`U_J` by :math:`x` and by
   :math:`t_K` on the slice :math:`\sum t_K = 1 - |x|`. By
   :eq:`skeleton-reduction-integral` with :math:`w_l = s_l t_l`,

   .. math::

      \prod_{l \in K} t_l \; p = \int_{0 \leq w_l \leq t_l} \Phi\Bigl(x, \sum_l w_l\Bigr) dw.

   Substitute this into :math:`\int p\, t^{\mathbf 1_J} g \, dU_J`, exchange the
   integrals, and write :math:`t_K = w + v` with :math:`v \geq 0` and
   :math:`\sum v = r = 1 - |x| - \sum w`. The integral of :math:`g(x, w + v)`
   over the simplex of such :math:`v` is :math:`r^{k-1}` times a polynomial in
   :math:`(x, w, r)` of degree at most :math:`D - |J|`. Integrating over
   :math:`w` on the slices :math:`\sum w = y` contributes a factor
   :math:`y^{k-1}`. The integral is therefore a combination of

   .. math::

      \int_{x, y \geq 0,\ |x| + y \leq 1} x^{\mathbf 1_I} \Phi(x, y)\, y^{k-1} r^{k-1} x^a y^b r^c \, dx\, dy,
      \qquad r = 1 - |x| - y, \quad |a| + b + c \leq D - |J|.

   *Factoring out the radius.* Substitute :math:`(x, y) = \tau\,\omega`, where
   :math:`\tau \in (0, 1]` and :math:`\omega` lies in the simplex :math:`S`
   with :math:`m + 2` vertices, so that :math:`dx\,dy = \tau^{m+1} d\tau\, d\omega`.
   Since :math:`\Phi` is homogeneous of degree :math:`D`, each integral
   factors into a :math:`\tau`-integral times

   .. math::

      \int_S \Phi(\omega)\, \omega_I^{\mathbf 1_I}\, \omega_I^a\, \omega_y^{b + k - 1} \, d\omega,

   and :math:`\omega_I^a \omega_y^{b+k-1}` has degree at most
   :math:`D - m - 2`. So (iii) holds for every :math:`J` once

   .. math::
      :label: skeleton-reduction-orthogonality

      \int_S \Phi\, \omega_I^{\mathbf 1_I} h \, d\omega = 0
      \quad \text{for every polynomial } h \text{ of degree at most } D - m - 2.

   *The face with one more vertex.* Run the same computation for
   :math:`J = I \cup \{j\}`, where :math:`k = 1`. There :math:`p = \phi`, because
   :math:`\int_0^1 \partial_y(y\phi)(x, sy)\, ds = \phi(x, y)`. The inner
   integral is :math:`g(x, 1 - |x|)`, an arbitrary polynomial in :math:`x` of
   degree at most :math:`D - m - 2`. Take :math:`g = x^a`. Then the whole
   integral is a positive :math:`\tau`-integral times the integral in
   :eq:`skeleton-reduction-orthogonality` with :math:`h = \omega_I^a`. The
   orthogonality of :math:`\phi` on :math:`I \cup \{j\}` therefore gives
   :eq:`skeleton-reduction-orthogonality` for these :math:`h`. These
   monomials span all polynomials of that degree on :math:`S`, because
   :math:`\omega_y = 1 - |\omega_I|` there.

Scalar evaluation of every coefficient
---------------------------------------

The propagation theorem leads to one formula for all dimensions, degrees
and skeletons. It separates the angular modes on a face from the moments
of the remaining coordinates. Closed forms below are specializations of
this formula; a finite sum is available even when no shorter evaluation
is known.

.. prf:lemma:: The outside-coordinate kernel
   :label: lem-skeleton-kernel

   Define the formal power series

   .. math::

      F(x,y)&=\sum_{h,l\geq0}\frac{\binom{h+l}{h}}{(h+1)(l+1)}x^hy^l\\
        &=\frac{(1-x-y)\log(1-x-y)-(1-x)\log(1-x)
            -(1-y)\log(1-y)}{xy},\\
      K_q(h,l)&=[x^hy^l]F(x,y)^q,\qquad q,h,l\in\mathbb N.

   The quotient is interpreted by its formal series, including on the
   coordinate axes. In particular,
   :math:`K_0(h,l)=\mathbf1_{h=l=0}` and
   :math:`K_1(h,l)=\binom{h+l}{h}/((h+1)(l+1))`. For fixed :math:`h,l`,
   the coefficient is a polynomial in :math:`q` of degree at most
   :math:`h+l`, given by the finite expansion

   .. math::
      :label: skeleton-kernel-polynomial

      K_q(h,l)=\sum_{j=0}^{h+l}\binom qj[x^hy^l](F-1)^j.

   Here :math:`\binom qj=0` for :math:`j>q`. A useful boundary value is
   :math:`K_2(h,0)=K_2(0,h)=2H_{h+1}/(h+2)`, with
   :math:`H_j=\sum_{i=1}^j1/i` and :math:`H_0=0`.

.. prf:proof::

   Integrating :math:`(1-x-y)^{-1}` formally once in each variable gives
   :math:`xyF` and the displayed logarithmic expression. Expand
   :math:`F^q=(1+(F-1))^q`; since :math:`F-1` has no constant term, only
   :math:`j\leq h+l` can contribute to the coefficient. On an axis,
   :math:`F(x,0)=\sum_{h\geq0}x^h/(h+1)`. Squaring and using
   :math:`1/((j+1)(h-j+1))=((j+1)^{-1}+(h-j+1)^{-1})/(h+2)`
   proves the boundary value.

We use shifted Jacobi polynomials
:math:`P_j^{(\alpha,\beta)}(2v-1)`, orthogonal for
:math:`(1-v)^\alpha v^\beta`, with endpoint normalization
:math:`P_j^{(\alpha,\beta)}(1)=\binom{j+\alpha}{j}`.

.. prf:theorem:: Scalar risk sum
   :label: thm-skeleton-scalar-sum

   Let :math:`1\leq s\leq\min(D,M)`, :math:`n=D-s`, and :math:`q=M-s`.
   Use the kernel of :prf:ref:`lem-skeleton-kernel`. Put

   .. math::

      d_s(k)&=\binom{k+s-2}{s-2}\quad(s\geq2),&
      d_1(k)&=\mathbf1_{k=0},\\
      A_{k,h}&=(-1)^h\binom{n-k}{h}\binom{n+k+3s-1}{h}
        \quad(0\leq h\leq n-k).

   Then

   .. math::
      :label: skeleton-scalar-sum

      a_{s-1}=\frac{\binom Ms^2}{(s-1)!\,(M)_{2D}}
      \sum_{k=0}^n d_s(k)\sum_{h,l=0}^{n-k}
      A_{k,h}A_{k,l}(2D+s-h-l-1)!\,h!l!K_q(h,l).

   All factorial arguments are nonnegative. Each coefficient extraction
   uses only terms of degree at most :math:`n` in each variable, so every
   operation is finite, also when :math:`q=0` or :math:`s=1`.

.. prf:proof::

   *Angular modes.* On the face with :math:`s\geq2` vertices, choose an
   orthogonal basis :math:`Q_{k,\nu}` of polynomials of exact orthogonal
   degree :math:`k` for the weight :math:`\prod_i z_i^2` under :math:`U_I`.
   Here exact orthogonal degree means orthogonal to all polynomials of
   degree less than :math:`k`; its dimension is
   :math:`d_s(k)=\binom{k+s-2}{s-2}`. The face bubbles
   :math:`f_{k,\nu}(z)=\prod_i z_i Q_{k,\nu}(z)` for
   :math:`0\leq k\leq n` are an orthogonal basis of all face bubbles of
   degree at most :math:`D`. For :math:`s=1`, use only :math:`Q_{0,1}=1`
   and :math:`f_{0,1}=1` on the one-point face.

   *Extension by one vertex.* Write :math:`x=rz`, :math:`r=\sum_i x_i`
   and let :math:`y` be the new coordinate. The homogeneous polynomial
   :math:`\Phi` for the boundary mode :math:`f_{k,\nu}` is

   .. math::
      :label: skeleton-mode-extension

      \Phi_{k,\nu}(x,y)=\prod_i x_i\,r^k Q_{k,\nu}(z)
        \sum_{h=0}^{n-k} A_{k,h}r^{n-k-h}y^h.

   Indeed the sum, restricted to :math:`r+y=1`, is
   :math:`P_{n-k}^{(0,\,3s+2k-1)}(2r-1)`. To verify
   :eq:`skeleton-reduction-orthogonality`, divide :math:`\Phi` by
   :math:`\prod_i x_i`. The weighted volume element is
   :math:`r^{3s-1}\prod_i z_i^2\,dr\,dz`. Expand a test polynomial of
   degree at most :math:`n-1` into homogeneous angular modes. Angular
   orthogonality leaves only its degree :math:`k` components. Their
   remaining radial degrees are at most :math:`n-k-1`, with weight
   :math:`r^{3s+2k-1}`. Jacobi orthogonality makes these integrals zero.
   At :math:`y=0` the radial sum is :math:`r^{n-k}`, giving the required
   boundary mode. This proves the extension by the characterization and
   reduction theorems. At :math:`n=0` there are no test polynomials.

   *The remaining coordinates.* By
   :eq:`skeleton-reduction-integral`, replace each :math:`y^h` by

   .. math::

      T_h(t_{I^c})=E_u\Bigl(\sum_{j\notin I}u_jt_j\Bigr)^h
        =\sum_{|b|=h}\frac{h!}{b!}\prod_{j\notin I}\frac{t_j^{b_j}}{b_j+1},

   where the :math:`u_j` are independent uniform variables on
   :math:`[0,1]`. If :math:`q\geq1`, write
   :math:`t_{I^c}=(1-r)v`, with :math:`v` uniform on its
   :math:`q`-vertex simplex. The Dirichlet moments give

   .. math::

      E_v[T_h(v)T_l(v)]=\frac{h!l!}{(q)_{h+l}}K_q(h,l).

   The aggregation :math:`t_I=rz` has independent :math:`z\sim U_I`
   and :math:`r\sim\operatorname{Beta}(s,q)`. Thus the ratio of the
   squared full-simplex norm of the extended mode to its face norm is

   .. math::

      \frac{1}{(s-1)!\,(M)_{2D}}
      \sum_{h,l=0}^{n-k} A_{k,h}A_{k,l}
        (2D+s-h-l-1)!\,h!l!K_q(h,l).

   Specifically, the radial beta integral has exponents
   :math:`2D+s-h-l-1` and :math:`q+h+l-1`; its factor
   :math:`(q+h+l-1)!` cancels the corresponding denominator in the
   preceding moment. The angular norm cancels as well. Trace is invariant
   under a change of bubble basis, so summing these norm ratios, with
   :math:`d_s(k)` modes, gives the trace for one face. Multiply by
   :math:`\binom Ms^2` for the equal faces and the allocation convention.

   When :math:`q=0` there is no extension. Only :math:`h=l=0` survives
   in the stated formula, each norm ratio equals one, and the sum is
   :math:`\sum_k d_s(k)=\binom{D-1}{s-1}`, including :math:`s=1`.

Closed forms near the degree
----------------------------

.. prf:proposition:: Top face
   :label: prop-skeleton-top

   For :math:`D \geq M`, :math:`a_{M-1} = \binom{D-1}{M-1}`.

.. prf:proof::

   For :math:`I = [M]`, :math:`E_I` embeds :math:`T_I` and :math:`G^I = G`, so
   the trace is :math:`\operatorname{tr}(G_{TT}^{-1} G_{TT}) = |T_I|`.

.. prf:proposition:: Faces with D vertices
   :label: prop-skeleton-degree-faces

   For :math:`D \leq M`,

   .. math::

      a_{D-1} = \binom{M}{D}^2 \frac{(3D-1)!}{(D-1)!\,(M)_{2D}}.

   In particular, :math:`a_0 = 2M/(M+1)` for :math:`D = 1`.

.. prf:proof::

   For :math:`|I| = D`, :math:`T_I = \{\mathbf 1_I\}`. No larger face has
   interior controls, so :math:`E_I` is a unit vector and the trace is
   :math:`G_{\mathbf 1_I \mathbf 1_I} / G^I_{\mathbf 1_I \mathbf 1_I}`. By
   :eq:`skeleton-moments` this equals
   :math:`\frac{(M-1)!}{(2D+M-1)!} \cdot \frac{(3D-1)!}{(D-1)!}`. Multiply by
   :math:`\binom{M}{D}` for the faces and by :math:`\binom{M}{D}` from the
   definition.

.. prf:proposition:: Faces with D - 1 vertices
   :label: prop-skeleton-second-faces

   For :math:`2 \leq D \leq M + 1`,

   .. math::

      a_{D-2} = \binom{M}{D-1}^2 \frac{(D-1)\,(3D-3)!}{4\,(D-2)!\,(M)_{2D}}
      \bigl(3M^2 - (6D+3) M + 3D^2 + 15D - 14\bigr).

.. prf:proof::

   Put :math:`s=D-1`, :math:`n=1` and :math:`q=M-s` in
   :prf:ref:`thm-skeleton-scalar-sum`. The angular degrees :math:`k=0,1`
   have multiplicities :math:`1,s-1`; the radial vectors are
   :math:`(1,-3s)` and :math:`(1)`. With
   :math:`K_q(0,0)=1`, :math:`K_q(1,0)=q/2` and
   :math:`K_q(1,1)=q(q+1)/4`, substitution gives

   .. math::

      P_{D,D-2}(M)=\frac{s(3s)!}{4(s-1)!}
                    \bigl(3q^2-9q+12s+4\bigr).

   Replacing :math:`s=D-1` and :math:`q=M-D+1` gives the stated
   quadratic. At :math:`q=0` it reduces to :math:`s`, agreeing with
   :prf:ref:`prop-skeleton-top`, including :math:`s=1`.

.. prf:theorem:: Faces with D - 2 vertices
   :label: thm-skeleton-third-faces

   Let :math:`D\geq3`, :math:`s=D-2`, and :math:`q=M-s\geq0`. Then

   .. math::

      a_{D-3}=\binom Ms^2\frac{(3s-1)!}{(s-1)!\,(M)_{2D}}
      \frac{s^2(3s+1)}{64}\,Q_s(q),

   where the following quartic applies to every such :math:`D,M`:

   .. math::

      Q_s(q)={}&9(3s+1)q^4-78(3s+1)q^3
        +(432s^2+1209s+547)q^2\\
        &-2(648s^2+1221s+623)q
        +288(s+1)^2(3s+2).

   In particular, this includes the full range :math:`D\geq4` requested
   for :math:`a_{D-3}`, and also :math:`a_0` at :math:`D=3`.

.. prf:proof::

   Apply :prf:ref:`thm-skeleton-scalar-sum` with :math:`n=2`.
   The angular degrees :math:`k=0,1,2` have multiplicities
   :math:`1,s-1,s(s-1)/2`. At :math:`s=1` only the first is present.
   The radial coefficient vectors :math:`(A_{k,h})_h` are

   .. math::

      (1,-2(3s+1),3s(3s+1)/2),\qquad (1,-(3s+2)),\qquad(1).

   The needed kernel coefficients, obtained by multiplying the constant,
   linear and quadratic terms of :math:`F^q`, are

   .. math::

      K_q(0,0)&=1,&K_q(1,0)&=q/2,\\
      K_q(2,0)&=q(3q+5)/24,&K_q(1,1)&=q(q+1)/4,\\
      K_q(2,1)&=q(q+2)(3q+5)/48,\\
      K_q(2,2)&=q(9q^3+66q^2+163q+146)/576.

   Use symmetry for the reversed arguments. Substitution into the scalar
   sum gives the displayed quartic after factoring out
   :math:`(3s-1)!s^2(3s+1)/(64(s-1)!)`. This is a symbolic evaluation
   with arbitrary :math:`s,q`, rather than an interpolation in the degree.
   At :math:`q=0` the expression is
   :math:`s(s+1)/2=\binom{D-1}{M-1}`, as required by
   :prf:ref:`prop-skeleton-top`.

Faces of codimension one
------------------------

.. prf:theorem:: Faces of codimension one
   :label: thm-skeleton-codimension-one

   For :math:`s\geq1`, :math:`M=s+1`, and :math:`D\geq s`, put
   :math:`n=D-s` and use :math:`d_s(k)` from
   :prf:ref:`thm-skeleton-scalar-sum`. Then

   .. math::

      a_{s-1}(s+1,D)=s(s+1)^2\sum_{k=0}^n
       \frac{d_s(k)}{(n-k+1)(n+k+3s)}.

   Thus the two radial sums and the kernel extraction reduce to a single
   sum of :math:`n+1` rational terms; there are no polynomial integrals,
   Gram matrices or propagation coefficients to compute.

.. prf:proof::

   In the mode extension of :prf:ref:`thm-skeleton-scalar-sum`, there is
   just one outside coordinate :math:`y`. Averaging :math:`y^h` multiplies
   its coefficient by :math:`1/(h+1)`. Write :math:`L=n-k` and
   :math:`\beta=3s+2k-1`. The homogeneous radial polynomial therefore
   becomes, on :math:`r+y=1`,

   .. math::

      \sum_{h=0}^L\frac{(-1)^h}{h+1}\binom Lh\binom{L+\beta}{h}
        r^{L-h}(1-r)^h
      =\frac{P_L^{(1,\beta)}(2r-1)}{L+1}.

   The beta density of :math:`r` is :math:`sr^{s-1}`. After the angular
   norm cancels, the full-to-face squared-norm ratio of this mode is

   .. math::

      \frac{s}{(L+1)^2}\int_0^1 r^\beta
         P_L^{(1,\beta)}(2r-1)^2\,dr
      =\frac{s}{(L+1)(L+\beta+1)}.

   For completeness, the last integral follows from the reproducing
   kernel for polynomials of degree at most :math:`L` with weight
   :math:`r^\beta`. The shifted polynomials
   :math:`P_j^{(0,\beta)}` have squared norms
   :math:`1/(2j+\beta+1)` and endpoint value one. Hence its kernel at
   endpoint one is
   :math:`C_L(r)=\sum_{j=0}^L(2j+\beta+1)P_j^{(0,\beta)}(2r-1)`,
   with :math:`\int r^\beta C_L(r)g(r)\,dr=g(1)`.
   Taking :math:`g=(1-r)h`, :math:`\deg h<L`, identifies
   :math:`C_L=(L+\beta+1)P_L^{(1,\beta)}` by orthogonality and
   :math:`C_L(1)=(L+1)(L+\beta+1)`. Taking :math:`g=C_L` proves
   :math:`\int r^\beta P_L^{(1,\beta)}(2r-1)^2\,dr
   =(L+1)/(L+\beta+1)`.
   Sum the mode ratios with their multiplicities and multiply by
   :math:`\binom{s+1}s^2=(s+1)^2`.

.. prf:theorem:: Codimension-one risk in harmonic numbers
   :label: thm-skeleton-facet-harmonics

   For :math:`s\geq2`, :math:`M=s+1` and :math:`D\geq s`, put
   :math:`n=D-s`, :math:`r=s-2` and :math:`B=n+3s`. Define

   .. math::

      U&=\binom{D-1}{r}\bigl(H_{D-1}-H_r\bigr),\\
      V&=(-1)^r\binom{B-1}{r}
       \left(H_{B+n}-H_{B-1}
         +\sum_{j=1}^r\frac{(-1)^j\binom{n+j}{j}}
                              {j\binom{B-1}{j}}\right).

   Then the codimension-one coefficient is

   .. math::
      :label: skeleton-facet-harmonics

      a_{s-1}(s+1,D)=\frac{s(s+1)^2}{2D+s+1}(U+V).

   Here :math:`H_j=\sum_{i=1}^j1/i`, with :math:`H_0=0`; the correction
   sum is empty when :math:`s=2`. Its length is :math:`s-2=M-3`, so for
   each fixed dimension the number of terms does not grow with the degree.
   All binomial coefficients in denominators are positive in the stated
   range, including :math:`D=s`.

   In particular, the triangular-facet coefficient on a tetrahedron is

   .. math::
      :label: skeleton-tetrahedron-facet-risk

      a_2(4,D)=\frac{24}{D+2}
       \left((D-1)H_{D-1}-1
             -(D+5)\bigl(H_{2D+3}-H_{D+5}\bigr)\right),
       \qquad D\geq3.

   For example, this coefficient is :math:`16/3,612/55,2364/143` at
   :math:`D=3,4,5`. At :math:`s=2`, :eq:`skeleton-facet-harmonics`
   gives the triangle edge formula :eq:`skeleton-triangle-edge-risk`.
   At :math:`s=1` the multiplicity is supported only at :math:`k=0`, and
   :prf:ref:`thm-skeleton-codimension-one` already gives the curve vertex
   coefficient :math:`4/(D(D+2))`.

.. prf:proof::

   The partial fraction identity

   .. math::

      \frac{1}{(n+1-k)(B+k)}
       =\frac{1}{2D+s+1}
          \left(\frac{1}{n+1-k}+\frac{1}{B+k}\right)

   splits :prf:ref:`thm-skeleton-codimension-one` into two sums with
   numerator :math:`\binom{k+r}{r}`. The first is

   .. math::

      \sum_{k=0}^n\frac{\binom{k+r}{r}}{n+1-k}
        =[z^{n+1}]\frac{-\log(1-z)}{(1-z)^{r+1}}
        =\binom{n+r+1}{r}\bigl(H_{n+r+1}-H_r\bigr)=U.

   The coefficient identity follows by differentiating the binomial
   series :math:`(1-z)^{-a}` in :math:`a` at :math:`a=r+1`.
   For the second sum write
   :math:`T_r=\sum_{k=0}^n\binom{k+r}{r}/(B+k)`. Since
   :math:`\binom{k+r}{r}=(k+r)\binom{k+r-1}{r-1}/r`,

   .. math::

      T_r=\frac{1}{r}\binom{n+r}{r}+\frac{r-B}{r}T_{r-1},
        \qquad r\geq1,\qquad
      T_0=H_{B+n}-H_{B-1}.

   Iteration gives

   .. math::

      T_r=(-1)^r\binom{B-1}{r}T_0+
       \sum_{j=1}^r
        \frac{(-1)^{r-j}\binom{B-1}{r}}{j\binom{B-1}{j}}
           \binom{n+j}{j}=V.

   This proves :eq:`skeleton-facet-harmonics`. Substituting
   :math:`s=3`, :math:`r=1`, :math:`B=D+6` and simplifying gives
   :eq:`skeleton-tetrahedron-facet-risk`.

.. prf:corollary:: Codimension-one risk at large degree
   :label: cor-skeleton-facet-large-degree

   For every fixed :math:`s\geq2`, as :math:`D\to\infty`,

   .. math::

      a_{s-1}(s+1,D)
       =\frac{s(s+1)^2}{2(s-2)!}\,D^{s-3}
           \bigl(\log D+O_s(1)\bigr).

   In particular, triangle edge risk decays as
   :math:`a_1(3,D)\sim9\log D/D`, whereas the tetrahedron facet
   coefficient grows as :math:`a_2(4,D)\sim24\log D`.

.. prf:proof::

   Fix :math:`r=s-2` in :prf:ref:`thm-skeleton-facet-harmonics`.
   Its first term is
   :math:`U=D^r\bigl(\log D+O_s(1)\bigr)/r!`.
   For its second term, :math:`H_{B+n}-H_{B-1}=O_s(1)` and
   :math:`\binom{n+j}{j}/\binom{B-1}{j}=O_s(1)` for each of the
   fixed number of correction terms. Hence :math:`V=O_s(D^r)`.
   Finally the prefactor is
   :math:`s(s+1)^2(1+O_s(D^{-1}))/(2D)`.

Curves
------

.. prf:theorem:: Curves
   :label: thm-skeleton-curves

   For :math:`M = 2`, :math:`a_0 = \dfrac{4}{D(D+2)}` and :math:`a_1 = D - 1`.
   A unit error at a vertex has the error polynomial
   :math:`\varphi(x) = \sum_k b_k \binom Dk x^k (1-x)^{D-k}` along the edge,
   where :math:`x` is the coordinate of that vertex and

   .. math::

      b_k = (-1)^{D-k} \frac{k}{D(D+2)} \binom{D+2}{k+1}, \qquad
      \varphi = \frac12 \Bigl(\frac{\tilde P_D'}{D(D+1)} + \frac{\tilde P_{D+1}'}{(D+1)(D+2)}\Bigr),

   with :math:`\tilde P_n(x) = P_n(2x - 1)` the shifted Legendre polynomials.

.. prf:proof::

   *The orthogonal complement of the bubbles.* The bubble functions of the
   edge are :math:`x(1-x) g` with :math:`\deg g \leq D - 2`. For
   :math:`n \geq D`, integration by parts gives
   :math:`\int_0^1 \tilde P_n'\, x(1-x) g \, dx = -\int_0^1 \tilde P_n \bigl(x(1-x) g\bigr)' dx = 0`.
   Both boundary terms vanish, and :math:`(x(1-x)g)'` has degree at most
   :math:`D - 1`. The polynomials :math:`\tilde P_D'` and
   :math:`\tilde P_{D+1}'` have degrees :math:`D - 1` and :math:`D`. They are
   independent and span the orthogonal complement of the bubbles in the
   polynomials of degree at most :math:`D`, which has dimension two.

   *The coefficients of φ.* :math:`\tilde P_n'(1) = n(n+1)` and
   :math:`\tilde P_n'(0) = (-1)^{n-1} n(n+1)`. So
   :math:`\varphi(1) = 1` and :math:`\varphi(0) = 0` give the stated
   combination.

   *The norm of φ.* :math:`\tilde P_D'` and :math:`\tilde P_{D+1}'` are
   orthogonal, because they have opposite parity about :math:`x = 1/2`.
   Integration by parts gives
   :math:`\int_{-1}^1 P_n'^2 = P_n(1)P_n'(1) - P_n(-1)P_n'(-1) = n(n+1)`, so
   :math:`\lVert \tilde P_n' \rVert^2 = 2n(n+1)`. Therefore
   :math:`\lVert \varphi \rVert^2 = \frac14 \bigl(\frac{2}{D(D+1)} + \frac{2}{(D+1)(D+2)}\bigr) = \frac{1}{D(D+2)}`.

   *The coefficients a_0 and a_1.* A vertex block of the Gram matrix is one,
   so :math:`a_0 = \binom21 \cdot 2 \lVert \varphi \rVert^2`: the binomial
   comes from the definition of :math:`a_m`, and the sum runs over the two
   vertices.
   :math:`a_1 = D - 1` by :prf:ref:`prop-skeleton-top`.

   *The Bernstein coefficients.* Rodrigues' formula and the Leibniz rule give
   :math:`\tilde P_n = \sum_k (-1)^{n-k} \binom nk \cdot \binom nk x^k (1-x)^{n-k}`,
   so the Bernstein coefficients of :math:`\tilde P_n` are
   :math:`(-1)^{n-k}\binom nk`. Differentiating, raising the degree of
   :math:`\tilde P_D'` to :math:`D`, and adding gives :math:`b_k`.

.. prf:corollary:: Inductive fitting beats all-at-once fitting on curves
   :label: cor-skeleton-curves-win

   For :math:`M = 2` and every degree, the optimal inductive risk
   :math:`(\sum_m \sqrt{a_m})^2 = \bigl(\sqrt{D-1} + 2/\sqrt{D(D+2)}\bigr)^2`
   is smaller than the all-at-once risk :math:`D + 1`.

.. prf:proof::

   Expanding, the inequality is
   :math:`2\sqrt{(D-1)D(D+2)} < D^2 + 2D - 2`. Both sides are nonnegative.
   Squaring gives :math:`0 < (D^2 - 2)^2`, which holds for every integer
   :math:`D`.

:prf:ref:`thm-skeleton-propagation-reduction` with :math:`m = 0` turns the curve
coefficients into the propagation from a vertex in any dimension.

.. prf:corollary:: Propagation from a vertex
   :label: cor-skeleton-vertex

   For a unit error at vertex :math:`i` and every control :math:`\alpha`
   with :math:`\alpha_i \geq 1`,

   .. math::

      \bigl(E_{\{i\}}\bigr)_\alpha = (D - \alpha_i + 1)\, b_{\alpha_i} \prod_{l \neq i} \frac{1}{\alpha_l + 1}.

   At the control :math:`\mathbf 1_J` of a face :math:`J \ni i` with
   :math:`D` vertices, the error is :math:`\gamma_D = (-1)^{D-1} (D+1) / 2^D`.

.. prf:proof::

   On the edge, :math:`e(k, D - k) = b_k`. For :math:`\mathbf 1_J`,
   :math:`\alpha_i = 1` and :math:`D b_1 = (-1)^{D-1} (D+1)/2`.

Vertex coefficients in arbitrary dimension
------------------------------------------

This specialization gives the vertex coefficient and its dimension
polynomial directly from the curve controls. It is useful for exact
evaluation at fixed :math:`M` or :math:`D`.

.. prf:theorem:: Vertex specialization and dimension polynomial
   :label: thm-skeleton-vertex-sum

   Put :math:`w_k=(D-k+1)b_k`, with :math:`b_k` as in
   :prf:ref:`thm-skeleton-curves`, and use :math:`F,K_q` from
   :prf:ref:`lem-skeleton-kernel`. For every :math:`D\geq1` and
   :math:`M\geq2`,

   .. math::
      :label: skeleton-vertex-double-sum

      a_0 = \frac{M^2(D!)^2}{(M)_{2D}}
      \sum_{k=1}^D\sum_{l=1}^D
      w_k w_l\binom{k+l}{k}K_{M-1}(D-k,D-l).

   In particular, the polynomial :math:`P_{D,0}` in
   :prf:ref:`thm-skeleton-rational` has the general finite expression

   .. math::
      :label: skeleton-vertex-polynomial

      P_{D,0}(M) = (D!)^2\sum_{s=0}^{2D-2}\binom{M-1}{s}
      \sum_{k,l=1}^D w_k w_l\binom{k+l}{k}
      [x^{D-k}y^{D-l}](F-1)^s.

.. prf:proof::

   Fix vertex 1 and write :math:`\alpha=(k,u)` and
   :math:`\beta=(l,v)`. By :prf:ref:`cor-skeleton-vertex`, their propagated
   errors are :math:`w_k/\prod_j(u_j+1)` and
   :math:`w_l/\prod_j(v_j+1)`. The full Gram entry is

   .. math::

      G_{\alpha\beta}=\frac{(D!)^2}{(M)_{2D}}
      \binom{k+l}{k}\prod_{j=1}^{M-1}\binom{u_j+v_j}{u_j}.

   Summing over :math:`|u|=D-k` and :math:`|v|=D-l` gives exactly
   :math:`K_{M-1}(D-k,D-l)`. The vertex Gram block is 1. The
   :math:`M` equal vertex contributions and the allocation factor
   :math:`\binom M1=M` give the factor :math:`M^2` in
   :eq:`skeleton-vertex-double-sum`.

   For the dimension polynomial, expand
   :math:`F^{M-1}=(1+(F-1))^{M-1}`. Since :math:`F-1` has no constant
   term, its power :math:`s` cannot contribute above :math:`s=p+q`.
   Here :math:`p+q\leq2D-2`, which proves
   :eq:`skeleton-vertex-polynomial`, including the degree bound. The
   coefficient extraction uses a polynomial of bounded degree for every
   fixed :math:`D`; it requires no propagation matrices or inverses.

Triangles: all degrees
----------------------

For :math:`M=3`, the following formulas determine every coefficient and
therefore the optimal risk for every degree. Edge and interior coefficients
are zero when the degree is too small to contain their controls.

.. prf:corollary:: Triangle edge risk in harmonic numbers
   :label: cor-skeleton-triangle-edge-risk

   For every :math:`D\geq2`,

   .. math::
      :label: skeleton-triangle-edge-risk

      a_1(3,D)=\frac{18}{2D+3}
        \bigl(H_{D-1}+H_{2D+2}-H_{D+3}\bigr),
      \qquad H_j=\sum_{i=1}^j\frac1i,\quad H_0=0.

   At :math:`D=1` there are no edge controls and :math:`a_1=0`.

.. prf:proof::

   Take :math:`s=2`, so every :math:`d_2(k)=1`, and set
   :math:`j=D-1-k` in :prf:ref:`thm-skeleton-codimension-one`. Its sum is
   :math:`18\sum_{j=1}^{D-1}1/(j(2D+3-j))`. Partial fractions give
   :math:`1/(j(C-j))=(1/j+1/(C-j))/C` with :math:`C=2D+3`.
   The two sums are :math:`H_{D-1}` and
   :math:`H_{2D+2}-H_{D+3}`, respectively.

.. prf:theorem:: Triangle vertex risk in harmonic numbers
   :label: thm-skeleton-triangle-vertex-risk

   For every integer :math:`D\geq1`,

   .. math::
      :label: skeleton-triangle-vertex-risk

      a_0(3,D)=\frac{18}{D(D+1)(D+2)(D+3)}
      \left(4H_D-2H_{2D+1}+\frac1D+\frac1{D+1}+\frac1{D+2}
        -\frac{1}{D(2D+1)\binom{2D}{D}}\right).

   Here :math:`H_j=\sum_{i=1}^j1/i`, as above. All denominators are
   positive throughout the stated range. In particular, the formula gives
   :math:`3/2,3/8,41/280,79/1120,1189/30800` at
   :math:`D=1,\ldots,5`. It uses a bounded number of harmonic numbers and
   rational operations, with no kernel coefficient extraction or
   propagation solve.

.. prf:proof::

   *Replace a squared norm by a mixed product.* Write :math:`p_D(x,y,z)`
   for the error from a unit error at the vertex :math:`x=1`, and
   :math:`\varphi_D(x)` for the curve error in
   :prf:ref:`thm-skeleton-curves`. On both incident edges their traces
   agree, and both vanish on :math:`x=0`. Therefore
   :math:`p_D-\varphi_D(x)` is a triangle bubble of degree at most
   :math:`D`. The propagation characterization gives

   .. math::

      \int p_D^2\,dU=\int p_D\varphi_D(x)\,dU.

   For :math:`D=1,2` the difference is zero, so this identity still holds.
   Put :math:`A_h=(-1)^h\binom{D-1}{h}\binom{D+1}{h}`.
   The scalar-mode extension with :math:`s=1` gives
   :math:`p_D=\sum_{h=0}^{D-1}A_h x^{D-h}T_h(y,z)`.
   Averaging over :math:`(y,z)=(1-x)(v,1-v)` yields

   .. math::

      \int_0^1 T_h(v,1-v)\,dv
        =\frac1{h+1}\sum_{j=0}^h\frac1{(j+1)(h-j+1)}
        =\frac{2H_{h+1}}{(h+1)(h+2)}.

   *Only one curve moment is needed.* Let
   :math:`I_h=\int_0^1(1-x)\varphi_D(x)x^{D-h}(1-x)^h\,dx`.
   Curve orthogonality implies :math:`I_h=(-1)^h I_0`: the difference
   :math:`x^{D-h-1}(1-x)^h-(-1)^h x^{D-1}` has degree at most
   :math:`D-2` and is tested against the bubble weight :math:`x(1-x)`.
   Also :math:`\varphi_D(x)=xP_{D-1}^{(1,2)}(2x-1)/D`.
   Rodrigues' formula followed by :math:`D-1` integrations by parts gives

   .. math::

      I_0=\frac1D\int_0^1 x^2(1-x)x^{D-1}
        P_{D-1}^{(1,2)}(2x-1)\,dx
        =\frac{(D+1)!(D-1)!}{(2D+2)!}.

   Every boundary term vanishes because the differentiated Rodrigues
   factor is :math:`x^{D+1}(1-x)^D`. The formula includes :math:`D=1`,
   where there is no integration by parts. The triangle has uniform
   density two in :math:`(x,y)`, and the allocation and vertex count give
   :math:`a_0=9\int p_D^2\,dU`. Consequently, writing

   .. math::

      w_h=\binom{D+1}{h+2}\binom{D+1}{h},\qquad
      W=\binom{2D+2}{D-1},

   gives the single sum

   .. math::
      :label: skeleton-triangle-vertex-single-sum

      a_0=\frac{36}{D(D+1)(D+2)(D+3)W}
          \sum_{h=0}^{D-1}w_h H_{h+1}.

   Here we used
   :math:`\binom{D-1}{h}/((h+1)(h+2))
   =\binom{D+1}{h+2}/(D(D+1))`.

   *Evaluate the weighted harmonic sum.* Differentiate the finite
   Vandermonde identity

   .. math::

      \sum_{h=0}^{D-1}\binom{D+1+u}{D-1-h}\binom{D+1}{h}
        =\binom{2D+2+u}{D-1}

   at :math:`u=0`. The generalized binomials here are finite products,
   so differentiating is elementary and gives

   .. math::

      \sum_{h=0}^{D-1} w_hH_{h+2}
        =W\bigl(H_{D+1}-H_{2D+2}+H_{D+3}\bigr).

   To subtract :math:`\sum_h w_h/(h+2)`, put

   .. math::

      c_D&=\frac{(2D+3)(D^2+2D-1)}{D(D+1)(D+2)(D+3)},\\
      R_h&=\frac{(h+2)((D^2+2D-1)h-1)}
                    {D(D+1)(D+2)(D+3)(h+1)}.

   With :math:`w_D=0`, direct cancellation proves the telescoping identity

   .. math::
      :label: skeleton-triangle-vertex-certificate

      w_h\left(\frac1{h+2}-c_D\right)
        =w_{h+1}R_{h+1}-w_hR_h,\qquad 0\leq h\leq D-1.

   One can check it by using
   :math:`w_{h+1}/w_h=(D-1-h)(D+1-h)/((h+1)(h+3))`.
   The boundary values are :math:`w_DR_D=0` and
   :math:`w_0R_0=-1/((D+2)(D+3))`, so

   .. math::

      \sum_{h=0}^{D-1}\frac{w_h}{h+2}
        =c_DW+\frac1{(D+2)(D+3)}.

   Subtract this equality from the differentiated Vandermonde identity
   and substitute in :eq:`skeleton-triangle-vertex-single-sum`.
   Simplifying, with
   :math:`W/\binom{2D}{D}=2D(2D+1)/((D+2)(D+3))`, gives the result.
   This proves the formula for all :math:`D\geq1`, including all boundary
   terms, without a computer-algebra dependency or an inferred recurrence.

Together with :prf:ref:`cor-skeleton-triangle-edge-risk` and
:prf:ref:`prop-skeleton-top`, this closes every triangle coefficient at
every degree: :math:`a_2(3,D)=\binom{D-1}{2}` for :math:`D\geq3`,
and :math:`a_2=0` otherwise. The optimal risk follows by substituting
these three coefficients in :prf:ref:`thm-skeleton-ratio`. In particular,

.. math::

   a_0(3,D)=\frac{36\log D+O(1)}{D^4},\qquad
   a_1(3,D)=\frac{9\log D+O(1)}D.

Explicit triangular propagation modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general extension also gives the following control-level formula, used
by the independent exact verification of the triangle risks.

.. prf:proposition:: Propagation from an edge via Jacobi polynomials
   :label: prop-skeleton-triangle-jacobi

   Let :math:`m=1`, :math:`D\geq2`, :math:`N=D-2`, and let
   :math:`I=\{1,2\}`. On the triangle with coordinates
   :math:`(x_1,x_2,y)`, put :math:`r=x_1+x_2` and
   :math:`z=x_1/r`. Write the boundary error polynomial as

   .. math::

      p_c(z,1-z,0)=z(1-z)\sum_{k=0}^N d_k P_k^{(2,2)}(2z-1).

   The polynomial :math:`\Phi` of
   :prf:ref:`thm-skeleton-propagation-reduction`, on this triangle, is

   .. math::
      :label: skeleton-triangle-jacobi

      \Phi(x_1,x_2,y)=x_1x_2\sum_{k=0}^N d_k r^k
      P_k^{(2,2)}(2z-1)P_{N-k}^{(0,2k+5)}(2r-1).

   The expression has a polynomial extension to :math:`r=0`. Homogenize it
   to degree :math:`D` before using
   :eq:`skeleton-reduction-integral`. More explicitly, the homogeneous
   polynomial is obtained by replacing the angular factor by

   .. math::

      H_k(x_1,x_2)=\sum_{h=0}^k(-1)^{k-h}
      \binom{k+2}{h}\binom{k+2}{k-h}x_1^h x_2^{k-h}

   and the radial factor by

   .. math::

      R_{N,k}(r,y)=\sum_{j=0}^{N-k}(-1)^{N-k-j}
      \binom{N-k}{j}\binom{N+k+5}{N-k-j}r^j y^{N-k-j}.

   Thus :math:`\Phi=x_1x_2\sum_k d_k H_k R_{N,k}` in homogeneous
   coordinates. If its coefficient of :math:`x_1^a x_2^b y^n` is
   :math:`A_{abn}`, the Bernstein control of the triangle error
   :math:`\phi` is

   .. math::
      :label: skeleton-triangle-jacobi-controls

      e((a,b),n)=\frac{A_{abn}\,a!b!n!}{D!(n+1)}.

.. prf:proof::

   Specialize :eq:`skeleton-mode-extension` to :math:`s=2` and use
   :math:`Q_k(z)=P_k^{(2,2)}(2z-1)`. Then :math:`n=N`, each angular
   multiplicity is one, and the radial parameter is :math:`2k+5`.
   The displayed homogeneous sums are the binomial expansions of those
   two Jacobi factors. Finally :math:`\partial_y(y\phi)=\Phi` divides
   the coefficient of :math:`y^n` by :math:`n+1`; conversion from
   monomials to Bernstein controls gives
   :eq:`skeleton-triangle-jacobi-controls`.

Fixed degree, growing dimension
-------------------------------

.. prf:theorem:: Rational dependence on the dimension
   :label: thm-skeleton-rational

   For fixed :math:`D` and :math:`m < D`, there is a polynomial
   :math:`P_{D,m}` of degree at most :math:`2(D - m - 1)` such that, for every
   :math:`M \geq m + 1`,

   .. math::

      a_m = \binom{M}{m+1}^2 \frac{P_{D,m}(M)}{(M)_{2D}}.

.. prf:proof::

   Set :math:`s=m+1` and :math:`n=D-s` in
   :eq:`skeleton-scalar-sum`. After multiplying by
   :math:`(M)_{2D}/\binom Ms^2`, all factors except
   :math:`K_{M-s}(h,l)` are independent of :math:`M`.
   By :eq:`skeleton-kernel-polynomial`, this kernel is a polynomial of
   degree at most :math:`h+l\leq2(n-k)\leq2n`. Summing gives the claim,
   including the boundary :math:`M=s`.

The polynomials through degree eight and their independent orbit-counting
verification are in :doc:`skeleton_risk_verification`. They provide exact
reference values and the finite calculations needed for the comparison
below; they are not substitutes for the all-degree formulas above.

.. prf:corollary:: Large dimension
   :label: cor-skeleton-large-dimension

   Fix :math:`D\geq1` and :math:`1\leq s\leq D`, and put :math:`n=D-s`.
   Every active coefficient has the positive limit

   .. math::
      :label: skeleton-large-dimension-limit

      \lim_{M\to\infty}a_{s-1}(M,D)
       =\frac{(3s-1)!}{(s-1)!(s!)^2\,4^n}
         \binom{n+3s-1}{n}^{\!2}.

   Thus the degree bound in :prf:ref:`thm-skeleton-rational` is attained
   for every coefficient. For example,
   :math:`\lim_{M\to\infty}a_0=2D^2(D+1)^2/4^D`.
   The optimal inductive risk :math:`(\sum_m\sqrt{a_m})^2` is bounded
   in :math:`M`, while the all-at-once risk :math:`\binom{D+M-1}{D}`
   grows without bound. Inductive fitting therefore wins in all
   sufficiently high dimensions for every degree.

.. prf:proof::

   Since :math:`F-1=(x+y)/2+O((|x|+|y|)^2)`, the coefficient of
   :math:`q^{h+l}` in :math:`K_q(h,l)` is
   :math:`1/(2^{h+l}h!l!)`. In :eq:`skeleton-scalar-sum`, the only
   term of degree :math:`2n` in :math:`q=M-s` has
   :math:`k=0` and :math:`h=l=n`. Its factors are
   :math:`d_s(0)=1`,
   :math:`A_{0,n}=(-1)^n\binom{n+3s-1}{n}` and
   :math:`(2D+s-2n-1)!=(3s-1)!`.
   Finally :math:`\binom Ms^2/(M)_{2D}\sim M^{-2n}/(s!)^2`.
   Substitution gives the stated limit, including :math:`n=0` and
   :math:`s=1`. Its positivity proves that the polynomial degree is
   exactly :math:`2n`. There are only :math:`D` active skeletons, so their
   optimal risk remains bounded as :math:`M` grows.

.. prf:proposition:: Where inductive fitting wins
   :label: prop-skeleton-wins

   The optimal inductive risk is smaller than the all-at-once risk exactly
   for

   - :math:`D = 1` and :math:`D = 2`: every :math:`M \geq 2`;
   - :math:`D = 3`: :math:`M = 2` and :math:`M \geq 11`;
   - :math:`D = 4`: :math:`M = 2` and :math:`M \geq 20`;
   - :math:`D = 5`: :math:`M \in \{2, 3\}` and :math:`M \geq 29`;
   - :math:`D = 6`: :math:`M \in \{2, 3\}` and :math:`M \geq 39`;
   - :math:`D = 7`: :math:`M \in \{2, 3\}` and :math:`M \geq 49`;
   - :math:`D = 8`: :math:`M \in \{2, 3, 4\}` and :math:`M \geq 60`.

   The cases :math:`D \geq 3` are computer-assisted.

.. prf:proof::

   *D = 1.* :math:`a_0 = 2M/(M+1) < 2 \leq M`.

   *D = 2.* By :prf:ref:`prop-skeleton-second-faces` and
   :prf:ref:`prop-skeleton-degree-faces`, :math:`a_0 = \frac{3M(3M^2 - 15M + 28)}{2(M+1)(M+2)(M+3)} < \frac92`
   and :math:`a_1 = \frac{30 M (M-1)^2}{(M+1)(M+2)(M+3)} < 30`. Clearing
   denominators shows both inequalities. So
   :math:`(\sqrt{a_0} + \sqrt{a_1})^2 < 58 < \binom{M+1}{2}` for
   :math:`M \geq 11`. The cases :math:`M \leq 10` are checked directly.

   *D ≥ 3.* :math:`\binom{M}{m+1}^2 \leq M^{2m+2} / (m+1)!^2` and
   :math:`(M)_{2D} \geq M^{2D}`. With the degree bound, this gives
   :math:`a_m \leq \sum_i |p_i| / (m+1)!^2` for all :math:`M \geq 1`, where
   :math:`p_i` are the coefficients of :math:`P_{D,m}`. The all-at-once risk
   exceeds the resulting bound on :math:`(\sum_m \sqrt{a_m})^2` for
   :math:`M \geq 55, 111, 190, 295, 429, 593` for
   :math:`D = 3, 4, 5, 6, 7, 8`, respectively. For every smaller :math:`M`,
   evaluating the reference polynomials
   in floating point with a bound on the rounding error gives the stated sets;
   the closest comparison differs by a relative :math:`4.5 \cdot 10^{-3}`.
   Converting each exact coefficient to float, taking square roots, summing,
   and squaring has relative error at most :math:`4(D+1)u`, where
   :math:`u = 2^{-53}`. Every comparison lies beyond this error bound.

These results sharpen the observation of :cite:t:`tanaka2020asymptotic` that
inductive skeleton fitting wins for degrees below three. That observation is
a theorem for every :math:`M`, and the comparison at higher degrees turns
around once the dimension is large enough.

Open questions
--------------

The asymptotic risk, optimal allocation and a finite scalar expression for
every coefficient are proved above. The questions below concern further
evaluation of those expressions and the comparison with all-at-once
fitting. The curves, all triangle coefficients, the top face, the
codimension-one faces at every fixed dimension, and the families
:math:`m=D-1,D-2,D-3` are already resolved in closed form.

The product formula :eq:`skeleton-product-coefficients` reduces coefficient
evaluation to the simplex factors. Winning against all-at-once fitting
does not require every factor to win: a cubic triangle loses against its
10 controls, while its product with a linear segment has optimal
dimension-based risk approximately :math:`14.2`, below its 20 controls.
The linear segment multiplies every nonzero triangle coefficient by
:math:`4/3`, and the two normalized optimal risks are approximately
:math:`10.65` and :math:`14.20`. In this example only the triangle has
active positive-dimensional faces, so the dimension and shape optima agree.

- **Shorter evaluations of the remaining nonvertex coefficients.**
  The remaining range is :math:`M\geq4`, :math:`D\geq5` and
  :math:`1\leq m\leq\min(D-4,M-3)`.
  :prf:ref:`thm-skeleton-scalar-sum` evaluates every such coefficient by
  a finite sum. The excluded codimension-one case :math:`m=M-2` has
  the harmonic evaluation :prf:ref:`thm-skeleton-facet-harmonics`, whose
  correction sum has length :math:`M-3`. A uniform evaluation in the
  remaining range without sums whose lengths grow
  with :math:`D`, for example in rational functions, harmonic numbers or
  standard hypergeometric functions, remains to be obtained in this range.
  For tetrahedron edges, :prf:ref:`thm-risk-tetrahedron-edge-representation`
  gives a bounded expression with two auxiliary recurrences, and
  :prf:ref:`cor-risk-tetrahedron-edge-scalar-recurrences` supplies explicit
  scalar degree recurrences. A representation using only classical
  harmonic or hypergeometric functions is a separate remaining question.
  For triangular faces on five-vertex simplices,
  :prf:ref:`thm-risk-five-vertex-triangle-state` also supplies a fixed-state
  representation with one additional auxiliary sequence.
  :prf:ref:`thm-risk-six-vertex-tetrahedral-state` extends the fixed-state
  evaluation and certified scalar recurrence to tetrahedral faces on six vertices.
  :prf:ref:`thm-risk-uniform-codimension-two-state` now gives a common
  fifteen-state construction for every nonvertex codimension-two face,
  rather than only those individual source sizes. The state bound is
  independent of the face size; its minimality and a bounded classical
  special-function expression remain separate questions.
  :prf:ref:`thm-risk-nonvertex-boundary-transfer` also bounds nonvertex
  coefficients using smaller simplices, with a trace-correct lift on every
  facet. Its five-vertex edge consequence
  :prf:ref:`thm-risk-five-vertex-edge-upper` is an all-degree upper bound,
  rather than an exact evaluation of the remaining codimension-three sums.
- **Finite-dimensional vertex coefficients beyond triangles.**
  For :math:`M\geq4` and :math:`D\geq4`, simplify
  :prf:ref:`thm-skeleton-vertex-sum` uniformly in the degree, or give an
  explicit fixed-order recurrence in :math:`D`, including its
  coefficients, initial values and exceptional indices. The triangle
  formula in :prf:ref:`thm-skeleton-triangle-vertex-risk` and the
  fixed-degree dimension polynomials do not supply that general
  evaluation. :prf:ref:`thm-risk-fixed-dimension-holonomic` proves that
  a degree recurrence exists at every fixed dimension, while
  :prf:ref:`thm-risk-vertex-boundary-pair` reduces its remaining kernel
  complexity. :prf:ref:`thm-risk-tetrahedron-vertex-single` now evaluates
  the tetrahedron vertex by one positive harmonic-binomial sum, with its
  leading large-degree constant. Its length still grows with :math:`D`;
  :prf:ref:`thm-risk-tetrahedron-vertex-recurrence` now supplies an explicit
  order-ten degree recurrence with initial values and no exceptional integers.
  :prf:ref:`thm-risk-five-vertex-single` similarly evaluates the
  five-vertex vertex coefficient with four harmonic states and one sum
  of length :math:`D`, and :prf:ref:`cor-risk-five-vertex-asymptotic`
  gives its leading constant. A bounded-length classical-special-function
  evaluation is still open in these dimensions.
  :prf:ref:`thm-risk-five-vertex-recurrence` now supplies the
  five-vertex order-seventeen operator as well, with all initial values
  and no positive integer exceptions.
  :prf:ref:`thm-risk-general-vertex-construction` now supplies rational
  univariate convolutions for every vertex count, and
  :prf:ref:`thm-risk-general-vertex-asymptotic` gives the leading constant
  at each fixed :math:`M\geq4` by a positive integral and a convergent
  series with certified truncation errors. The convolution lengths still
  grow with the degree; these results do not give a bounded-length
  classical-special-function expression or a limit uniform in growing
  :math:`M`.
  :prf:ref:`thm-risk-general-vertex-explicit-error` adds a computable
  bound :math:`R_M(1+\log D)^{M-2}/D^{M+2}` for every
  :math:`M\geq4,D\geq256`. Its relative error tends to zero at
  each fixed :math:`M`; this still does not give a uniform growing-M limit.
- **The full comparison beyond the classified degrees.**
  :prf:ref:`prop-skeleton-wins` classifies every :math:`M` at
  :math:`D\leq8`, and :prf:ref:`thm-risk-comparisons-nine-twenty`
  classifies every :math:`M` at :math:`9\leq D\leq20`.
  :prf:ref:`thm-risk-comparisons-twentyone-thirty` extends this through degree thirty.
  For :math:`D\geq31`, determine the exact set of
  winning vertex counts, including any small-dimensional exceptions,
  and the growth of the eventual crossover as :math:`D\to\infty`.
  :prf:ref:`cor-skeleton-large-dimension` proves eventual winning for
  each fixed degree; it gives no exact crossover law.
  :prf:ref:`thm-risk-uniform-dimension-tail` now gives a sufficient linear
  region valid simultaneously for all positive degrees: :math:`M\geq9720D`
  implies :math:`R_{\mathrm{ind}}<(4/5)R_{\mathrm{all}}`. The constant
  is conservative and is not an exact crossover. Closed formulas
  for a fixed dimension also require a separate inequality analysis
  before they yield a classification over all degrees.
  At :math:`M=4`, :prf:ref:`thm-risk-tetrahedron-comparison` now supplies
  this separate analysis and completes the comparison at every degree.
  At :math:`M=5`, :prf:ref:`thm-risk-five-vertex-comparison` also completes
  every degree: inductive fitting wins at :math:`D=1,2` and :math:`D\geq12`,
  while all-at-once fitting wins exactly at :math:`3\leq D\leq11`.
- **Higher-degree self-identifications in higher dimensions.**
  :prf:ref:`thm-skeleton-regular-incidence` removes assembled matrices on
  regular complexes, and :prf:ref:`thm-skeleton-linear-assembled` and
  :prf:ref:`thm-skeleton-graph-risk` handle arbitrary identifications at
  linear factor degrees and in dimension one, respectively. Outside
  those ranges, the general finite formula
  :prf:ref:`thm-skeleton-assembled-risk` remains available. Further
  evaluation must retain the cross terms between distinct occurrences
  of an identified face and the pullbacks from collapsed faces.
