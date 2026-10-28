Theory of the skeleton risk
===========================

This chapter proves the results behind :doc:`skeleton_sampling`:

- the asymptotic risk of inductive skeleton fitting,
- its optimal ratio,
- a structure theorem for how estimation errors propagate from face to face,
- the closed forms that follow from that theorem.

:cite:t:`tanaka2020asymptotic` derived the risk through a recursion and
evaluated it numerically for degrees two and three. The proofs here are
self-contained and hold for every dimension and degree. Statements that rest
on a finite computation are labeled computer-assisted. Questions that remain
open are collected at the end. ``tests/sampling.py`` checks the formulas,
the quoted values and the computations on this page against exact rational
arithmetic, and ``tests/fitting.py`` checks the asymptotic risks by
simulation.

Setting
-------

Let the simplex have the vertex set :math:`[M] = \{1, \dots, M\}` and degree
:math:`D \geq 1`.

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
   :math:`(\sum_m \sqrt{a_m})^2`.

.. prf:proof::

   By the Cauchy-Schwarz inequality,
   :math:`(\sum_m \sqrt{a_m})^2 = (\sum_m \sqrt{a_m / \rho_m} \sqrt{\rho_m})^2
   \leq \sum_m (a_m / \rho_m) \sum_m \rho_m`. Equality holds exactly when
   :math:`\sqrt{a_m/\rho_m}` is proportional to :math:`\sqrt{\rho_m}`.

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

Closed forms
------------

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

   *The error polynomial.* Let :math:`|I| = D - 1`. Then
   :math:`T_I = \{\mathbf 1_I + \mathbf e_k : k \in I\}`; write
   :math:`c_k` for the coefficient of its :math:`k`-th control. Only the
   faces :math:`J = I \cup \{j\}` have interior controls above :math:`I`, each
   the single control :math:`\mathbf 1_J`. By :eq:`skeleton-moments`,
   :math:`G^J_{\mathbf 1_J,\, \mathbf 1_I + \mathbf e_k} / G^J_{\mathbf 1_J \mathbf 1_J} = 3/4`.
   So by :eq:`skeleton-propagation`, :math:`\mathbf 1_J` has error
   :math:`-\frac34 \sum_k c_k`, and the error polynomial is

   .. math::

      p(t) = D!\, t^{\mathbf 1_I} \Bigl(\tfrac12 \sum_{k \in I} c_k t_k - \tfrac34 \bigl(\textstyle\sum_k c_k\bigr) u\Bigr),
      \qquad u = \sum_{j \notin I} t_j.

   *The trace as an expectation.* By :eq:`skeleton-moments`,
   :math:`G^I_{T_I T_I} = \kappa (3\,\mathrm{Id} + 9 \mathbf 1 \mathbf 1^\top)` with
   :math:`\kappa = (D!)^2 (D-2)!\, 2^{D-1} / (4 (3D-2)!)`. Its inverse is
   :math:`\frac{1}{3\kappa}(\mathrm{Id} - \frac{3}{3D-2} \mathbf 1 \mathbf 1^\top)`.
   Hence the trace in :math:`a_{D-2}` equals :math:`E \int p^2 \, dU_{[M]}` for
   :math:`c` with this covariance, that is, the integral of
   :math:`(D!)^2 t^{2 \cdot \mathbf 1_I} v^\top (G^I_{T_I T_I})^{-1} v`, where
   :math:`v_k = \frac12 t_k - \frac34 u`.

   *The integral.* For :math:`M \geq D`, the vector :math:`(t_I, u)` is
   Dirichlet distributed with parameters :math:`(1, \dots, 1, M - D + 1)`, so

   .. math::

      \int t_I^\gamma u^b \, dU_{[M]} = \frac{(M-1)!\, \gamma!\, (M-D+1)_b}{(M - 1 + |\gamma| + b)!}.

   Expanding the quadratic form needs only the moments of
   :math:`t^{2 \cdot \mathbf 1_I}` times :math:`t_k^2`, :math:`t_k t_l`,
   :math:`t_k u`, :math:`u^2`, :math:`u` and :math:`1`, using
   :math:`\sum_k t_k = 1 - u`. Collecting terms gives the formula. For
   :math:`M = D - 1` the formula reduces to :math:`D - 1`, which agrees with
   :prf:ref:`prop-skeleton-top`.

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

.. prf:theorem:: A generating function for the vertex risk
   :label: thm-skeleton-vertex-sum

   Put :math:`w_k=(D-k+1)b_k`, with :math:`b_k` as in
   :prf:ref:`thm-skeleton-curves`, and define the formal power series

   .. math::

      F(x,y) &= \sum_{p,q\geq0}\frac{\binom{p+q}{p}}{(p+1)(q+1)}x^p y^q \\
      &= \frac{(1-x-y)\log(1-x-y)-(1-x)\log(1-x)
         -(1-y)\log(1-y)}{xy}, \\
      K_r(p,q) &= [x^p y^q]F(x,y)^r.

   The quotient denotes its formal series, including at either zero
   coordinate. For every :math:`D\geq1` and :math:`M\geq2`,

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

   Integrating :math:`(1-x-y)^{-1}` formally once in each variable gives
   the displayed expression for :math:`xyF`. Finally expand
   :math:`F^{M-1}=(1+(F-1))^{M-1}`. Since :math:`F-1` has no constant
   term, its power :math:`s` cannot contribute above :math:`s=p+q`.
   Here :math:`p+q\leq2D-2`, which proves
   :eq:`skeleton-vertex-polynomial`, including the degree bound. The
   coefficient extraction uses a polynomial of bounded degree for every
   fixed :math:`D`; it requires no propagation matrices or inverses.

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

   By symmetry every face with :math:`m + 1` vertices contributes equally.
   Fix one such face :math:`I`. The entries of :math:`E_I` on the faces
   :math:`J \supseteq I` with :math:`|J| \leq D` are computed from
   :math:`G^J` alone, so they do not depend on :math:`M`. They are equivariant
   under permutations that fix :math:`I` pointwise. With
   :math:`C = E_I (G^I_{T_I T_I})^{-1} E_I^\top`, :eq:`skeleton-moments` gives

   .. math::

      \operatorname{tr}\bigl((G^I_{T_I T_I})^{-1} E_I^\top G E_I\bigr)
      = \sum_{\alpha, \beta} C_{\alpha\beta} \frac{(D!)^2 (\alpha + \beta)!}{\alpha!\,\beta!} \cdot \frac{1}{(M)_{2D}}.

   Group the pairs :math:`(\alpha, \beta)` into classes under these
   permutations. A class whose union of supports has :math:`s` vertices
   outside :math:`I` has
   :math:`(M - m - 1)(M - m - 2) \cdots (M - m - s)` members up to a constant
   factor, a polynomial in :math:`M` of degree :math:`s`. Every member makes
   the same contribution, which does not depend on :math:`M`. Each support has
   at most :math:`D - m - 1` vertices outside :math:`I`, so
   :math:`s \leq 2(D - m - 1)`.

.. list-table:: :math:`P_{D,m}(M)` for :math:`D \leq 8`
   :header-rows: 1
   :stub-columns: 2

   * - :math:`D`
     - :math:`m`
     - :math:`P_{D,m}(M)`
   * - 1
     - 0
     - :math:`2`
   * - 2
     - 0
     - :math:`\frac32 (3M^2 - 15M + 28)`
   * - 2
     - 1
     - :math:`120`
   * - 3
     - 0
     - :math:`\frac12 (9M^4 - 114M^3 + 835M^2 - 2610M + 3320)`
   * - 3
     - 1
     - :math:`360 (3M^2 - 21M + 58)`
   * - 3
     - 2
     - :math:`20160`
   * - 4
     - 0
     - :math:`\frac58 (5M^6 - 115M^5 + 1825M^4 - 14765M^3 + 71250M^2 - 174840M + 181152)`
   * - 4
     - 1
     - :math:`\frac{105}{2} (63M^4 - 1050M^3 + 9481M^2 - 38654M + 67512)`
   * - 4
     - 2
     - :math:`136080 (3M^2 - 27M + 94)`
   * - 4
     - 3
     - :math:`6652800`
   * - 5
     - 0
     - :math:`\frac{1}{128} (225M^8 - 8100M^7 + 223650M^6 - 3352440M^5 + 33885265M^4 - 210684820M^3 + 810530924M^2 - 1692775952M + 1526667648)`
   * - 5
     - 1
     - :math:`840 (7M^6 - 203M^5 + 3573M^4 - 34441M^3 + 201644M^2 - 629700M + 867312)`
   * - 5
     - 2
     - :math:`56700 (45M^4 - 930M^3 + 9971M^2 - 49694M + 106152)`
   * - 5
     - 3
     - :math:`79833600 (3M^2 - 33M + 136)`
   * - 5
     - 4
     - :math:`3632428800`

   * - 6
     - 0
     - :math:`\frac{7}{1536} (189M^{10} - 9765M^{9} + 416010M^{8} - 9917418M^{7} + 170241981M^{6} - 1939736421M^{5} + 15319824604M^{4} - 79461497164M^{3} + 261855225632M^{2} - 485208011456M + 394380100608)`
   * - 6
     - 1
     - :math:`\frac{63}{40} (4725M^{8} - 207900M^{7} + 5933250M^{6} - 99912240M^{5} + 1132957965M^{4} - 8261943780M^{3} + 38361752764M^{2} - 100723167440M + 118578731904)`
   * - 6
     - 2
     - :math:`155925 (55M^{6} - 1925M^{5} + 38401M^{4} - 431519M^{3} + 2952160M^{2} - 11026484M + 18273744)`
   * - 6
     - 3
     - :math:`21621600 (117M^{4} - 2886M^{3} + 35695M^{2} - 208734M + 519368)`
   * - 6
     - 4
     - :math:`68108040000 (3M^{2} - 39M + 184)`
   * - 6
     - 5
     - :math:`2964061900800`
   * - 7
     - 0
     - :math:`\frac{1}{10368} (3969M^{12} - 277830M^{11} + 16911909M^{10} - 585892314M^{9} + 15245701191M^{8} - 274563067842M^{7} + 3631411308511M^{6} - 34078439471286M^{5} + 227106974533684M^{4} - 1028193132036840M^{3} + 3014866953859072M^{2} - 5083595030097792M + 3804385671687168)`
   * - 7
     - 1
     - :math:`\frac{315}{8} (189M^{10} - 11655M^{9} + 488100M^{8} - 12550518M^{7} + 229047537M^{6} - 2894482743M^{5} + 25831659094M^{4} - 156730040300M^{3} + 621915070424M^{2} - 1437930406240M + 1507450665216)`
   * - 7
     - 2
     - :math:`\frac{31185}{4} (2475M^{8} - 128700M^{7} + 4016430M^{6} - 76100640M^{5} + 969732275M^{4} - 8107098420M^{3} + 43614175604M^{2} - 135327928048M + 189756997248)`
   * - 7
     - 3
     - :math:`151351200 (91M^{6} - 3731M^{5} + 83699M^{4} - 1072645M^{3} + 8358906M^{2} - 36008544M + 68795424)`
   * - 7
     - 4
     - :math:`363242880000 (9M^{4} - 258M^{3} + 3607M^{2} - 24086M + 67896)`
   * - 7
     - 5
     - :math:`80029671321600 (3M^{2} - 45M + 238)`
   * - 7
     - 6
     - :math:`3379030566912000`
   * - 8
     - 0
     - :math:`\frac{1}{1536} (243M^{14} - 22113M^{13} + 1822905M^{12} - 86413257M^{11} + 3170012139M^{10} - 82654759071M^{9} + 1640539616043M^{8} - 24125599635171M^{7} + 266382561743494M^{6} - 2157428183555988M^{5} + 12638903353801912M^{4} - 51381774261439680M^{3} + 137262734136374272M^{2} - 214224590422872576M + 149755742713294848)`
   * - 8
     - 1
     - :math:`\frac{55}{384} (43659M^{12} - 3580038M^{11} + 206363619M^{10} - 7485895494M^{9} + 199202107797M^{8} - 3823026542874M^{7} + 54535243641473M^{6} - 568795558812642M^{5} + 4306623724542524M^{4} - 22811488486847928M^{3} + 80426058871064384M^{2} - 168310145729971584M + 161038581507542016)`
   * - 8
     - 2
     - :math:`\frac{27027}{16} (19305M^{10} - 1383525M^{9} + 61604400M^{8} - 1734571410M^{7} + 34500231285M^{6} - 483666978405M^{5} + 4831926907690M^{4} - 33375332254980M^{3} + 152661774969416M^{2} - 413966621739552M + 513284796940032)`
   * - 8
     - 3
     - :math:`\frac{4729725}{2} (20475M^{8} - 1228500M^{7} + 42200550M^{6} - 891274440M^{5} + 12638555275M^{4} - 118800088180M^{3} + 722231576084M^{2} - 2559256843376M + 4107713331840)`
   * - 8
     - 4
     - :math:`1543782240000 (17M^{6} - 799M^{5} + 19941M^{4} - 286433M^{3} + 2496802M^{2} - 12115512M + 26028000)`
   * - 8
     - 5
     - :math:`31678411564800 (171M^{4} - 5586M^{3} + 86965M^{2} - 650590M + 2040000)`
   * - 8
     - 6
     - :math:`124179373334016000 (3M^{2} - 51M + 298)`
   * - 8
     - 7
     - :math:`5129368400572416000`

The table is computer-assisted: it is computed exactly from the counting in
the proof, starting from one face per class on a simplex with
:math:`2D - m - 1` vertices, which contains every class. ``tests/sampling.py``
repeats this computation, using :prf:ref:`thm-skeleton-propagation-reduction`
to solve on only :math:`m+2` vertices. In the pair sum it keeps one first
control per orbit under permutations outside the fixed face. For each first
control it keeps one second control per orbit of the stabilizer, which
permutes positions with equal outside exponents. If the outside exponents
have multiplicities
:math:`n_k` (including :math:`n_0`), the orbit size is
:math:`(2D-2m-2)!/\prod_k n_k!`; the second-control multiplicity is the
product of the same formula within the equal-exponent groups. It compares
the entries with an independent exact
evaluation of the definition for small :math:`M`. The entries for
:math:`m = D-1` and :math:`m = D-2` agree with
:prf:ref:`prop-skeleton-degree-faces` and
:prf:ref:`prop-skeleton-second-faces`.

.. prf:corollary:: Large dimension
   :label: cor-skeleton-large-dimension

   For fixed :math:`D`, every :math:`a_m` converges as :math:`M \to \infty`.
   In particular

   .. math::

      \lim_{M \to \infty} a_0 = \frac{2 D^2 (D+1)^2}{4^D}, \qquad
      \lim_{M \to \infty} a_{D-1} = \frac{(3D-1)!}{(D-1)!\,(D!)^2}, \qquad
      \lim_{M \to \infty} a_{D-2} = \frac{3 (3D-3)!}{4\,(D-2)!^2\,(D-1)!}.

   The optimal inductive risk :math:`(\sum_m \sqrt{a_m})^2` is therefore
   bounded in :math:`M`, while the all-at-once risk
   :math:`\binom{D+M-1}{D}` grows without bound. For every degree, inductive
   skeleton fitting wins in all sufficiently high dimensions.

.. prf:proof::

   :math:`\binom{M}{m+1}^2 / (M)_{2D}` has degree :math:`2(m + 1) - 2D`, so
   the degree bound of :prf:ref:`thm-skeleton-rational` gives convergence.
   The last two limits follow from :prf:ref:`prop-skeleton-degree-faces` and
   :prf:ref:`prop-skeleton-second-faces`.

   For :math:`a_0`, the leading term counts the pairs with
   :math:`s = 2(D-1)`. These are exactly :math:`\alpha = \mathbf 1_J` and
   :math:`\beta = \mathbf 1_K` with :math:`|J| = |K| = D` and
   :math:`J \cap K = I`. Each such pair has :math:`C_{\alpha\beta} = \gamma_D^2`
   by :prf:ref:`cor-skeleton-vertex` and
   :math:`(\alpha+\beta)!/(\alpha!\,\beta!) = 2`. There are
   :math:`M^{2D-2} / (D-1)!^2 + O(M^{2D-3})` ordered pairs. Hence
   :math:`a_0 = M^2 \cdot 2 (D!)^2 \gamma_D^2 M^{2D-2} / ((D-1)!^2 M^{2D}) + O(1/M)
   \to 2 D^2 \gamma_D^2`.

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

   *D = 2.* By the table, :math:`a_0 = \frac{3M(3M^2 - 15M + 28)}{2(M+1)(M+2)(M+3)} < \frac92`
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
   :math:`D = 3, 4, 5, 6, 7, 8`, respectively. For every smaller :math:`M`, evaluating the table
   in floating point with a bound on the rounding error gives the stated sets;
   the closest comparison differs by a relative :math:`4.5 \cdot 10^{-3}`.
   Converting each exact coefficient to float, taking square roots, summing,
   and squaring has relative error at most :math:`4(D+1)u`, where
   :math:`u = 2^{-53}`. Every comparison lies beyond this error bound.

   On the development machine, regenerating all polynomials took about
   :math:`0.09`, :math:`0.75` and :math:`6.6` seconds for :math:`D=6,7,8`.
   The unreduced :math:`D=6,m=0` computation alone took :math:`31.6` seconds;
   the reduced version took about :math:`0.04` seconds. Full exact regeneration
   remains in the test suite; the additional degree-eight cases dominate
   its added runtime.

These results sharpen the observation of :cite:t:`tanaka2020asymptotic` that
inductive skeleton fitting wins for degrees below three. That observation is
a theorem for every :math:`M`, and the comparison at higher degrees turns
around once the dimension is large enough.

Open questions
--------------

- **A closed form in D for 1 ≤ m ≤ D − 3.** By
  :prf:ref:`thm-skeleton-propagation-reduction`, :math:`a_m` is determined by the
  propagation onto a face with :math:`m + 2 \geq 3` vertices. For
  :math:`m = 0` this is an edge, and Legendre polynomials solve it. For
  larger :math:`m`, the bubble functions live on a simplex of dimension at
  least two, and no product formula is known. Exact values such as
  :math:`a_1 = 99/28, 71/20, 188/55, 81259/25025` for :math:`M = 3` and
  :math:`D = 3, \dots, 6` acquire large prime factors, such as
  :math:`81259 = 23 \cdot 3533`. This suggests that no single
  hypergeometric term gives them.
- **A simpler form of a_0 for finite M ≥ 3.**
  :eq:`skeleton-vertex-double-sum` reduces the control-pair sum to a
  :math:`D\times D` sum with an explicit generating function. For example
  it gives :math:`41/280,79/1120,1189/30800` at :math:`M=3` and
  :math:`D=3,4,5`. :eq:`skeleton-vertex-polynomial` gives every fixed-degree
  polynomial, but a scalar closed form in :math:`D`, or a linear
  recurrence in :math:`D` of bounded order, remains open even for
  :math:`M=3`.

  Two simplification attempts give the following concrete partial results.
  Removing the factors :math:`(p+1)^{-1}(q+1)^{-1}` from :math:`F`
  would give :math:`(1-x-y)^{-1}`, whose powers have coefficients
  :math:`(r)_{p+q}/(p!q!)` by Vandermonde convolution. Keeping these
  factors instead gives the logarithmic generating function above;
  the same unweighted convolution therefore does not evaluate
  :math:`K_r`.

  For :math:`M=3`, the boundary coefficient can be simplified by partial
  fractions:

  .. math::

     K_2(p,0)=\sum_{j=0}^p\frac{1}{(j+1)(p-j+1)}
       =\frac{2H_{p+1}}{p+2},\qquad
     H_n=\sum_{j=1}^n\frac1j.

  Indeed the summand is
  :math:`((j+1)^{-1}+(p-j+1)^{-1})/(p+2)`.
  Gosper summation of this summand with symbolic :math:`p` returns no
  hypergeometric antidifference (SymPy ``gosper_sum`` returns ``None``).
  This limits that direct indefinite-summation attempt; it does not rule
  out a definite-sum identity or creative telescoping. The boundary
  identity only evaluates terms with :math:`k=D` or :math:`l=D` in
  :eq:`skeleton-vertex-double-sum`; interior coefficients still involve
  both variables. No WZ certificate or bounded-order recurrence is
  asserted.
- **Exact crossover dimensions for every degree.**
  :prf:ref:`cor-skeleton-large-dimension` shows that a crossover dimension
  exists for every :math:`D`, and :prf:ref:`prop-skeleton-wins` determines it
  for :math:`D \leq 8`. For :math:`D=3,\ldots,8`, the crossover vertex
  counts are :math:`11,20,29,39,49,60`. In particular the apparent arithmetic
  progression :math:`9D-16` fails already at :math:`D=6`, where it predicts
  :math:`38` instead of :math:`39`. The successive differences
  :math:`9,9,10,10,11` do not establish a general formula or an asymptotic
  law as :math:`D\to\infty`; those remain open. The exceptional wins below
  the crossover are :math:`\{2\}` for :math:`D=3,4`,
  :math:`\{2,3\}` for :math:`D=5,6,7`, and :math:`\{2,3,4\}` for
  :math:`D=8`.
