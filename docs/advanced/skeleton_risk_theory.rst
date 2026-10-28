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

Triangular Jacobi extension
---------------------------

The orthogonal-polynomial construction does extend beyond curves, although
it does not yet give a scalar closed form for :math:`a_1`.

.. prf:proposition:: Propagation from an edge via Jacobi polynomials
   :label: prop-skeleton-triangle-jacobi

   Let :math:`m=1`, :math:`D\geq2`, :math:`N=D-2`, and let
   :math:`I=\{1,2\}`. On the triangle with coordinates
   :math:`(x_1,x_2,y)`, put :math:`r=x_1+x_2` and
   :math:`z=x_1/r`. Write the boundary error polynomial as

   .. math::

      p_c(z,1-z,0)=z(1-z)\sum_{k=0}^N d_k P_k^{(2,2)}(2z-1).

   Here :math:`P_n^{(\alpha,\beta)}(2v-1)` denotes the shifted Jacobi
   polynomial orthogonal for the weight :math:`(1-v)^\alpha v^\beta`,
   normalized by :math:`P_n^{(\alpha,\beta)}(1)=\binom{n+\alpha}{n}`.
   Then the polynomial :math:`\Phi` of
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

   Every error polynomial in question is divisible by :math:`x_1x_2`.
   Write :math:`\Phi=x_1x_2 Q`. The orthogonality condition
   :eq:`skeleton-reduction-orthogonality` becomes orthogonality of
   :math:`Q` to all polynomials of degree at most :math:`N-1` for the
   weight :math:`x_1^2x_2^2`. Under
   :math:`(x_1,x_2)=(rz,r(1-z))`, this weighted area element is
   :math:`r^5z^2(1-z)^2\,dr\,dz`.

   Set :math:`A_k(z)=P_k^{(2,2)}(2z-1)`. Polynomials of total degree
   at most :math:`N-1` are spanned by
   :math:`r^l A_l(z)q(r)`, with :math:`0\leq l\leq N-1` and
   :math:`\deg q\leq N-1-l`: this follows by expanding each monomial
   :math:`r^j z^a` in the angular Jacobi basis, with :math:`a\leq j`.
   For the :math:`k`-th summand of :math:`Q`, angular orthogonality
   kills every :math:`l\ne k`. When :math:`l=k`, the remaining
   integral has weight :math:`r^{2k+5}` and vanishes by the radial
   Jacobi orthogonality. Consequently every displayed summand satisfies
   :eq:`skeleton-reduction-orthogonality`.

   The space of polynomials of degree at most :math:`N` orthogonal to
   all degrees at most :math:`N-1` has dimension :math:`N+1`.
   The displayed :math:`N+1` polynomials are independent: at
   :math:`r=1`, their radial factors equal one and their restrictions
   are the angular Jacobi basis. They therefore give the unique
   orthogonal extension of the prescribed boundary data.
   Since :math:`\Phi(x_1,x_2,0)=\phi(x_1,x_2,0)`, the polynomial from
   :prf:ref:`thm-skeleton-propagation-reduction` has exactly these
   boundary values and orthogonality, proving the formula.
   Finally :math:`\partial_y(y\phi)=\Phi` divides each monomial
   coefficient by :math:`n+1` when recovering :math:`\phi`;
   conversion to Bernstein coefficients gives
   :eq:`skeleton-triangle-jacobi-controls`.

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

.. list-table:: :math:`P_{D,m}(M)` for :math:`D \leq 5`
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

The table is computer-assisted: it is computed exactly from the counting in
the proof, starting from one face per class on a simplex with
:math:`2D - m - 1` vertices, which contains every class. ``tests/sampling.py``
repeats this computation, and compares the entries with an independent exact
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
   - :math:`D = 5`: :math:`M \in \{2, 3\}` and :math:`M \geq 29`.

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
   :math:`M \geq 55`, :math:`111` and :math:`190` when :math:`D = 3`,
   :math:`4` and :math:`5`. For every smaller :math:`M`, evaluating the table
   in floating point with a bound on the rounding error gives the stated sets;
   the closest comparison differs by a relative :math:`4.5 \cdot 10^{-3}`.

These results sharpen the observation of :cite:t:`tanaka2020asymptotic` that
inductive skeleton fitting wins for degrees below three. That observation is
a theorem for every :math:`M`, and the comparison at higher degrees turns
around once the dimension is large enough.

Open questions
--------------

- **A scalar closed form in D for 1 ≤ m ≤ D − 3.**
  :prf:ref:`prop-skeleton-triangle-jacobi` solves the triangular
  orthogonal-extension step for :math:`m=1`, using separated angular
  and radial Jacobi polynomials. Substitution into the Gram trace gives
  a finite exact procedure, but the resulting sums have not been
  evaluated to a scalar closed form for :math:`a_1`.

  A single radial polynomial shared by all angular modes does not
  provide this extension. Already at :math:`D=4` (:math:`N=2`), the
  normalized radial factors for :math:`k=0,1,2` are respectively
  :math:`36r^2-56r+21`, :math:`9r-8`, and :math:`1`; they are different
  polynomials, each equal to one at :math:`r=1`. Thus this particular
  separation ansatz fails, while the mode-dependent construction above
  succeeds. This rules out only that ansatz, not other closed forms.

  Exact rational checks of every angular mode for
  :math:`D=2,\dots,7` reproduce the independently solved propagation
  matrices and, after reconstructing the Gram trace, give
  :math:`a_1=3,99/28,71/20,188/55,81259/25025,\dots` for :math:`M=3`.
  The complete checked values are fixed in ``tests/sampling.py``.
  Large prime factors in these values do not rule out a hypergeometric
  formula: no nonexistence claim or conjecture of that kind is made.
  Extending the angular construction to larger faces and evaluating
  the scalar risk sums remain open.
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
  for :math:`D \leq 5`. A formula for it is not known.
