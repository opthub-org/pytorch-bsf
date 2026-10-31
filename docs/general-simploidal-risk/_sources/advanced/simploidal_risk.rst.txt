.. _skeleton-risk-assembled:

Risk on an assembled simploidal set
===================================

The product formulas in :ref:`skeleton-risk-products` extend to any finite
simploidal set with compatible positive factor degrees. Shared faces are
fitted once, and their errors enter every incident cell. Consequently,
adding the risks of independently fitted maximal cells gives the wrong
answer. The formula below uses the global proper controls of
:meth:`~torch_bsf.BezierSpace.on_set`, including repeated controls caused by
self-identifications and degenerate boundary faces.

This chapter proves the formula and optimal allocations. The public
:func:`~torch_bsf.sampling.skeleton_risk_coefficients`,
:func:`~torch_bsf.sampling.sample_skeleton` and
:func:`~torch_bsf.fitting.fit_skeleton_regression` still accept only a single
simplex or product cell. The assembled estimator defined here can be
implemented by successive calls to :func:`~torch_bsf.fit` with
:class:`~torch_bsf.FixedControls`; the verification tests do exactly that.

Global controls and moments
---------------------------

Let :math:`\mathcal S` be the finite collection of non-degenerate cells,
with maximal dimension :math:`d`. The set may be disconnected, have cells of
different dimensions or types, and have identified or degenerate faces.
Use a compatible degree assignment as in :doc:`simploidal`. For a cell
:math:`s` of type :math:`(d_{s,1},\ldots,d_{s,k_s})` and factor degrees
:math:`D_{s,i}\geq1`, let :math:`T_s` be its proper controls, namely the
local Bernstein indices that are strictly positive in every coordinate.
They are distinct global controls, and

.. math::

   p_s=|T_s|=\prod_i\binom{D_{s,i}-1}{d_{s,i}},\qquad
   p=\sum_{s\in\mathcal S}p_s.

A binomial coefficient is zero when its lower index exceeds its upper
index. A vertex has an empty product and one control. The sets :math:`T_s`
partition the :math:`p` global proper controls. Call a cell *active* if
:math:`p_s>0`, and write :math:`\mathcal A` for the active cells.
Degenerate cells add no controls or fitting stages: their coefficients are
pulled back from their non-degenerate base.

Let :math:`\beta_s(t)` be the column vector of the full local product
Bernstein basis. Define the local-to-global matrix :math:`A_s` by

.. math::

   (A_s)_{\alpha j}=\begin{cases}1&\text{local control }\alpha
          \text{ is global control }j,\\0&\text{otherwise},\end{cases}
   \qquad \phi_s(t)=A_s^\top\beta_s(t).

Each row of :math:`A_s` has one 1, but a column may have several.
Thus the corresponding entries of :math:`\beta_s` must be **summed**.
This includes multiplicities from every face map; replacing them by a set
of distinct control IDs would lose cross terms.

Use the product uniform probability measure :math:`U_s` on each chart.
The local and assembled Gram matrices are

.. math::
   :label: assembled-risk-grams

   Q_s=\int\beta_s\beta_s^\top\,dU_s,\qquad
   G_s=A_s^\top Q_s A_s=\int\phi_s\phi_s^\top\,dU_s.

Every entry follows directly from Dirichlet moments:

.. math::

   (Q_s)_{\alpha\gamma}
   =\prod_i\frac{(D_{s,i}!)^2d_{s,i}!
                       \prod_j(\alpha_{ij}+\gamma_{ij})!}
                      {(2D_{s,i}+d_{s,i})!
                       \prod_j\alpha_{ij}!\gamma_{ij}!}.

For risk evaluation, choose fixed weights :math:`w_s\geq0` summing to one,
and put :math:`\mu=\sum_s w_sU_s`, interpreting each chart measure on the
identified domain. Then

.. math::

   G_\mu=\sum_s w_sG_s,\qquad
   R_N=E\sum_s w_s\int\|\hat b_s(t)-b_s(t)\|^2\,dU_s(t).

There is no canonical uniform probability measure across cells of different
dimensions. Equal weights on maximal cells, geometric-volume weights, and
weights on lower cells describe different risks. The evaluation weights
are independent of the sampling fractions below.

Estimator and finite propagation formula
----------------------------------------

Assume the correctly specified model :math:`b_s(t)=\phi_s(t)^\top b` with
:math:`L` outputs. On each active cell :math:`s`, take a deterministic
number :math:`n_s` of independent :math:`U_s` samples, with independent
draws across cells, including cells that share a face. All noise vectors
are mutually independent and independent of all design rows, with mean
zero and covariance :math:`\sigma^2I_L`, with :math:`\sigma>0`.

Visit active cells in increasing dimension. On :math:`s`, estimate only
:math:`b_{T_s}` by ordinary least squares, holding all proper controls on
its boundary at their earlier estimates. Every nonzero entry of
:math:`\phi_s` belongs either to :math:`T_s` or to these boundary controls,
whose set of distinct global IDs is denoted by :math:`B_s`.
No second fit is made for another occurrence of an identified face.

Set :math:`H_s=G_s[T_s,T_s]`. This block equals the local proper-control
Gram block and is positive definite, because those local Bernstein
polynomials are linearly independent. Order the global controls by their
owning cell's dimension, and define the :math:`p\times p` matrix :math:`C`
by the blocks

.. math::
   :label: assembled-risk-propagation

   C[T_s,B_s]=H_s^{-1}G_s[T_s,B_s],\qquad
   C[T_s,j]=0\quad(j\notin B_s),\qquad
   P=(I+C)^{-1}=\sum_{\ell=0}^{d}(-C)^\ell.

Indeed, every nonzero block has a strictly higher row dimension than
column dimension, so
:math:`C^{d+1}=0`. No inverse of a global Gram matrix is needed.
Write :math:`E_s=P[:,T_s]` and define the nonnegative cell coefficients

.. math::
   :label: assembled-risk-traces

   c_s(\mu)=\operatorname{tr}\bigl(H_s^{-1}E_s^\top G_\mu E_s\bigr).

Equations :eq:`assembled-risk-grams`--:eq:`assembled-risk-traces` are a
finite expression for every compatible degree and finite simploidal set.
They require only local proper-block solves and a finite polynomial in
:math:`C`. Rational evaluation weights give rational coefficients.
In general the matrices do not factor globally over the simplex factors:
the identifications and the evaluation measure are part of the answer.

.. prf:theorem:: Assembled asymptotic risk
   :label: thm-skeleton-assembled-risk

   With the estimator and assumptions above, let
   :math:`N=\sum_{s\in\mathcal A}n_s` and suppose
   :math:`n_s/N\to q_s>0` for every active cell. Then :math:`R_N` is finite
   for all sufficiently large :math:`N`, and

   .. math::
      :label: assembled-risk-limit

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
      =\sum_{s\in\mathcal A}\frac{c_s(\mu)}{q_s}.

.. prf:proof::

   Let :math:`Z_{s,T}` and :math:`Z_{s,B}` be the design blocks on
   cell :math:`s`. With empirical moments denoted by hats, its exact error
   equation is

   .. math::

      e_{T_s}=\eta_s-\hat H_s^{-1}\hat G_s[T_s,B_s]e_{B_s},\qquad
      \eta_s=(Z_{s,T}^\top Z_{s,T})^{-1}Z_{s,T}^\top\varepsilon_s.

   Hence :math:`e=\hat P\eta=\sum_s\hat E_s\eta_s`, using the same finite
   polynomial as :eq:`assembled-risk-propagation`. Conditional on all
   design rows, the innovations are independent, and each output column
   of :math:`\eta_s` has covariance :math:`\sigma^2\hat H_s^{-1}/n_s`.
   Therefore, exactly,

   .. math::

      \frac{NR_N}{\sigma^2L}=E X_N,\qquad
      X_N=\sum_s\frac{N}{n_s}
                \operatorname{tr}(\hat H_s^{-1}\hat E_s^\top G_\mu\hat E_s).

   The strong law on each of the finitely many charts gives
   :math:`\hat H_s\to H_s`, :math:`\hat C\to C` and :math:`\hat P\to P`
   almost surely, proving the corresponding limit of :math:`X_N`.

   For expectation convergence, each chart is convex in its affine span
   and its basis has bounded total polynomial degree
   :math:`\sum_iD_{s,i}`. Thus
   :prf:ref:`lem-skeleton-inverse-moments` applies on every positive-dimensional
   active cell. Vertices have the deterministic block :math:`H_s=1`.
   The assembled basis entries are sums of finitely many Bernstein
   polynomials and are bounded. The finite propagation polynomial bounds
   :math:`X_N` by a fixed polynomial in the local inverse norms, since
   :math:`N/n_s` is bounded. Hölder's inequality and the inverse-moment lemma
   give a uniform second-moment bound, exactly as in
   :prf:ref:`thm-skeleton-risk`. This proves finiteness and uniform
   integrability, and hence :eq:`assembled-risk-limit`.

Optimal allocation and all-at-once comparison
----------------------------------------------

If :math:`G_\mu` is positive definite, every active :math:`c_s` is positive:
:math:`E_s[T_s]=I`, so it has full column rank. A sufficient condition is
positive evaluation weight on every maximal cell. Each global control
occurs in at least one maximal chart, and every local Bernstein basis is
linearly independent, so their assembled Gram sum is positive definite.

.. prf:corollary:: Cell and grouped allocations
   :label: cor-skeleton-assembled-allocation

   When :math:`G_\mu` is positive definite, the optimal individual-cell
   allocation and normalized risk are

   .. math::

      q_s^*=\frac{\sqrt{c_s}}{\sum_t\sqrt{c_t}},\qquad
      C_{\rm cell}^*=\left(\sum_s\sqrt{c_s}\right)^2.

   More generally, partition the active cells into groups
   :math:`\mathcal A_g` and require an equal split within each group.
   With :math:`K_g=|\mathcal A_g|` and total group fraction :math:`\rho_g`,

   .. math::
      :label: assembled-risk-grouped

      a_g=K_g\sum_{s\in\mathcal A_g}c_s,\qquad
      C(\rho)=\sum_g\frac{a_g}{\rho_g},\qquad
      \rho_g^*=\frac{\sqrt{a_g}}{\sum_h\sqrt{a_h}},\qquad
      C_{\rm group}^*=\left(\sum_g\sqrt{a_g}\right)^2\geq C_{\rm cell}^*.

.. prf:proof::

   Substitute :math:`q_s=\rho_g/K_g` in :eq:`assembled-risk-limit`.
   Cauchy-Schwarz gives both minima as in :prf:ref:`thm-skeleton-ratio`.
   Individual-cell allocations include the group-constrained ones.

Grouping by :math:`\dim s=m` gives the general skeleton coefficients
:math:`a_m=K_m\sum_{\dim s=m}c_s`; an empty group has coefficient zero.
There is generally no symmetry making all :math:`c_s` in a group equal.
For one unglued product cell and uniform evaluation on it, the propagation
matrices reduce to :prf:ref:`lem-skeleton-product-propagation`; this recovers
:eq:`skeleton-product-coefficients` and the face-shape optimum.

If :math:`G_\mu` is only positive semidefinite, the risk theorem still
requires strictly positive fractions on **all** active cells. For individual
allocations, the square-root expression is an infimum approached by making
the fractions of zero-:math:`c_s` cells arbitrarily small and positive; it is
attained when every :math:`c_s>0`. For equal-within-group allocations, only
groups with :math:`a_g=0` receive vanishing fractions in this limiting
argument. If every nonempty group has :math:`a_g>0`, the grouped minimum is
attained, even when some cells within a group have zero trace: those cells
receive the same positive fraction as the other members of their group.
Neither limiting prescription justifies leaving active cells unfitted, or
a simultaneous limit with vanishing fractions. The estimator still needs
their controls, even when the evaluation measure cannot see them.

.. prf:corollary:: Global all-at-once fitting
   :label: cor-skeleton-assembled-aao

   Fit all :math:`p` proper controls jointly, with deterministic chart counts
   :math:`n_s/N\to v_s>0` on a fixed collection :math:`\mathcal D` of
   non-degenerate cells. Suppose
   :math:`G_\nu=\sum_{s\in\mathcal D}v_sG_s` is positive definite.
   With the same model and independent noise,

   .. math::

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
       =\operatorname{tr}(G_\nu^{-1}G_\mu).

   In particular, matching the design and evaluation weights gives
   :math:`p`, the number of **global proper controls**, whenever
   :math:`G_\mu` is positive definite.

.. prf:proof::

   Conditional covariance is :math:`\sigma^2(Z^\top Z)^{-1}` per output,
   and :math:`Z^\top Z/N\to G_\nu`. To justify the expectation limit,
   apply the inverse-moment lemma to the full local Bernstein basis on
   each sampled chart. If :math:`\lambda_s` is the smallest eigenvalue of
   its whitened empirical Gram matrix, then, for all sufficiently large
   :math:`N`,

   .. math::

      Z^\top Z/N\succeq\tfrac12(\min_{s\in\mathcal D}\lambda_s)G_\nu.

   The inverse moments of the minimum are bounded by sums of the finitely
   many local inverse moments. This gives uniform integrability; the trace
   limit follows. For :math:`G_\nu=G_\mu`, the trace is :math:`p`.

The deterministic chart counts matter for this statement. Randomly selecting
chart labels can leave a whole connected component unobserved with positive
probability at finite :math:`N`; invertibility then needs a separate argument.

Two small examples
-------------------

Take two quadratic segments meeting at one endpoint, and evaluate equally
on the two segments. A single quadratic segment has vertex propagation
polynomials
:math:`f_0(t)=(1-t)^2-\tfrac32t(1-t)` and
:math:`f_1(t)=t^2-\tfrac32t(1-t)`, each with squared integral :math:`1/8`.
The shared vertex propagates onto both segments. Consequently the three
vertex traces are :math:`1/16,1/8,1/16`, while each segment-interior trace
is :math:`1/2`. Thus

.. math::

   a_0=3/4,\qquad a_1=2,\qquad
   C_{\rm cell}^*=\left(\tfrac12+\tfrac{5}{2\sqrt2}\right)^2
     <\left(\sqrt{3/4}+\sqrt2\right)^2=C_{\rm dimension}^*,

and matching all-at-once fitting has normalized risk :math:`p=5`.
The optimal shared-vertex fraction is :math:`\sqrt2` times either
outer-vertex fraction, despite all three cells having dimension zero.

For a single quadratic segment with its two endpoints identified, both
endpoint occurrences carry the same innovation. Its vertex propagation
is :math:`f_0+f_1=1-5t(1-t)`, whose squared integral is :math:`1/6`.
The coefficients are therefore :math:`(a_0,a_1)=(1/6,1)` and the all-at-once
count is :math:`p=2`. Treating the endpoint occurrences as independent
would instead give :math:`1/4` for their combined vertex trace and miss
the cross term. At degree one this loop has only one global control,
:math:`\phi=1`, so its normalized risk is exactly :math:`1`.

Exact checks of these examples, mixed triangle-square domains, shared
triangles, a torus and a sphere with collapsed edges are recorded in
:doc:`skeleton_risk_verification`.
