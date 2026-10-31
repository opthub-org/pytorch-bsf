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

This chapter proves the formula and optimal allocations implemented by
:func:`~torch_bsf.sampling.skeleton_risk_coefficients`,
:func:`~torch_bsf.sampling.optimal_skeleton_ratio`,
:func:`~torch_bsf.sampling.sample_skeleton` and
:func:`~torch_bsf.fitting.fit_skeleton_regression` for spaces constructed with
:meth:`~torch_bsf.BezierSpace.on_set`. The single-cell constructors retain
their factorized risk calculation and original sampling behavior.

Using the assembled APIs
-------------------------

Two quadratic intervals with one shared endpoint illustrate the whole path:

.. testcode:: assembled-skeleton

    import torch
    from torch_bsf import BezierModel, BezierSpace, Samples, SimploidalSet
    from torch_bsf.fitting import fit_skeleton_regression
    from torch_bsf.sampling import (
        optimal_skeleton_ratio, sample_skeleton, skeleton_risk_coefficients,
    )

    domain = SimploidalSet()
    left, right = domain.add_simplex(1), domain.add_simplex(1)
    domain.identify(left.face(0, 0), right.face(0, 1))
    space = BezierSpace.on_set(domain, degree=2)
    print(tuple(round(a, 6) for a in skeleton_risk_coefficients(space)))
    ratio = optimal_skeleton_ratio(space)
    points = sample_skeleton(space, 120, ratio=ratio, seed=0)
    truth = BezierModel(space, torch.arange(space.n_controls, dtype=torch.float64)[:, None])
    result = fit_skeleton_regression(space, Samples(points, truth(points).detach())).require_success()
    assert torch.allclose(result.model.coefficients, truth.coefficients)
    assert torch.allclose(result.model(points), truth(points))
    assert all(isinstance(cid, int) for cid in result.faces)
    assert sum(face.diagnostics.n_samples for face in result.faces.values()) == 120

.. testoutput:: assembled-skeleton

    (0.75, 2.0)

The default evaluation measure gives equal weight to maximal cells. Specify
``evaluation_weights`` separately from the sampling allocation to evaluate
another measure, including a mixture with lower-dimensional cells:

.. testcode:: assembled-skeleton

    evaluation = {left: 3.0, right: 1.0}
    c = skeleton_risk_coefficients(space, evaluation_weights=evaluation, allocation="cell")
    q = optimal_skeleton_ratio(space, evaluation_weights=evaluation, allocation="cell")
    points = sample_skeleton(space, 120, cell_weights=q, seed=0)
    assert set(c) == set(q) == set(result.faces)
    assert abs(sum(q.values()) - 1) < 1e-14
    risk_factor = sum(c[cid] / q[cid] for cid in c)
    assert abs(risk_factor - sum(value**0.5 for value in c.values())**2) < 1e-12

``allocation="dimension"`` (the default) returns a tuple indexed by dimension,
including zero entries for dimensions without active cells. It uses
:math:`a_m=K_m\sum_{\dim s=m}c_s` and assigns equal shares to the
:math:`K_m` active cells of each dimension. ``allocation="cell"`` returns an
increasing-dimension, then increasing-cell-ID dictionary and optimizes the
original cell objective. Cell mode needs addressable faces: use ``on_set``
for positive-dimensional spaces; a bare ``BezierSpace.simplex`` or
``simploid`` has only one chart ID and cannot name its faces this way.
Zero-dimensional spaces support both modes.

``evaluation_weights`` and ``cell_weights`` accept integer IDs or the original
builder handles. An ID and its handle are aliases whose weights add. Missing
keys have weight zero. Weights must be real, finite, nonnegative, with a finite
positive total, and are normalized to sum to one. Real values are converted to
float64 before accumulation; values outside its range and normalization that
underflows a positive cell weight raise ``ValueError``. Unknown cells, booleans and
invalid weights raise ``ValueError``. ``cell_weights`` may give positive mass
only to active cells and is mutually exclusive with ``ratio``. Evaluation
weights may include any compiled cell. They never change an explicit sampling
allocation. Pass the same evaluation weights when requesting coefficients and
an automatic ratio.

Sampling first rounds dimension budgets by largest remainder (dimension order
breaks ties), then splits each budget equally among its active cells, randomly
breaking remainder ties with a local generator. Explicit cell shares use one
largest-remainder rounding in fitting order. Both paths return exactly ``n``
rows, including for ``n=0`` or budgets smaller than the active-cell count.
Positive shares on empty dimensions are rejected. Small budgets and rounded
zero counts do not guarantee that fitting has enough observations or full
rank. Within each cell, ``method="random"`` and ``method="sobol"`` retain the
existing product-uniform exponential normalization; neither changes global
random state. Reusing a seed reproduces coordinates, IDs and row order.

For fitting, a directly tagged lower-cell row and a boundary row expressed in
an upper chart both reach their nondegenerate cell. Boundary coordinates must
be exactly zero. Face restriction and degeneracy maps sum coordinates of
repeated vertices and discard collapsed factors. Each input row is assigned
once, and every shared cell is estimated once. The constrained solves retain
the original chart coordinates, so membership-validation tolerances do not
change the observed design when a nearly unit factor is collapsed. Cells without proper controls
are ignored and counted in the result message. Result ``faces`` keys are
integer cell IDs and each value has the ordinary fit diagnostics. The original
single-cell coordinate-tuple keys are preserved. Missing positive-weight
observations raise an error naming the cells; deficient rank remains a
successful least-squares solve with explicit rank diagnostics. A solver failure
stops assembly, identifies the failed cell, and leaves later stages absent and
their proper controls zero.

A zero evaluation weight need not make a cell coefficient zero: boundary
errors can propagate into evaluated cells. If a coefficient is zero, that cell
still owns controls and needs positive sampling share for the theorem's
estimator. With a zero-coefficient active cell, the cell formula describes
only an infimum over strictly positive allocations, so automatic cell
allocation raises ``ValueError``. Dimension allocation remains feasible when
such a cell shares a dimension with a positive coefficient; the equal split
still samples it. It raises when an entire active dimension has zero
coefficient. Supply strictly positive explicit shares to estimate all active
controls in either case. Explicit zero shares are allowed for sampling but do
not promise that fitting all controls will succeed.

The assembled risk implementation uses float64 local moment/Cholesky solves
and a unit triangular propagation solve. It retains repeated-control cross
terms. It checks local Gram and propagation condition numbers, rejecting a
first-order roundoff budget above ``1e-6``. Positive subnormal evaluation
weights or weighted trace contributions are rejected as numerically unreliable;
they are never reported as structural zeros or as a mathematical infimum.
The first-order screen budgets moment/reduction errors of order
:math:`(2w+p+2)u` across :math:`d+1` propagation levels, multiplied by the
largest local Gram condition and the squared propagation condition, with a
safety factor of eight for whitening, trace rounding and higher orders. Here
:math:`w` is the maximum chart width, :math:`p` the global control count and
:math:`u=2^{-53}`. This conservative screen is not an all-degree certificate.
Its conservative work limits are
512 global controls and a total of 4,000,000 local Gram entries. Dense
propagation costs quadratic global memory and cubic global time, with extra
local moment and trace contractions. Explicit ``ratio`` or ``cell_weights``
lets sampling proceed without this risk calculation; fitting uses the usual
solver and its diagnostics. Positive-dimensional degree-zero factors remain
unsupported. These risk formulas assume the correctly specified model,
independent homoscedastic noise, independent product-uniform observations and
unregularized least squares; Sobol sampling is a design option rather than a
claim that the independent-design risk theorem applies to its dependence.

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
