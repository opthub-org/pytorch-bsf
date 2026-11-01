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

The later analytic results eliminate assembled matrices for regular glued
complexes, for arbitrary sets at linear factor degrees, and for every
one-dimensional set at every positive degree. The regular case reduces to
weighted face incidences; the other two cases explicitly retain the cross
terms from identified corners. The public assembled risk implementation uses
these closed forms directly and retains checked matrix propagation for the
remaining nonlinear identified sets.

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
    plan = optimal_skeleton_ratio(space)
    assert plan.strategy == "cell" and plan.fallback_reason is None
    points = sample_skeleton(space, 120, allocation=plan, seed=0)
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
    points = sample_skeleton(space, 120, allocation=q, seed=0)
    assert set(c) == set(q.cell_weights) == set(result.faces)
    assert abs(sum(q.cell_weights.values()) - 1) < 1e-14
    risk_factor = sum(c[cid] / q.cell_weights[cid] for cid in c)
    assert abs(risk_factor - sum(value**0.5 for value in c.values())**2) < 1e-12

``optimal_skeleton_ratio`` and ``sample_skeleton`` default to
``allocation="auto"``: prefer individual cells, falling back to dimension
shares only for zero-cell coefficients with positive active-group coefficients.
Manual ``"cell"`` and ``"dimension"`` modes remain strict. Every optimized
allocation has the same :class:`~torch_bsf.SkeletonAllocation` result type;
inspect ``strategy`` and ``fallback_reason`` and pass the whole plan when
sampling. See :ref:`skeleton-auto-allocation`.

The coefficient function keeps ``allocation="dimension"`` as its default,
returning a tuple indexed by dimension, including zero entries for empty
active dimensions. It uses :math:`a_m=K_m\sum_{\dim s=m}c_s` for equal shares
within each dimension. ``allocation="cell"`` returns an increasing-dimension,
then increasing-cell-ID dictionary and describes the individual-cell objective.
Cell mode needs addressable faces: use ``on_set`` for positive-dimensional
spaces; bare ``BezierSpace.simplex``/``simploid`` retain their fast dimension
path. Zero-dimensional spaces support both modes.

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

Dimension plans and explicit ``ratio`` first round dimension budgets by largest
remainder (dimension order breaks ties), then split each budget equally among
its active cells, randomly breaking remainder ties with a local generator.
Cell plans, including the usual automatic selection, and explicit
``cell_weights`` use one largest-remainder rounding in fitting order.
Both paths return exactly ``n``
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
only an infimum over strictly positive allocations, so strict
``allocation="cell"`` raises ``ValueError``. Dimension allocation remains
feasible when such a cell shares a dimension with a positive coefficient;
the equal split still samples it. ``allocation="auto"`` selects this rescue
when every active dimension has a positive coefficient, reporting
``fallback_reason="zero_cell_coefficients"``. Both automatic and dimension
allocation raise when an entire active dimension has zero coefficient.
Supply strictly positive explicit shares to estimate all active
controls in either case. Explicit zero shares are allowed for sampling but do
not promise that fitting all controls will succeed.

The remaining nonlinear identified sets use float64 local moment/Cholesky
solves and a unit triangular propagation solve. This dense path retains
repeated-control cross terms and checks local Gram and propagation condition
numbers, rejecting a first-order roundoff budget above ``1e-6``.
Across all implementation paths, positive subnormal evaluation
weights or weighted trace contributions are rejected as numerically unreliable;
they are never reported as structural zeros or as a mathematical infimum.
For the dense path, the first-order screen budgets moment/reduction errors of order
:math:`(2w+p+2)u` across :math:`d+1` propagation levels, multiplied by the
largest local Gram condition and the squared propagation condition, with a
safety factor of eight for whitening, trace rounding and higher orders. Here
:math:`w` is the maximum chart width, :math:`p` the global control count and
:math:`u=2^{-53}`. This conservative screen is not an all-degree certificate.
The dense path's conservative work limits are 512 global controls and a total
of 4,000,000 local Gram entries. Its propagation costs quadratic global memory
and cubic global time, with extra
local moment and trace contractions. Explicit ``ratio`` or ``cell_weights``
lets sampling proceed without this risk calculation; fitting uses the usual
solver and its diagnostics. Positive-dimensional degree-zero factors remain
unsupported. These risk formulas assume the correctly specified model,
independent homoscedastic noise, independent product-uniform observations and
unregularized least squares; Sobol sampling is a design option rather than a
claim that the independent-design risk theorem applies to its dependence.


Closed-form implementation paths
--------------------------------

The public risk and allocation APIs use the updated theory directly:

- Arbitrary graphs, including loops, parallel edges and isolated vertices,
  use :prf:ref:`thm-skeleton-graph-risk`. Endpoint/loop norms and interior
  ranks need only scalar arithmetic and incidence accumulation. Large graphs
  and high-degree graph edges avoid the global dense limit.
- At linear factor degrees, :prf:ref:`thm-skeleton-linear-assembled` sums
  corner collisions, retaining cross terms from identified or collapsed
  corners. There is no global matrix. Local corner-pair work is limited to
  4,000,000 pairs per chart; a chart with a single global corner is exactly
  constant and needs no pair expansion.
- Regular chart closures use :prf:ref:`thm-skeleton-regular-incidence` and
  conditioned single-simplex factor traces. Thus large regular meshes avoid
  global Gram/propagation matrices while retaining the existing factor-level
  numerical safeguards. Each factor is limited to 512 controls; compounded
  factor/accumulation error must fit the ``1e-6`` first-order screen.
- Other nonlinear identified sets retain checked dense propagation and its
  global work limits above.

Closed formulas can evaluate theoretical coefficients when a numerical fit
at the same degree is poorly conditioned. They do not certify that a finite
sample or floating-point regression identifies all controls; inspect the
ordinary fit rank/condition and solver diagnostics separately.

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

Incidence formulas on regular complexes
---------------------------------------

The assembled propagation matrices can be eliminated on ordinary glued
meshes. Here *regular* means that every local face of every chart maps
isomorphically to a non-degenerate global cell, and distinct local faces
of one chart map to distinct global cells. Different charts may share
faces, with any number of incident charts; no manifold assumption is
needed. Identifications within the closure of a single chart and collapsed
faces are excluded from this particular result.

.. prf:theorem:: Risk from weighted face incidences
   :label: thm-skeleton-regular-incidence

   Suppose the simploidal set is regular. For an active cell :math:`s`
   which is a face of a chart :math:`x`, write its unique local face as
   :math:`F_{s,x}=\prod_i I_i`, and put
   :math:`M_i=d_{x,i}+1`, :math:`r_i=|I_i|-1`. Define

   .. math::
      :label: assembled-risk-incidence

      \lambda_{s,x}
       =\prod_i\frac{a_{r_i}(M_i,D_{x,i})}
                         {\binom{M_i}{r_i+1}^2},\qquad
      c_s(\mu)=\sum_{x:\,s\preceq x}w_x\lambda_{s,x}.

   The empty product for a vertex chart is one. The simplex coefficients
   :math:`a_r(M,D)` are those of :prf:ref:`thm-skeleton-scalar-sum`.
   Consequently this formula uses finite scalar sums and face incidences,
   with no assembled matrix or Gram inverse. It allows different cell
   dimensions and types, compatible factor degrees, and evaluation weights
   on lower-dimensional charts as well as maximal ones.

.. prf:proof::

   Fix a source innovation in :math:`T_s`, setting all other innovations
   to zero. Its error is zero on every cell that does not contain
   :math:`s`: induction over dimension in the error equations proves this
   support statement. On a chart :math:`x` containing :math:`s`, all its
   faces occur exactly once and are non-degenerate. Restricting the global
   recursion to these faces therefore gives exactly the normal equations
   of the unglued product chart, with source face :math:`F_{s,x}`.
   Positive definite proper-control blocks give the same unique extension.
   A reordering of source controls induced by a face identification also
   reorders :math:`H_s`, leaving the covariance trace unchanged.

   The product face trace is :math:`\lambda_{s,x}` by
   :eq:`skeleton-product-face-trace`. Finally, the evaluation metric is a
   weighted sum of chart metrics, and the trace is linear in that metric.
   Summing the local contributions proves :eq:`assembled-risk-incidence`.

This is stronger than adding independently fitted cell risks: a shared
source is sampled and fitted once, and its single coefficient adds all
incident evaluation weights before the allocation is optimized.

.. prf:corollary:: The face vector of a pure simplicial complex
   :label: cor-skeleton-pure-incidence

   Suppose a regular simplicial complex is pure of dimension :math:`M-1`,
   every simplex has the same degree :math:`D`, and evaluation is supported
   on its maximal simplices. Their weights may be arbitrary, summing to
   one. Let :math:`K_m` be the number of global :math:`m`-faces and
   :math:`f_m=\binom M{m+1}`. For active dimensions,

   .. math::
      :label: assembled-risk-face-vector

      d_w(s)&=\sum_{x:\,s\preceq x}w_x,\qquad
      c_s=\frac{a_m(M,D)}{f_m^2}d_w(s)\quad(\dim s=m),\\
      A_m&=\frac{K_m}{f_m}a_m(M,D).

   Thus the dimension-grouped risk and its optimum depend only on the
   face counts and the single-simplex coefficients, and are independent of
   the distribution of evaluation weights among maximal simplices.
   Individual-cell optimal allocations still depend on :math:`d_w(s)`.

.. prf:proof::

   Every maximal chart has :math:`f_m` local :math:`m`-faces. Double
   counting gives :math:`\sum_{\dim s=m}d_w(s)=f_m\sum_xw_x=f_m`.
   Substitution into :math:`A_m=K_m\sum_{\dim s=m}c_s` proves the result.

For example, an arbitrary pure regular triangular complex of degree two,
with :math:`V` vertices and :math:`E` edges, has

.. math::
   :label: assembled-risk-triangle-mesh

   c_v=d_w(v)/24,\qquad c_e=d_w(e)/3,\qquad
   (A_0,A_1,A_2)=(V/8,E,0).

The triangle coefficients are :math:`a_0=3/8`, :math:`a_1=3`.
There are :math:`p=V+E` global controls. If the evaluation Gram matrix is
positive definite, dimension-grouped skeleton fitting therefore beats
matching all-at-once fitting precisely when

.. math::

   \left(\sqrt{V/8}+\sqrt E\right)^2<V+E
      \quad\Longleftrightarrow\quad E/V<49/32.

At :math:`E/V=49/32` the two risks agree. This comparison concerns the
dimension-grouped design; allowing separate cell allocations can improve
its skeleton risk further.

Linear degrees on arbitrary identified sets
-------------------------------------------

When every factor degree is one, all proper controls belong to vertices.
There is then an explicit formula even with repeated faces and
degeneracies. For a chart :math:`x` put :math:`M_i=d_{x,i}+1` and label
its local corners by :math:`a\in\prod_i[M_i]`. Let :math:`\pi_x(a)` be
the global vertex of that corner, with repetitions retained, and write
:math:`\mathcal F_{x,v}=\{a:\pi_x(a)=v\}`.

.. prf:theorem:: Corner collisions at linear factor degrees
   :label: thm-skeleton-linear-assembled

   On any finite simploidal set with all factor degrees one, the vertex
   coefficient is

   .. math::
      :label: assembled-risk-linear-corners

      c_v(\mu)=\sum_x
        \frac{w_x}{\prod_iM_i(M_i+1)}
        \sum_{a,b\in\mathcal F_{x,v}}
                 2^{|\{i:a_i=b_i\}|}.

   Empty products and the single corner of a vertex chart have their
   usual values. With :math:`V` global vertices, only dimension zero is
   active, so :math:`A_0=V\sum_vc_v`. The individually optimal normalized
   risk satisfies

   .. math::
      :label: assembled-risk-linear-bound

      C_{\rm cell}^*=
        \left(\sum_v\sqrt{c_v}\right)^2
           \leq A_0\leq V=p.

   The first expression is interpreted as an infimum if some coefficients
   vanish. If :math:`G_\mu` is positive definite, :math:`p` is the matching
   all-at-once risk. Both skeleton designs have strictly smaller risk than
   :math:`p` if some positive-weight chart has at least two distinct global
   corner vertices.

.. prf:proof::

   For a positive-dimensional cell at least one factor has
   :math:`\binom{1-1}{d_i}=0`, so it has no proper controls. Thus
   :math:`C=0`, :math:`H_v=1`, and the propagated vertex polynomial is just
   its assembled basis function
   :math:`\phi_{x,v}=\sum_{a\in\mathcal F_{x,v}}\prod_i t_{i,a_i}`.
   Uniform simplex moments give

   .. math::

      E\,[t_{i,a_i}t_{i,b_i}]
         =\frac{1+\mathbf1_{a_i=b_i}}{M_i(M_i+1)}.

   Squaring the assembled basis function and using independence of the
   factors proves :eq:`assembled-risk-linear-corners`, including every
   cross term between identified corners. Cauchy--Schwarz gives the first
   inequality in :eq:`assembled-risk-linear-bound`. The nonnegative
   assembled functions form a partition of unity on each chart; hence
   :math:`\sum_v\phi_{x,v}^2\leq1`, proving the second inequality after
   integration. If a positive-weight chart has two distinct corner
   vertices, both assembled functions are positive in its interior, so
   this last inequality is strict on a set of positive measure.

For a single unglued product, every corner fiber is a singleton and this
recovers :math:`A_0=\prod_i2M_i/(M_i+1)`. At the other extreme, if all
corners of every chart are the same single global vertex, the entire
model is constant and :math:`c_v=A_0=C_{\rm cell}^*=1`.

Closed forms on arbitrary graphs
--------------------------------

Every one-dimensional simploidal set is a finite multigraph, allowing
loops, parallel edges and isolated vertices. Here edges mean the
non-degenerate one-cells; a collapsed edge contributes through its base
vertex. Assign each edge :math:`e` a positive degree :math:`D_e`. Let
:math:`w_v` and :math:`w_e` be the evaluation weights on vertices and edges,
respectively, with their sum equal to one.

.. prf:theorem:: Graph risk coefficients
   :label: thm-skeleton-graph-risk

   Define

   .. math::

      h_D=\frac{1}{D(D+2)},\qquad
      j_D=\frac{(-1)^{D-1}}{D(D+1)(D+2)},\qquad
      \ell_D=2(h_D+j_D)
         =\frac{2}{(D+1)(D+1+(-1)^D)}.

   The assembled coefficients have the closed forms

   .. math::
      :label: assembled-risk-graph-coefficients

      c_v=w_v+
         \sum_{\substack{e\ni v\\e\text{ is not a loop}}}w_eh_{D_e}
         +\sum_{\substack{e\text{ is a loop}\\\text{at }v}}w_e\ell_{D_e},
      \qquad
      c_e=w_e(D_e-1)\quad(D_e\geq2).

   Each ordinary incident edge occurs once in the first sum. Each loop
   occurs once in the second sum. Edges of degree one have no proper
   controls and no fitting stage, although their evaluation weights still
   contribute to the vertex coefficients.

   Thus, under the assumptions of
   :prf:ref:`thm-skeleton-assembled-risk`,

   .. math::

      \lim_{N\to\infty}\frac{NR_N}{\sigma^2L}
      =\sum_v\frac{c_v}{q_v}
       +\sum_{e:D_e\geq2}\frac{w_e(D_e-1)}{q_e}.

   Whenever all active coefficients are positive, the individual-cell
   optimum is the square of the sum of their square roots. The general
   qualifications for zero coefficients remain those in
   :prf:ref:`cor-skeleton-assembled-allocation`.

.. prf:proof::

   On an edge of degree :math:`D`, write the endpoint propagation
   polynomials as :math:`\varphi_1(x)=\varphi(x)` and
   :math:`\varphi_0(x)=\varphi(1-x)`, where :math:`\varphi` is given in
   :prf:ref:`thm-skeleton-curves`. Set

   .. math::

      A=\frac{\widetilde P_D'}{D(D+1)},\qquad
      B=\frac{\widetilde P_{D+1}'}{(D+1)(D+2)}.

   Then :math:`\varphi_1=(A+B)/2` and
   :math:`\varphi_0=(-1)^{D-1}(A-B)/2`. The opposite parities make
   :math:`A` and :math:`B` orthogonal. Their squared norms are
   :math:`2/[D(D+1)]` and :math:`2/[(D+1)(D+2)]`, respectively.
   Consequently,

   .. math::

      \lVert\varphi_0\rVert^2=\lVert\varphi_1\rVert^2=h_D,\qquad
      \langle\varphi_0,\varphi_1\rangle=j_D.

   An ordinary edge carries independent innovations at its two distinct
   vertices. A loop carries the same innovation at both endpoint
   occurrences, giving squared norm
   :math:`\lVert\varphi_0+\varphi_1\rVert^2=\ell_D`.
   The edge-interior innovation has trace :math:`D-1`, and there are no
   higher-dimensional cells to receive it. Weighting these contributions
   by the evaluation measure proves the formulas.

The loop coefficient has an exact parity pattern:

.. math::

   \ell_1=1,\qquad
   \ell_{2k}=\ell_{2k+1}
      =\frac{1}{(k+1)(2k+1)}\quad(k\geq1).

For positive degrees the reciprocal sequence is
:math:`1,6,6,15,15,28,28,\ldots`. Its distinct values are the hexagonal
numbers, identified in `OEIS A000384 <https://oeis.org/A000384>`_.
The identity above proves the pattern independently of the sequence
identification.

.. prf:corollary:: Graph allocations by dimension
   :label: cor-skeleton-graph-dimension

   Suppose the graph is loop-free, has :math:`V` vertices and :math:`E`
   edges, has no isolated vertices, and every edge has the same degree
   :math:`D\geq2`. Evaluate only on edges, with positive weights summing
   to one. Equal sampling shares within each dimension give

   .. math::

      a_0=\frac{2V}{D(D+2)},\qquad
      a_1=E(D-1),\qquad
      C_{\rm dimension}^*
         =\left(\sqrt{\frac{2V}{D(D+2)}}+
                     \sqrt{E(D-1)}\right)^2.

   These grouped coefficients do not depend on the edge weights or
   connectivity. With :math:`K=D(D+2)`, this optimum is smaller than the
   matching all-at-once risk :math:`p=V+E(D-1)` exactly when

   .. math::

      \frac EV<\frac{(K-2)^2}{8K(D-1)}.

.. prf:proof::

   Every edge contributes twice to the sum of vertex coefficients, so
   :math:`\sum_v c_v=2/K`, while
   :math:`\sum_e c_e=D-1`. Apply the grouped allocation formula.
   Since :math:`p-a_0-a_1=V(1-2/K)>0`, squaring the equivalent inequality
   :math:`2\sqrt{a_0a_1}<p-a_0-a_1` gives the stated threshold.

For the individual optimum, the incidence pattern still matters. Writing
:math:`\delta_v=\sum_{e\ni v}w_e`, the same loop-free setting gives

.. math::

   C_{\rm cell}^*
      =\left(
         \frac{\sum_v\sqrt{\delta_v}}{\sqrt{D(D+2)}}+
         \sqrt{D-1}\sum_e\sqrt{w_e}
        \right)^2.

For example, two quadratic segments meeting at one endpoint, evaluated
equally, have

.. math::

   C_{\rm cell}^*=\frac{27}{8}+\frac{5}{2\sqrt2}>5,\qquad
   C_{\rm dimension}^*=\frac{11}{4}+\sqrt6>5.

Thus even the individual optimum loses to the matching all-at-once risk
:math:`p=5` on this graph, although a single quadratic segment wins.

.. prf:corollary:: Complete comparison for a single loop
   :label: cor-skeleton-loop-comparison

   For a single degree-:math:`D` loop, evaluated uniformly on its edge,

   .. math::

      C_{\rm loop}^*=(\sqrt{\ell_D}+\sqrt{D-1})^2,\qquad p=D.

   The two risks are equal at :math:`D=1`. Inductive fitting loses at
   :math:`D=3,5`, and wins at :math:`D=2,4` and every :math:`D\geq6`.

.. prf:proof::

   For :math:`D>1`, put :math:`T=1/\ell_D>1`.
   The sign of :math:`p-C_{\rm loop}^*` is the sign of

   .. math::

      \Delta=(T-1)^2-4(D-1)T.

   For :math:`D=2k`, :math:`k\geq1`,

   .. math::

      T=(k+1)(2k+1),\qquad
      \Delta=(2k^2-k-2)^2>0.

   The quadratic inside the square has discriminant 17, so it has no
   integer root. For :math:`D=2k+1`, :math:`k\geq1`,

   .. math::

      \Delta=k(4k^3-4k^2-15k-8).

   This is negative for :math:`k=1,2`. For :math:`k=u+3`,
   :math:`u\geq0`, the cubic factor becomes
   :math:`4u^3+32u^2+69u+19>0`.
   At degree one the loop has only one constant global control, and both
   normalized risks equal one.

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
