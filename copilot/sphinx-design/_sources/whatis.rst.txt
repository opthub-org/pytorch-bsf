What is Bézier simplex fitting?
================================

You are probably familiar with Bézier curves (1-D) and Bézier triangles (2-D) from computer graphics and CAD software. A Bézier simplex is their natural generalization to any number of dimensions: the same elegant polynomial construction, extended to a hypersurface of arbitrary dimension defined over a standard simplex.

At its core, **Bézier simplex fitting is a general-purpose regression technique**. Just as a regular Bézier curve smoothly interpolates or approximates a 1-D point cloud using a small set of *control points*, a Bézier simplex can approximate any continuous map from a standard simplex to a high-dimensional Euclidean space. Given a point cloud dataset defined over simplex coordinates, it can fit a highly flexible and mathematically well-behaved parametric surface to the data.

This page introduces the formal definition of Bézier simplices, the least-squares fitting algorithm used by PyTorch-BSF, and its most prominent real-world applications.


Bézier Simplex
--------------

Let :math:`D, M, N` be non-negative integers, :math:`\mathbb N` the set of non-negative integers (including zero), and :math:`\mathbb R^N` the :math:`N`-dimensional Euclidean space.
We define the *index set* by

.. math:: \mathbb N_D^M = \left\{\mathbf d=(d_1,\ldots,d_M)\in\mathbb N^M\ \Bigg|\ \sum_{m=1}^M d_m=D\right\},

and the *simplex* by

.. math:: \Delta^{M-1} = \left\{\mathbf t=(t_1,\ldots,t_M)\in[0,1]^M\ \Bigg|\ \sum_{m=1}^M t_m=1\right\}.

An :math:`(M-1)`-dimensional *Bézier simplex* of degree :math:`D` in :math:`\mathbb R^N` is a polynomial map :math:`\mathbf b: \Delta^{M-1}\to\mathbb R^N` defined by

.. math:: \mathbf b(\mathbf t\mid\mathbf p) = \sum_{\mathbf d\in\mathbb N_D^M} \binom{D}{\mathbf d} \mathbf t^{\mathbf d} \mathbf p_{\mathbf d},

where :math:`\mathbf t^{\mathbf d} = t_1^{d_1} t_2^{d_2}\cdots t_M^{d_M}`, :math:`\binom{D}{\mathbf d}=D! / (d_1!d_2!\cdots d_M!)`, and :math:`\mathbf p_{\mathbf d}\in\mathbb R^N\ (\mathbf d\in\mathbb N_D^M)` are parameters called the *control points*.

.. figure:: _static/bezier-simplex.png
   :width: 33%
   :align: center

   A 2-D Bézier simplex of degree 3 in :math:`\mathbb R^3` and its control points. The shape of the simplex is determined by the control points :math:`\mathbf p_{(3,0,0)}, \mathbf p_{(2,1,0)}, \ldots, \mathbf p_{(0,0,3)}`.


Bézier Simplex Fitting
----------------------

Assume we have a finite dataset :math:`B\subset\Delta^{M-1}\times\mathbb R^N` and want to fit a Bézier simplex to the dataset. What we are trying can be formulated as a problem of finding the best vector of control points :math:`\mathbf p=(\mathbf p_{\mathbf d})_{\mathbf d\in\mathbb N_D^M}` that minimizes the least squares error between the Bézier simplex and the dataset:

.. math:: \arg\min_{\mathbf p} \sum_{(\mathbf t,\mathbf x)\in B}\|\mathbf b(\mathbf t\mid\mathbf p)-\mathbf x\|^2.

PyTorch-BSF provides an algorithm for solving this optimization problem with the L-BFGS algorithm.

.. figure:: _static/bezier-simplex-fitting.png
   :width: 66%
   :align: center

   A Bézier simplex fitted to a dataset. The control points are determined by the least squares fitting algorithm.


Approximation Theorem
---------------------

Any continuous map from a simplex to a Euclidean space can be approximated by a Bézier simplex. More precisely, the following theorem holds :cite:p:`kobayashi2019bezier`:

.. prf:theorem:: Universal Approximation Theorem

   For any continuous map :math:`\phi: \Delta^{M-1} \to \mathbb{R}^N` and any :math:`\epsilon > 0`, there exists a degree :math:`D` and control points :math:`\mathbf{p}` such that the Bézier simplex :math:`\mathbf{b}(\mathbf{t} \mid \mathbf{p})` satisfies :math:`\max_{\mathbf{t} \in \Delta^{M-1}} \| \phi(\mathbf{t}) - \mathbf{b}(\mathbf{t} \mid \mathbf{p}) \| < \epsilon`.

This guarantees that Bézier simplices are universal approximators for any continuous simplex-domain function.


Relation to Multi-Objective Optimization
----------------------------------------

Data suitable for modeling with a Bézier simplex are point clouds distributed along a low-dimensional (e.g., 1 to 10 dimensions) curved simplex lying within a high-dimensional ambient space (e.g., tens to thousands of dimensions). Such data can be regarded as samples from the solution set of a specific class of multi-objective optimization problems.


Multi-Objective Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many real-world applications, from engineering design to machine learning, we often need to optimize multiple conflicting criteria simultaneously (e.g., maximizing performance while minimizing cost). Because these objectives naturally compete with one another, it is generally impossible to find a single perfect solution. Instead, multi-objective optimization seeks to find a set of optimal trade-offs, providing the mathematical foundation to explore and select the best compromise.

.. prf:definition:: Multi-Objective Optimization Problem

   Let :math:`X` be a feasible decision space. A multi-objective optimization problem with :math:`M` objectives aims to minimize a vector-valued function:

   .. math:: \min_{x \in X} f(x) = (f_1(x), \ldots, f_M(x))^\top.

Since the objectives typically conflict with one another, there is rarely a single solution that minimizes all :math:`M` objectives simultaneously. Instead, optimality is defined in terms of trade-offs.

.. prf:definition:: Pareto Set and Pareto Front

   A solution :math:`x \in X` is said to *dominate* another solution :math:`x' \in X` if :math:`f_m(x) \le f_m(x')` for all :math:`m \in \{1, \ldots, M\}` and :math:`f_j(x) < f_j(x')` for at least one index :math:`j`.

   A solution :math:`x^* \in X` is *Pareto optimal* if no other solution within :math:`X` dominates it.

   * The *Pareto set* is the set of all Pareto optimal solutions in the decision space :math:`X`.
   * The *Pareto front* is the image of the Pareto set in the objective space :math:`f(X) \subset \mathbb{R}^M`.


Weakly Simplicial Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^

In many real-world multi-objective optimization problems, the Pareto set and Pareto front exhibit a highly structured, continuous shape. Specifically, they often mirror the topological structure of a standard simplex (e.g., a curve for a two-objective problem, a curved triangle for a three-objective problem, and so on). The concept of a *weakly simplicial problem* formally captures this property, ensuring that the optimal trade-off surfaces are well-behaved and can be elegantly approximated by Bézier simplices.

.. prf:definition:: Weakly Simplicial Problem :cite:p:`mizota2021unconstrained`

   Let :math:`f:X\to\mathbb R^M` be a mapping, where :math:`X` is a subset of :math:`\mathbb R^N`. The problem of minimizing :math:`f` is :math:`C^r`\ *-simplicial* if there exists a :math:`C^r`-mapping :math:`\Phi:\Delta^{M-1} \to X^\star(f)` such that both the mappings :math:`\Phi|_{\Delta_I}:\Delta_I\to X^\star(f_I)` and :math:`f|_{X^\star(f_I)}:X^\star(f_I)\to f(X^\star(f_I))` are :math:`C^r`-diffeomorphisms for any nonempty subset :math:`I` of :math:`\{1,\ldots,M\}`, where :math:`0\le r\le\infty`. The problem of minimizing :math:`f` is :math:`C^r`\ *-weakly simplicial* if there exists a :math:`C^r`-mapping :math:`\phi:\Delta^{M-1}\to X^\star(f)` such that :math:`\phi(\Delta_I)=X^\star(f_I)` for any nonempty subset :math:`I` of :math:`\{1,\ldots,M\}`, where :math:`0\le r\le\infty`.

.. figure:: _static/simplicial-problem.png
   :width: 100%
   :align: center

   A simplicial problem: the Pareto set and Pareto front are homeomorphic to a simplex, i.e., they have no pinched topology.

.. figure:: _static/weakly-simplicial-problem.png
   :width: 33%
   :align: center

   A weakly simplicial problem: the Pareto set and Pareto front are a continuous image of a simplex, i.e., they may have a pinched topology.

In weakly simplicial problems, there exists a continuous map from a simplex to the Pareto set and Pareto front, which is guaranteed to be approximable by the Universal Approximation Theorem.


Strongly Convex Problems
^^^^^^^^^^^^^^^^^^^^^^^^

While the concept of weakly simplicial problems provides a powerful framework, verifying this property for an arbitrary problem can be challenging. Fortunately, a broad and highly practical class of optimization problems—strongly convex problems—are mathematically guaranteed to be weakly simplicial. This means that if an optimization problem is strongly convex, its Pareto front inherently possesses a simplex-like structure, making it an ideal candidate for Bézier simplex fitting. 
As outlined by :cite:t:`hamada2020topology`, this simplicial topology of the Pareto set is a fundamental, rigorous mathematical property of strongly convex multi-objective problems.

Strong convexity commonly arises in practice through four natural structural mechanisms: 
**(1) L2 regularization**, such as ridge penalties; **(2) Positive-definite quadratic forms**, like risk covariances or LQR stage costs; **(3) Squared Euclidean distances**, prevalent in facility location; and **(4) Composite structures**, where a strongly convex smooth term is paired with a convex nonsmooth penalty.

.. prf:definition:: Strongly Convex Problem

   A multi-objective optimization problem is *strongly convex* if its feasible decision space :math:`X` is convex, and every objective function :math:`f_m` (:math:`m=1,\ldots,M`) is strongly convex. 
   Formally, a function :math:`f_m` is strongly convex with parameter :math:`\mu > 0` if for all :math:`x, y \in X` and :math:`t \in [0, 1]`:

   .. math:: f_m(tx + (1-t)y) \le t f_m(x) + (1-t)f_m(y) - \frac{\mu}{2} t(1-t) \|x - y\|^2.


The formal mathematical foundation for this connection is given by the following theorems, which prove that strongly convex problems are inherently weakly simplicial and that their optimal solutions can be continuously parameterized by a standard simplex.

.. prf:theorem:: Theorems 1 and 2 in :cite:p:`mizota2021unconstrained`

   Let :math:`f: \mathbb{R}^n \to \mathbb{R}^m` be a :math:`C^r`-strongly convex mapping (:math:`0 \le r \le \infty`).
   Then, the problem of minimizing :math:`f` is :math:`C^{r-1}`-weakly simplicial for :math:`r > 0` and :math:`C^0`-weakly simplicial for :math:`r = 0`.



.. prf:theorem:: Proposition 1 of :cite:p:`mizota2021unconstrained`

   Let :math:`f: \mathbb{R}^n \to \mathbb{R}^m` be a strongly convex mapping.
   Then, the mapping :math:`x^*: \Delta^{m-1} \to X^*(f)` defined by

   .. math:: x^*(w) = \arg\min_x \sum_{i=1}^m w_i f_i(x)

   is surjective and continuous.


This guarantees that for strongly convex models, their Pareto fronts admit a simplex structure and can be efficiently reconstructed using Bézier simplex fitting.


Statistical Test for Simplicial Topology
----------------------------------------

When the underlying problem class is not analytically known in advance (e.g., in complex real-world designs using simulations), it is not immediately clear whether the Pareto set admits a simplex structure. In such cases, data-driven statistical tests based on persistent homology can determine whether this assumption is warranted before committing to a Bézier simplex model.

Earlier work introduced the formal concept of a *simple problem*:

.. prf:definition:: Simple Problem :cite:p:`hamada2018data`

   A problem is *simple* if it satisfies two conditions:
   
   | (S1) The Pareto set is homeomorphic to a standard simplex, and 
   | (S2) the objective mapping restricted to the Pareto set is a :math:`C^0`-embedding.


.. prf:theorem:: Equivalence of Simple and Simplicial Problems

   All :math:`C^r`-simplicial problems are simple (:math:`0\le r \le \infty`). A simple problem is :math:`C^0`-simplicial if the interiors of Pareto sets for any two subproblems do not intersect.

Furthermore, it is conjectured that "simple" and "simplicial" are equivalent even without this extra condition. Therefore, testing whether a problem is simple serves as a practical test for whether a problem is simplicial. Notably, the non-intersection condition itself can also be tested statistically.

To strictly determine whether a problem is simple, two complementary statistical tests are required. In standard statistical testing, failing to reject a null hypothesis does not allow one to affirmatively adopt it. Therefore, we need both a test to reject simplicity and a test to affirmatively confirm it.

**1. Testing that a problem is not simple**

:cite:t:`hamada2018data` introduced a data-driven method using persistent homology and its confidence sets to test for violations of the simplicity conditions. Specifically, we can reject the simplicity hypothesis if the estimated Betti numbers do not match those of a topological simplex:

.. prf:theorem:: Theorem 3.1 in :cite:p:`hamada2018data` : Test for (S1) violation

   Let :math:`K(\mathcal{F})` be a simplicial complex representing the approximated Pareto set. Assuming its geometric realization is homotopy equivalent to the true Pareto set, if one of the following homological conditions is satisfied:

   * :math:`H_0(K(\mathcal{F})) \not\cong \mathbb{Z}`
   * :math:`\exists i > 0 : H_i(K(\mathcal{F})) \not\cong 0`

   then the problem does not satisfy the simplicity condition (S1).

**2. Testing that a problem is simple**

Building upon this, :cite:t:`hamada2020test` demonstrated that this homology-based approach can also be used to affirmatively test whether a problem *is* simple. By leveraging the h-cobordism theorem from algebraic topology, they established a mathematical characterization showing that if the Pareto set (modeled as a compact smooth manifold) and its boundary are simply connected, and its estimated homology matches that of a standard simplex, then the Pareto set is guaranteed to be homeomorphic to a simplex.
This affirmative test complements the non-simpliciality test, allowing practitioners to strictly verify the simpliciality of a multi-objective optimization problem from sampled Pareto solutions before applying Bézier simplex fitting.

.. prf:theorem:: Theorem 3 in :cite:p:`hamada2020test`

   Let :math:`M` be an :math:`n`-dimensional compact :math:`C^\infty` manifold. If both :math:`M` and its boundary :math:`\partial M` are simply connected, and its homology groups satisfy:

   .. math::

      H_q(M) \cong H_q(\Delta^n) \cong \begin{cases} \mathbb{Z} & (q=0) \\ 0 & (q \neq 0) \end{cases}

   then :math:`M` is homeomorphic to the standard simplex :math:`\Delta^n`.


