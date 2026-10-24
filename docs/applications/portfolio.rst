Robust Portfolio Management
===========================

In financial engineering and asset management, the most classical real-world application is Markowitz's mean-variance portfolio optimization :cite:p:`markowitz1952portfolio`. Investors and fund managers seek to simultaneously maximize expected returns and minimize risk (often modeled as the variance of returns). This framework is widely utilized by institutional pension funds (e.g., GPIF), treasury departments, and risk management teams to quantify trade-offs and build consensus on asset allocation decisions.

However, in practice, estimating the covariance matrix from limited observations or highly collinear assets often leads to numerical instability. To resolve this, it is common to introduce a strongly convex regularization term, such as an :math:`L_2` norm penalty on the asset allocation weights or a turnover penalty to suppress excessive trading compared to a previous portfolio :math:`x^{\text{prev}}` :cite:p:`qi2026optimal`.

This formulation effectively creates a three-objective optimization problem over the allocation weights :math:`x \in \mathbb{R}^n`: 

.. math::

   \text{Expected Return: } & f_1(x) = -\mu^T x \\
   \text{Risk (Variance): } & f_2(x) = x^T \Sigma x \\
   \text{Stability (Turnover): } & f_3(x) = \lambda \|x - x^{\text{prev}}\|_2^2

Subject to classical constraints such as the budget constraint :math:`\mathbf{1}^T x = 1` and long-only constraints :math:`x \ge 0`.

For weights :math:`w`, the scalarized Hessian is
:math:`2w_2\Sigma+2w_3\lambda I`. A positive turnover weight guarantees
strong convexity, but that guarantee can disappear on the face
:math:`w_3=0`; the return objective alone is linear. The unconstrained
strongly convex theorem in :doc:`../whatis` therefore cannot be applied to
this entire constrained three-objective formulation without further work.

Fitting a Bézier simplex gives a continuous approximation of sampled
trade-offs. Uniqueness comes from the reference problem's assumptions,
rather than the surrogate representation. Weighted quadratic subproblems
can be solved offline, with their decisions and objective values retained
as training responses. Extensions such as CVaR or uncertain market inputs
change the reference problem and require their own parameterization and
validation.

Numerical Experiments
---------------------

To illustrate Bézier simplex fitting on a portfolio Pareto front, we conducted an experiment using a three-asset mean-variance problem with L2 regularization to ensure strong convexity.

**Problem Setup:**

- Decision variable: :math:`x \in \mathbb{R}^3` (portfolio weights, unconstrained)
- Expected returns: :math:`\mu = [0.05, 0.12, 0.08]`
- Asset variances: :math:`\text{diag}(\Sigma) = [0.10, 0.15, 0.08]`
- Regularized objectives:

.. math::

   f_1(x) &= -\mu^T x + 0.2\|x\|^2 \quad \text{(negative return + L2)} \\
   f_2(x) &= x^T \mathrm{diag}(\Sigma) x + 0.05\|x\|^2 \quad \text{(variance + L2)}

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` uniformly on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`x^*(w) = \arg\min_x [w_1 f_1(x) + w_2 f_2(x)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(x^*(w)), f_2(x^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–objective pairs and another to the weight–decision pairs.
5. Evaluate the fitted Bézier curve on 101 test weights.

The maximum error on the test weights is about :math:`10^{-4}`, i.e., below 0.5% of the range of each objective.
The regularization weight of :math:`f_1` matters: the weighted-sum solution :math:`x^*(w)` is a rational function of :math:`w`, and with a weaker regularization (e.g., :math:`0.05\|x\|^2`) its poles come close to the weight simplex and a cubic Bézier curve visibly deviates from the Pareto front.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/portfolio_pareto_set.png
         :alt: Pareto set for 3-asset portfolio (asset allocation space)
         :width: 100%
     - .. image:: ../_static/portfolio_pareto.png
         :alt: Bézier simplex fitting to 3-asset portfolio Pareto front
         :width: 100%
   * - Pareto set.
     - Pareto front.

The complete example script is available at :file:`examples/generate_portfolio_pareto.py`.

Inputs, allocations and objective values
----------------------------------------

The numerical example uses two regularized objectives, so its input is a
segment of priority weights ``[w1, w2]``. It differs from the three-objective,
budget-constrained formulation above: it has no budget or nonnegativity
constraints, and its first objective includes an allocation penalty. Its
three decision outputs are unconstrained coefficients rather than guaranteed
portfolio percentages. The figure illustrates surrogate construction for
that stated toy problem.

For a diagonal covariance with entries :math:`s_j`, stationarity gives

.. math::

   x_j^*(w)=\frac{w_1\mu_j}
       {2\left(0.2w_1+(s_j+0.05)w_2\right)}.

The denominator stays positive on the weight segment, but the solution is
rational rather than polynomial. A cubic fit approximates this map. It does
not become exact merely because the optimization objectives are quadratic.
Reducing the regularization can make the map vary sharply near an endpoint,
which explains why the same degree and sample count can then perform worse.

Fit and validate the decision map
---------------------------------

Store one row of weights and three allocation outputs for each reference
solve. Fit them with ``BezierSpace.simplex(1, degree=3)`` and
``Samples(weights, allocations)``. A separate two-output fit gives the front.
Keep objective scaling explicit: variance and return have different units,
and changing their scales changes the interpretation of the input weights.
See :doc:`../advanced/normalization` for output transformations.

At held-out weights, compare the predicted allocations with the reference
optimizer, then evaluate the original objectives on those allocations.
Report the maximum error in each output in addition to average error. The
quoted test-grid result describes these constants and this parameter range;
it is not a guarantee for another covariance estimate or asset universe.

For a constrained extension, check the budget residual, minimum holding and
any turnover or exposure limits. Least squares does not automatically
preserve such inequalities between training rows. Projecting a prediction
onto a feasible set can restore feasibility but changes its objectives;
evaluate the projected allocation as well.

Choosing capacity and updating the model
----------------------------------------

Increasing degree gives more flexibility, while adding samples improves
coverage of the same chosen function class. Select them using fresh
reference solves, especially near active-set transitions in a constrained
portfolio. Adding ridge to the Bézier fit shrinks its controls; it does not
replace the allocation regularization in the original portfolio objective.

The fitted map assumes fixed expected returns, covariance and objective
definitions. When those inputs change, the old surrogate represents the
old problem. Refit it, or introduce a bounded independent scenario parameter
as a product factor when an explicit scenario model is available. Arbitrary
market changes are not represented by moving along the priority segment.

The workflow is a numerical approximation example. Evidence for a practical
allocation model also requires evaluation under the intended data and
constraints. See :doc:`../advanced/sampling` for boundary coverage and
:doc:`../advanced/auto_degree` for validation-based capacity selection.
