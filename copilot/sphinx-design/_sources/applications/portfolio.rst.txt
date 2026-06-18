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

Because the regularization term is strongly convex, the resulting scalarized objective function becomes strictly strongly convex. Even if the covariance matrix :math:`\Sigma` is only positive semi-definite, the addition of the turnover penalty guarantees that the Hessian :math:`\nabla^2 f(x) \succeq 2\lambda I` is strictly positive definite. According to the theorems in :cite:p:`mizota2021unconstrained`, this strongly convex problem is guaranteed to be weakly simplicial. 

Fitting a Bézier simplex to this problem allows practitioners to continuously map the entire robust Pareto front—namely, the continuous efficient frontier—guaranteeing unique solutions. Instead of computing disconnected discrete point clouds using weighted sums, analysts obtain a functionally continuous mapping of allocations versus risk. Utilizing sensitivity-based Newton path-following, the full Pareto front can be computed with an extremely efficient iteration complexity of :math:`O(p \log(1/\varepsilon))` :cite:p:`bergou2021complexity`. 
Solvers like MOSEK, Gurobi, and OSQP are widely used to efficiently compute these strongly convex QP subproblems. Modern extensions also include CVaR (Conditional Value-at-Risk) portfolio optimization and robust multi-objective portfolio frameworks under data uncertainty.
