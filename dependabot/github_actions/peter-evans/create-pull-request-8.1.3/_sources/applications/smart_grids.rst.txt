Distributed Smart Grids and Energy Operations
=============================================

Modern power management systems face uniquely high-stakes trade-offs. The Multi-Objective Optimal Power Flow (MO-OPF) model mathematically formulates the challenges of simultaneously minimizing the economic costs of power generation, restricting environmental emission rates, and regulating line transmission losses or voltage deviations. 

For traditional generators, the expected fuel and operating cost across an interconnected network is naturally modeled as a strongly convex quadratic function of power output :math:`P_i`:

.. math:: C_i(P_i) = a_i P_i^2 + b_i P_i + c_i \quad (a_i > 0)

Equally, in multi-agent microgrids integrating photovoltaic systems (PV), energy storage batteries, and Electric Vehicle (EV) charging stations, researchers utilize game-theoretic models to continuously distribute power allocations. To stabilize the energy demand across multiple EV chargers, the network actively minimizes the demand mismatch penalty against a target demand :math:`p_k^*` :cite:p:`yan2021two`. Evaluated as an independent utility function for each node :math:`l`, this forms:

.. math:: u_l = -\frac{1}{2}(p_l - p_l^*)^2

From a mathematical perspective, modifying the power allocations to maximize the utility :math:`u_l` is mathematically identical to minimizing the positive-definite squared deviation between allocated and requested charge. Because the error metrics natively employ bounded quadratic structures (with the positive coefficient :math:`a_i > 0`), the resulting system cost matrix is structurally strongly convex across the sub-groups.

Positive quadratic curvature can support efficient convex dispatch solvers.
Convergence rates for a distributed method additionally depend on the
constraints, coupling and algorithm settings. Convex generator costs alone
do not make nonlinear AC power-flow constraints convex.

A fitted Bézier map can reduce the cost of exploring priorities for a fixed
dispatch problem. Demand, weather and prices must be represented explicitly
or held fixed; changing the priority weights does not update those inputs.

Numerical Experiments
---------------------

We illustrate Bézier simplex fitting on a three-generator economic–emission dispatch problem balancing generation cost against emissions under a fixed demand.

**Problem Setup:**

- Power outputs: :math:`P = [P_1, P_2, P_3] \in \mathbb{R}^3_{\ge 0}`, subject to the demand balance :math:`P_1 + P_2 + P_3 = D = 5`
- Generator 1 is cheap but dirty, generator 3 is expensive but clean.
- Generation cost (strongly convex):

.. math::

   f_1(P) = 0.10 P_1^2 + 0.12 P_2^2 + 0.15 P_3^2 + 0.2 P_1 + 0.3 P_2 + 0.4 P_3

- Emissions (strongly convex):

.. math::

   f_2(P) = 0.20 P_1^2 + 0.10 P_2^2 + 0.05 P_3^2 + 0.3 P_1 + 0.2 P_2 + 0.1 P_3

The demand constraint is what makes the objectives conflict: without it, both would be minimized by :math:`P = 0`.

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`P^*(w) = \arg\min_P [w_1 f_1(P) + w_2 f_2(P)]` subject to the demand balance using SLSQP.
3. Collect the Pareto front points :math:`(f_1(P^*(w)), f_2(P^*(w)))`.
4. Fit a degree-4 Bézier simplex to the weight–objective pairs and another to the weight–decision pairs.
5. Evaluate the fitted Bézier curve on 101 test weights.

The maximum error on the test weights is about :math:`0.003`, i.e., 0.2% of the range of :math:`f_1`.
The Pareto set lies on the plane :math:`P_1 + P_2 + P_3 = 5` and is shown by its projection onto :math:`(P_1, P_2)`.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/smart_grids_pareto_set.png
         :alt: Pareto set for smart grid (power output space)
         :width: 100%
     - .. image:: ../_static/smart_grids_pareto.png
         :alt: Bézier simplex fitting to smart grid Pareto front
         :width: 100%
   * - Pareto set.
     - Pareto front.

The complete example script is available at :file:`examples/generate_smart_grids_pareto.py`.

Separate priorities from generator shares
-----------------------------------------

The model input ``[w1, w2]`` specifies cost versus emissions, not power shares.
It is a point on a one-dimensional simplex. The decision map has three
outputs ``[P1, P2, P3]`` and the objective map has two. Although generator
shares also satisfy a sum constraint, they are responses of this optimization,
not additional coordinates in the weight segment.

Every reference solve enforces demand balance and nonnegative generation.
The two quadratic objective Hessians are positive definite, so their convex
combinations remain positive definite, including at the endpoints. On the
fixed convex feasible set, this gives a unique dispatch for each weight.
It does not make the polynomial surrogate an exact dispatch solver.

What the plots and error measure
--------------------------------

The left panel projects a three-output map to ``(P1, P2)``. Recovering the
third output from balance is possible for the reference solutions, but the
fit should still be checked in all three columns. The right panel describes
the corresponding cost and emissions. A close front fit does not prove that
a separately fitted generation vector satisfies balance or operating limits.

Reproduce the figures from the repository root with
``python examples/generate_smart_grids_pareto.py`` after installing the plot,
SciPy and scikit-learn dependencies used by the script. It uses 10 training
weights and evaluates 101 weights. Distinguish training points from new
weights when interpreting the reported maximum error; also check each
objective rather than normalizing both by only the cost range.

For the decision surrogate, measure
:math:`|\sum_i\widehat P_i-D|` and
:math:`\max(0,-\min_i\widehat P_i)` across the evaluation set. If capacity
bounds are added, inspect those too. Exact reference feasibility at sampled
weights does not establish feasibility everywhere between them. Validate
objective values by recomputing them from the predicted dispatch.

Demand variation and operating limits
-------------------------------------

The example keeps :math:`D=5` fixed. For demand in a bounded interval,
normalize it to a second barycentric block and use a square of type
``(1, 1)``: one factor for priorities and one for demand. Then a parameter
row can be ``[1-u, u, 1-v, v]`` with
:math:`D=(1-v)D_{\min}+vD_{\max}`. Each block sums to one independently.
The :ref:`simploid figure <simploidal-space-overview>` illustrates this
independent-factor construction.

Capacity constraints can change which generators saturate as priorities or
demand vary. Such transitions can create derivative changes in an otherwise
continuous solution map. Add reference solves near them and compare held-out
errors before increasing degree. Multiple local patches may be appropriate
if a single polynomial needs too much capacity; their coordinate conventions
and continuity requirements must then be explicit.

This experiment covers an economic-emission dispatch model. It does not
include voltage constraints, line limits, AC network equations or a
closed-loop grid controller. Those conditions belong in the reference
problem and in the validation of any proposed extension. For fitting
mechanics, see :doc:`../quickstart` and :doc:`../advanced/simploidal`.
