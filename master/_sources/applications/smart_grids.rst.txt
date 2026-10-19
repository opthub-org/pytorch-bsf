Distributed Smart Grids and Energy Operations
=============================================

Modern power management systems face uniquely high-stakes trade-offs. The Multi-Objective Optimal Power Flow (MO-OPF) model mathematically formulates the challenges of simultaneously minimizing the economic costs of power generation, restricting environmental emission rates, and regulating line transmission losses or voltage deviations. 

For traditional generators, the expected fuel and operating cost across an interconnected network is naturally modeled as a strongly convex quadratic function of power output :math:`P_i`:

.. math:: C_i(P_i) = a_i P_i^2 + b_i P_i + c_i \quad (a_i > 0)

Equally, in multi-agent microgrids integrating photovoltaic systems (PV), energy storage batteries, and Electric Vehicle (EV) charging stations, researchers utilize game-theoretic models to continuously distribute power allocations. To stabilize the energy demand across multiple EV chargers, the network actively minimizes the demand mismatch penalty against a target demand :math:`p_k^*` :cite:p:`yan2021two`. Evaluated as an independent utility function for each node :math:`l`, this forms:

.. math:: u_l = -\frac{1}{2}(p_l - p_l^*)^2

From a mathematical perspective, modifying the power allocations to maximize the utility :math:`u_l` is mathematically identical to minimizing the positive-definite squared deviation between allocated and requested charge. Because the error metrics natively employ bounded quadratic structures (with the positive coefficient :math:`a_i > 0`), the resulting system cost matrix is structurally strongly convex across the sub-groups.

These strongly convex definitions enable massive decentralized networking algorithms—like accelerated Alternating Direction Method of Multipliers (ADMM)—to secure linear convergence uniformly across geographically vast nodes.

Additionally, as weather profiles and market electricity pricing constantly fluctuate, network operators must instantaneously reroute their balance policies. Equipping a distributed smart grid operator with a pre-computed Bézier simplex mapping allows the grid to frictionlessly evaluate these high-dimensional, competitive variables in real-time, executing continuous equilibrium adjustments across millions of endpoints securely.

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
