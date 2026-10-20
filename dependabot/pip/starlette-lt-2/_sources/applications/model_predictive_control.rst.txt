Multi-Objective Model Predictive Control
========================================

In industrial automation and robotics, Model Predictive Control (MPC) and Linear Quadratic Regulators (LQR) are foundational algorithms used to optimize future behavior in real-time. Operating chemical plant processes, for instance, requires rigorously balancing product quality tracking (setpoint proximity) against the suppression of abrupt heater and valve modifications (machinery wear). 

Similarly, omnidirectional mobile robots traversing complex environments (e.g., hospitals or warehouses) must optimize trajectory tracking while minimizing energy consumption and avoiding collisions. However, they face significant physical constraints like nonlinear friction modes. Over a predictive horizon :math:`H`, these systems evaluate competing objectives to find the strictly optimal control sequence :math:`U`:

.. math:: 
   
   f_{track}(U) &= \sum_{k=1}^H \|y_k - r_k\|_Q^2 \\
   f_{smooth}(U) &= \sum_{k=0}^{H-1} \|\Delta u_k\|_R^2 \\
   f_{energy}(U) &= \sum_{k=0}^{H-1} \|u_k\|_S^2

where :math:`y_k` denotes system states/outputs, :math:`r_k` the target setpoints, :math:`u_k` the operational inputs, and :math:`\Delta u_k` the input change rates.

Crucially, as long as the weighting matrices (e.g., :math:`R`) acting on the independent control inputs are uniformly positive definite (:math:`R \succ 0`), the resulting multi-objective performance stage cost is strongly convex with respect to the continuous input decisions :math:`U`. Even in complex scenarios where discrete mode switching (like robot friction statuses) transforms the formulation into a Mixed-Integer Quadratic Program (MIQP), the continuous sub-component remains undeniably strongly convex, stabilizing bounded-node optimization searches.

In practice, resolving these multi-objective QP or MIQP problems during an active millisecond control loop requires prohibitive amounts of computational power. Since the strong convexity naturally spans a continuous simplicial Pareto front, controllers equipped with a Bézier simplex representation can bypass the online QP solver entirely. The polynomial evaluation provides an analytic, instantaneous multi-objective control response, enabling ultra-fast, high-frequency continuous re-optimization in highly unstable physical environments.

Numerical Experiments
---------------------

An MPC controller solves the same problem again at every sampling instant, each time from a different measured state.
The optimal inputs therefore depend not only on the objective weights :math:`w` but also on the initial state :math:`x_0`.
Their joint parameter space is the product of the weight simplex and an interval, i.e., the square :math:`\Delta^1 \times \Delta^1`,
which is a *simploid* rather than a simplex (see :doc:`../advanced/simploidal`).
We therefore fit a Bézier simploid of type :math:`(1, 1)` that represents, in one polynomial, the Pareto sets and fronts for all initial states — an explicit multi-objective MPC law.

**Problem Setup:**

- Stable 1D system: :math:`x(k+1) = 0.5\,x(k) + u_k`, target :math:`r = 1`, horizon :math:`H = 5`
- Initial state: :math:`x(0) = x_0 \in [-1, 1]`
- Control sequence: :math:`U = [u_0, u_1, u_2, u_3, u_4] \in \mathbb{R}^5`
- Competing objectives:

.. math::

   f_1(U; x_0) &= \sum_{k=0}^{4} (x(k+1) - r)^2 \quad \text{(tracking error)} \\
   f_2(U) &= \sum_{k=0}^{4} u_k^2 \quad \text{(control effort)}

- Parameter of the simploid: :math:`(w_1, w_2, v_1, v_2) \in \Delta^1 \times \Delta^1` with :math:`x_0 = -v_1 + v_2`

Both objectives are strongly convex in :math:`U` with comparable curvature, so :math:`U^*(w, x_0)` is a smooth rational function of :math:`w`.
For a fixed :math:`w`, it is affine in :math:`x_0`, and the objective values are quadratic in :math:`x_0`.

**Experiment Procedure:**

1. Sample an :math:`11 \times 5` grid on the square: 11 weight vectors from :math:`w = (1,0)` to :math:`(0,1)` and 5 initial states from :math:`x_0 = -1` to :math:`1`.
2. For each :math:`(w, x_0)`, solve :math:`U^*(w, x_0) = \arg\min_U [w_1 f_1(U; x_0) + w_2 f_2(U)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1, f_2)` and the Pareto set points :math:`U^*`.
4. Fit Bézier simploids of type :math:`(1, 1)` with :func:`torch_bsf.fit`: degree :math:`(6, 2)` for the Pareto front and :math:`(6, 1)` for the Pareto set.
5. Evaluate the fitted simploids on a :math:`41 \times 21` test grid.

On the test grid, the maximum error is about :math:`0.01` on the Pareto front (0.2% of the range of :math:`f_1`) and below :math:`10^{-3}` on the Pareto set.
The figures show the curves of the fitted simploids for the five sampled initial states.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/mpc_pareto_set.png
         :alt: Pareto sets for parametric MPC (control input space)
         :width: 100%
     - .. image:: ../_static/mpc_pareto.png
         :alt: Bézier simploid fitting to parametric MPC Pareto fronts
         :width: 100%
   * - Pareto sets.
     - Pareto fronts.

The complete example script is available at :file:`examples/generate_mpc_pareto.py`.
