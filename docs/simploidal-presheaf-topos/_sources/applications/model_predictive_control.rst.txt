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
