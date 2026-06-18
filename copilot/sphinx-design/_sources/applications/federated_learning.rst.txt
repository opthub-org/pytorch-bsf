Multi-Task and Federated Learning
=================================

In modern machine learning, models are frequently tasked with balancing strictly competing performance indicators. In multi-task learning (MTL), a single shared neural representation must simultaneously optimize losses across different tasks, often leading to "gradient conflict" where improving one task substantially degrades another. This same underlying tension appears when training models with explicit fairness constraints, where the expected loss over different population subgroups forms a multi-objective optimization problem. 

Furthermore, this principle is critically important in Federated Learning (FL), such as the MOCHA framework, where massive data is distributed across edge devices and a global model must balance diverse local losses against consensus.

By imposing strongly convex regularizations (e.g., Tikhonov/Ridge) on the model parameters alongside the local losses, the optimization formulations become naturally resolvable:

.. math:: 
   
   f_1(\theta) &= \mathcal{L}_1(\theta) + \lambda\|\theta\|^2 \\
   f_2(\theta) &= \mathcal{L}_2(\theta) + \lambda\|\theta\|^2 \\
               &\vdots \\
   f_m(\theta) &= \mathcal{L}_m(\theta) + \lambda\|\theta\|^2

The :math:`L_2` regularization penalty :math:`\lambda\|\theta\|^2` ensures that the Hessian incorporates a positive definite matrix :math:`2\lambda I \succ 0`. This global penalty forcefully transforms the generic convex loss landscapes (common in logistic/ridge regressions or SVMs) into strictly strongly convex objectives. Under strong convexity, algorithms like the Stochastic Multi-Gradient (SMG) descent achieve dramatic convergence accelerations, improving from sub-linear to linear :math:`O(1/n)` or :math:`O(\exp(-\mu T))` convergence rates :cite:p:`liu2024stochastic`.

Rather than computing single compromised aggregations, the global model's Pareto front is modeled as a continuous Bézier simplex. This facilitates a "train once, customize anywhere" paradigm. A global Bézier simplex allows individual edge devices or network nodes to download the analytic continuous trade-off surface, enabling them to instantly select an optimal model variant tailored precisely to their local power limits, fairness constraints, or inference needs, simply by navigating the simplex variables—representing a drastic improvement over constant network-wide retraining.
