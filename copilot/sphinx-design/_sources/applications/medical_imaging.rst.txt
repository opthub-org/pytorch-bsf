Medical Imaging and Radiation Therapy
=====================================

Optimizing diagnostic clarity and therapeutic safety simultaneously is paramount in medical fields, making multi-objective optimization frameworks indispensable.

For instance, Computed Tomography (CT) image reconstruction represents an ill-posed multi-objective inverse problem where doctors must minimize a data fidelity term (noise residual) against a regularization term safeguarding spatial continuity. Utilizing the L-Curve method :cite:p:`hansen1993use`, this is routinely formulated mathematically as balancing:

.. math:: 
   
   f_1(x) &= \|Kx - y\|_2^2 \quad \text{(Fidelity)} \\
   f_2(x) &= \|Lx\|_2^2     \quad \text{(Regularization)}

Similarly, Intensity-Modulated Radiation Therapy (IMRT) requires treating patients via highly constrained inverse planning :cite:p:`breedveld2007novel`. Planners must aggressively deliver prescriptive radiation doses to tumors while strictly repressing the doses affecting adjacent healthy Organs At Risk (OAR). The optimization minimizes heavily competing quadratic objectives:

.. math:: 
   
   s(f) = \sum_{v \in V} \xi_v (Hf - d_v^p)^T \tilde{\eta}_v (Hf - d_v^p) + \kappa (Mf)^T (Mf)

where the first component penalizes dose errors via beamlet fluence :math:`f`, and the second component applies spatial smoothing. 

In both clinical environments, strong convexity critically anchors the computation. For CT reconstructions, the explicit :math:`L_2` regularizations (via the parameter :math:`\lambda^2 \|Lx\|^2_2`) stabilize singular projection matrices, resulting in a definitively positive-definite system matrix :math:`\nabla^2 f \succeq \lambda I`. Correspondingly, in IMRT, smoothing parameters (like :math:`\kappa`) combined with dense beamlet mappings force the objective Hessian :math:`2(H^T \tilde{\eta} H + \kappa M^T M)` to remain strictly positive definite. 

In traditional practices, resolving the exact trade-offs requires hours of discrete algorithmic recalculations, delaying urgent therapies. However, by substituting the Pareto front mapping with a Bézier simplex, radiation oncologists and radiologists are granted an interactive, fully continuous digital slider. They can instantaneously sweep through the entire geometric space—trading off tumor destruction versus organ preservation precisely—in real-time without recalculating a single QP iteration.
