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

Numerical Experiments
---------------------

We illustrate Bézier simplex fitting on a small CT reconstruction problem, tracing the L-curve between data fidelity and spatial smoothness.

**Problem Setup:**

- Image: :math:`x \in \mathbb{R}^{256}` (flattened 16×16 circular phantom)
- Forward operator: :math:`K \in \mathbb{R}^{128 \times 256}` (simplified 2D Radon transform, 8 angles × 16 rays per angle)
- Finite-difference operator: :math:`L` (2D anisotropic total variation, horizontal and vertical differences)
- Noisy measurement: :math:`y = Kx^* + \epsilon` (:math:`\epsilon \sim \mathcal{N}(0, 0.05^2)`)
- Regularized objectives:

.. math::

   f_1(x) &= \|Kx - y\|^2 + 0.01\|x\|^2 \quad \text{(fidelity)} \\
   f_2(x) &= \|Lx\|^2 + 0.01\|x\|^2 \quad \text{(smoothness)}

**Parametrization of the trade-off:**

The weighted-sum solution :math:`x^*(w)` depends only on the regularization parameter :math:`\lambda = w_2 / w_1`.
Because :math:`K^\top K` and :math:`L^\top L` have nearly complementary spectra (smooth images are barely penalized by :math:`L` but strongly seen by :math:`K`, and vice versa),
:math:`x^*(w)` has poles just outside the weight simplex: with uniformly spaced weights, almost all of the L-curve is squeezed into a tiny neighborhood of :math:`w = (1, 0)`, where no low-degree polynomial can follow it.
As in the classical L-curve analysis, we therefore sweep :math:`\lambda` on a logarithmic scale and fit the Bézier simplex on the parameter :math:`(1 - t, t)` of the 1-simplex:

.. math::

   \lambda(t) = 10^{-2 + 3t}, \qquad w(t) = \left(\frac{1}{1 + \lambda(t)}, \frac{\lambda(t)}{1 + \lambda(t)}\right), \qquad t \in [0, 1].

**Experiment Procedure:**

1. Sample 41 equally spaced parameters :math:`t \in [0, 1]`, i.e., :math:`\lambda` from :math:`10^{-2}` to :math:`10^{1}`.
2. For each :math:`t`, solve :math:`x^*(t) = \arg\min_x [w_1(t) f_1(x) + w_2(t) f_2(x)]` using L-BFGS-B with the analytic gradient.
3. Collect the Pareto front points :math:`(f_1(x^*(t)), f_2(x^*(t)))` and the reconstructions :math:`x^*(t)`.
4. Fit degree-10 Bézier simplices to the Pareto front and to the Pareto set (the 256 pixels) by exact linear least squares (:func:`~torch_bsf.simploidal.fit_simploid` with type :math:`(1)`).
5. Evaluate the fitted Bézier curves on 20 held-out parameters between the training ones.

On the held-out parameters, the maximum error is about :math:`10^{-3}` on the Pareto front (whose :math:`f_1` ranges over :math:`[0.4, 31]`) and below :math:`10^{-4}` per pixel on the Pareto set.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/medical_imaging_pareto_set.png
         :alt: Pareto set for CT reconstruction (pixel space)
         :width: 100%
     - .. image:: ../_static/medical_imaging_pareto.png
         :alt: Bézier simplex fitting to CT reconstruction L-curve
         :width: 100%
   * - Pareto set (two background pixels).
     - Pareto front.

.. figure:: ../_static/medical_imaging_pareto_samples.png
   :alt: Reconstructions predicted by the fitted Bézier simplex
   :width: 100%
   :align: center

   Reconstructions predicted by the Bézier simplex fitted to the Pareto set, from weak (left) to strong (right) smoothing.

The complete example script is available at :file:`examples/generate_medical_imaging_pareto.py`.
