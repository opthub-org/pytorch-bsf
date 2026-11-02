Elastic Net Model Selection
===========================

A canonical and highly practical application of this theory is hyperparameter optimization for the **Elastic Net**. In machine learning and statistics, selecting the correct regularization parameters is critical to balancing data fidelity, model sparsity, and numerical stability. 

Traditional approaches rely on exhaustive grid searches or cross-validation over a discrete set of hyperparameter combinations :math:`(\alpha, \lambda)`. This process is computationally expensive. By formulating the hyperparameter tuning as a multi-objective optimization problem over a **regularization map** :cite:p:`bonnel2019post`, PyTorch-BSF allows for a continuous exploration of the model space, often reducing the computational cost by orders of magnitude (e.g., achieving speeds up to 2,000 times faster than exhaustive search for a 3-objective problem).

Unified Framework for Sparse Modeling
--------------------------------------

The Elastic Net example is part of a broader framework for **generalized sparse modeling**. Any optimization problem expressed as a convex combination of several strongly convex functions can be analyzed using this approach:

.. math:: 
   
   \min_{\beta \in \mathbb{R}^N} f_w(\beta) = \sum_{m=1}^M w_m f_m(\beta), \quad w \in \Delta^{M-1}

where :math:`f_m` represent different aspects of the model, such as data fidelity (loss functions) or structural priors (regularization terms). By ensuring each :math:`f_m` is strongly convex (e.g., by adding a small :math:`L_2` penalty :math:`\frac{\epsilon}{2} \|\beta\|_2^2`), the resulting solution map :math:`x^*(w)` is guaranteed to be weakly simplicial :cite:p:`mizota2021unconstrained`, allowing for high-quality approximation via Bézier simplices.

This unified framework extends far beyond the standard Elastic Net:

* **Generalized Linear Models (GLMs)**: The loss term can be any negative log-likelihood with a canonical link function (e.g., Logistic, Poisson, or Gamma regression).
* **Structural Regularization**: Practitioners can incorporate structural priors like **Group Lasso** :cite:p:`yuan2006model`, **Fused Lasso** :cite:p:`tibshirani2005sparsity`, or **Smoothed Lasso** to encode spatial or group relationships between features.
* **Transfer Learning and Covariate Shift**: By using importance-weighted empirical risk as the loss function, the framework can handle scenarios where the training and test data distributions differ (covariate shift).
* **Robust Estimation**: Replacing the standard squared error with robust loss functions like the **Huber loss** allows for model selection that is resilient to outliers.

Elastic Net Formulation
-----------------------

The standard Elastic Net regression problem is formulated as:

.. math::

   \min_{\beta \in \mathbb{R}^N} \frac{1}{2n}\|y - X\beta\|_2^2
   + \lambda \Bigl(\alpha \|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\Bigr)

where :math:`\lambda \ge 0` is the overall regularization strength and :math:`\alpha \in [0, 1]` controls the L1/L2 mixing ratio. Setting :math:`\alpha = 1` recovers the Lasso, and :math:`\alpha = 0` gives Ridge regression.

To cast this into the multi-objective framework required by PyTorch-BSF, we identify three objectives over the model weights :math:`\beta \in \mathbb{R}^N`:

.. math::

   f_{\text{data}}(\beta) &= \frac{1}{2n}\|y - X\beta\|_2^2 + \frac{\epsilon}{2}\|\beta\|_2^2 \\
   f_{\text{sparse}}(\beta) &= \|\beta\|_1 + \frac{\epsilon}{2}\|\beta\|_2^2 \\
   f_{\text{smooth}}(\beta) &= \frac{1 + \epsilon}{2}\|\beta\|_2^2

where :math:`n` is the number of observations and :math:`\epsilon > 0` is a small constant.
These definitions introduce an :math:`\epsilon`-regularized (strongly convex) surrogate of the
classical Elastic Net objective: the additional :math:`\frac{\epsilon}{2}\|\beta\|_2^2` terms in
:math:`f_{\text{data}}` and :math:`f_{\text{sparse}}` (and the corresponding adjustment in
:math:`f_{\text{smooth}}`) ensure that all three objectives are strongly convex, which is required
for the solution map to be weakly simplicial :cite:p:`mizota2021unconstrained`.
This :math:`\epsilon`-perturbation slightly changes the minimizer for fixed :math:`(\alpha, \lambda)`,
but the classical Elastic Net formulation is recovered in the limit as :math:`\epsilon \to 0`.
We can then express this :math:`\epsilon`-regularized objective as a convex combination of these
three functions:

.. math::

   \min_{\beta \in \mathbb{R}^N} \; w_1 \, f_{\text{data}}(\beta)
   + w_2 \, f_{\text{sparse}}(\beta)
   + w_3 \, f_{\text{smooth}}(\beta),
   \quad (w_1, w_2, w_3) \in \Delta^2.

The conventional elastic-net parameters :math:`\lambda` and :math:`\alpha` relate to the
simplex weight vector :math:`w = (w_1, w_2, w_3)` by:

.. math::

   w_1 = \frac{1}{1 + \lambda}, \qquad
   w_2 = \frac{\lambda\,\alpha}{1 + \lambda}, \qquad
   w_3 = \frac{\lambda\,(1-\alpha)}{1 + \lambda}.

With these weights the convex-combination objective equals the :math:`\epsilon`-regularized
elastic net objective divided by the positive constant :math:`(1 + \lambda)`.
Because this scale factor does not depend on :math:`\beta`, both formulations share the
same minimizer.

This maps the semi-infinite rectangle :math:`[0, \infty) \times [0, 1]` in
:math:`(\lambda, \alpha)` onto the 2-simplex :math:`\Delta^2`. When
:math:`\lambda = 0` the entire edge :math:`\{0\} \times [0, 1]` collapses to
the single vertex :math:`(1, 0, 0)` (all regularization vanishes), so the
mapping is not injective at :math:`\lambda = 0`. For :math:`\lambda > 0`,
different :math:`(\lambda, \alpha)` pairs map to distinct interior points,
enabling a single Bézier simplex to represent the entire elastic-net
regularization path.

By training the model on a sparse subset of weight vectors :math:`w` and fitting a Bézier simplex, we obtain a continuous **solution map** :math:`(x^*, f \circ x^*): \Delta^{M-1} \to G^*(f)` that maps any weight :math:`w` to the optimal weights :math:`\beta` and the corresponding objective values.

Regularization map of the elastic net:

.. grid:: 3
   :gutter: 2

   .. grid-item-card::

      .. figure:: ../_static/figure1/wine,weight_102_102_1_1000,W.png
         :width: 100%

         Weight space :math:`\Delta^{2}`

   .. grid-item-card::

      .. figure:: ../_static/figure1/wine,weight_102_102_1_1000,X123.png
         :width: 100%

         Parameter space :math:`\Theta^*(f)`

   .. grid-item-card::

      .. figure:: ../_static/figure1/wine,weight_102_102_1_1000,F.png
         :width: 100%

         Objective space :math:`f(\Theta^*(f))`


Model Selection on the Regularization Map
-----------------------------------------

The resulting Bézier simplex provides a continuous surrogate of the performance surface, enabling users to apply various statistical model selection criteria analytically:

1. **Minimum Cross-Validation Error (min profile rule)**: Locates the model :math:`w` that minimizes the mean cross-validation error across folds, prioritizing predictive accuracy.
2. **One Standard Error Rule (1se profile rule)**: Selects the most parsimonious (sparsest) model whose mean error is within one standard error of the minimum. This heuristic is widely used in tools like `glmnet` to gain stability and interpretability.
3. **AICc Profile Rule**: Balances goodness-of-fit and model complexity using the **Akaike Information Criterion with Finite Sample Correction**. This allows for selecting models with a theoretically grounded trade-off without the noise of fold-wise cross-validation variability.


Interactive Exploration and Insights
------------------------------------

The Bézier simplex approximation of the regularization map provides more than just a tool for optimization. It offers a **face structure** that naturally reflects the combinations of subsets of loss and regularization terms. By observing the solutions on each face of the simplex, users can:

* **Obtain Insights**: Gain a deep understanding of the trade-offs between different modeling assumptions, such as :math:`L_1` vs :math:`L_2` regularization.
* **Perform Exploratory Analysis**: Test how sensitive the optimal model is to changes in hyperparameters without the trial-and-error of retraining.
* **Support Model Selection**: Make better decisions *a posteriori* by seeing the entire landscape of potential models, rather than relying on a single fixed structure chosen *a priori*.

Empirical Evaluation
--------------------

The effectiveness of PyTorch-BSF for Elastic Net is demonstrated using the **Wine** dataset from the UCI Machine Learning Repository. Experiments show that even with a limited number of training points (e.g., 51 points for a degree-6 simplex), the Bézier simplex accurately approximates the entire solution map, maintaining low Mean Squared Error (MSE) across the continuous hyperparameter space.

The following figures compare the results obtained through an exhaustive grid search (ground truth) and the Bézier simplex approximation. The high similarity between the performance surfaces confirms the fidelity of the surrogate model.

Ground truth results from exhaustive grid search (102x102 grid):

.. grid:: 2
   :gutter: 2

   .. grid-item-card::

      .. figure:: ../_static/grid/wine,weight_102_102_1_1000,mean,F.png
         :width: 100%

         Mean CV error

   .. grid-item-card::

      .. figure:: ../_static/grid/wine,weight_102_102_1_1000,std,F.png
         :width: 100%

         Std dev of CV error

   .. grid-item-card::

      .. figure:: ../_static/grid/wine,weight_102_102_1_1000,aicc,F.png
         :width: 100%

         AICc

   .. grid-item-card::

      .. figure:: ../_static/grid/wine,weight_102_102_1_1000,nonzero,F.png
         :width: 100%

         Number of nonzero coefficients

Approximation results for the Wine dataset with a Bézier simplex of degree :math:`d = 6`:

.. grid:: 2
   :gutter: 2

   .. grid-item-card::

      .. figure:: ../_static/mesh/wine,weight_7_7_1_1000,meshgrid,d_6,f,mean,std,nonzero,aicc,x1-6.tsv,mean,F.png
         :width: 100%

         Mean CV error

   .. grid-item-card::

      .. figure:: ../_static/mesh/wine,weight_7_7_1_1000,meshgrid,d_6,f,mean,std,nonzero,aicc,x1-6.tsv,std,F.png
         :width: 100%

         Std dev of CV error

   .. grid-item-card::

      .. figure:: ../_static/mesh/wine,weight_7_7_1_1000,meshgrid,d_6,f,mean,std,nonzero,aicc,x1-6.tsv,aicc,F.png
         :width: 100%

         AICc

   .. grid-item-card::

      .. figure:: ../_static/mesh/wine,weight_7_7_1_1000,meshgrid,d_6,f,mean,std,nonzero,aicc,x1-6.tsv,nonzero,F.png
         :width: 100%

         Nonzero coefficients
