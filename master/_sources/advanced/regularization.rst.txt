Smoothness Regularization
=========================

Fitting a Bézier simplex to a noisy dataset can sometimes result in an "oscillation" or "unstable" surface. To produce a smooth, visually appealing manifold that truly represents the underlying structure, PyTorch-BSF provides **Smoothness Regularization**.

This technique adds a penalty term to the loss function based on the distance between adjacent control points in the simplex.

The Smoothness Penalty
----------------------

The smoothness penalty :math:`\mathcal L_{smooth}` is defined as:

.. math::
   \mathcal L_{smooth} = \sum_{\text{adjacent } i, j} \|\mathbf x_i - \mathbf x_j\|^2

where :math:`\mathbf x_i` and :math:`\mathbf x_j` are adjacent control points in the Bézier simplex. Two control points are considered adjacent if their multi-indices differ by 1 in two components and are identical in others.

How to Use
----------

You can enable smoothness regularization by passing the ``smoothness_weight`` argument to the ``fit()`` function.

.. code-block:: python

   import torch_bsf

   # Fit with a smoothness weight (e.g., 0.1)
   bs = torch_bsf.fit(
       params=ts, 
       values=xs, 
       degree=3, 
       smoothness_weight=0.1
   )

CLI / MLflow
------------

You can also specify the smoothness weight via the command-line interface or MLflow using the ``--smoothness_weight`` flag.

**CLI Example:**

.. code-block:: bash

   python -m torch_bsf --params params.csv --values values.csv --degree 3 --smoothness_weight 0.1

**MLflow Example:**

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf -P params=params.csv -P values=values.csv -P degree=3 -P smoothness_weight=0.1

Bézier Simploids
----------------

:func:`~torch_bsf.simploidal.fit_simploid` and :func:`~torch_bsf.simploidal.fit_simploidal_set` take the same ``smoothness_weight``.
Two control points of a Bézier simploid are adjacent if their multi-indices differ in a single factor, by moving one unit between two components of that factor (see :meth:`~torch_bsf.simploidal.BezierSimploid.adjacency`); for a Bézier simplex, this is the definition above.
In a simploidal Bézier space, the stored control points adjacent in the control net of some simploid are adjacent, and a pair on a shared face counts once (see :meth:`~torch_bsf.simploidal.SimploidalBezierSpace.adjacency`).

Since the penalty is quadratic, the default ``"lstsq"`` solver minimizes the regularized loss exactly. With :math:`n` samples of :math:`m` values, the design matrix :math:`A`, the targets :math:`Y` and the Laplacian :math:`L` of the adjacency graph, it solves

.. math::
   (A^\top A + nm\,w\,L)\,P = A^\top Y,

which is the minimizer of :math:`\frac{1}{nm}\|AP - Y\|^2 + w\,\mathcal L_{smooth}`, the loss minimized by ``solver="trainer"``.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf.simploidal import fit_simploid, sample_simploid

   t = sample_simploid((2, 1), 200, seed=0)  # parameters on a prism
   x = torch.sin(3 * t[:, :2]) + 0.05 * torch.randn(200, 2)  # noisy values
   bs = fit_simploid(t, x, (2, 1), degree=3, smoothness_weight=0.01)

Choosing the Weight
-------------------

The optimal value for ``smoothness_weight`` depends on the noise level of your data:

*   **0.0 (Default)**: No regularization. Best for noise-free data.
*   **0.01 - 0.1**: Light regularization to smooth out minor jitter.
*   **1.0 or more**: Strong regularization. Use this if your data is very noisy or you want a very "flat" manifold.

Benefits
--------

1.  **Noise Reduction**: Effectively filters out high-frequency noise in the training set.
2.  **Manifold Stability**: Prevents the Bézier simplex from "overfitting" to individual scattered observations.
3.  **Visual Quality**: Produces smoother surfaces that are better for visualization and Pareto front approximation.
