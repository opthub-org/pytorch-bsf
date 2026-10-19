Bézier Simplex Visualization
============================

Visualizing high-dimensional Bézier simplices is crucial for understanding the manifold structure and analyzing Pareto fronts. PyTorch-BSF provides high-level utilities to project and plot 2D and 3D Bézier manifolds.

The ``torch_bsf.plotting`` Module
---------------------------------

The ``plot_bezier_simplex()`` function provides a high-level API for visualization. It automatically handles the dimensionality of your model.

.. code-block:: python

   import torch_bsf
   from torch_bsf.plotting import plot_bezier_simplex
   import matplotlib.pyplot as plt

   # Fit a model (e.g., a Bézier curve in 3D)
   bs = torch_bsf.fit(params=ts, values=xs, degree=3)

   # Plot the model
   fig = plt.figure(figsize=(10, 8))
   ax = plot_bezier_simplex(bs, num=100)
   plt.title("Bézier Manifold in 3D Space")
   plt.show()

Supported Plot Types
--------------------

Bézier Curves (:math:`n\_params=2`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Bézier curves, the tool plots a smooth path.

*   **2D Values**: Plots a path in the XY plane.
*   **3D Values**: Plots a 3D path in XYZ space.
*   **Optional**: Control points and the "control polygon" can be displayed using ``show_control_points=True``.

Bézier Triangles (:math:`n\_params=3`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Bézier triangles (surfaces), the tool uses triangulation to render a smooth manifold.

*   **2D Values**: Plots a projection of the surface into 2D space.
*   **3D Values**: Renders a smooth 3D surface using ``plot_trisurf``.
*   **Optional**: Control points and the underlying mesh can be displayed.

High-Dimensional Bézier Simplices (:math:`n\_params \geq 4`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For high-dimensional Bézier simplices, the function creates a **pairwise scatter
plot** (pair plot) of the output values.

*   **Diagonal panels**: Histograms showing the distribution of each output
    dimension individually.
*   **Off-diagonal panels**: Scatter plots showing the relationship between
    every pair of output dimensions.
*   **Optional**: Control points are overlaid on off-diagonal panels and as
    vertical lines on diagonal panels when ``show_control_points=True``.

The return value is a 2-D ``numpy.ndarray`` of ``matplotlib.axes.Axes`` with
shape ``(n_values, n_values)``.

.. code-block:: python

   from torch_bsf import bezier_simplex
   from torch_bsf.plotting import plot_bezier_simplex
   import matplotlib.pyplot as plt

   # Build a random 4-parameter, 3-output model (n_params=4 means a 3-simplex input)
   model = bezier_simplex.randn(n_params=4, n_values=3, degree=2)

   # Plot returns an (n_values, n_values) grid of axes
   axes = plot_bezier_simplex(model, num=30)
   axes[0, 1].set_xlabel("Objective 2")
   axes[1, 0].set_ylabel("Objective 1")
   plt.suptitle("Pairwise Plot of High-Dimensional Bézier Simplex")
   plt.tight_layout()
   plt.show()

Customizing the Plot
--------------------

The ``plot_bezier_simplex()`` function returns a standard Matplotlib axes object (``Axes`` or ``Axes3D``) for low-dimensional models, and a 2-D array of ``Axes`` for high-dimensional models. You can customize labels, titles, and styles using the standard Matplotlib API.

.. code-block:: python

   ax = plot_bezier_simplex(bs, num=100, color="blue", alpha=0.5)
   ax.set_xlabel("Objective 1")
   ax.set_ylabel("Objective 2")
   ax.set_zlabel("Objective 3")
   ax.view_init(elev=20, azim=45)

Why Visualize?
--------------

1.  **Validation**: See how well the Bézier simplex fits your training observations.
2.  **Pareto Front Analysis**: Analyze the shape and tradeoff of your Pareto front in multi-objective optimization.
3.  **Communication**: Clearly communicate the structure of high-dimensional manifolds to stakeholders.
