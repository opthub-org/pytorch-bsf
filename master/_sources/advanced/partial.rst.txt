Partial Training
================

By default, ``fit()`` trains all control points of the Bézier simplex simultaneously. The ``freeze`` argument lets you hold a subset of control points constant during training, so that only the remaining (free) control points are updated.

This is useful when:

*   **Boundary constraints are known** — If single-objective optimizations have already determined the vertex solutions, you can freeze those vertices and train only the interior and edge control points.
*   **Incremental refinement** — Fit a low-degree model first, reuse its control points as the initialization for a higher-degree model, and freeze the already-accurate points to stabilize training.
*   **Encoding prior knowledge** — Pin control points whose values are theoretically or physically determined.

The example below demonstrates how to freeze the two vertices of a Bézier curve while training its interior control points.

The ``freeze`` argument takes a list of multi-index lists (e.g., ``[[3, 0], [0, 3]]``). Any control point whose multi-index appears in ``freeze`` is excluded from gradient updates.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf

   ts = torch.tensor(  # parameters on a simplex
      [
         [8/8, 0/8],
         [7/8, 1/8],
         [6/8, 2/8],
         [5/8, 3/8],
         [4/8, 4/8],
         [3/8, 5/8],
         [2/8, 6/8],
         [1/8, 7/8],
         [0/8, 8/8],
      ]
   )
   xs = 1 - ts * ts  # values corresponding to the parameters

   # Initialize 2D control points of a Bézier curve of degree 3
   init = {
      # index: value
      (3, 0): [0.0, 0.1],
      (2, 1): [1.0, 1.1],
      (1, 2): [2.0, 2.1],
      (0, 3): [3.0, 3.1],
   }

   # Or, generate random control points in [0, 1)
   init = torch_bsf.bezier_simplex.rand(n_params=2, n_values=2, degree=3)

   # Or, load control points from a file
   init = torch_bsf.bezier_simplex.load("control_points.yml")

   # Train the edge of a Bézier curve while its vertices are frozen
   bs = torch_bsf.fit(
      params=ts,  # input observations (training data)
      values=xs,  # output observations (training data)
      init=init,  # initial values of control points
      freeze=[[3, 0], [0, 3]],  # freeze vertices of the Bézier curve
   )

   # Predict with the trained model
   t = [
      [0.2, 0.8],
      [0.7, 0.3],
   ]
   x = bs(t)
   print(x)

.. testoutput::
   :hide:

   tensor([[...]], grad_fn=<...>)

Bézier Simploids
----------------

:func:`~torch_bsf.simploidal.fit_simploid` takes ``init`` and ``freeze`` too.
The indices in ``freeze`` are those of the Bézier simploid, one multi-index per factor (e.g., ``((2, 0, 0), (2, 0))``), and ``init`` is required since it gives the values of the frozen control points.
The default ``"lstsq"`` solver then solves the least-squares problem exactly for the free control points, and ``solver="trainer"`` starts from ``init`` and keeps the frozen control points fixed.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf.simploidal import BezierSimploid, fit_simploid, sample_simploid

   t = sample_simploid((2, 1), 100, seed=0)  # parameters on a prism
   x = t[:, :3] * t[:, 3:4]

   init = BezierSimploid.zeros((2, 1), 2, n_values=3)
   corner = ((2, 0, 0), (2, 0))
   with torch.no_grad():
       init[corner].copy_(torch.tensor([1.0, 0.0, 0.0]))

   bs = fit_simploid(t, x, (2, 1), degree=2, init=init, freeze=[corner])
   print(bs[corner].tolist())

.. testoutput::

   [1.0, 0.0, 0.0]

For :func:`~torch_bsf.simploidal.fit_simploidal_set`, ``init`` is a simploidal Bézier space (or its proper control points), and each element of ``freeze`` is a pair ``(simploid, index)`` of any simploid of the set and a control point index of its Bézier simploid.
A control point on a shared face is the same whichever simploid names it.
