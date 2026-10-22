Partial Training
================

By default, ``fit()`` fits all control points of the Bézier simplex simultaneously. The ``freeze`` argument lets you hold a subset of control points constant at their values in ``init``, so that only the remaining (free) control points are fitted.

This is useful when:

*   **Boundary constraints are known** — If single-objective optimizations have already determined the vertex solutions, you can freeze those vertices and fit only the interior and edge control points.
*   **Incremental refinement** — Fit a low-degree model first, reuse its control points as the initialization for a higher-degree model, and freeze the already-accurate points to stabilize fitting.
*   **Encoding prior knowledge** — Pin control points whose values are theoretically or physically determined.

The example below demonstrates how to freeze the two vertices of a Bézier curve while fitting its interior control points.

The ``freeze`` argument takes a list of multi-indices (e.g., ``[[3, 0], [0, 3]]``), and requires ``init``, which gives the values of the frozen control points (see :doc:`initialization`).
The default ``"lstsq"`` solver solves the least-squares problem exactly for the free control points, and ``solver="trainer"`` (which needs the ``lightning`` extra; see :doc:`../installation`) starts from ``init`` and excludes the frozen control points from gradient updates.

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
   init = torch_bsf.BezierSimploid.rand((1,), 3, n_values=2)

   # Or, load control points from a file saved by torch_bsf.simploidal.save
   # init = torch_bsf.simploidal.load("control_points.yml")

   # Fit the edge of a Bézier curve while its vertices are frozen
   bs = torch_bsf.fit(
      params=ts,  # input observations (training data)
      values=xs,  # output observations (training data)
      init=init,  # values of the frozen control points; the degree is taken from it
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

``freeze`` works in the same way for a Bézier simploid (see :doc:`simploidal`), whose indices have one multi-index per factor (e.g., ``((2, 0, 0), (2, 0))``).

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf import BezierSimploid, fit
   from torch_bsf.simploidal import sample_simploid

   t = sample_simploid((2, 1), 100, seed=0)  # parameters on a prism
   x = t[:, :3] * t[:, 3:4]

   init = BezierSimploid.zeros((2, 1), 2, n_values=3)
   corner = ((2, 0, 0), (2, 0))
   with torch.no_grad():
       init[corner].copy_(torch.tensor([1.0, 0.0, 0.0]))

   bs = fit(t, x, init=init, freeze=[corner])  # the type and degree are taken from init
   print(bs[corner].tolist())

.. testoutput::

   [1.0, 0.0, 0.0]

For :func:`~torch_bsf.simploidal.fit_simploidal_set`, ``init`` is a simploidal Bézier space (or its proper control points), and each element of ``freeze`` is a pair ``(simploid, index)`` of any simploid of the set and a control point index of its Bézier simploid.
A control point on a shared face is the same whichever simploid names it.
