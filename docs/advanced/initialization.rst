Initial Control Points
======================

By default, ``fit()`` solves the least-squares problem exactly (``solver="lstsq"``), so it needs no initial control points.
When you train with the Lightning trainer instead (``solver="trainer"``), ``fit()`` starts from random control points unless you supply your own via the ``init`` argument, which may help the optimizer converge faster or to a better solution when you have prior knowledge about the shape of the target manifold.
With either solver, ``init`` also gives the values of the control points held fixed by ``freeze`` (see :doc:`partial`).

The ``init`` argument accepts:

*   a :class:`~torch_bsf.simploidal.BezierSimploid`, e.g., a model returned by an earlier ``fit()`` or loaded from a file; its degree is then used, so ``degree`` may be omitted;
*   a dictionary mapping multi-index tuples to value vectors, where each key is a multi-index :math:`\mathbf{d} \in \mathbb{N}_D^M` (e.g., ``(2, 0)``) and each value is the corresponding control point :math:`\mathbf{p}_{\mathbf{d}} \in \mathbb{R}^N`;
*   a tensor of shape ``(n_control_points, n_values)`` whose rows follow :meth:`~torch_bsf.simploidal.BezierSimploid.indices`.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf

   ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
   xs = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]])

   init = {
      (2, 0): [0.0, 0.1],
      (1, 1): [1.0, 1.1],
      (0, 2): [2.0, 2.1],
   }
   bs = torch_bsf.fit(ts, xs, degree=2, init=init, solver="trainer", max_epochs=1,
                      enable_progress_bar=False, enable_model_summary=False, logger=False)
   print(bs.degree)

.. testoutput::

   (2,)

Saving and Loading
------------------

Save a fitted model with :func:`torch_bsf.simploidal.save` and load it with :func:`torch_bsf.simploidal.load`; the loaded model can be passed as ``init``.
The command-line interface instead accepts a file path directly via ``--init``, whose type and degree are then used in place of ``--degree``.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import tempfile
   from pathlib import Path

   from torch_bsf.simploidal import load, save

   with tempfile.TemporaryDirectory() as d:
      save(Path(d) / "control_points.yml", bs)
      init = load(Path(d) / "control_points.yml")
   print(init.degree)

.. testoutput::

   (2,)

The file format is chosen by the extension, as described below.
In every format except ``.pt``, the key of a control point is the string representation of its index, which is a tuple with one multi-index per factor of the Bézier simploid: a Bézier simplex has a single factor, so the key of the control point at :math:`(2, 0)` is ``"((2, 0),)"``.
The degree and the dimension are recovered from the keys, and every control point must be present exactly once.
Files written by versions before the unification of Bézier simplices and simploids (whose keys look like ``"(2, 0)"``) are rejected.

Pickled PyTorch
---------------

A dictionary with the keys ``"simploid_type"``, ``"degree"`` and ``"control_points"`` saved with :func:`torch.save`.
It is loaded with ``weights_only=True``, so loading never executes arbitrary code.
See PyTorch documentation for details:

- https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
- https://pytorch.org/docs/stable/generated/torch.load.html#torch.load

CSV
---

.. code-block::

   "((2, 0),)",0.0,0.1
   "((1, 1),)",1.0,1.1
   "((0, 2),)",2.0,2.1

TSV
---

.. code-block::

   ((2, 0),)	0.0	0.1
   ((1, 1),)	1.0	1.1
   ((0, 2),)	2.0	2.1

JSON
----

.. code-block:: json

   {
     "((2, 0),)": [0.0, 0.1],
     "((1, 1),)": [1.0, 1.1],
     "((0, 2),)": [2.0, 2.1]
   }

YAML
----

.. code-block:: yaml

   ((2, 0),): [0.0, 0.1]
   ((1, 1),): [1.0, 1.1]
   ((0, 2),): [2.0, 2.1]

Bézier Simploids
----------------

The same formats hold Bézier simploids, with one multi-index per factor in each key, e.g., ``"((2, 0, 0), (1, 0))"`` for a prism of type :math:`(2, 1)`.
See :doc:`simploidal`.
