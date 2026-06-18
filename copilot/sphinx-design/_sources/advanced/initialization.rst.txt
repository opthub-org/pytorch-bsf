Initial Control Points
======================

By default, ``fit()`` generates random initial control points for the Bézier simplex. You can override this by supplying your own initial control points via the ``init`` argument, which may help the optimizer converge faster or to a better solution when you have prior knowledge about the shape of the target manifold.

The ``init`` argument accepts a dictionary mapping multi-index tuples to value vectors, or an already-constructed object (such as a :class:`BezierSimplex` or control-points data) that you have loaded in Python. Each key is a multi-index :math:`\mathbf{d} \in \mathbb{N}_D^M` and each value is the corresponding control point :math:`\mathbf{p}_{\mathbf{d}} \in \mathbb{R}^N`.

When working from a file in Python, load the control points or Bézier simplex first (for example, via :func:`torch_bsf.bezier_simplex.load(path)`) and then pass the resulting object as ``init``. The command-line interface instead accepts a file path directly via ``--init``.

Supported file formats for saving and loading control points (and for use with the CLI ``--init`` option) are described below. In every format the key is the string representation of the multi-index tuple (e.g., ``"(2, 0)"`` for the control point at index :math:`(2, 0)`) and the value is a list of floats.

Pickled PyTorch
---------------

See PyTorch documentation for details:

- https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
- https://pytorch.org/docs/stable/generated/torch.load.html#torch.load

CSV
---

.. code-block::

   "(2, 0)", 0.0, 0.1
   "(1, 1)", 1.0, 1.1
   "(0, 2)", 2.0, 2.1

TSV
---

.. code-block::

   "(2, 0)"	0.0	0.1
   "(1, 1)"	1.0	1.1
   "(0, 2)"	2.0	2.1

JSON
----

.. code-block:: json

   {
     "(2, 0)": [0.0, 0.1],
     "(1, 1)": [1.0, 1.1],
     "(0, 2)": [2.0, 2.1]
   }

YAML
----

.. code-block:: yaml

   (2, 0): [0.0, 0.1]
   (1, 1): [1.0, 1.1]
   (0, 2): [2.0, 2.1]
