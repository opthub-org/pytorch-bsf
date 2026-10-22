Hardware Acceleration
=====================

PyTorch-BSF can train with `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ (``solver="trainer"``), which provides seamless support for hardware acceleration (GPUs, TPUs, etc.) and distributed training (multi-GPU, multi-node).

This page explains how to leverage these features to speed up your Bézier simplex (and simploid) fitting tasks.

Installation
------------

Lightning is an optional dependency, which only ``solver="trainer"`` needs. Install it with the ``lightning`` extra:

.. code-block:: bash

   pip install "pytorch-bsf[lightning]"

Without Lightning, ``solver="trainer"`` (and ``--solver=trainer`` on the command line) raises an :exc:`ImportError` that tells you to run this command; the default ``solver="lstsq"`` and :func:`torch_bsf.fit_sgd` do not need it.
See :doc:`../installation` for all extras and for upgrading from a version that installed Lightning by default.

.. note::
   By default, ``fit()`` solves the least-squares problem exactly on the CPU in double precision (``solver="lstsq"``), which is usually the fastest and most accurate choice, so the options on this page do not apply.
   They take effect when you train with the Lightning trainer instead: pass ``solver="trainer"`` to ``fit()``, or ``--solver=trainer`` on the command line.
   Passing a trainer option without ``solver="trainer"`` raises a ``ValueError`` in the Python API; the command line ignores the trainer options unless ``--solver=trainer`` is given.

CLI / MLflow
------------

When using the command-line interface or MLflow with ``--solver=trainer``, you can control hardware usage via several flags. These flags are passed directly to the underlying PyTorch Lightning Trainer.

Accelerator and Devices
^^^^^^^^^^^^^^^^^^^^^^^

The ``--accelerator`` and ``--devices`` flags are the primary way to specify your hardware.

*   ``--accelerator``: Choose the hardware backend. Supported values include ``cpu``, ``gpu``, ``tpu``, ``hpu``, ``mps`` (for Apple Silicon), or ``auto``.
*   ``--devices``: Specify which devices to use. You can provide an integer for the number of devices (e.g., ``1``, ``2``), a list of device IDs (e.g., ``0,1``), or ``-1`` / ``auto`` to use all available devices.

Example: Use all available GPUs
.. code-block:: bash

   python -m torch_bsf --params data.csv --values results.csv --degree 3 --solver trainer --accelerator gpu --devices -1

Precision
^^^^^^^^^

You can reduce memory usage and increase training speed by using lower-precision floating-point arithmetic. This is particularly effective on modern GPUs (e.g., NVIDIA A100, RTX 30/40 series).

*   ``--precision``: Supports ``16-mixed``, ``bf16-mixed``, ``32-true`` (default), or ``64-true``.

Example: Mixed precision training
.. code-block:: bash

   python -m torch_bsf --params data.csv --values results.csv --degree 3 --solver trainer --accelerator gpu --precision 16-mixed

Multi-Node Training
^^^^^^^^^^^^^^^^^^^

To scale training across multiple machines, use the ``--num_nodes`` flag.

*   ``--num_nodes``: The number of machines (nodes) to use.

Example: Training on 4 nodes with 4 GPUs each
.. code-block:: bash

   # Run this on each node (usually handled by a cluster manager like SLURM)
   python -m torch_bsf --params data.csv --values results.csv --degree 3 --solver trainer --accelerator gpu --devices 4 --num_nodes 4

Python API
----------

Pass ``solver="trainer"`` together with any PyTorch Lightning `Trainer arguments <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_ as keyword arguments to the ``fit()`` function.

.. code-block:: python

   import torch_bsf

   # Use multi-GPU and mixed precision via API
   bs = torch_bsf.fit(
       params=ts,
       values=xs,
       degree=3,
       solver="trainer",
       accelerator="gpu",
       devices=2,
       precision="16-mixed",
   )

The same holds for a Bézier simploid (see :doc:`simploidal`):

.. code-block:: python

   bs = torch_bsf.fit(ts, xs, degree=3, simploid_type=(2, 1), solver="trainer", accelerator="gpu", devices=2)

The returned model can be moved to any device for prediction, like any PyTorch module (e.g., ``bs.to("cuda")``).

Detailed Use Cases
------------------

Large-Scale Pareto Front Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When dealing with thousands of Pareto optimal points and high-degree Bézier simplices (e.g., degree 10+), the number of control points increases significantly. The exact ``"lstsq"`` solver then solves a large dense linear system on the CPU; if it becomes too slow or too large for memory, train with ``solver="trainer"`` and:

1.  **Use GPUs**: Fitting is highly parallelizable.
2.  **Vectorized Forward**: PyTorch-BSF uses a highly efficient, fully vectorized `forward` pass. This means that instead of looping through each control point, it uses matrix operations, which significantly speeds up computation on GPUs.
3.  **Use Mixed Precision**: Set ``precision="16-mixed"`` to fit larger models into GPU memory.
4.  **Increase Batch Size**: Use ``--batch_size`` to optimize GPU throughput.

Distributed Training on Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your dataset is massive or you are performing an extensive search over hyperparameters, you can use multi-node training. PyTorch-BSF supports all distributed strategies provided by Lightning (DDP, FSDP, etc.).

.. tip::
   For most cases, ``strategy="auto"`` is sufficient. If you encounter issues on specialized clusters, you might need to specify ``strategy="ddp"``.

See Also
--------

For more advanced configuration details, please refer to the official documentation:

*   `PyTorch Lightning Accelerator Documentation <https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html>`_
*   `PyTorch Lightning Multi-GPU Training <https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html>`_
*   `Mixed Precision Training in Lightning <https://lightning.ai/docs/pytorch/stable/common/precision_basic.html>`_
