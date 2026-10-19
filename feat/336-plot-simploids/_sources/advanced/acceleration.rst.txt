Hardware Acceleration
=====================

PyTorch-BSF is built on top of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_, which means it provides seamless support for hardware acceleration (GPUs, TPUs, etc.) and distributed training (multi-GPU, multi-node) out of the box.

This page explains how to leverage these features to speed up your Bézier simplex fitting tasks.

CLI / MLflow
------------

When using the command-line interface or MLflow, you can control hardware usage via several flags. These flags are passed directly to the underlying PyTorch Lightning Trainer.

Accelerator and Devices
^^^^^^^^^^^^^^^^^^^^^^^

The ``--accelerator`` and ``--devices`` flags are the primary way to specify your hardware.

*   ``--accelerator``: Choose the hardware backend. Supported values include ``cpu``, ``gpu``, ``tpu``, ``hpu``, ``mps`` (for Apple Silicon), or ``auto``.
*   ``--devices``: Specify which devices to use. You can provide an integer for the number of devices (e.g., ``1``, ``2``), a list of device IDs (e.g., ``0,1``), or ``-1`` / ``auto`` to use all available devices.

Example: Use all available GPUs
.. code-block:: bash

   python -m torch_bsf --params data.csv --values results.csv --accelerator gpu --devices -1

Precision
^^^^^^^^^

You can reduce memory usage and increase training speed by using lower-precision floating-point arithmetic. This is particularly effective on modern GPUs (e.g., NVIDIA A100, RTX 30/40 series).

*   ``--precision``: Supports ``16-mixed``, ``bf16-mixed``, ``32-true`` (default), or ``64-true``.

Example: Mixed precision training
.. code-block:: bash

   python -m torch_bsf --params data.csv --values results.csv --accelerator gpu --precision 16-mixed

Multi-Node Training
^^^^^^^^^^^^^^^^^^^

To scale training across multiple machines, use the ``--num_nodes`` flag.

*   ``--num_nodes``: The number of machines (nodes) to use.

Example: Training on 4 nodes with 4 GPUs each
.. code-block:: bash

   # Run this on each node (usually handled by a cluster manager like SLURM)
   python -m torch_bsf --params data.csv --values results.csv --accelerator gpu --devices 4 --num_nodes 4

Python API
----------

The Python API provides two ways to handle hardware acceleration: by moving data to the device manually or by passing trainer configuration to ``fit()``.

Manual Device Management
^^^^^^^^^^^^^^^^^^^^^^^^

If you move your training tensors to a specific device before calling ``fit()``, PyTorch-BSF will attempt to run on that device.

.. code-block:: python

   import torch
   import torch_bsf

   # Assume ts and xs are your parameter and value tensors
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Move tensors to GPU
   ts = ts.to(device)
   xs = xs.to(device)

   # Fit the model (it will detect the device from the tensors)
   bs = torch_bsf.fit(params=ts, values=xs, degree=3)

Passing Trainer Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^

For more granular control, you can pass any PyTorch Lightning `Trainer arguments <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_ directly as keyword arguments to the ``fit()`` function.

.. code-block:: python

   import torch_bsf

   # Use multi-GPU and mixed precision via API
   bs = torch_bsf.fit(
       params=ts, 
       values=xs, 
       degree=3,
       accelerator="gpu",
       devices=2,
       precision="16-mixed"
   )

Detailed Use Cases
------------------

Large-Scale Pareto Front Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When dealing with thousands of Pareto optimal points and high-degree Bézier simplices (e.g., degree 10+), the number of control points increases significantly. In such cases:

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
