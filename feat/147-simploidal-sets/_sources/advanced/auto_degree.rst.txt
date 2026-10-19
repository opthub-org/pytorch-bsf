Automatic Degree Selection
==========================

Choosing the optimal degree for a Bézier simplex can be challenging. A degree that is too low may underfit and fail to capture the manifold's complexity, while a degree that is too high may overfit to noise. 

PyTorch-BSF provides an automated tool to select the best degree based on **k-fold cross-validation**.

How to Use
----------

The ``select_degree()`` function iterates through multiple degrees and evaluates the mean squared error (MSE) using cross-validation.

.. code-block:: python

   import torch_bsf
   from torch_bsf.model_selection.degree_selection import select_degree

   # Your data
   ts = ... # parameters
   xs = ... # values

   # Select the best degree between 1 and 5
   best_d = select_degree(
       params=ts, 
       values=xs, 
       min_degree=1, 
       max_degree=5, 
       num_folds=5, 
       max_epochs=2,
       accelerator="auto" # use GPU if available
   )

   # Use the best degree for final fitting
   bs = torch_bsf.fit(params=ts, values=xs, degree=best_d)

How It Works
------------

1.  **Iteration**: The tool trains Bézier simplices for degrees from ``min_degree`` to ``max_degree``.
2.  **K-Fold CV**: For each degree, it performs :math:`k`-fold cross-validation. The mean MSE across all validation folds is calculated.
3.  **Heuristic Search**: If the CV error increases for a degree compared to the previous one, the search stops early to save time.
4.  **Optimal Selection**: The degree with the lowest mean validation MSE is returned.

Configuration
-------------

The ``select_degree()`` function accepts any keyword arguments that the standard PyTorch Lightning `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_ supports. 

Important parameters include:

*   ``min_degree`` / ``max_degree``: Range of degrees to check.
*   ``num_folds``: Number of cross-validation folds.
*   ``max_epochs``: Number of epochs to train for each fold (smaller is faster, but might be less accurate for selection).
*   ``accelerator`` / ``devices``: Hardware settings for training.

When to Use
-----------

1.  **Exploratory Data Analysis**: When you are unsure about the underlying manifold's complexity.
2.  **Automated Processes**: If you are building an automated pipeline for Pareto front approximation.
3.  **Model Optimization**: When accuracy is more important than training time, and you want the most precisely tuned model.

Command-Line Interface
----------------------

``degree_selection`` can also be invoked directly from the command line as a Python module.

As a Python Module (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the module directly to search for the best degree and print it to *stdout*:

.. code-block:: bash

   python -m torch_bsf.model_selection.degree_selection \
       --params params.csv \
       --values values.csv \
       --min_degree 1 \
       --max_degree 5 \
       --num_folds 5 \
       --max_epochs 2 \
       --accelerator auto

All options except ``--params`` and ``--values`` are optional and fall back to their
defaults (``--min_degree 1``, ``--max_degree 5``, ``--num_folds 5``,
``--max_epochs 2``, ``--accelerator auto``).

Example Output
~~~~~~~~~~~~~~

With the default ``--loglevel INFO``, the command prints per-degree progress to
*stderr* and writes the selected degree to *stdout*:

.. code-block:: text

   INFO:torch_bsf.model_selection.degree_selection:Checking degree 1...
   INFO:torch_bsf.model_selection.degree_selection:Degree 1: Mean MSE = 0.012345
   INFO:torch_bsf.model_selection.degree_selection:Checking degree 2...
   INFO:torch_bsf.model_selection.degree_selection:Degree 2: Mean MSE = 0.007891
   INFO:torch_bsf.model_selection.degree_selection:Checking degree 3...
   INFO:torch_bsf.model_selection.degree_selection:Degree 3: Mean MSE = 0.009012
   INFO:torch_bsf.model_selection.degree_selection:MSE increased at degree 3, stopping.
   Best degree: 2

The final line ``Best degree: <N>`` is the only output written to *stdout*.
Pass ``--loglevel WARNING`` to suppress the per-degree progress messages.

Available Options
~~~~~~~~~~~~~~~~~

Run with ``--help`` to see all available options:

.. code-block:: text

   usage: python -m torch_bsf.model_selection.degree_selection [-h]
                                                                --params PARAMS
                                                                --values VALUES
                                                                [--header HEADER]
                                                                [--min_degree MIN_DEGREE]
                                                                [--max_degree MAX_DEGREE]
                                                                [--num_folds NUM_FOLDS]
                                                                [--batch_size BATCH_SIZE]
                                                                [--max_epochs MAX_EPOCHS]
                                                                [--accelerator ACCELERATOR]
                                                                [--devices DEVICES]
                                                                [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

   Automatic degree selection for Bezier simplex via k-fold cross-validation

   options:
     -h, --help            show this help message and exit
     --params PARAMS       Path to the input parameters CSV file
     --values VALUES       Path to the output values CSV file
     --header HEADER       Number of header rows to skip (default: 0)
     --min_degree MIN_DEGREE
                           Minimum degree to search (default: 1)
     --max_degree MAX_DEGREE
                           Maximum degree to search (default: 5)
     --num_folds NUM_FOLDS
                           Number of cross-validation folds (default: 5)
     --batch_size BATCH_SIZE
                           Batch size for training (default: full-batch)
     --max_epochs MAX_EPOCHS
                           Training epochs per fold (default: 2)
     --accelerator ACCELERATOR
                           Accelerator type (default: auto)
     --devices DEVICES     Devices to use, integer or 'auto' (default: auto)
     --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                           Python logging level for degree selection progress
                           (default: INFO)

Via MLproject
~~~~~~~~~~~~~

The ``degree_selection`` entry point in ``MLproject`` calls the same module:

.. code-block:: bash

   mlflow run https://github.com/opthub-org/pytorch-bsf \
       -e degree_selection \
       -P params=params.csv \
       -P values=values.csv \
       -P min_degree=1 \
       -P max_degree=5 \
       -P num_folds=5 \
       -P max_epochs=2 \
       -P accelerator=auto \
       -P devices=auto

The best degree is printed to the run's stdout and can be reviewed in the MLflow UI.
The ``header``, ``accelerator``, and ``devices`` parameters are also exposed and
default to ``0``, ``"auto"``, and ``"auto"`` respectively.  The ``batch_size`` option
is not exposed through the MLproject entry point; the default full-batch loading is
used for all MLflow runs.

.. seealso::

   * :func:`torch_bsf.model_selection.degree_selection.select_degree` – Python API with
     full parameter descriptions.
   * :doc:`grid_sampling` – elastic net grid generation and k-fold cross-validation.
