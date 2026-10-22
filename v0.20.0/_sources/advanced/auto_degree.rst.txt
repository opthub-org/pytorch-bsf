Automatic Degree Selection
==========================

Choosing the optimal degree for a Bézier simplex can be challenging. A degree that is too low may underfit and fail to capture the manifold's complexity, while a degree that is too high may overfit to noise. 

PyTorch-BSF provides an automated tool to select the best degree based on **k-fold cross-validation**.

How to Use
----------

The ``select_degree()`` function iterates through multiple degrees and evaluates the mean squared error (MSE) using cross-validation.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf
   from torch_bsf.model_selection.degree_selection import select_degree
   from torch_bsf.sampling import simplex_random

   # Your data
   ts = simplex_random(n_params=3, n_samples=60, seed=0)  # parameters
   xs = torch.sin(3 * ts[:, :2])  # values

   # Select the best degree between 1 and 5
   best_d = select_degree(
       params=ts,
       values=xs,
       min_degree=1,
       max_degree=5,
       num_folds=5,
       seed=0,  # shuffle the samples into the same folds every time
   )

   # Use the best degree for final fitting
   bs = torch_bsf.fit(params=ts, values=xs, degree=best_d)
   print(1 <= best_d <= 5)

.. testoutput::

   True

How It Works
------------

1.  **Iteration**: The tool fits Bézier simplices with ``torch_bsf.fit()`` for degrees from ``min_degree`` to ``max_degree``.
2.  **K-Fold CV**: For each degree, it performs :math:`k`-fold cross-validation: the samples are shuffled (unless ``shuffle=False``) and split into ``num_folds`` folds, a model is fitted to all folds but one, and its MSE is measured on the held-out fold. The mean MSE across all validation folds is calculated.
3.  **Heuristic Search**: If the CV error does not improve for ``patience`` consecutive degrees (``1`` by default), the search stops early to save time.
4.  **Optimal Selection**: The degree with the lowest mean validation MSE is returned.

Configuration
-------------

Important parameters include:

*   ``min_degree`` / ``max_degree``: Range of degrees to check.
*   ``num_folds``: Number of cross-validation folds (capped at the number of samples).
*   ``patience``: Number of consecutive degrees without improvement before stopping.
*   ``shuffle`` / ``seed``: Whether to shuffle the samples before splitting them into folds, and the random seed for shuffling (also passed to ``fit()``).

Any other keyword arguments are passed to ``torch_bsf.fit()``, e.g., ``smoothness_weight``, ``normalize`` or ``ridge``.
By default, each fold is fitted exactly by the ``"lstsq"`` solver.
To train with the Lightning trainer instead (which needs the ``lightning`` extra; see :doc:`../installation`), pass ``solver="trainer"`` together with any `Trainer arguments <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags>`_, such as ``max_epochs`` (smaller is faster, but might be less accurate for selection) and ``accelerator`` / ``devices``.

To select the degree of a Bézier simploid (see :doc:`simploidal`), pass its ``simploid_type``; every factor then gets the same degree.

.. code-block:: python

   best_d = select_degree(params=ts, values=xs, simploid_type=(2, 1))

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
       --seed 0

All options except ``--params`` and ``--values`` are optional and fall back to their
defaults (``--min_degree 1``, ``--max_degree 5``, ``--num_folds 5``, ``--patience 1``,
``--solver lstsq``).
Pass ``--simploid-type`` (e.g., ``--simploid-type 2,1``) to select the degree of a Bézier simploid.
The trainer options (``--batch_size``, ``--max_epochs``, ``--accelerator`` and ``--devices``)
are used only with ``--solver trainer``.

Example Output
~~~~~~~~~~~~~~

With the default ``--loglevel INFO``, the command prints per-degree progress to
*stderr* and writes the selected degree to *stdout*:

.. code-block:: text

   INFO:__main__:Checking degree 1...
   INFO:__main__:Degree 1: Mean MSE = 0.043644
   INFO:__main__:Checking degree 2...
   INFO:__main__:Degree 2: Mean MSE = 0.000454
   INFO:__main__:Checking degree 3...
   INFO:__main__:Degree 3: Mean MSE = 0.000177
   INFO:__main__:Checking degree 4...
   INFO:__main__:Degree 4: Mean MSE = 0.000115
   INFO:__main__:Checking degree 5...
   INFO:__main__:Degree 5: Mean MSE = 0.000278
   INFO:__main__:No improvement for 1 consecutive degree(s) at degree 5, stopping.
   Best degree: 4

The final line ``Best degree: <N>`` is the only output written to *stdout*.
Pass ``--loglevel WARNING`` to suppress the per-degree progress messages.

Available Options
~~~~~~~~~~~~~~~~~

Run with ``--help`` to see all available options:

.. code-block:: text

   usage: python -m torch_bsf.model_selection.degree_selection
          [-h] --params PARAMS --values VALUES [--header HEADER]
          [--simploid-type A1,...,An] [--min_degree MIN_DEGREE]
          [--max_degree MAX_DEGREE] [--num_folds NUM_FOLDS] [--patience PATIENCE]
          [--seed SEED] [--solver {lstsq,trainer}] [--batch_size BATCH_SIZE]
          [--max_epochs MAX_EPOCHS] [--accelerator ACCELERATOR]
          [--devices DEVICES] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

   Automatic degree selection for a Bezier simplex (or simploid) via k-fold
   cross-validation

   options:
     -h, --help            show this help message and exit
     --params PARAMS       Path to the input parameters CSV file
     --values VALUES       Path to the output values CSV file
     --header HEADER       Number of header rows to skip (default: 0)
     --simploid-type, --simploid_type A1,...,An
                           Type of the Bezier simploid (default: a Bezier simplex
                           of dimension n_params - 1)
     --min_degree MIN_DEGREE
                           Minimum degree to search (default: 1)
     --max_degree MAX_DEGREE
                           Maximum degree to search (default: 5)
     --num_folds NUM_FOLDS
                           Number of cross-validation folds (default: 5)
     --patience PATIENCE   Consecutive non-improving degrees before stopping
                           (default: 1)
     --seed SEED           Random seed for the folds and training (default: none)
     --solver {lstsq,trainer}
                           'lstsq' solves the least-squares problem exactly,
                           'trainer' trains with Lightning (default: lstsq)
     --batch_size BATCH_SIZE
                           Batch size for --solver=trainer (default: full-batch)
     --max_epochs MAX_EPOCHS
                           Training epochs per fold for --solver=trainer
                           (default: 2)
     --accelerator ACCELERATOR
                           Accelerator for --solver=trainer (default: auto)
     --devices DEVICES     Devices for --solver=trainer, integer or 'auto'
                           (default: auto)
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
       -P num_folds=5

The best degree is printed to the run's stdout and can be reviewed in the MLflow UI.
The ``simploid_type`` (default ``""``, i.e., a Bézier simplex), ``solver`` (default
``lstsq``), ``header`` (default ``0``), and, for ``solver=trainer``,
``max_epochs`` (default ``2``), ``accelerator`` and ``devices`` (default ``"auto"``)
parameters are also exposed.  The ``seed`` and ``batch_size`` options are not exposed through
the MLproject entry point; MLflow runs shuffle the folds at random and use full-batch loading.

.. seealso::

   * :func:`torch_bsf.model_selection.degree_selection.select_degree` – Python API with
     full parameter descriptions.
   * :doc:`grid_sampling` – elastic net grid generation and k-fold cross-validation.
