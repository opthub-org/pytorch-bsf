Quickstart
==========

PyTorch-BSF can be used in four ways depending on your workflow: as a **Docker container** (no installation required), as a zero-install **MLflow project** (great for one-off experiments with experiment tracking), as a **CLI module** (scriptable, no Python coding required), or as a **Python library** (for full programmatic control).
Pick the option that best fits your setup.


Run as a Docker Container
--------------------------

A pre-built image is available on GHCR, built on ``continuumio/miniconda3`` with PyTorch installed via the ``pytorch`` conda channel — providing Intel MKL as the BLAS backend.

**Prerequisites:** `Docker`_.

.. _Docker: https://docs.docker.com/get-docker/


Training
^^^^^^^^

Let's prepare sample parameters and values files for training:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:CreateFiles]
   :end-before: [TAG:CreateFiles_End]

.. warning::
   The parameters file and the values file must have the same number of lines.

Mount the current directory as ``/workspace`` inside the container and run training:

.. code-block:: bash

   docker run --rm \
     --user "$(id -u)":"$(id -g)" \
     -v "$(pwd)":/workspace \
     ghcr.io/opthub-org/pytorch-bsf \
     python -m torch_bsf \
     --params=params.csv \
     --values=values.csv \
     --degree=3

The trained model will be saved under ``mlruns/`` in the current directory.


Run as an MLflow Project
------------------------

MLflow can run PyTorch-BSF directly from its GitHub repository without a manual installation step. It automatically creates a conda environment from the project's ``environment.yml``, which uses the ``pytorch`` conda channel and provides Intel MKL as the BLAS backend. This makes it the easiest way to get started with experiment tracking.

.. _MLflow: https://www.mlflow.org/docs/latest/


Installation
^^^^^^^^^^^^

Install `Miniconda`_.
Then, install ``mlflow`` package from ``conda-forge`` channel:

.. code-block:: bash

   conda install -c conda-forge mlflow

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html


Training
^^^^^^^^

Let's prepare sample parameters and values files for training:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:CreateFiles]
   :end-before: [TAG:CreateFiles_End]

.. warning::
   The parameters file and the values file must have the same number of lines.

Now, you can fit a Bézier simplex model using the latest version of PyTorch-BSF directly from its GitHub repository:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:MLflowURLDefine]
   :end-before: [TAG:MLflowURLDefine_End]

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:RunMLflowTraining]
   :end-before: [TAG:RunMLflowTraining_End]

After the command finishes, the trained model will be saved in ``mlruns/0`` directory.
Note the **Run ID** automatically set to the command execution, as you will need it for prediction.


Prediction
^^^^^^^^^^

To make predictions, MLflow may use ``virtualenv`` and ``pyenv`` to create an isolated environment for the model. Please ensure it's available in your system.

First, find the **Run ID** (e.g., `47a7...`) from the previous training step.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:FetchLatestRunID]
   :end-before: [TAG:FetchLatestRunID_End]


Next, you can predict with the model and output the results to a specified file (in this example, `test_values.json`).

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:MakePrediction]
   :end-before: [TAG:MakePrediction_End]

See https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-models-predict for details.


Serve Prediction API
^^^^^^^^^^^^^^^^^^^^

You can also serve a Web API for prediction.

First, find the Run ID (e.g., `a1b2c3...`) set to the model training.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:FetchLatestRunID]
   :end-before: [TAG:FetchLatestRunID_End]


Then, start a prediction server using the Run ID.

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:ServeAPI]
   :end-before: [TAG:ServeAPI_End]


Now, you can request a prediction with HTTP POST method:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:PredictWithHTTPPost]
   :end-before: [TAG:PredictWithHTTPPost_End]


See https://mlflow.org/docs/latest/genai/serving/ for details.


Run as a Python Package
-----------------------

First, install the package (requires Python 3.10 or above):

.. code-block:: bash

   pip install pytorch-bsf

Then run ``torch_bsf`` as a CLI module:

.. literalinclude:: ../examples/quickstart/run.sh
   :language: bash
   :start-after: [TAG:RunPackageTraining]
   :end-before: [TAG:RunPackageTraining_End]


Run as Python Code
------------------

First, install the package (requires Python 3.10 or above):

.. code-block:: bash

   pip install pytorch-bsf

Train a model by calling ``fit()``, then call the model to predict.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf

   # Prepare training parameters
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

   # Train a model
   bs = torch_bsf.fit(params=ts, values=xs, degree=3)

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
