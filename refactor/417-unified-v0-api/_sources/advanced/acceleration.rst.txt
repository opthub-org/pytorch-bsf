Solvers and optional execution
==============================

Use typed solver settings in core regression::

    from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares
    result = fit(space, samples, solver=SparseLeastSquares(atol=1e-10, btol=1e-10))

SciPy is imported only when a sparse solve executes. The sparse backend
assembles local basis blocks and penalty entries directly into CSR without a
global dense design matrix. Dense solves use augmented SVD, not normal equations.

Lightning is a separate opt-in execution adapter::

    from torch_bsf.integrations.lightning import regression_module
    module = regression_module(initial_model, samples, learning_rate=.01)
    # Trainer is owned by the application; use one device and a one-item loader.
    trainer.fit(module, train_dataloaders=loader)
    trained_model = module.bezier_model()

The adapter uses the same complete weighted penalized regression objective.
Trainer precision, scheduling and device settings do not leak into ``fit``.
The adapter rejects distributed execution rather than silently changing weights.

MLflow recording is explicit::

    from torch_bsf.integrations.mlflow import log_result
    run_id = log_result(result, tracking_uri="sqlite:///experiments.db")

Metrics, diagnostics and the versioned model artifact are recorded. Importing
or executing core fitting never enables autologging or starts an experiment.
