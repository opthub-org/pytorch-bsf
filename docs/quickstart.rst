Quickstart
==========

Create a polynomial space, observations, and an explicit fit result::

    import torch
    from torch_bsf import BezierSpace, Samples, fit

    space = BezierSpace.simplex(dimension=1, degree=2)
    t = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])
    result = fit(space, Samples(t, [[0.], [1.], [4.]]))
    result.require_success()
    predictions = result.model(t)
    print(result.metrics)

``BezierSpace`` contains immutable geometry and coefficient ordering.
``BezierModel`` contains the learned coefficients and is a PyTorch module.
``fit`` always returns ``FitResult``; predict through ``result.model``.
No Lightning, MLflow or Conda installation is required for this example.

The command line uses the same operations::

    python -m torch_bsf fit --params params.csv --values values.csv --degree 2 --save model.bsf
    python -m torch_bsf predict --model model.bsf --params params.csv --save predictions.csv
    python -m torch_bsf cross-validate --params params.csv --values values.csv --degree 2 --folds 3
    python -m torch_bsf select-degree --params params.csv --values values.csv --degree 1 --degrees 1 2 3 --folds 3

Outputs include metrics and numerical status. An unsuccessful numerical solve
returns a nonzero exit code. Tracking is opt-in with ``fit --track``.
Run ``bash examples/quickstart/run.sh`` for a complete local example.

Persistence uses one versioned, non-pickled format::

    from torch_bsf.io import save_model, load_model
    save_model("model.bsf", result.model)
    restored = load_model("model.bsf")

The extension does not choose a different format. CSV export is a separate
``export_control_points`` operation and does not preserve geometry.

Verified Python workflow
------------------------

The following example exercises fixed constraints, cross-validation, objective
minimization, and persistence using the same space and model types.

.. testcode::

    import tempfile
    from pathlib import Path
    import torch
    from torch_bsf import (
        BezierSpace, FixedControls, Samples, fit, minimize, PreconditionedSGD,
    )
    from torch_bsf.sampling import grid
    from torch_bsf.model_selection import KFold, cross_validate
    from torch_bsf.io import save_model, load_model

    space = BezierSpace.simplex(1, degree=2)
    points = grid(space, resolution=12)
    observations = Samples(points, points.parameters[:, :1] ** 2)
    fixed = FixedControls({space.control_index((2, 0)): [1.]})
    result = fit(space, observations, fixed=fixed)
    assert result.success and result.metrics["data_mse"] < 1e-24
    validation = cross_validate(space, observations, cv=KFold(3, seed=7))
    assert validation.success and validation.metrics["pooled_mse"] < 1e-24
    optimized = minimize(
        result.model, lambda y, p: (y[:, 0] - p.parameters[:, 0] ** 2).square(),
        points=points, method=PreconditionedSGD(gradient_tolerance=1e-8),
    )
    assert optimized.success
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "model.bsf"
        save_model(destination, optimized.model)
        restored = load_model(destination)
        assert torch.equal(restored(points), optimized.model(points))

To predict a saved multi-cell model, pass ``--cell-ids cells.csv`` with one
integer cell ID per row and pad parameter rows to the saved space width.

A one-column parameter CSV denotes a constant domain when dimensions are
inferred. The column is a dummy input and is ignored for both fitting and
prediction of zero-dimensional models.
