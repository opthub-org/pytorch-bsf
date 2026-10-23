Explicit initial models
=======================

Least-squares regression has no initial-model argument: geometry, fixed
constraints and observations completely specify its problem.

Objective minimization accepts an explicit model::

    import torch
    from torch_bsf import BezierModel
    initial = BezierModel(space, torch.zeros(space.n_controls, n_outputs, dtype=torch.float64))

Coefficients have shape ``(space.n_controls, n_outputs)``. Model construction
copies the input tensor. Forward evaluation is differentiable with respect to
coefficients and barycentric parameters.
