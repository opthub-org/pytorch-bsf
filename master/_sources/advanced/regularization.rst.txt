Regularization and numerical evidence
=====================================

The regression objective in original output units is

.. math::

   \frac{\sum_i w_i\|B(t_i;P)-y_i\|^2}{\sum_i w_i}
   +\lambda\|P\|^2+\mu\|DP\|^2.

``D`` is the shared coefficient-net incidence operator. Fixed rows are eliminated
exactly, including their contribution to penalty right-hand sides::

    from torch_bsf import FixedControls, QuadraticPenalty, fit
    from torch_bsf.solvers import DenseLeastSquares
    result = fit(space, samples,
                 fixed=FixedControls({0: [1., 2.]}),
                 penalty=QuadraticPenalty(ridge=.01, smoothness=.02),
                 solver=DenseLeastSquares(diagnostics="full"))

Dense SVD and sparse LSMR solve the same augmented problem without forming
normal equations. Both return a CPU float64 model. Replicating every sample or
scaling every observation weight equally leaves the objective unchanged.

``result.diagnostics.data`` describes identifiability from observations alone;
``system`` describes the regularized system. Ridge can stabilize the latter
without making the former identifiable. Basic diagnostics leave expensive
extra data-only SVD fields as ``None``. Full dense diagnostics compute them.
Sparse estimates never pretend to be exact SVD ranks or condition numbers.

``result.success`` concerns solver termination, not goodness of fit or
identifiability. LSMR reports stop codes and iterations for each output. Failed
convergence emits ``ConvergenceWarning`` and stays visible in the result.
Use ``result.require_success()`` when failure should raise.
