Solvers and optional execution
==============================

Use typed solver settings in core regression::

    from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares
    result = fit(space, samples, solver=SparseLeastSquares(atol=1e-10, btol=1e-10))

SciPy is imported only when a sparse solve executes. The sparse backend
assembles local basis blocks and penalty entries directly into CSR without a
global dense design matrix. Dense solves use augmented SVD, not normal equations.

Choosing a solver
-----------------

Start with ``DenseLeastSquares()`` for small and medium problems. It computes
a direct SVD solution with rank and singular values. ``SparseLeastSquares`` is
useful when there are many cells and each observation touches only a few of the
global controls. Install ``pytorch-bsf[sparse]`` to execute it. Constructing the
configuration object requires no SciPy import.

Switching solvers preserves the Bezier space: domain, degrees, shared controls,
fixed values and the mathematical objective all stay the same. It changes how
the linear problem is stored and solved. Sparse predictions approach the dense
solution as LSMR converges; the tolerance and stopping budget can affect the
approximation. The comparison in :doc:`regularization` uses both backends and
checks their predictions before drawing each curve.

For an augmented matrix with ``R`` rows, ``C`` free columns and ``nnz`` nonzero
entries, dense storage grows as ``R * C``. Sparse CSR stores nonzero entries and
indices, so it helps when ``nnz`` is much smaller. It is assembled directly from
local Bernstein blocks and penalty entries, summing contributions that refer
to the same shared row. A single large simploid can have a generally dense
observation matrix and need not benefit from the sparse backend.

LSMR settings and termination
-----------------------------

``atol`` and ``btol`` are backward-error tolerances, not requested digits in the
control points or prediction MSE. For the augmented matrix ``A``, right-hand
side ``b`` and residual ``r=b-Ax``, compatible-system termination tests
``norm(r) <= atol*norm(A)*norm(x) + btol*norm(b)``; least-squares termination
tests ``norm(A.T@r) <= atol*norm(A)*norm(r)``. Set tolerances in light of input
accuracy; tighter tolerances can require more iterations without improving
statistical prediction accuracy.

``conlim`` stops on an estimated condition-number limit; zero disables that
limit. Raising it permits more ill-conditioned iterations, which does not
repair missing data. ``max_iter`` is a per-output budget; ``None`` uses LSMR's
default, the smaller augmented matrix dimension. Increase the budget after
examining why an earlier run stopped. There is no automatic dense fallback.

These settings follow the
`SciPy LSMR specification <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html>`_.
Ridge is represented by explicit augmented rows shared with the dense solver;
an additional LSMR damping term is not applied.

.. list-table:: LSMR termination reported for every output column
   :header-rows: 1
   :widths: 15 60 25

   * - Stop code
     - Meaning
     - Output success
   * - 0
     - Zero is a solution
     - True
   * - 1 / 2
     - Compatible / least-squares backward-error tolerance reached
     - True
   * - 4 / 5
     - Corresponding machine-precision tolerance reached
     - True
   * - 3 / 6
     - Configured / machine-precision condition limit reached
     - False
   * - 7
     - Iteration budget exhausted
     - False

Non-finite output or diagnostic quantities override any apparent success.
Inspect ``diagnostics.output_success``, ``output_statuses``, ``stop_codes``,
``iterations``, ``residual_norms``, ``normal_residual_norms`` and
``condition_estimates``. These residuals refer to the *augmented* problem, not
the unpenalized observation error. ``result.success`` is true only if every
output succeeds; ``require_success()`` raises with the result attached otherwise.

Sparse condition numbers are estimates. Rank and singular values remain
``None``; requesting a sparse fit never densifies the matrix for diagnostics.
With zero initial iterate, an unregularized rank-deficient or underdetermined
problem is approached in the minimum-norm free-coefficient subspace. Compare
predictions and objective values first: finite-precision stopping can leave
weakly determined coefficients different from a truncated-SVD solution.

.. testcode::
    :skipif: __import__("importlib.util").util.find_spec("scipy") is None

    import torch
    from torch_bsf import BezierSpace, Samples, fit
    from torch_bsf.solvers import SparseLeastSquares

    space = BezierSpace.simplex(1, degree=1)
    data = Samples(torch.eye(2), [[1., 2.], [3., 4.]])
    result = fit(space, data, solver=SparseLeastSquares(atol=1e-10, btol=1e-10))
    result.require_success()
    assert result.diagnostics.driver == "lsmr"
    assert result.diagnostics.output_success == (True, True)
    assert result.diagnostics.system.rank is None

Measured scaling on a connected cell chain
------------------------------------------

``examples/benchmark_least_squares.py`` measures each backend in a fresh process,
including basis/penalty assembly and solving in elapsed fit time. Inputs and
imports precede the timer. Peak RSS is the OS process high-water resident set
through the solve, including Python, imports and inputs; it is not just matrix
storage, GPU memory or Python ``tracemalloc`` allocations. The script captures
peak RSS before inspecting ``nnz``. Each process runs one fit with one CPU thread.

.. code-block:: bash

    pip install "pytorch-bsf[sparse]" psutil
    python examples/benchmark_least_squares.py --cells 16 64 256 --repeats 3 --output benchmark.json

The following medians are one Windows measurement with Python 3.12.6,
PyTorch 2.4.1, NumPy 2.1.1 and SciPy 1.14.1, an Intel Family 6 Model 191 CPU
(16 logical processors, one compute thread). All trials succeeded. The chain
has degree 3, two outputs and 20 observations per cell, with ridge ``1e-4``
and smoothness ``1e-3``. Sparse tolerances were ``atol=btol=1e-10`` and the
per-output iteration budget was 2000.

.. list-table:: Median of three fresh-process trials
   :header-rows: 1

   * - Cells
     - Observations
     - Controls
     - nnz
     - Dense seconds
     - Sparse seconds
     - Dense peak MiB
     - Sparse peak MiB
   * - 16
     - 320
     - 49
     - 1,329
     - 0.012
     - 0.364
     - 182.3
     - 193.3
   * - 64
     - 1,280
     - 193
     - 5,313
     - 0.040
     - 0.292
     - 193.5
     - 194.8
   * - 256
     - 5,120
     - 769
     - 21,249
     - 1.746
     - 0.383
     - 301.4
     - 199.4

The sparse backend saved memory and time at 256 cells in this run; the dense
backend was faster for the smaller cases. Startup costs and the machine affect
the crossover. This is evidence for this connected chain, not a general speedup
claim for all spaces. The complete per-trial records, objective values, stop
codes, matrix dimensions and machine metadata are available in
:download:`the raw JSON <../benchmarks/least_squares_windows.json>`.

Optional execution and recording
--------------------------------

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
