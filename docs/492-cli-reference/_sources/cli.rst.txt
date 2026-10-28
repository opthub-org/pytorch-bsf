Command-line reference
======================

Run ``python -m torch_bsf COMMAND`` in an environment with PyTorch-BSF installed.
``--help`` works at the top level and on every subcommand. Paths are relative to
the current working directory. The examples below use Python to create files,
so they work with either a POSIX shell or PowerShell.

.. contents:: On this page
   :local:
   :depth: 2

Regression input and shared options
-----------------------------------

``fit``, ``cross-validate`` and ``select-degree`` share these options. Every
subcommand also accepts ``-h`` / ``--help`` (print help and exit successfully).

.. list-table:: Regression options
   :header-rows: 1
   :widths: 27 18 55

   * - Option
     - Default
     - Meaning and constraints
   * - ``--params PATH``
     - Required
     - Numeric comma-separated parameter rows; one observation per row.
   * - ``--values PATH``
     - Required
     - Numeric output rows in the same order; at least one output column.
   * - ``--weights PATH``
     - Omitted (all ones)
     - One finite nonnegative weight per observation; total must be positive
       and finite. Use a single column. The loader flattens this file.
   * - ``--header INTEGER``
     - ``0``
     - Number of initial rows to skip in **all** supplied regression CSVs,
       including weights. It does not select columns by name.
   * - ``--dimensions D1,D2,...``
     - Inferred from columns
     - Positive simplex dimensions of product factors, with no spaces inside
       the comma-separated argument. Required width is ``sum(Di + 1)``.
   * - ``--degree INTEGER``
     - Required
     - Nonnegative polynomial degree, applied to every product factor.
   * - ``--solver {dense,sparse}``
     - ``dense``
     - CPU float64 SVD or optional SciPy LSMR. Install the ``sparse`` extra for
       LSMR. Tolerances and iteration budgets use solver defaults; the CLI
       does not expose them.
   * - ``--diagnostics {basic,full}``
     - ``basic``
     - ``full`` additionally diagnoses the weighted data matrix where needed.
       Only ``dense`` supports ``full``; the sparse/full combination is an error.
   * - ``--ridge FLOAT``
     - ``0``
     - Finite nonnegative coefficient penalty strength.
   * - ``--smoothness FLOAT``
     - ``0``
     - Finite nonnegative adjacent-control difference penalty strength.

Parameter and value tables must have equal, positive row counts. Values must
be finite. Each simplex factor uses nonnegative barycentric coordinates that
sum to one (validation permits small floating-point roundoff). Concatenate
factor blocks in the order of ``--dimensions``. For ``--dimensions 2,1``, the
columns are ``triangle_0,triangle_1,triangle_2,segment_0,segment_1``.

Without ``--dimensions``, a table with ``K > 1`` columns means a single
``(K-1)``-dimensional simplex. Four columns therefore do **not** automatically
describe a square: specify ``--dimensions 1,1`` and supply two independently
normalized coordinate pairs. A one-column file instead denotes a constant,
zero-dimensional domain; its dummy column is ignored. Regression CLI commands
construct single-cell spaces. Use Python to fit a glued multi-cell space, then
use :ref:`cli-predict` to evaluate its saved model.

Weights scale squared residuals, not the coordinates or responses. Zero weights
exclude observations from fitting metrics; they do not permit missing or
non-finite values. For validation, each train and test fold must retain positive
weight. See :doc:`advanced/regularization` for the objective and diagnostics.

.. testsetup:: cli

    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from torch_bsf import BezierModel, BezierSpace, Points
    from torch_bsf.io import load_model, save_model

    scratch = tempfile.TemporaryDirectory()
    previous_directory = Path.cwd()
    os.chdir(scratch.name)

    def cli(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "torch_bsf", *arguments],
            capture_output=True, text=True, check=False,
        )

fit
---

``fit`` solves weighted least squares. In addition to the shared options:

.. list-table:: Fit options
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--save PATH``
     - Required
     - Model archive destination, independent of its filename extension.
   * - ``--track``
     - Disabled
     - Log the result using the optional MLflow integration. Requires the
       ``mlflow`` extra; merely installing MLflow does not enable tracking.
   * - ``--tracking-uri URI``
     - MLflow configuration
     - Explicit tracking URI; requires ``--track``.

Header and weighted product example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this Python code in a fresh working directory. It writes a square's four
corners and center, a scalar response ``x + 2*y``, and matching weights. All
three training files have one header row:

.. testcode:: cli

    from pathlib import Path
    Path("params.csv").write_text(
        "u0,u1,v0,v1\n1,0,1,0\n0,1,1,0\n1,0,0,1\n0,1,0,1\n0.5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )
    Path("values.csv").write_text("response\n0\n1\n2\n3\n1.5\n", encoding="utf-8")
    Path("weights.csv").write_text("weight\n1\n1\n1\n1\n4\n", encoding="utf-8")
    Path("query.csv").write_text("0.75,0.25,0.5,0.5\n", encoding="utf-8")

.. code-block:: console

    python -m torch_bsf fit --params params.csv --values values.csv --weights weights.csv --header 1 --dimensions 1,1 --degree 1 --save square.bsf
    python -m torch_bsf predict --model square.bsf --params query.csv --save prediction.csv

The prediction file contains one numeric value, approximately ``1.25``.
Prediction input has **no header**; ``predict`` has no ``--header`` option.

.. testcode:: cli
    :hide:

    regression = ["--params", "params.csv", "--values", "values.csv",
                  "--weights", "weights.csv", "--header", "1",
                  "--dimensions", "1,1", "--degree", "1"]
    fitted = cli("fit", *regression, "--save", "square.bsf")
    assert fitted.returncode == 0, fitted.stderr
    report = json.loads(fitted.stdout)
    assert report["success"] and report["model_saved"]
    assert report["diagnostics"]["solution_available"]
    predicted = cli("predict", "--model", "square.bsf", "--params", "query.csv",
                    "--save", "prediction.csv")
    assert predicted.returncode == 0 and predicted.stdout == "", predicted.stderr
    np.testing.assert_allclose(np.loadtxt("prediction.csv"), 1.25, atol=1e-12)

cross-validate
--------------

``cross-validate`` uses the shared regression options and the following options:

.. list-table:: Validation options (also used by select-degree)
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--folds INTEGER``
     - ``5``
     - Number of shuffled folds, at least two and at most the observation count.
   * - ``--seed INTEGER``
     - ``0``
     - Local random seed for reproducible splits.

Using the files above:

.. code-block:: console

    python -m torch_bsf cross-validate --params params.csv --values values.csv --weights weights.csv --header 1 --dimensions 1,1 --degree 1 --folds 3 --seed 7

The JSON contains ``success``, aggregate ``metrics`` (``mean_mse``, ``std_mse``,
``pooled_mse``), and ``folds``. Each fold includes zero-based ``train`` and
``test`` row indices after header removal, held-out ``mse`` / ``mae`` metrics,
and solver ``status``. No model is saved, and there is no ``--save`` option.

select-degree
-------------

``select-degree`` adds required ``--degrees INTEGER [INTEGER ...]`` to the shared
regression and validation options. Candidates are nonnegative integers, each
applied to every product factor. ``--degree`` is still required to construct
the initial space; it does not restrict the candidate list.

.. code-block:: console

    python -m torch_bsf select-degree --params params.csv --values values.csv --weights weights.csv --header 1 --dimensions 1,1 --degree 1 --degrees 0 1 --folds 3 --seed 7 > selection.json

The JSON contains ``success``, ``best_degree`` and ``candidates`` (each with
``degree`` and validation ``metrics``). All candidates use identical splits.
The eligible candidate with smallest pooled held-out MSE wins; input order
breaks exact ties. Failed candidates remain visible but cannot win. If none
is eligible, ``best_degree`` is ``null`` and the exit code is 1.

Selection does not refit or save a model. Read ``best_degree`` from the report,
then explicitly fit that degree on all observations. This Python snippet runs
the refit without shell-specific JSON processing:

.. testcode:: cli
    :hide:

    validated = cli("cross-validate", *regression, "--folds", "3", "--seed", "7")
    assert validated.returncode == 0, validated.stderr
    validation = json.loads(validated.stdout)
    assert len(validation["folds"]) == 3 and validation["success"]
    selected = cli("select-degree", *regression, "--degrees", "0", "1",
                   "--folds", "3", "--seed", "7")
    assert selected.returncode == 0, selected.stderr
    Path("selection.json").write_text(selected.stdout, encoding="utf-8")

.. testcode:: cli

    import json
    import subprocess
    import sys
    selection = json.loads(Path("selection.json").read_text(encoding="utf-8"))
    if not selection["success"]:
        raise RuntimeError("No candidate passed validation")
    refit = subprocess.run([
        sys.executable, "-m", "torch_bsf", "fit",
        "--params", "params.csv", "--values", "values.csv",
        "--weights", "weights.csv", "--header", "1", "--dimensions", "1,1",
        "--degree", str(selection["best_degree"]), "--save", "selected.bsf",
    ], capture_output=True, text=True, check=True)

.. _cli-predict:

predict
-------

``predict`` reads the saved space and coefficients; it accepts none of the
shared regression options except ``--params``.

.. list-table:: Prediction options
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--model PATH``
     - Required
     - Saved model archive.
   * - ``--params PATH``
     - Required
     - Headerless numeric CSV of barycentric coordinates.
   * - ``--cell-ids PATH``
     - Omitted
     - Headerless single column of integer cell IDs, one per parameter row.
       Required for a multi-cell model; optional for a single-cell model.
   * - ``--save PATH``
     - Required
     - Headerless numeric CSV destination, one row per input and one column per
       model output. Row order is preserved; values use NumPy scientific notation.

Multi-cell prediction
~~~~~~~~~~~~~~~~~~~~~

Create a model in Python before invoking the CLI. This example uses a segment
and a triangle with constant outputs ``3`` and ``7``, respectively, so that
cell routing and padding can be checked independently of fitting:

.. testcode:: cli

    import numpy as np
    import torch
    from torch_bsf import BezierModel, BezierSpace, Points
    from torch_bsf.geometry import SimploidalSet
    from torch_bsf.io import save_model

    domain = SimploidalSet()
    segment = domain.add_simplex(1)
    triangle = domain.add_simplex(2)
    space = BezierSpace.on_set(domain, degree=1)
    coefficients = torch.full((space.n_controls, 1), 7., dtype=torch.float64)
    coefficients[list(space.cells[space.cell_id(segment)].controls)] = 3.
    model = BezierModel(space, coefficients)
    save_model("multi.bsf", model)
    points = Points.from_cells(space, [
        (segment, [[0.25, 0.75]]),
        (triangle, [[0.2, 0.3, 0.5]]),
    ])
    np.savetxt("multi-params.csv", points.parameters.numpy(), delimiter=",")
    np.savetxt("cells.csv", points.cell_ids.numpy(), fmt="%d", delimiter=",")

``multi-params.csv`` contains rows ``0.25,0.75,0`` and ``0.2,0.3,0.5``. The
segment row is padded to the saved space width, three columns. Only the active
coordinates for the selected cell are evaluated. Use zero padding for portable
files. Resolve IDs with ``space.cell_id(handle)`` or export ``Points.cell_ids``
as above: compiled IDs include faces and need not equal builder insertion order.

.. code-block:: console

    python -m torch_bsf predict --model multi.bsf --params multi-params.csv --cell-ids cells.csv --save multi-prediction.csv

This writes two rows, approximately ``3`` and ``7``. Omitting ``--cell-ids`` for this
model is an input error, even when all requested cells have the same width.

.. testcode:: cli
    :hide:

    predicted = cli("predict", "--model", "multi.bsf", "--params", "multi-params.csv",
                    "--cell-ids", "cells.csv", "--save", "multi-prediction.csv")
    assert predicted.returncode == 0 and predicted.stdout == "", predicted.stderr
    np.testing.assert_allclose(np.loadtxt("multi-prediction.csv"), [3., 7.], atol=1e-12)
    assert np.array_equal(points.parameters[0].numpy(), [0.25, 0.75, 0.])

minimize
--------

``minimize`` optimizes a copied initial model using simplex-weighted objectives.
It accepts the following options, and none of the shared regression options:

.. list-table:: Minimization options
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``--init PATH``
     - Required
     - Initial saved model.
   * - ``--objective MODULE:FUNCTION``
     - Required
     - Importable differentiable vector objective; see shape contract below.
   * - ``--save PATH``
     - Required
     - Destination for the returned model, including after iteration exhaustion.
   * - ``--max-iter INTEGER``
     - ``100``
     - Positive number of updates.
   * - ``--step-size FLOAT``
     - ``0.05``
     - Finite positive constant step size.
   * - ``--n-samples INTEGER``
     - ``100``
     - Positive number of fixed domain samples used for optimization.
   * - ``--seed INTEGER``
     - ``0``
     - Local seed for samples and update ordering.

Save an initial segment model and write the objective module into the working
directory using Python:

.. testcode:: cli

    from torch_bsf import BezierModel, BezierSpace
    from torch_bsf.io import save_model

    initial = BezierModel(BezierSpace.simplex(1, degree=1), [[0.], [1.]])
    save_model("initial.bsf", initial)
    Path("objectives.py").write_text(
        "import torch\n\n"
        "def costs(values):\n"
        "    x = values[:, 0]\n"
        "    return torch.stack((x.square(), (x - 1).square()), dim=1)\n",
        encoding="utf-8",
    )

.. code-block:: console

    python -m torch_bsf minimize --init initial.bsf --objective objectives:costs --save optimized.bsf --max-iter 20 --step-size 0.05 --n-samples 100 --seed 0

The module must be on Python's import path (the current directory works for
this command). ``costs`` receives predicted values with shape
``(n_samples, n_outputs)`` and returns differentiable finite objective values
with shape ``(n_samples, dimension + 1)`` for a positive-dimensional simplex
of that dimension.
Here one predicted scalar produces two competing costs, matching the segment's
two barycentric coordinates. The CLI wraps the function with
``linear_scalarization`` and minimizes ``sum_i coordinate_i * cost_i`` per row.
Keep operations in PyTorch to preserve gradients.

This callable takes **one** argument; the Python API's general
``loss(values, points)`` takes two. Ordinary product coordinates have one
normalization per factor and do not satisfy the scalarization's single-simplex
sum/shape requirement. Use the Python API with an explicit loss for those
domains or for custom stopping criteria.

Zero-dimensional models are supported by regression and prediction, but CLI
minimization does not support them. Their parameter width is zero: returning
one objective column fails the scalarization's shape check, and returning zero
columns fails its coordinate-sum check. Use the Python API with a general
``loss(values, points)`` for a constant domain.

The CLI sets an iteration budget without a gradient tolerance. Consequently
this example returns ``success: false``, ``status: "iteration_limit"``, exit
code 1, and a ``ConvergenceWarning`` on stderr, even if the objective decreased.
It still saves ``optimized.bsf``. Inspect the report before deciding to use it.

.. testcode:: cli
    :hide:

    optimized = cli("minimize", "--init", "initial.bsf", "--objective", "objectives:costs",
                    "--save", "optimized.bsf", "--max-iter", "20", "--step-size", "0.05",
                    "--n-samples", "100", "--seed", "0")
    assert optimized.returncode == 1, optimized.stderr
    optimization = json.loads(optimized.stdout)
    assert not optimization["success"] and optimization["status"] == "iteration_limit"
    assert "ConvergenceWarning" in optimized.stderr
    assert Path("optimized.bsf").is_file() and load_model("optimized.bsf").n_outputs == 1
    constant = BezierModel(BezierSpace.simplex(0, degree=0), [[0.]])
    save_model("constant.bsf", constant)
    Path("constant_objective.py").write_text(
        "def costs(values):\n    return values.square()\n", encoding="utf-8",
    )
    rejected = cli("minimize", "--init", "constant.bsf", "--objective", "constant_objective:costs",
                   "--save", "constant-optimized.bsf")
    assert rejected.returncode == 2 and rejected.stdout == ""
    assert "linear scalarization requires simplex coordinates" in rejected.stderr
    assert not Path("constant-optimized.bsf").exists()

Output, failures and saved files
--------------------------------

``fit``, ``cross-validate``, ``select-degree`` and ``minimize`` print one JSON
object on stdout after returning a result. Warnings go to stderr. ``predict``
writes its CSV and produces no stdout on success. Redirect stdout to capture
JSON independently of warnings. Help text is printed on stdout and exits 0.

For example, a successful ``fit`` report has the following structure (additional
diagnostic fields are omitted here; numeric values depend on the data):

.. code-block:: json

    {
      "success": true,
      "status": "solved",
      "message": "Least-squares optimum computed.",
      "model_saved": true,
      "metrics": {"data_mse": 0.0, "objective": 0.0},
      "diagnostics": {"solution_available": true}
    }

An exhausted ``minimize`` report is
``{"success": false, "status": "iteration_limit", "objective": 0.16}``,
with an illustrative objective value. A selection report has the form
``{"success": true, "best_degree": 1, "candidates": [...]}``.
JSON output uses Python's ``allow_nan=True``: unavailable or overflowing metrics
may appear as ``NaN``, ``Infinity`` or ``-Infinity``. These tokens are accepted
by Python's JSON reader but are outside strict JSON; consumers must handle them.

.. list-table:: Exit and persistence contracts
   :header-rows: 1
   :widths: 15 40 45

   * - Exit code
     - Meaning / output
     - File behavior
   * - ``0``
     - Successful result, prediction, or help.
     - ``fit`` and ``minimize`` save model archives; ``predict`` saves CSV.
       Validation and selection save no model.
   * - ``1``
     - A returned numerical result has ``success: false``; JSON is printed.
     - ``fit`` saves only if ``diagnostics.solution_available`` is true,
       reported as ``model_saved``. A finite but nonconverged solution or a
       diagnostic/metric failure may still be saved. An unavailable solution
       neither creates nor overwrites the destination. ``minimize`` saves its
       returned model on iteration exhaustion. Validation/selection save none.
   * - ``2``
     - Argument parsing or caught ``ValueError``, ``OSError`` or ``ImportError``;
       usage and ``error: ...`` on stderr, no result JSON.
     - Invalid inputs normally fail before saving. This is not an atomic
       transaction: a later write or tracking error may occur after an artifact
       was created. Existing files are not removed or rolled back.

These codes describe handled paths. Unexpected exceptions, including uncaught
``ArithmeticError`` in minimization, can produce a traceback and process failure
without JSON or a saved returned model. Do not interpret every exit 1 as a
structured numerical result. Check stdout and the report fields.

For a reproducible input error using the headered files above:

.. code-block:: console

    python -m torch_bsf fit --params params.csv --values values.csv --header 1 --dimensions 1,1 --degree 1 --solver sparse --diagnostics full --save invalid.bsf

This reports ``error: full SVD diagnostics require --solver=dense`` on stderr,
exits 2, prints no JSON and creates no ``invalid.bsf``. Likewise, headers in a
prediction CSV fail numeric loading; strip them before prediction.

.. testcode:: cli
    :hide:

    invalid = cli("fit", *regression, "--solver", "sparse", "--diagnostics", "full",
                  "--save", "invalid.bsf")
    assert invalid.returncode == 2 and invalid.stdout == ""
    assert "full SVD diagnostics require --solver=dense" in invalid.stderr
    assert not Path("invalid.bsf").exists()
.. testcleanup:: cli

    os.chdir(previous_directory)
    scratch.cleanup()
