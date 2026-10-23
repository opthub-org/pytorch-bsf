"""Compare dense SVD and sparse LSMR in fresh CPU processes.

Run from a checkout with the sparse extra (and psutil on Windows)::

    python examples/benchmark_least_squares.py --cells 16 64 256 --repeats 3 --output benchmark.json

RSS includes Python, imported libraries, input geometry and fit temporaries.
Each process runs one solve; timings exclude imports and input construction.
No speed or memory assertion is made: examine scaling on your own machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from torch_bsf import BezierSpace, QuadraticPenalty, Samples, SimploidalSet, fit
from torch_bsf._problem import LeastSquaresProblem
from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares


def peak_rss_bytes() -> int:
    """OS high-water resident set, including native tensor allocations."""
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def make_problem(n_cells: int, per_cell: int, degree: int) -> tuple[BezierSpace, Samples]:
    """A connected chain with deterministic interior samples and two outputs."""
    domain = SimploidalSet()
    cells = [domain.add_simplex(1) for _ in range(n_cells)]
    for left, right in zip(cells[:-1], cells[1:], strict=True):
        domain.identify(left.face(0, 0), right.face(0, 1))
    space = BezierSpace.on_set(domain, degree=degree)
    x = torch.linspace(0, 1, per_cell, dtype=torch.float64)
    t = torch.stack([1 - x, x], 1)
    samples = Samples.from_cells(
        space,
        [(cell, t, torch.stack([torch.sin((k + x) / n_cells * 6), (k + x) / n_cells], 1)) for k, cell in enumerate(cells)],
    )
    return space, samples


def measure(args: argparse.Namespace) -> dict[str, object]:
    import numpy
    import scipy

    torch.set_num_threads(1)
    space, samples = make_problem(args.cells[0], args.per_cell, args.degree)
    penalty = QuadraticPenalty(ridge=1e-4, smoothness=1e-3)
    solver = DenseLeastSquares() if args.worker == "dense" else SparseLeastSquares(atol=1e-10, btol=1e-10, max_iter=2000)
    baseline_peak = peak_rss_bytes()
    started = time.perf_counter()
    result = fit(space, samples, solver=solver, penalty=penalty)
    elapsed = time.perf_counter() - started
    peak = peak_rss_bytes()
    # Inspect sparsity after capturing peak memory and elapsed fit time.
    problem = LeastSquaresProblem(space, samples, penalty, None)
    matrix = problem.sparse()
    return {
        "backend": args.worker,
        "maximal_cells": args.cells[0],
        "addressable_cells": space.n_cells,
        "observations": len(samples),
        "controls": space.n_controls,
        "outputs": 2,
        "system_shape": list(problem.shape),
        "nnz": matrix.nnz,
        "dtype": "float64",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "seconds": elapsed,
        "process_peak_rss_bytes": peak,
        "before_fit_peak_rss_bytes": baseline_peak,
        "success": result.success,
        "objective": result.metrics["objective"],
        "iterations": result.diagnostics.iterations,
        "stop_codes": result.diagnostics.stop_codes,
        "penalty": {"ridge": penalty.ridge, "smoothness": penalty.smoothness},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--per-cell", type=int, default=20)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--worker", choices=["dense", "sparse"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(*args.cells, args.per_cell, args.degree, args.repeats) < 1:
        parser.error("counts and degree must be positive")
    if args.worker:
        print(json.dumps(measure(args)))
        return
    runs = []
    for cells in args.cells:
        for backend in ("dense", "sparse"):
            for trial in range(args.repeats):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    backend,
                    "--cells",
                    str(cells),
                    "--per-cell",
                    str(args.per_cell),
                    "--degree",
                    str(args.degree),
                ]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                run = json.loads(completed.stdout)
                run["trial"] = trial
                runs.append(run)
                print(
                    f"{cells} cells / {backend} / trial {trial}: {run['seconds']:.3f}s, "
                    f"peak RSS {run['process_peak_rss_bytes'] / 2**20:.1f} MiB",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    if not all(run["success"] for run in runs):
        raise SystemExit("One or more solvers did not converge; inspect the recorded status.")


if __name__ == "__main__":
    main()
