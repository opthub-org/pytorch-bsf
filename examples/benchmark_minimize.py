"""Measure public minimize time and peak RSS in independent CPU processes.

Run from the checkout::

    PYTHONPATH=. python examples/benchmark_minimize.py --controls 16 64 256 --iterations 5 20 --repeats 3

RSS includes Python, imports, the model and optimizer temporaries. Each fresh
worker runs one optimizer; elapsed time excludes imports and input construction.
Compare JSON from two checkouts on the same machine. No timing threshold is used.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

from torch_bsf import BezierModel, BezierSpace, Points, PreconditionedSGD, SimploidalSet, minimize
from torch_bsf.results import ConvergenceWarning


def peak_rss_bytes() -> int:
    """OS high-water RSS, including native tensor storage."""
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def make_problem(n_controls: int) -> tuple[BezierModel, Points]:
    """A chain of degree-one cells and two output columns."""
    domain = SimploidalSet()
    cells = [domain.add_simplex(1) for _ in range(n_controls - 1)]
    for left, right in zip(cells[:-1], cells[1:], strict=True):
        domain.identify(left.face(0, 0), right.face(0, 1))
    space = BezierSpace.on_set(domain, degree=1)
    parameters = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64)
    points = Points.from_cells(space, [(cell, parameters) for cell in cells])
    return BezierModel(space, torch.zeros(space.n_controls, 2, dtype=torch.float64)), points


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Measure one run; separately count fixed-sample basis construction calls."""
    torch.set_num_threads(1)
    model, points = make_problem(args.controls[0])
    method = PreconditionedSGD(max_iter=args.iterations[0], preconditioner=args.worker, step_size=0.01)
    baseline = peak_rss_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        started = time.perf_counter()
        result = minimize(model, lambda y, p: (y - 1).square().sum(1), points=points, method=method, seed=7)
        elapsed = time.perf_counter() - started
        peak = peak_rss_bytes()
        # Profiling is outside timed/RSS measurements so patch overhead cannot
        # distort the measured path. Local basis blocks remain cell-local.
        with patch.object(BezierSpace, "basis", autospec=True, side_effect=BezierSpace.basis) as basis:
            minimize(model, lambda y, p: (y - 1).square().sum(1), points=points, method=method, seed=7)
        basis_calls = basis.call_count
    return {
        "preconditioner": args.worker,
        "controls": model.space.n_controls,
        "iterations": result.n_iterations,
        "samples": len(points),
        "outputs": model.n_outputs,
        "dtype": "float64",
        "device": "cpu",
        "seconds": elapsed,
        "before_minimize_peak_rss_bytes": baseline,
        "process_peak_rss_bytes": peak,
        "basis_constructions": basis_calls,
        "objective": result.objective,
        "status": result.status,
        "threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }


def main() -> None:
    """Run every size/iteration/preconditioner combination in fresh workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--iterations", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("minimize-benchmark.json"))
    parser.add_argument("--worker", choices=["none", "gram"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.controls) < 2 or min(args.iterations) < 1 or args.repeats < 1:
        parser.error("controls must be >=2; iterations and repeats must be positive")
    if args.worker:
        print(json.dumps(measure(args)))
        return
    runs = []
    for controls in args.controls:
        for iterations in args.iterations:
            for preconditioner in ("none", "gram"):
                for trial in range(args.repeats):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        preconditioner,
                        "--controls",
                        str(controls),
                        "--iterations",
                        str(iterations),
                    ]
                    completed = subprocess.run(command, check=True, capture_output=True, text=True)
                    run = json.loads(completed.stdout)
                    run["trial"] = trial
                    runs.append(run)
                    print(
                        f"{controls} controls / {iterations} updates / {preconditioner}: {run['seconds']:.3f}s, "
                        f"peak RSS {run['process_peak_rss_bytes'] / 2**20:.1f} MiB",
                        flush=True,
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "runs": runs}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
