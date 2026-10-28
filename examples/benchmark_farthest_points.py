"""Compare dense and chunked acquisition in fresh CPU processes.

Run from a checkout with PYTHONPATH=. (and psutil on Windows)::

    python examples/benchmark_farthest_points.py --candidates 10000 --observed 2000 --chunks 256 1024

Each worker reports acquisition time and process peak RSS, including imports
and inputs. There is no timing assertion; compare measurements on one machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from torch_bsf import Points
from torch_bsf.active_learning import farthest_points


def peak_rss_bytes() -> int:
    """Include native tensor allocations in the OS high-water measurement."""
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def dense_reference(candidates: Points, observed: Points, n: int) -> Points:
    """Original one-cell full distance matrix, for comparison only."""
    coordinates, existing = candidates.parameters, observed.parameters
    distances = torch.cdist(coordinates, existing).min(1).values
    selected: list[int] = []
    for _ in range(n):
        index = int(distances.argmax())
        selected.append(index)
        distances = torch.minimum(distances, (coordinates - coordinates[index]).norm(dim=1))
        distances[selected] = -1
    return candidates.take(torch.tensor(selected))


def measure(args: argparse.Namespace) -> dict[str, object]:
    """One measurement; worker zero selects the dense reference."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(459)
    candidates = Points(torch.rand(args.candidates, args.dimensions, generator=generator, dtype=torch.float64))
    observed = Points(torch.rand(args.observed, args.dimensions, generator=generator, dtype=torch.float64))
    baseline = peak_rss_bytes()
    start = time.perf_counter()
    if args.worker == 0:
        result = dense_reference(candidates, observed, args.select)
    else:
        result = farthest_points(candidates, observed, args.select, chunk_size=args.worker)
    seconds = time.perf_counter() - start
    return {
        "backend": "dense" if args.worker == 0 else "chunked",
        "candidates": args.candidates,
        "observed": args.observed,
        "coordinate_dimensions": args.dimensions,
        "cells": 1,
        "select": args.select,
        "chunk_size": args.worker or None,
        "seconds": seconds,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "before_acquisition_peak_rss_bytes": baseline,
        "selected_parameters": result.parameters.tolist(),
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=10000)
    parser.add_argument("--observed", type=int, default=2000)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--select", type=int, default=8)
    parser.add_argument("--chunks", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.candidates, args.observed, args.dimensions, args.select, args.repeats, *args.chunks) < 1:
        parser.error("counts, dimensions and chunk sizes must be positive")
    if args.select > args.candidates:
        parser.error("select must not exceed candidates")
    if args.worker is not None:
        print(json.dumps(measure(args)))
        return
    measurements = []
    for chunk in [0, *args.chunks]:
        for _ in range(args.repeats):
            command = [
                sys.executable,
                __file__,
                "--worker",
                str(chunk),
                "--candidates",
                str(args.candidates),
                "--observed",
                str(args.observed),
                "--dimensions",
                str(args.dimensions),
                "--select",
                str(args.select),
            ]
            worker = subprocess.run(command, check=True, capture_output=True, text=True)
            measurements.append(json.loads(worker.stdout))
    reference = measurements[0]["selected_parameters"]
    report = {
        "same_selection": all(row["selected_parameters"] == reference for row in measurements),
        "measurements": measurements,
    }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
