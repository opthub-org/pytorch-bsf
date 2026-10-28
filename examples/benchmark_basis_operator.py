"""Measure cell-local basis assembly and products against the pre-grouping implementation.

Run ``python examples/benchmark_basis_operator.py --output basis_operator.json``.
Each case runs in a fresh process with one CPU thread. Peak RSS includes Python,
imports, inputs, warmups and profiling. Cumulative full-output copy bytes are profiler call counts times output size;
profiler-attributed net memory is recorded separately. Neither is peak RAM.
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from torch_bsf import BezierSpace, Points
from torch_bsf._kernels.combinatorics import bernstein_basis
from torch_bsf.core import BasisOperator, Cell


class LegacyOperator(BasisOperator):
    """The previous full-observation cell scans and full-output copy products."""

    def __init__(self, space: BezierSpace, points: Points):
        self.space, self.points = space, points
        self.shape = (len(points), space.n_controls)
        self.blocks = []
        for cid in torch.unique(points.cell_ids).tolist():
            cell = space.cells[cid]
            rows = torch.nonzero(points.cell_ids == cid).flatten()
            columns = torch.tensor(cell.controls, dtype=torch.long)
            basis = bernstein_basis(cell.dimensions, cell.degree, points.parameters[rows, : cell.width])
            self.blocks.append((rows, columns, basis))

    def matmat(self, coefficients: torch.Tensor) -> torch.Tensor:
        out = coefficients.new_zeros(self.shape[0], coefficients.shape[1])
        for rows, columns, basis in self.blocks:
            out = out.index_copy(0, rows, basis @ coefficients[columns])
        return out

    def rmatmat(self, values: torch.Tensor) -> torch.Tensor:
        out = values.new_zeros(self.shape[1], values.shape[1])
        for rows, columns, basis in self.blocks:
            out = out.index_add(0, columns, basis.T @ values[rows])
        return out


def _median_ms(function: Any, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def _worker(cells: int, rows: int, outputs: int, implementation: str, repeats: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    space = BezierSpace(tuple(Cell((1,), (1,), (2 * k, 2 * k + 1)) for k in range(cells)), 2 * cells, tuple(range(cells)), ())
    # Interleaving makes restoring original observation order necessary.
    points = Points(
        torch.tensor([[0.25, 0.75]], dtype=torch.float64).repeat(cells * rows, 1), torch.arange(cells).repeat(rows)
    )
    coefficients = torch.arange(space.n_controls * outputs, dtype=torch.float64).reshape(-1, outputs)
    values = torch.arange(len(points) * outputs, dtype=torch.float64).reshape(-1, outputs)
    factory = LegacyOperator if implementation == "legacy" else BasisOperator
    operator = factory(space, points)
    result: dict[str, Any] = dict(cells=cells, rows_per_cell=rows, outputs=outputs, implementation=implementation)
    with torch.no_grad():
        result["assembly_ms"] = _median_ms(lambda: factory(space, points), repeats)
        for name, operand in (("matmat", coefficients), ("rmatmat", values)):
            action = getattr(operator, name)
            action(operand)
            result[f"{name}_ms"] = _median_ms(lambda action=action, operand=operand: action(operand), repeats)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
                action(operand)
            events = [event for event in profile.key_averages() if event.key in ("aten::index_copy", "aten::index_add")]
            calls = sum(event.count for event in events)
            output_rows = operator.shape[0 if name == "matmat" else 1]
            result[f"{name}_full_copy_bytes"] = calls * output_rows * outputs * operand.element_size()
            result[f"{name}_profiler_copy_net_bytes"] = sum(event.cpu_memory_usage for event in events)
            result[f"{name}_full_copy_calls"] = calls
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--rows-per-cell", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--outputs", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("basis_operator.json"))
    parser.add_argument("--worker", nargs=4, metavar=("CELLS", "ROWS", "OUTPUTS", "IMPLEMENTATION"))
    args = parser.parse_args()
    if args.worker:
        cells, rows, outputs, implementation = args.worker
        print(json.dumps(_worker(int(cells), int(rows), int(outputs), implementation, args.repeats)))
        return
    results = []
    for cells in args.cells:
        for rows in args.rows_per_cell:
            for outputs in args.outputs:
                for implementation in ("legacy", "grouped"):
                    process = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            str(cells),
                            str(rows),
                            str(outputs),
                            implementation,
                            "--repeats",
                            str(args.repeats),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    results.append(json.loads(process.stdout))
    report = dict(
        environment=dict(
            python=sys.version,
            torch=torch.__version__,
            platform=platform.platform(),
            processor=platform.processor(),
            threads=1,
            device="cpu",
        ),
        repeats=args.repeats,
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
