"""Exact fixed-vertex experiments using integer polynomial kernel products.

This is an implementation of the proved finite scalar sum, not a closed-form
conjecture or a new proof. Clearing denominators before Kronecker substitution
avoids rational arithmetic inside both the kernel and the final scalar sum.
"""

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IntegerKernel:
    """A rectangle of [x^h y^ell]F(x,y)^power with one denominator."""

    power: int
    maximum: int
    denominator: int
    coefficients: list[list[int]]

    def value(self, h: int, ell: int) -> Fraction:
        """Read one exact coefficient."""
        return Fraction(self.coefficients[h][ell], self.denominator)


def integer_kernel(power: int, maximum: int, *, backend: str = "flint") -> IntegerKernel:
    """Multiply F^power after clearing all relevant denominators.

    F_ij=binom(i+j,i)/((i+1)(j+1)). Terms beyond the requested rectangle
    cannot contribute to it. With L=lcm(1,...,maximum+1), L^2 F has integer
    coefficients. Each multiplication uses x=z^stride,y=z with
    stride=2*maximum+1, then extracts the requested rectangle. Nonnegative
    exponents ensure discarded terms can never contribute in later products.
    """
    if type(power) is not int or power < 0 or type(maximum) is not int or maximum < 0:
        raise ValueError("power and maximum must be nonnegative integers")
    if backend not in {"python", "flint"}:
        raise ValueError("backend must be python or flint")
    if power == 0:
        return IntegerKernel(0, maximum, 1, [[int(h == ell == 0) for ell in range(maximum + 1)] for h in range(maximum + 1)])
    denominator = math.lcm(*range(1, maximum + 2)) ** 2
    base = [
        [math.comb(h + ell, h) * (denominator // ((h + 1) * (ell + 1))) for ell in range(maximum + 1)]
        for h in range(maximum + 1)
    ]
    if backend == "flint":
        fmpz_poly = import_module("flint").fmpz_poly
        stride = 2 * maximum + 1
        packed = [0] * (maximum * stride + maximum + 1)
        for h, row in enumerate(base):
            packed[h * stride : h * stride + maximum + 1] = row
        polynomial = fmpz_poly(packed)
        coefficients = base
        for _ in range(power - 1):
            for h, row in enumerate(coefficients):
                packed[h * stride : h * stride + maximum + 1] = row
            product = fmpz_poly(packed) * polynomial
            coefficients = [[int(product[h * stride + ell]) for ell in range(maximum + 1)] for h in range(maximum + 1)]
    else:
        coefficients = [[int(h == ell == 0) for ell in range(maximum + 1)] for h in range(maximum + 1)]
        for _ in range(power):
            coefficients = [
                [
                    sum(coefficients[i][j] * base[h - i][ell - j] for i in range(h + 1) for j in range(ell + 1))
                    for ell in range(maximum + 1)
                ]
                for h in range(maximum + 1)
            ]
    return IntegerKernel(power, maximum, denominator**power, coefficients)


def scalar_coefficient(vertices: int, degree: int, skeleton: int, kernel: IntegerKernel) -> Fraction:
    """Evaluate the existing exact scalar sum using only integer inner sums."""
    if any(type(value) is not int for value in (vertices, degree, skeleton)):
        raise ValueError("case parameters must be integers")
    if vertices < 2 or degree < 1 or not 0 <= skeleton < vertices:
        raise ValueError("invalid vertex count, degree, or skeleton")
    size = skeleton + 1
    if degree <= skeleton:
        return Fraction(0)
    n = degree - size
    if kernel.power != vertices - size or kernel.maximum < n:
        raise ValueError("kernel does not cover this scalar sum")
    factorials = [math.factorial(index) for index in range(2 * degree + size)]
    total = 0
    for k in range(n + 1 if size >= 2 else 1):
        multiplicity = math.comb(k + size - 2, size - 2) if size >= 2 else 1
        radial = [
            (-1) ** h * math.comb(n - k, h) * math.comb(n + k + 3 * size - 1, h) * factorials[h] for h in range(n - k + 1)
        ]
        total += multiplicity * sum(
            radial[h]
            * radial[ell]
            * factorials[2 * degree + size - h - ell - 1]
            * kernel.coefficients[h][ell]
            * (1 if h == ell else 2)
            for h in range(n - k + 1)
            for ell in range(h + 1)
        )
    return Fraction(
        math.comb(vertices, size) ** 2 * total,
        math.factorial(size - 1) * math.prod(range(vertices, vertices + 2 * degree)) * kernel.denominator,
    )


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(arguments: argparse.Namespace) -> None:
    flint = import_module("flint")

    source = Path(__file__).read_bytes()
    started = time.monotonic()
    cpu_started = time.process_time()
    maximum = max(arguments.degrees) - arguments.skeleton - 1
    kernel_started = time.monotonic()
    kernel = integer_kernel(arguments.vertices - arguments.skeleton - 1, max(0, maximum))
    kernel_seconds = time.monotonic() - kernel_started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = arguments.output.with_suffix(".source.py")
    source_path.write_bytes(source)
    document: dict[str, Any] = {
        "schema": "skeleton-risk-integer-experiment-v1",
        "proof_status": "evaluation_of_proved_finite_sum_only",
        "algorithm": "integer_denominator_clearing_and_kronecker_polynomial_power",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_snapshot": source_path.name,
        "python": platform.python_version(),
        "python_flint": flint.__version__,
        "created_utc": datetime.now(UTC).isoformat(),
        "vertices": arguments.vertices,
        "skeleton": arguments.skeleton,
        "maximum_kernel_degree": max(0, maximum),
        "kernel_seconds": kernel_seconds,
        "records": [],
    }
    for degree in sorted(set(arguments.degrees)):
        case_started, case_cpu = time.monotonic(), time.process_time()
        value = scalar_coefficient(arguments.vertices, degree, arguments.skeleton, kernel)
        record = {
            "degree": degree,
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
            "elapsed_seconds": time.monotonic() - case_started,
            "cpu_seconds": time.process_time() - case_cpu,
        }
        document["records"].append(record)
        document["elapsed_seconds"] = time.monotonic() - started
        document["cpu_seconds"] = time.process_time() - cpu_started
        # ru_maxrss is bytes on macOS, KiB on other supported Unix systems.
        document["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        _write(arguments.output, document)
        print(
            json.dumps({"skeleton": arguments.skeleton, **{key: record[key] for key in ("degree", "elapsed_seconds")}}),
            flush=True,
        )
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != document["source_sha256"]:
        raise RuntimeError("experiment source changed during evaluation; reject this run")


def main() -> None:
    """Run a bounded child process and save exact records incrementally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=4)
    parser.add_argument("--skeleton", type=int, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(arguments)
        return
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    try:
        subprocess.run(command, check=True, timeout=arguments.timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"experiment exceeded {arguments.timeout_seconds}s; partial output, if any, is preserved", file=sys.stderr)
        raise SystemExit(124) from None


if __name__ == "__main__":
    main()
