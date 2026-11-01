"""Screen recurrence bounds modulo a prime before exact held-out fitting.

A full-rank matrix modulo a prime excludes a rational recurrence in that
particular polynomial/order space, provided every denominator is invertible.
It says nothing about recurrences beyond the searched bounds. Every accepted
exact candidate is still only a conjecture until independently proved.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any

from .exact import peak_rss_bytes, record_fraction


def read_experiment(path: Path, *, rising: int = 0, central_binomial: bool = False) -> dict[int, Fraction]:
    """Load exact records, rejecting conflicting values and invalid indices."""
    if type(rising) is not int or rising < 0:
        raise ValueError("rising normalization must be a nonnegative integer")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema") != "skeleton-risk-integer-experiment-v1"
        or not isinstance(document.get("records"), list)
    ):
        raise ValueError("expected an integer experiment manifest with a records list")
    sequence: dict[int, Fraction] = {}
    for record in document["records"]:
        if not isinstance(record, dict) or "degree" not in record:
            raise ValueError("each exact record must contain a degree")
        index = record["degree"]
        if type(index) is not int or index < 1:
            raise ValueError("degree indices must be positive integers")
        try:
            value = record_fraction(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed exact-data record: {error}") from error
        value *= math.prod(range(index, index + rising))
        if central_binomial:
            value *= math.comb(2 * index, index)
        if index in sequence and sequence[index] != value:
            raise ValueError(f"conflicting exact values at degree {index}")
        sequence[index] = value
    if not sequence:
        raise ValueError("need nonempty exact data")
    return sequence


def modular_attempt(
    sequence: Mapping[int, Fraction], *, order: int, degree: int, train_through: int, prime: int
) -> dict[str, Any]:
    """Certify full rank over a prime field or identify a possible nullspace."""
    flint = import_module("flint")
    if not flint.fmpz(prime).is_prime():
        raise ValueError("modulus must be prime")
    if order < 1 or degree < 0:
        raise ValueError("order must be positive and polynomial degree nonnegative")
    training = [
        index
        for index in sorted(sequence)
        if index + order <= train_through and all(index + j in sequence for j in range(order + 1))
    ]
    unknowns = (order + 1) * (degree + 1)
    result: dict[str, Any] = {
        "order": order,
        "polynomial_degree": degree,
        "unknowns": unknowns,
        "training_windows": len(training),
    }
    if len(training) < unknowns - 1:
        return {**result, "status": "insufficient_data"}
    values = {index: value for index, value in sequence.items() if index <= train_through}
    if any(value.denominator % prime == 0 for value in values.values()):
        return {**result, "status": "noninvertible_denominator"}
    residues = {index: value.numerator % prime * pow(value.denominator, -1, prime) % prime for index, value in values.items()}
    matrix = flint.nmod_mat(
        [
            [residues[index + j] * pow(index, k, prime) % prime for j in range(order + 1) for k in range(degree + 1)]
            for index in training
        ],
        prime,
    )
    rank = matrix.rank()
    return {**result, "rank_mod_prime": rank, "status": "excluded_by_full_rank" if rank == unknowns else "possible_nullspace"}


def _worker(arguments: argparse.Namespace) -> None:
    from .recurrences import guess_recurrence

    source = Path(__file__).read_bytes()
    input_sha256 = hashlib.sha256(arguments.input.read_bytes()).hexdigest()
    started, cpu_started = time.monotonic(), time.process_time()
    sequence = read_experiment(arguments.input, rising=arguments.rising, central_binomial=arguments.central_binomial)
    if not any(index > arguments.train_through + 1 for index in sequence):
        raise ValueError("need complete held-out degrees after training")
    document: dict[str, Any] = {
        "schema": "skeleton-risk-modular-recurrence-scan-v1",
        "status": "no_candidate_in_searched_bounds",
        "input_sha256": input_sha256,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "train_through": arguments.train_through,
        "prime": arguments.prime,
        "normalization": {"rising": arguments.rising, "central_binomial": arguments.central_binomial},
        "scope": "exclusion applies only to each recorded order and degree; finite candidate checks are not a proof",
        "attempts": [],
        "candidates": [],
    }
    bounds = sorted(
        (
            (order, degree)
            for order in range(1, arguments.max_order + 1)
            for degree in range(arguments.max_degree + 1)
            if (order + 1) * (degree + 1) <= arguments.max_unknowns
        ),
        key=lambda pair: ((pair[0] + 1) * (pair[1] + 1), pair),
    )
    for order, degree in bounds:
        attempt = modular_attempt(
            sequence, order=order, degree=degree, train_through=arguments.train_through, prime=arguments.prime
        )
        document["attempts"].append(attempt)
        if attempt["status"] == "possible_nullspace":
            try:
                candidate = guess_recurrence(
                    sequence, order=order, polynomial_degree=degree, train_through=arguments.train_through
                )
            except ValueError as error:
                attempt["exact_fit_error"] = str(error)
            else:
                document["candidates"].append(candidate)
                if candidate["status"] == "conjecture":
                    document["status"] = "conjecture_found"
                    break
    document["excluded_spaces"] = sum(item["status"] == "excluded_by_full_rank" for item in document["attempts"])
    document["skipped_spaces"] = sum(
        item["status"] in {"insufficient_data", "noninvertible_denominator"} for item in document["attempts"]
    )
    if document["status"] != "conjecture_found" and document["skipped_spaces"]:
        document["status"] = "incomplete_search"
    document["elapsed_seconds"] = time.monotonic() - started
    document["cpu_seconds"] = time.process_time() - cpu_started
    document["peak_rss_bytes"] = peak_rss_bytes()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != document["source_sha256"]:
        raise RuntimeError("experiment source changed during evaluation; reject this run")
    if hashlib.sha256(arguments.input.read_bytes()).hexdigest() != input_sha256:
        raise RuntimeError("input data changed during evaluation; reject this run")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".source.py").write_bytes(source)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: document[key] for key in ("status", "elapsed_seconds", "cpu_seconds")}))


def main() -> None:
    """Run one prime-field screen and exact fitting under a hard timeout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-through", type=int, required=True)
    parser.add_argument("--max-order", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--max-unknowns", type=int, default=185)
    parser.add_argument("--prime", type=int, default=1000000007)
    parser.add_argument("--rising", type=int, default=0)
    parser.add_argument("--central-binomial", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(arguments)
        return
    command = [sys.executable, "-m", "tools.risk_research.research_scan", *sys.argv[1:], "--worker"]
    try:
        subprocess.run(command, check=True, timeout=arguments.timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"recurrence scan exceeded {arguments.timeout_seconds}s", file=sys.stderr)
        raise SystemExit(124) from None


if __name__ == "__main__":
    main()
