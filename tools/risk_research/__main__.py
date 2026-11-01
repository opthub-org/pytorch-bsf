"""Run exact risk experiments and conjecture searches from a source checkout."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .exact import SCHEMA, Case, atomic_json, provenance, run_case, worker


def integer_list(text: str) -> list[int]:
    """Parse comma-separated integers or one inclusive first:last interval."""
    try:
        if ":" in text:
            first, last = (int(value) for value in text.split(":"))
            if last < first:
                raise ValueError("descending interval")
            result = list(range(first, last + 1))
        else:
            result = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers or an inclusive first:last interval") from error
    if not result:
        raise argparse.ArgumentTypeError("the integer list must be nonempty")
    return sorted(set(result))


def parser() -> argparse.ArgumentParser:
    """Build the research command-line parser."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate resumable exact simplex coefficient data")
    generate.add_argument("--vertices", type=int, required=True)
    generate.add_argument("--degrees", type=integer_list, required=True)
    generate.add_argument("--skeletons", type=integer_list, default=[0])
    generate.add_argument("--oracle", choices=("scalar", "gram"), default="scalar")
    generate.add_argument("--cache", type=Path, default=Path(".research/skeleton-risk/cache"))
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--timeout", type=float, default=60)
    generate.add_argument("--memory-mib", type=int)
    generate.add_argument("--role", choices=("exploration", "validation"), default="exploration")
    guess = commands.add_parser("guess", help="guess a recurrence and validate exclusively on unseen degrees")
    guess.add_argument("input", type=Path)
    guess.add_argument("--skeleton", type=int, required=True)
    guess.add_argument("--vertices", type=int)
    guess.add_argument("--order", type=int, required=True)
    guess.add_argument("--poly-degree", type=int, required=True)
    guess.add_argument("--train-through", type=int, required=True)
    guess.add_argument("--output", type=Path, required=True)
    compute = commands.add_parser("_worker", help=argparse.SUPPRESS)
    compute.add_argument("case")
    compute.add_argument("--memory-mib", type=int)
    compute.add_argument("--source-fingerprint")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run a bounded experiment or record a finitely tested conjecture."""
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "_worker":
            worker(Case(**json.loads(arguments.case)), arguments.memory_mib, arguments.source_fingerprint)
            return 0
        if arguments.command == "generate":
            cases = [
                Case(arguments.vertices, degree, skeleton, arguments.oracle)
                for degree in arguments.degrees
                for skeleton in arguments.skeletons
            ]
            metadata = provenance()
            records = []
            for case in cases:
                record = run_case(
                    case,
                    arguments.cache,
                    metadata,
                    timeout=arguments.timeout,
                    memory_mib=arguments.memory_mib,
                    role=arguments.role,
                )
                records.append(record)
                # Save after every case so an interrupted batch retains its results.
                atomic_json(arguments.output, {"schema": SCHEMA, "records": records})
                print(json.dumps(record, sort_keys=True), flush=True)
            return int(any(record["status"] != "exact" for record in records))
        from .recurrences import guess_recurrence, sequence_from_manifest

        input_fingerprint = hashlib.sha256(arguments.input.read_bytes()).hexdigest()
        vertices, sequence = sequence_from_manifest(arguments.input, arguments.skeleton, arguments.vertices)
        if hashlib.sha256(arguments.input.read_bytes()).hexdigest() != input_fingerprint:
            raise ValueError("input data changed while loading; retry the search")
        result = guess_recurrence(
            sequence,
            order=arguments.order,
            polynomial_degree=arguments.poly_degree,
            train_through=arguments.train_through,
        )
        result.update(
            {
                "vertices": vertices,
                "skeleton": arguments.skeleton,
                "input": str(arguments.input),
                "input_sha256": input_fingerprint,
                **provenance(),
            }
        )
        atomic_json(arguments.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(result["status"] == "rejected")
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"research error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
