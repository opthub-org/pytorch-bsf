"""Certify coefficient-height bounds for the finite leading-constant compiler.

The bounds apply to symbolic stored rational results at every fixed index.
They exclude numerical MZV truncation, runtime and native temporary buffers.
"""

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

from .exact import atomic_json
from .leading_constant_mzv import DEFAULT_WORK_LIMIT, _fraction, _integer, leading_constant_expression
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-leading-constant-height-v1"
PROOF = "docs/advanced/skeleton_risk_leading_constant_height.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"


def height_bounds(index: int) -> dict[str, int]:
    """Return proved common-denominator and bit bounds without compiling words.

    All integers index >= 3 are allowed. The denominator base is index!,
    represented symbolically; no factorial or exponential-sized old bound
    is allocated here. Primitive bits refer to elementary rational arithmetic,
    not Python's internal implementation buffers.
    """
    _integer(index, 3, "constant index")
    q = index
    size = q + 1
    logarithm = q.bit_length()
    stored_bits = 5 * q * size * logarithm + 8 * size * (logarithm + 1) + 2
    return {
        "denominator_factorial_base": q,
        "denominator_power": 5 * size,
        "factorial_log2_upper": q * logarithm,
        "stored_numerator_or_denominator_bits": stored_bits,
        "primitive_integer_product_or_sum_bits": 2 * stored_bits + 1,
    }


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "tools/risk_research/leading_constant_height.py",
        "tools/risk_research/leading_constant_mzv.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        PROOF,
        "docs/advanced/skeleton_risk_leading_constant_mzv.rst",
        "docs/advanced/skeleton_risk_vertex_general.rst",
    )
    return {name: (root / name).read_bytes() for name in paths}


def build_record(index: int = 7, work_limit: int = DEFAULT_WORK_LIMIT) -> dict[str, object]:
    """Record all-index bounds and final coefficient observations separately."""
    sources = _sources()
    bounds = height_bounds(index)
    expression = leading_constant_expression(index, work_limit)
    maximum_numerator = max((abs(value.numerator).bit_length() for value in expression.values()), default=0)
    maximum_denominator = max((value.denominator.bit_length() for value in expression.values()), default=0)
    # Divisibility is checked prime by prime against the symbolic factorial
    # power, so there is no need to allocate (index!)**denominator_power.
    denominator_divisibility = True
    for value in expression.values():
        remainder = value.denominator
        for factor in range(2, index + 1):
            for _ in range(bounds["denominator_power"]):
                common = gcd(remainder, factor)
                remainder //= common
                if remainder == 1:
                    break
            if remainder == 1:
                break
        denominator_divisibility &= remainder == 1
    if (
        not denominator_divisibility
        or max(maximum_numerator, maximum_denominator) > bounds["stored_numerator_or_denominator_bits"]
    ):
        raise ArithmeticError("constructed coefficients violate the proved height bound")
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_common_denominator_and_symbolic_coefficient_bits",
        "proof": PROOF,
        "constant_index": index,
        "work_limit": work_limit,
        "bounds": bounds,
        "observed_final_coefficients": {
            "term_count": len(expression),
            "maximum_numerator_bits": maximum_numerator,
            "maximum_denominator_bits": maximum_denominator,
            "denominators_divide_factorial_power": denominator_divisibility,
        },
        "mzv_terms": [{"indices": list(word), "coefficient": _fraction(value)} for word, value in expression.items()],
        "observations_include_intermediates": False,
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": (
            "symbolic coefficient heights at every fixed index; unchanged exponential word count; "
            "no numerical MZV prefix, elapsed-time, RSS or native-internal-buffer bound"
        ),
    }
    if _sources() != sources:
        raise RuntimeError("a height-bound construction or proof source changed during calculation")
    return result


def verify_record(record: object) -> None:
    """Restore all construction options and compare the complete typed record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported leading-constant height record")
    index = _integer(record.get("constant_index"), 3, "constant index")
    limit = _integer(record.get("work_limit"), 1, "work limit")
    if canonical_json(record) != canonical_json(build_record(index, limit)):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace construction or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a source-bound coefficient-height record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant-index", type=int)
    parser.add_argument("--work-limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        if args.verify is not None:
            if any(value is not None for value in (args.constant_index, args.work_limit, args.output)):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(args.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            if _sources() != sources:
                raise RuntimeError("a height-bound construction or proof source changed during calculation")
            print("verified leading-constant height record")
            return 0
        if args.output is None:
            raise ValueError("generation requires --output")
        _protect_output(args.output, sources)
        record = build_record(
            7 if args.constant_index is None else args.constant_index,
            DEFAULT_WORK_LIMIT if args.work_limit is None else args.work_limit,
        )
        if _sources() != sources:
            raise RuntimeError("a height-bound construction or proof source changed during calculation")
        output = args.output.parent.resolve() / args.output.name
        _protect_output(output, sources)
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
