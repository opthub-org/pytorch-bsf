"""Enclose the proved finite-zeta expression for c6 and the C7 vertex constant.

See ``skeleton_risk_c6_closed_form.rst``. Exact rational intervals include
the signed product zeta(2)*zeta(3); no optional backend is required.
These are fixed-vertex leading constants, not finite-degree risks.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from .c5_closed_form import _fraction, _integer, zeta_enclosure
from .exact import atomic_json
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-c6-closed-form-v1"
PROOF = "docs/advanced/skeleton_risk_c6_closed_form.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
ZETA_COEFFICIENTS = ((2, 600), (3, 2520), (4, 13980), (5, -10080))
ZETA_PRODUCT_COEFFICIENTS = ((2, 3, -4320),)
MZV_COEFFICIENTS = (
    ((2, 1, 1, 1), -14400),
    ((2, 1, 2), -4320),
    ((2, 2, 1), -4320),
    ((2, 1, 1), 11280),
    ((2, 2), 3600),
    ((2, 1), 2520),
    ((2,), 600),
)


def c6_enclosure(terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose c6 by ordinary zeta values and their proved signed product."""
    _integer(terms, 1, "terms")
    values = {order: zeta_enclosure(order, terms) for order in range(2, 6)}
    lower, upper = Fraction(0), Fraction(0)
    for order, coefficient in ZETA_COEFFICIENTS:
        left, right = values[order]
        lower += coefficient * (left if coefficient > 0 else right)
        upper += coefficient * (right if coefficient > 0 else left)
    for first, second, coefficient in ZETA_PRODUCT_COEFFICIENTS:
        left = values[first][0] * values[second][0]
        right = values[first][1] * values[second][1]
        lower += coefficient * (left if coefficient > 0 else right)
        upper += coefficient * (right if coefficient > 0 else left)
    return max(Fraction(0), lower), upper


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: (root / name).read_bytes()
        for name in (
            "tools/risk_research/c6_closed_form.py",
            "tools/risk_research/c5_closed_form.py",
            "tools/risk_research/exact.py",
            "tools/risk_research/saved_records.py",
            PROOF,
            "docs/advanced/skeleton_risk_c5_closed_form.rst",
            "docs/advanced/skeleton_risk_vertex_general.rst",
        )
    }


def build_record(terms: int = 64) -> dict[str, object]:
    """Record c6/C7 intervals, finite coefficients and complete provenance."""
    sources = _sources()
    lower, upper = c6_enclosure(terms)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_classical_closed_form",
        "proof": PROOF,
        "terms": terms,
        "vertices": 7,
        "zeta_coefficients": {str(order): coefficient for order, coefficient in ZETA_COEFFICIENTS},
        "zeta_product_coefficients": {
            f"{first},{second}": coefficient for first, second, coefficient in ZETA_PRODUCT_COEFFICIENTS
        },
        "mzv_coefficients_descending": {",".join(map(str, indices)): coefficient for indices, coefficient in MZV_COEFFICIENTS},
        "c6": {"lower": _fraction(lower), "upper": _fraction(upper)},
        "C7": {"lower": _fraction(2205 * lower), "upper": _fraction(2205 * upper)},
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": "c6 and the fixed-seven-vertex leading constant only; no finite-degree remainder or joint limit",
    }
    if _sources() != sources:
        raise RuntimeError("a closed-form source changed during calculation")
    return result


def verify_record(record: object) -> None:
    """Restore saved terms and verify the entire strictly typed JSON record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported c6 closed-form record")
    terms = _integer(record.get("terms"), 1, "terms")
    if canonical_json(record) != canonical_json(build_record(terms)):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a source-bound leading-constant record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        if args.verify is not None:
            if any(value is not None for value in (args.terms, args.output)):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(args.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            print("verified c6 closed-form record")
            return 0
        if args.output is None:
            raise ValueError("generation requires --output")
        _protect_output(args.output, sources)
        record = build_record(64 if args.terms is None else args.terms)
        if _sources() != sources:
            raise RuntimeError("a closed-form source changed during calculation")
        output = args.output.parent.resolve() / args.output.name
        _protect_output(output, sources)
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
