"""Enclose the proved classical-zeta formula for c7 and the C8 constant.

This is a fixed-eight-vertex leading constant, not a finite-degree risk.
Rational endpoints include signed products and repeated zeta factors.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from .c5_closed_form import _fraction, _integer, zeta_enclosure
from .exact import atomic_json
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-c7-closed-form-v1"
PROOF = "docs/advanced/skeleton_risk_c7_closed_form.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
ZETA_COEFFICIENTS = ((2, -3920), (3, -18900), (4, -75250), (5, -189000), (6, 211680))
ZETA_MONOMIAL_COEFFICIENTS = (
    ((2, 3), -78960),
    ((2, 4), 85680),
    ((3, 3), 50400),
    ((2, 2, 2), 5040),
)
MZV_COEFFICIENTS = (
    ((2,), -3920),
    ((2, 1), -18900),
    ((2, 1, 1), -62020),
    ((2, 1, 1, 1), -267960),
    ((2, 1, 1, 1, 1), 352800),
    ((2, 1, 1, 2), 100800),
    ((2, 1, 2), -78960),
    ((2, 1, 2, 1), 100800),
    ((2, 2), -17640),
    ((2, 2, 1), -78960),
    ((2, 2, 1, 1), 100800),
    ((2, 2, 2), 30240),
)


def c7_enclosure(terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose c7 with signed rational ordinary-zeta monomials."""
    _integer(terms, 1, "terms")
    values = {order: zeta_enclosure(order, terms) for order in range(2, 7)}
    lower, upper = Fraction(0), Fraction(0)
    for order, coefficient in ZETA_COEFFICIENTS:
        left, right = values[order]
        lower += coefficient * (left if coefficient > 0 else right)
        upper += coefficient * (right if coefficient > 0 else left)
    for factors, coefficient in ZETA_MONOMIAL_COEFFICIENTS:
        left = right = Fraction(1)
        for order in factors:
            left *= values[order][0]
            right *= values[order][1]
        lower += coefficient * (left if coefficient > 0 else right)
        upper += coefficient * (right if coefficient > 0 else left)
    return max(Fraction(0), lower), upper


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: (root / name).read_bytes()
        for name in (
            "tools/risk_research/c7_closed_form.py",
            "tools/risk_research/c5_closed_form.py",
            "tools/risk_research/leading_constant_mzv.py",
            "tools/risk_research/exact.py",
            "tools/risk_research/saved_records.py",
            PROOF,
            "docs/advanced/skeleton_risk_c5_closed_form.rst",
            "docs/advanced/skeleton_risk_leading_constant_mzv.rst",
            "docs/advanced/skeleton_risk_vertex_general.rst",
        )
    }


def build_record(terms: int = 64) -> dict[str, object]:
    """Record c7/C8 intervals, finite coefficients and source provenance."""
    sources = _sources()
    lower, upper = c7_enclosure(terms)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_classical_closed_form",
        "proof": PROOF,
        "terms": terms,
        "vertices": 8,
        "zeta_coefficients": {str(order): coefficient for order, coefficient in ZETA_COEFFICIENTS},
        "zeta_monomial_coefficients": {
            ",".join(map(str, factors)): coefficient for factors, coefficient in ZETA_MONOMIAL_COEFFICIENTS
        },
        "mzv_coefficients_descending": {",".join(map(str, word)): coefficient for word, coefficient in MZV_COEFFICIENTS},
        "c7": {"lower": _fraction(lower), "upper": _fraction(upper)},
        "C8": {"lower": _fraction(10080 * lower), "upper": _fraction(10080 * upper)},
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": "c7 and the fixed-eight-vertex leading constant only; no finite-degree remainder or joint limit",
    }
    if _sources() != sources:
        raise RuntimeError("a closed-form source changed during calculation")
    return result


def verify_record(record: object) -> None:
    """Restore saved terms and verify the complete strictly typed record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported c7 closed-form record")
    terms = _integer(record.get("terms"), 1, "terms")
    if canonical_json(record) != canonical_json(build_record(terms)):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a source-bound c7 leading-constant record."""
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
            if _sources() != sources:
                raise RuntimeError("a closed-form source changed during calculation")
            print("verified c7 closed-form record")
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
