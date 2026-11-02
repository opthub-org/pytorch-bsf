"""Enclose the proved finite-zeta expression for c5, C6 and q4 face constants.

See ``skeleton_risk_c5_closed_form.rst``. All arithmetic is rational;
neither symbolic algebra nor an optional numerical backend is required.
These intervals describe leading constants, not finite-degree risks.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

from .exact import atomic_json
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-c5-closed-form-v1"
PROOF = "docs/advanced/skeleton_risk_c5_closed_form.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
ZETA_COEFFICIENTS = ((2, -120), (3, -600), (4, 900))


def _integer(value: object, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _decimal(value: int) -> str:
    """Render arbitrary fraction endpoints without changing process limits."""
    if value == 0:
        return "0"
    sign, value = ("-", -value) if value < 0 else ("", value)
    chunks = []
    while value:
        value, chunk = divmod(value, 10**500)
        chunks.append(f"{chunk:0500d}" if value else str(chunk))
    return sign + "".join(reversed(chunks))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def zeta_enclosure(order: int, terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose zeta(order) with a proved positive Euler-transform tail.

    The tail is at most 2**(-terms)/(1-2**(1-order)). For fixed order,
    the complete-homogeneous recurrence takes O(terms) rational operations;
    integer bit lengths grow. No approximate zeta function is called.
    """
    _integer(order, 2, "order")
    _integer(terms, 1, "terms")
    homogeneous = [Fraction(1)] + [Fraction(0)] * (order - 1)
    partial = Fraction(0)
    for n in range(1, terms + 1):
        for k in range(1, order):
            homogeneous[k] += homogeneous[k - 1] / n
        partial += homogeneous[-1] / (n * 2**n)
    multiplier = Fraction(1, 1 - Fraction(1, 2 ** (order - 1)))
    lower = multiplier * partial
    return lower, lower + multiplier / 2**terms


def c5_enclosure(terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose 900*zeta(4)-600*zeta(3)-120*zeta(2) with signed endpoints."""
    lower, upper = Fraction(0), Fraction(0)
    for order, coefficient in ZETA_COEFFICIENTS:
        left, right = zeta_enclosure(order, terms)
        lower += coefficient * (left if coefficient > 0 else right)
        upper += coefficient * (right if coefficient > 0 else left)
    return max(Fraction(0), lower), upper


def q4_scale(face_size: int) -> Fraction:
    """Return A_(s,4)/c5 for any fixed source-face vertex count s>=2."""
    _integer(face_size, 2, "face_size")
    return Fraction(comb(face_size + 4, face_size) ** 2 * prod(range(face_size, face_size + 4)), 80 * factorial(face_size - 2))


def nonvertex_leading_constant_enclosure(face_size: int, terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose the q4 leading constant, without a finite-degree error claim."""
    scale = q4_scale(face_size)
    lower, upper = c5_enclosure(terms)
    return scale * lower, scale * upper


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: (root / name).read_bytes()
        for name in (
            "tools/risk_research/c5_closed_form.py",
            "tools/risk_research/exact.py",
            "tools/risk_research/saved_records.py",
            PROOF,
            "docs/advanced/skeleton_risk_vertex_general.rst",
            "docs/advanced/skeleton_risk_codimension_three_exact.rst",
        )
    }


def build_record(terms: int = 64, face_size: int = 2) -> dict[str, object]:
    """Record exact finite-expression intervals and their complete provenance."""
    sources = _sources()
    scale = q4_scale(face_size)
    lower, upper = c5_enclosure(terms)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_classical_closed_form",
        "proof": PROOF,
        "terms": terms,
        "face_size": face_size,
        "outside": 4,
        "zeta_coefficients": {str(order): coefficient for order, coefficient in ZETA_COEFFICIENTS},
        "c5": {"lower": _fraction(lower), "upper": _fraction(upper)},
        "C6": {"lower": _fraction(540 * lower), "upper": _fraction(540 * upper)},
        "q4_leading_constant": {"lower": _fraction(scale * lower), "upper": _fraction(scale * upper)},
        "q4_scale": _fraction(scale),
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": "c5 and fixed-s q4 leading constants only; no finite-degree remainder or joint limit",
    }
    if _sources() != sources:
        raise RuntimeError("a closed-form source changed during calculation")
    return result


def verify_record(record: object) -> None:
    """Restore saved nondefault parameters and verify the entire typed record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported c5 closed-form record")
    terms = _integer(record.get("terms"), 1, "terms")
    face_size = _integer(record.get("face_size"), 2, "face_size")
    if canonical_json(record) != canonical_json(build_record(terms, face_size)):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a source-bound leading-constant record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int)
    parser.add_argument("--face-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        if args.verify is not None:
            if any(value is not None for value in (args.terms, args.face_size, args.output)):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(args.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            print("verified c5 closed-form record")
            return 0
        if args.output is None:
            raise ValueError("generation requires --output")
        _protect_output(args.output, sources)
        record = build_record(64 if args.terms is None else args.terms, 2 if args.face_size is None else args.face_size)
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
