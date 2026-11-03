"""Certify logarithmic finite-degree errors for fixed nonvertex q4 faces.

An explicit q4 discrete derivative proves O(log(D)/D) with all-tail
thresholds. Exact rational evaluation builds no degree-sized table;
uncertainty in the leading constant remains a separate error budget.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import prod
from pathlib import Path

from .nonvertex_q3_error import (
    _canonical,
    _decimal_integer,
    _integer,
    _target_argument,
    _unique,
    atomic_json,
    fraction_record,
)
from .nonvertex_q4_error import leading_constant_enclosure

SCHEMA = "skeleton-risk-nonvertex-q4-refined-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_nonvertex_q4_refined.rst"
DIAGONAL_MAJORANT = 5400
DIFFERENCE_MAJORANT = 179568


def bound_constants(face_size: int) -> tuple[Fraction, Fraction]:
    """Return K_s and B_s in the proved (K_s*(1+log(D))+B_s)/D bound."""
    _integer(face_size, 2, "face_size")
    logarithmic = Fraction(DIFFERENCE_MAJORANT) + Fraction(DIAGONAL_MAJORANT * (face_size - 2), 4)
    constant = 2 * face_size + 3 + Fraction(DIAGONAL_MAJORANT * (2 + (face_size - 1) * (face_size - 2)), 4)
    return logarithmic, constant


def averaging_error_bound(face_size: int, degree: int) -> Fraction:
    """Bound the summed q4 hypergeometric averaging errors at active degrees."""
    _integer(face_size, 2, "face_size")
    _integer(degree, 1, "degree")
    if degree < face_size:
        raise ValueError("averaging requires degree >= face_size")
    count = degree - face_size + 1
    harmonic = 1 + (count - 1).bit_length()
    return Fraction(4 * DIFFERENCE_MAJORANT, degree + 2 * face_size + 2) * (harmonic + Fraction(2, count + 1) - 2)


def dyadic_tail_bound(face_size: int, exponent: int) -> Fraction:
    """Bound the relative error at EVERY D>=2**exponent in the proved domain."""
    _integer(face_size, 2, "face_size")
    _integer(exponent, 1, "exponent")
    denominator = 1 << exponent
    if denominator < 2 * (face_size - 1):
        raise ValueError("dyadic degree must be >= 2*(face_size-1)")
    logarithmic, constant = bound_constants(face_size)
    return (logarithmic * (exponent + 1) + constant) / denominator


def relative_error_threshold(face_size: int, target: Fraction) -> int:
    """Return a sufficient degree for the target at every larger degree."""
    _integer(face_size, 2, "face_size")
    if type(target) is not Fraction or target <= 0:
        raise ValueError("target must be a positive Fraction")
    exponent = max(1, (2 * face_size - 3).bit_length())
    while dyadic_tail_bound(face_size, exponent) > target:
        exponent += 1
    return 1 << exponent


def relative_error_bound(face_size: int, degree: int) -> Fraction:
    """Bound abs(a/(A_(s,4)*D**(s-6))-1) without evaluating the risk."""
    _integer(face_size, 2, "face_size")
    _integer(degree, 1, "degree")
    if degree < face_size:
        return Fraction(1)
    count = degree - face_size + 1
    harmonic = 1 + (count - 1).bit_length()
    tail = Fraction(DIAGONAL_MAJORANT, count)
    angular = Fraction(DIAGONAL_MAJORANT * ((face_size - 2) * harmonic + (face_size - 1) * (face_size - 2)), degree)
    prefactor = Fraction((2 * degree) ** 4, prod(range(2 * degree + face_size, 2 * degree + face_size + 4)))
    bound = 1 - prefactor + (averaging_error_bound(face_size, degree) + tail + angular) / 4
    exponent = degree.bit_length() - 1
    if exponent < 1 or 2**exponent < 2 * (face_size - 1):
        return bound
    return min(bound, dyadic_tail_bound(face_size, exponent))


def coefficient_enclosure(face_size: int, degree: int, terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose the finite-D coefficient with separate constant truncation."""
    lower, upper = leading_constant_enclosure(face_size, terms)
    error = relative_error_bound(face_size, degree)
    if degree < face_size:
        return Fraction(0), Fraction(0)
    scale = Fraction(degree) ** (face_size - 6)
    return lower * max(Fraction(0), 1 - error) * scale, upper * (1 + error) * scale


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "tools/risk_research/nonvertex_q4_refined.py",
        "tools/risk_research/nonvertex_q4_error.py",
        "tools/risk_research/nonvertex_q3_error.py",
        "tools/risk_research/codimension_three_exact.py",
        "tools/risk_research/vertex_general.py",
        "tools/risk_research/vertex_general_asymptotic.py",
        "tools/risk_research/vertex_refined_asymptotic.py",
        "tools/risk_research/exact.py",
        PROOF,
        "docs/advanced/skeleton_risk_nonvertex_q4_error.rst",
        "docs/advanced/skeleton_risk_codimension_three_exact.rst",
        "docs/advanced/skeleton_risk_vertex_general.rst",
        "docs/advanced/skeleton_risk_vertex_general_asymptotics.rst",
        "docs/advanced/skeleton_risk_vertex_refined_asymptotics.rst",
    ]
    return {name: (root / name).read_bytes() for name in names}


def certificate(face_size: int, degree: int, terms: int = 64, target: Fraction | None = None) -> dict[str, object]:
    """Specialize the proved logarithmic bound with source-bound exact intervals."""
    raw_sources = _sources()
    leading = leading_constant_enclosure(face_size, terms)
    interval = coefficient_enclosure(face_size, degree, terms)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_logarithmic_finite_degree_error_and_all_tail_threshold",
        "proof": PROOF,
        "scope": "each fixed s>=2, q=4; no finite-sample or growing-s joint limit",
        "face_size": face_size,
        "vertices": face_size + 4,
        "outside": 4,
        "degree": degree,
        "active": degree >= face_size,
        "constant_terms": terms,
        "diagonal_majorant": DIAGONAL_MAJORANT,
        "difference_majorant": DIFFERENCE_MAJORANT,
        "leading_constant": {"lower": fraction_record(leading[0]), "upper": fraction_record(leading[1])},
        "coefficient_enclosure": {"lower": fraction_record(interval[0]), "upper": fraction_record(interval[1])},
        "relative_error_bound": fraction_record(relative_error_bound(face_size, degree)),
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in raw_sources.items()
        },
    }
    if target is not None:
        threshold = relative_error_threshold(face_size, target)
        report["relative_target"] = fraction_record(target)
        report["all_tail"] = {
            "from_degree": threshold,
            "relative_error_bound": fraction_record(dyadic_tail_bound(face_size, threshold.bit_length() - 1)),
            "requested_degree_in_tail": degree >= threshold,
        }
    if _sources() != raw_sources:
        raise RuntimeError("nonvertex q4 refined sources changed during calculation")
    return report


def _record_parameters(record: object) -> tuple[int, int, int, Fraction | None]:
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported nonvertex q4 refined certificate")
    face_size = _integer(record.get("face_size"), 2, "face_size")
    degree = _integer(record.get("degree"), 1, "degree")
    terms = _integer(record.get("constant_terms"), 1, "constant_terms")
    target = None
    if "relative_target" in record:
        value = record["relative_target"]
        if (
            type(value) is not dict
            or set(value) != {"numerator", "denominator"}
            or any(type(item) is not str for item in value.values())
        ):
            raise ValueError("invalid recorded relative target")
        target = Fraction(_decimal_integer(value["numerator"]), _decimal_integer(value["denominator"]))
        if target <= 0:
            raise ValueError("target must be positive")
    return face_size, degree, terms, target


def verify_certificate(record: object) -> None:
    """Regenerate the entire record using its own saved nondefault options."""
    parameters = _record_parameters(record)
    if _canonical(record) != _canonical(certificate(*parameters)):
        raise ValueError("saved certificate differs from source or exact recomputation")


def _protect_output(path: Path | None, sources: dict[str, bytes]) -> None:
    if path is not None:
        root = Path(__file__).resolve().parents[2]
        if path.resolve() in {(root / name).resolve() for name in sources}:
            raise ValueError("output must not overwrite evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Save or replay a complete source-bound q4 logarithmic certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=_decimal_integer)
    parser.add_argument("--degree", type=_decimal_integer)
    parser.add_argument("--constant-terms", type=_decimal_integer)
    parser.add_argument("--relative-target", type=_target_argument)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    sources = _sources()
    try:
        _protect_output(arguments.output, sources)
        if arguments.verify is not None:
            previous = json.loads(arguments.verify.read_bytes(), object_pairs_hook=_unique, parse_int=_decimal_integer)
            saved = _record_parameters(previous)
            supplied = (arguments.face_size, arguments.degree, arguments.constant_terms, arguments.relative_target)
            if any(value is not None and value != recorded for value, recorded in zip(supplied, saved, strict=True)):
                raise ValueError("generation override differs from the saved certificate")
            verify_certificate(previous)
            report = previous
        else:
            if arguments.face_size is None or arguments.degree is None or arguments.output is None:
                raise ValueError("generation requires --face-size, --degree and --output")
            terms = 64 if arguments.constant_terms is None else arguments.constant_terms
            report = certificate(arguments.face_size, arguments.degree, terms, arguments.relative_target)
        if _sources() != sources:
            raise RuntimeError("nonvertex q4 refined sources changed during calculation")
        output = None if arguments.output is None else arguments.output.parent.resolve() / arguments.output.name
        _protect_output(output, sources)
        if output is not None:
            atomic_json(output, report)
        else:
            print("verified saved nonvertex q4 refined certificate")
    except (ValueError, TypeError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
