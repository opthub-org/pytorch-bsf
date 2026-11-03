"""Certify finite-degree errors for every fixed nonvertex codimension.

The proof covers all integers s,q>=2. For q>=3 the explicit relative
error is O(D**(-1/2)); q=2 has O(log(D)/sqrt(D)). Arithmetic is exact,
and the leading-constant interval is separate from this degree error.
The selected-work limit bounds accepted evaluator inputs, not CPU/RSS.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import isqrt, prod
from pathlib import Path

from .codimension_three_exact import nonvertex_leading_constant_enclosure
from .nonvertex_q3_error import (
    _canonical,
    _decimal_integer,
    _integer,
    _target_argument,
    _unique,
    atomic_json,
    fraction_record,
)
from .vertex_general_asymptotic import error_constants as general_error_constants
from .vertex_refined_asymptotic import error_constants as refined_error_constants

SCHEMA = "skeleton-risk-nonvertex-general-error-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_nonvertex_general_error.rst"
DEFAULT_WORK_LIMIT = 200_000
MAX_WORK_LIMIT = 1_000_000


def _budget(face_size: int, outside: int, terms: int = 0, bits: int = 0, work_limit: int = DEFAULT_WORK_LIMIT) -> int:
    _integer(face_size, 2, "face_size")
    _integer(outside, 2, "outside")
    _integer(terms, 0, "terms")
    _integer(bits, 0, "bits")
    _integer(work_limit, 1, "work_limit")
    if work_limit > MAX_WORK_LIMIT:
        raise ValueError("work_limit exceeds the evaluator maximum")
    if max(face_size, outside, terms, bits) > work_limit:
        raise ValueError("selected work exceeds work_limit")
    selected = (outside + 1) ** 3 + (outside + 1) ** 2 * terms + face_size + outside
    selected += bits * (outside + abs(face_size - outside - 2) + 2)
    if selected > work_limit:
        raise ValueError("selected work exceeds work_limit")
    return selected


def diagonal_majorant(outside: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> tuple[Fraction, bool]:
    """Return Q_q and whether its inverse-square bound includes 1+log(n)."""
    _budget(2, outside, work_limit=work_limit)
    if outside == 2:
        return Fraction(4), True
    if outside == 3:
        constants = general_error_constants(4)
        return Fraction(constants.limiting + 4 * constants.diagonal), False
    refined = refined_error_constants(outside + 1)
    return refined.limiting + refined.diagonal, False


def leading_constant_enclosure(
    face_size: int, outside: int, terms: int = 64, *, work_limit: int = DEFAULT_WORK_LIMIT
) -> tuple[Fraction, Fraction]:
    """Enclose A_(s,q), independently of the finite-degree error budget."""
    _integer(terms, 1, "terms")
    _budget(face_size, outside, terms, work_limit=work_limit)
    return nonvertex_leading_constant_enclosure(face_size, outside, terms)


def averaging_error_bound(face_size: int, outside: int, degree: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> Fraction:
    """Bound the sum of all positive-diagonal averaging errors."""
    _integer(degree, 1, "degree")
    _budget(face_size, outside, bits=degree.bit_length(), work_limit=work_limit)
    if degree < face_size:
        raise ValueError("averaging requires degree >= face_size")
    majorant, logarithmic = diagonal_majorant(outside, work_limit=work_limit)
    logarithm = 1 + (degree - 1).bit_length() if logarithmic else 1
    cutoff = isqrt(degree)
    return majorant * logarithm * (Fraction(9 * cutoff, 2 * degree) + Fraction(10, cutoff + 1))


def _dyadic_bound(face_size: int, outside: int, exponent: int, majorant: Fraction, logarithmic: bool) -> Fraction:
    logarithm = exponent + 1
    factor = logarithm if logarithmic else 1
    tail = logarithm + 1 if logarithmic else 1
    prefactor = Fraction(outside * (2 * face_size + outside - 1), 4)
    angular = factor * ((face_size - 2) * logarithm + (face_size - 1) * (face_size - 2))
    square_root_lower = Fraction(1 << (exponent // 2)) * (Fraction(4, 3) if exponent % 2 else 1)
    return Fraction(29, 2) * majorant * factor / square_root_lower + (prefactor + majorant * (2 * tail + angular)) / (
        1 << exponent
    )


def dyadic_tail_bound(face_size: int, outside: int, exponent: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> Fraction:
    """Bound the relative error at EVERY D>=2**exponent in the proved domain."""
    _integer(exponent, 2, "exponent")
    _budget(face_size, outside, bits=exponent + 1, work_limit=work_limit)
    if (exponent + 1 < face_size.bit_length()) or (1 << exponent) < 2 * (face_size - 1):
        raise ValueError("dyadic degree must be >= 2*(face_size-1)")
    majorant, logarithmic = diagonal_majorant(outside, work_limit=work_limit)
    return _dyadic_bound(face_size, outside, exponent, majorant, logarithmic)


def relative_error_threshold(face_size: int, outside: int, target: Fraction, *, work_limit: int = DEFAULT_WORK_LIMIT) -> int:
    """Return a sufficient all-tail degree, or reject its selected work."""
    if type(target) is not Fraction or target <= 0:
        raise ValueError("target must be a positive Fraction")
    _budget(
        face_size, outside, bits=max(target.numerator.bit_length(), target.denominator.bit_length()), work_limit=work_limit
    )
    majorant, logarithmic = diagonal_majorant(outside, work_limit=work_limit)
    exponent = max(2, (2 * face_size - 3).bit_length())
    while True:
        _budget(face_size, outside, bits=exponent + 1, work_limit=work_limit)
        if _dyadic_bound(face_size, outside, exponent, majorant, logarithmic) <= target:
            return 1 << exponent
        exponent += 1


def relative_error_bound(face_size: int, outside: int, degree: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> Fraction:
    """Bound abs(a/(A_(s,q)*D**(s-q-2))-1) without a degree-sized table."""
    _integer(degree, 1, "degree")
    _budget(face_size, outside, bits=degree.bit_length(), work_limit=work_limit)
    if degree < face_size:
        return Fraction(1)
    majorant, logarithmic = diagonal_majorant(outside, work_limit=work_limit)
    logarithm = 1 + (degree - 1).bit_length()
    factor = logarithm if logarithmic else 1
    count = degree - face_size + 1
    harmonic = 1 + (count - 1).bit_length()
    tail = majorant * (logarithm + 1 if logarithmic else 1) / count
    angular = majorant * factor * ((face_size - 2) * harmonic + (face_size - 1) * (face_size - 2)) / degree
    prefactor = Fraction((2 * degree) ** outside, prod(range(2 * degree + face_size, 2 * degree + face_size + outside)))
    bound = 1 - prefactor + averaging_error_bound(face_size, outside, degree, work_limit=work_limit) + tail + angular
    exponent = degree.bit_length() - 1
    if exponent >= 2 and 2**exponent >= 2 * (face_size - 1):
        bound = min(bound, _dyadic_bound(face_size, outside, exponent, majorant, logarithmic))
    return bound


def coefficient_enclosure(
    face_size: int, outside: int, degree: int, terms: int = 64, *, work_limit: int = DEFAULT_WORK_LIMIT
) -> tuple[Fraction, Fraction]:
    """Combine the independent constant interval and proved degree error."""
    _integer(degree, 1, "degree")
    _integer(terms, 1, "terms")
    _budget(face_size, outside, terms, degree.bit_length(), work_limit)
    if degree < face_size:
        return Fraction(0), Fraction(0)
    lower, upper = leading_constant_enclosure(face_size, outside, terms, work_limit=work_limit)
    error = relative_error_bound(face_size, outside, degree, work_limit=work_limit)
    scale = Fraction(degree) ** (face_size - outside - 2)
    return lower * max(Fraction(0), 1 - error) * scale, upper * (1 + error) * scale


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "tools/risk_research/nonvertex_general_error.py",
        "tools/risk_research/nonvertex_q3_error.py",
        "tools/risk_research/codimension_three_exact.py",
        "tools/risk_research/vertex_general.py",
        "tools/risk_research/vertex_general_asymptotic.py",
        "tools/risk_research/vertex_refined_asymptotic.py",
        "tools/risk_research/exact.py",
        PROOF,
        "docs/advanced/skeleton_risk_codimension_three_exact.rst",
        "docs/advanced/skeleton_risk_vertex_general.rst",
        "docs/advanced/skeleton_risk_vertex_general_asymptotics.rst",
        "docs/advanced/skeleton_risk_vertex_refined_asymptotics.rst",
    )
    return {name: (root / name).read_bytes() for name in names}


def certificate(
    face_size: int,
    outside: int,
    degree: int,
    terms: int = 64,
    target: Fraction | None = None,
    work_limit: int = DEFAULT_WORK_LIMIT,
) -> dict[str, object]:
    """Build a source-bound record with separate constant and degree errors."""
    sources = _sources()
    _integer(degree, 1, "degree")
    selected = _budget(face_size, outside, terms, degree.bit_length(), work_limit)
    leading = leading_constant_enclosure(face_size, outside, terms, work_limit=work_limit)
    interval = coefficient_enclosure(face_size, outside, degree, terms, work_limit=work_limit)
    majorant, logarithmic = diagonal_majorant(outside, work_limit=work_limit)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_general_fixed_codimension_error_and_all_tail_threshold",
        "proof": PROOF,
        "scope": "all fixed integers s,q>=2; no finite-sample or growing-s,q,D uniform limit",
        "face_size": face_size,
        "outside": outside,
        "vertices": face_size + outside,
        "degree": degree,
        "active": degree >= face_size,
        "constant_terms": terms,
        "work_limit": work_limit,
        "selected_work": selected,
        "work_scope": "selected input counter only; not CPU, RSS or integer bit-complexity",
        "diagonal_majorant": fraction_record(majorant),
        "diagonal_logarithm_power": int(logarithmic),
        "leading_constant": {"lower": fraction_record(leading[0]), "upper": fraction_record(leading[1])},
        "coefficient_enclosure": {"lower": fraction_record(interval[0]), "upper": fraction_record(interval[1])},
        "relative_error_bound": fraction_record(relative_error_bound(face_size, outside, degree, work_limit=work_limit)),
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
    }
    if target is not None:
        threshold = relative_error_threshold(face_size, outside, target, work_limit=work_limit)
        result["relative_target"] = fraction_record(target)
        result["all_tail"] = {
            "from_degree": threshold,
            "relative_error_bound": fraction_record(
                dyadic_tail_bound(face_size, outside, threshold.bit_length() - 1, work_limit=work_limit)
            ),
            "requested_degree_in_tail": degree >= threshold,
        }
    if _sources() != sources:
        raise RuntimeError("nonvertex general sources changed during calculation")
    return result


def _record_parameters(record: object) -> tuple[int, int, int, int, Fraction | None, int]:
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported nonvertex general certificate")
    values = tuple(
        _integer(record.get(name), minimum, name)
        for name, minimum in (("face_size", 2), ("outside", 2), ("degree", 1), ("constant_terms", 1), ("work_limit", 1))
    )
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
    return values[0], values[1], values[2], values[3], target, values[4]


def verify_certificate(record: object) -> None:
    """Regenerate the entire strict record with its saved nondefault options."""
    sources = _sources()
    if _canonical(record) != _canonical(certificate(*_record_parameters(record))):
        raise ValueError("saved certificate differs from source or exact recomputation")
    if _sources() != sources:
        raise RuntimeError("nonvertex general sources changed during verification")


def _protect_output(path: Path | None, sources: dict[str, bytes]) -> None:
    if path is not None:
        root = Path(__file__).resolve().parents[2]
        if path.resolve() in {(root / name).resolve() for name in sources}:
            raise ValueError("output must not overwrite evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Save or independently replay a complete general nonvertex certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=_decimal_integer)
    parser.add_argument("--outside", type=_decimal_integer)
    parser.add_argument("--degree", type=_decimal_integer)
    parser.add_argument("--constant-terms", type=_decimal_integer)
    parser.add_argument("--relative-target", type=_target_argument)
    parser.add_argument("--work-limit", type=_decimal_integer)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        _protect_output(args.output, sources)
        if args.verify is not None:
            record = json.loads(args.verify.read_bytes(), object_pairs_hook=_unique, parse_int=_decimal_integer)
            saved = _record_parameters(record)
            supplied = (args.face_size, args.outside, args.degree, args.constant_terms, args.relative_target, args.work_limit)
            if any(value is not None and value != previous for value, previous in zip(supplied, saved, strict=True)):
                raise ValueError("generation override differs from the saved certificate")
            verify_certificate(record)
        else:
            if args.face_size is None or args.outside is None or args.degree is None or args.output is None:
                raise ValueError("generation requires --face-size, --outside, --degree and --output")
            record = certificate(
                args.face_size,
                args.outside,
                args.degree,
                64 if args.constant_terms is None else args.constant_terms,
                args.relative_target,
                DEFAULT_WORK_LIMIT if args.work_limit is None else args.work_limit,
            )
        if _sources() != sources:
            raise RuntimeError("nonvertex general sources changed during calculation")
        output = None if args.output is None else args.output.parent.resolve() / args.output.name
        _protect_output(output, sources)
        if output is None:
            print("verified saved general nonvertex certificate")
        else:
            atomic_json(output, record)
    except (ValueError, TypeError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
