"""Certify errors uniformly over all 2<=s<=floor(D**(1/4)), at fixed q.

The moving face range is covered for every D>=16. Exact rational dyadic
envelopes also certify the whole infinite degree tail, with no enumeration
of faces or degrees. The selected-work counter is not a CPU/RSS bound.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

from . import nonvertex_general_error as general
from .nonvertex_q3_error import (
    _canonical,
    _decimal_integer,
    _integer,
    _target_argument,
    _unique,
    atomic_json,
    fraction_record,
)

SCHEMA = "skeleton-risk-nonvertex-growing-faces-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_nonvertex_growing_faces.rst"
DEFAULT_WORK_LIMIT = general.DEFAULT_WORK_LIMIT
MAX_WORK_LIMIT = general.MAX_WORK_LIMIT


def _budget(outside: int, degree_bits: int = 0, target_bits: int = 0, *, work_limit: int = DEFAULT_WORK_LIMIT) -> int:
    _integer(degree_bits, 0, "degree_bits")
    _integer(target_bits, 0, "target_bits")
    return general._budget(2, outside, bits=degree_bits + target_bits, work_limit=work_limit)


def _target_bits(target: Fraction | None) -> int:
    if target is None:
        return 0
    if type(target) is not Fraction or target <= 0:
        raise ValueError("target must be a positive Fraction")
    return max(target.numerator.bit_length(), target.denominator.bit_length())


def face_limit(degree: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> int:
    """Compute the exact floor fourth root at degrees in the proved domain."""
    _integer(degree, 16, "degree")
    _budget(2, degree.bit_length(), work_limit=work_limit)
    return isqrt(isqrt(degree))


def _dyadic_bound(outside: int, exponent: int, majorant: Fraction) -> Fraction:
    root_lower = Fraction(1 << (exponent // 2)) * (Fraction(4, 3) if exponent % 2 else 1)
    numerator = 88 * (exponent + 1) if outside == 2 else Fraction(39, 2) * majorant + Fraction(outside * (outside + 1), 4)
    return numerator / root_lower


def dyadic_tail_bound(outside: int, exponent: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> Fraction:
    """Bound every eligible face at EVERY D>=2**exponent, exponent>=4."""
    _integer(exponent, 4, "exponent")
    _budget(outside, exponent + 1, work_limit=work_limit)
    majorant, _ = general.diagonal_majorant(outside, work_limit=work_limit)
    return _dyadic_bound(outside, exponent, majorant)


def relative_error_bound(outside: int, degree: int, *, work_limit: int = DEFAULT_WORK_LIMIT) -> Fraction:
    """Bound all 2<=s<=floor(D**(1/4)) using the containing dyadic block."""
    _integer(degree, 16, "degree")
    _budget(outside, degree.bit_length(), work_limit=work_limit)
    return dyadic_tail_bound(outside, degree.bit_length() - 1, work_limit=work_limit)


def relative_error_threshold(outside: int, target: Fraction, *, work_limit: int = DEFAULT_WORK_LIMIT) -> int:
    """Return a sufficient all-tail degree for the moving face range."""
    if target is None:
        raise ValueError("target must be a positive Fraction")
    bits = _target_bits(target)
    _budget(outside, 5, bits, work_limit=work_limit)
    majorant, _ = general.diagonal_majorant(outside, work_limit=work_limit)
    exponent = 4
    while True:
        _budget(outside, exponent + 1, bits, work_limit=work_limit)
        if _dyadic_bound(outside, exponent, majorant) <= target:
            return 1 << exponent
        exponent += 1


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        **general._sources(),
        "tools/risk_research/nonvertex_growing_faces.py": Path(__file__).read_bytes(),
        PROOF: (root / PROOF).read_bytes(),
    }


def certificate(
    outside: int, degree: int, target: Fraction | None = None, work_limit: int = DEFAULT_WORK_LIMIT
) -> dict[str, object]:
    """Record a source-bound uniform error and an optional moving-range tail."""
    sources = _sources()
    _integer(degree, 16, "degree")
    bits = _target_bits(target)
    selected = _budget(outside, degree.bit_length(), bits, work_limit=work_limit)
    majorant, _ = general.diagonal_majorant(outside, work_limit=work_limit)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_uniform_growing_face_error_and_all_tail_threshold",
        "proof": PROOF,
        "scope": "fixed integer outside q>=2; every D>=16 and every 2<=s<=floor(D**(1/4))",
        "outside": outside,
        "degree": degree,
        "minimum_face_size": 2,
        "maximum_face_size": isqrt(isqrt(degree)),
        "diagonal_majorant": fraction_record(majorant),
        "diagonal_logarithm_power": int(outside == 2),
        "relative_error_bound": fraction_record(_dyadic_bound(outside, degree.bit_length() - 1, majorant)),
        "work_limit": work_limit,
        "selected_input_work": selected,
        "work_scope": "selected input counter only; not CPU, RSS or integer bit-complexity",
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
    }
    if target is not None:
        threshold = relative_error_threshold(outside, target, work_limit=work_limit)
        tail_selected = _budget(outside, threshold.bit_length(), bits, work_limit=work_limit)
        result["selected_input_work"] = max(selected, tail_selected)
        result["relative_target"] = fraction_record(target)
        result["all_tail"] = {
            "from_degree": threshold,
            "relative_error_bound": fraction_record(_dyadic_bound(outside, threshold.bit_length() - 1, majorant)),
            "face_range": "every 2<=s<=floor(D**(1/4)), recomputed for each D>=from_degree",
            "requested_degree_in_tail": degree >= threshold,
        }
    if _sources() != sources:
        raise RuntimeError("growing-face sources changed during calculation")
    return result


def _record_parameters(record: object) -> tuple[int, int, Fraction | None, int]:
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported growing-face certificate")
    outside, degree, work_limit = (
        _integer(record.get(name), minimum, name) for name, minimum in (("outside", 2), ("degree", 16), ("work_limit", 1))
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
    return outside, degree, target, work_limit


def verify_certificate(record: object) -> None:
    """Regenerate every field with saved nondefault options and exact types."""
    sources = _sources()
    if _canonical(record) != _canonical(certificate(*_record_parameters(record))):
        raise ValueError("saved certificate differs from source or exact recomputation")
    if _sources() != sources:
        raise RuntimeError("growing-face sources changed during verification")


def main(argv: list[str] | None = None) -> int:
    """Generate or replay a uniform growing-face record with the stdlib."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outside", type=_decimal_integer)
    parser.add_argument("--degree", type=_decimal_integer)
    parser.add_argument("--relative-target", type=_target_argument)
    parser.add_argument("--work-limit", type=_decimal_integer)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        general._protect_output(args.output, sources)
        if args.verify is not None:
            raw_record = args.verify.read_bytes()
            record = json.loads(raw_record, object_pairs_hook=_unique, parse_int=_decimal_integer)
            saved = _record_parameters(record)
            supplied = (args.outside, args.degree, args.relative_target, args.work_limit)
            if any(value is not None and value != old for value, old in zip(supplied, saved, strict=True)):
                raise ValueError("generation override differs from the saved certificate")
            verify_certificate(record)
            if args.verify.read_bytes() != raw_record:
                raise RuntimeError("saved input changed during verification")
        else:
            if args.outside is None or args.degree is None or args.output is None:
                raise ValueError("generation requires --outside, --degree and --output")
            record = certificate(
                args.outside,
                args.degree,
                args.relative_target,
                DEFAULT_WORK_LIMIT if args.work_limit is None else args.work_limit,
            )
        if _sources() != sources:
            raise RuntimeError("growing-face sources changed during calculation")
        output = None if args.output is None else args.output.parent.resolve() / args.output.name
        general._protect_output(output, sources)
        if output is None:
            print("verified saved uniform growing-face certificate")
        else:
            atomic_json(output, record)
    except (ValueError, TypeError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
