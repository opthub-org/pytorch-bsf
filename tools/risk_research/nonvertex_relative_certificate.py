"""Certify a requested relative accuracy for one finite-degree coefficient.

This research tool composes the general nonvertex analytic error and one
dyadic leading-constant enclosure. Relative error uses the true positive
coefficient as denominator. An inactive face has an exact zero result,
without a relative-error assertion. Operational refusals make no claim
that the actual coefficient is inaccurate.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

from . import mzv_dyadic, nonvertex_general_error
from .checkout_harness import checkout_state
from .leading_constant_mzv import _budget as numeric_budget
from .nonvertex_q3_error import (
    _canonical,
    _decimal_integer,
    _integer,
    _target_argument,
    _unique,
    atomic_json,
    fraction_record,
)

SCHEMA = "skeleton-risk-nonvertex-relative-certificate-v1"
PROOF = "docs/advanced/skeleton_risk_nonvertex_relative_certificate.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
DEFAULT_WORK_LIMIT = 200_000


def _validate(
    face_size: int, outside: int, degree: int, target: Fraction, analytic_work_limit: int, numeric_work_limit: int
) -> int:
    _integer(degree, 1, "degree")
    if type(target) is not Fraction or not 0 < target < 1:
        raise ValueError("relative_target must be a Fraction strictly between zero and one")
    # Validate both resource policies and all scalar bit lengths before any
    # diagonal construction or MZV evaluation, including inactive requests.
    numeric_budget(numeric_work_limit)
    return nonvertex_general_error._budget(
        face_size,
        outside,
        bits=max(degree.bit_length(), target.numerator.bit_length(), target.denominator.bit_length()),
        work_limit=analytic_work_limit,
    )


def _precision_bits(width: Fraction) -> int:
    bits = max(1, width.denominator.bit_length() - width.numerator.bit_length())
    return bits + int(Fraction(1, 1 << bits) > width)


def _interval_record(interval: tuple[Fraction, Fraction]) -> dict[str, object]:
    return {"lower": fraction_record(interval[0]), "upper": fraction_record(interval[1])}


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        **mzv_dyadic._sources(),
        **nonvertex_general_error._sources(),
        **{
            name: (root / name).read_bytes()
            for name in (
                "tools/risk_research/nonvertex_relative_certificate.py",
                "tools/risk_research/checkout_harness.py",
                PROOF,
            )
        },
    }


def _calculate(
    face_size: int,
    outside: int,
    degree: int,
    relative_target: Fraction,
    analytic_work_limit: int,
    numeric_work_limit: int,
    sources: dict[str, bytes],
) -> dict[str, object]:
    """Return a complete success record, or refuse without a partial record."""
    selected = _validate(face_size, outside, degree, relative_target, analytic_work_limit, numeric_work_limit)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof": PROOF,
        "proof_status": "proved_fixed_degree_relative_accuracy_composition",
        "scope": "uniform-simplex sample-size-asymptotic coefficient; fixed D; no finite-sample risk",
        "face_size": face_size,
        "outside": outside,
        "vertices": face_size + outside,
        "degree": degree,
        "relative_target": fraction_record(relative_target),
        "analytic_work_limit": analytic_work_limit,
        "analytic_selected_work": selected,
        "numeric_work_limit": numeric_work_limit,
        "work_scope": "separate analytic selected-input and numeric additive counters; not CPU, RSS or bit-complexity",
    }
    if degree < face_size:
        result.update(
            active=False,
            status="exact_zero_inactive_face",
            constant_index=outside + 1,
            precision_bits=None,
            constant_terms=0,
            numeric_work_units=0,
            constant_interval=None,
            leading_constant=None,
            analytic_relative_error=None,
            coefficient_enclosure=_interval_record((Fraction(0), Fraction(0))),
            approximation=fraction_record(Fraction(0)),
            certified_relative_error=None,
        )
    else:
        error = nonvertex_general_error.relative_error_bound(face_size, outside, degree, work_limit=analytic_work_limit)
        if error >= relative_target:
            raise ValueError("the current analytic bound cannot certify relative_target (requires bound < target)")
        allowance = (relative_target - error) / (1 - relative_target * error)
        bits = _precision_bits(2 * (outside + 1) * allowance)
        if bits > numeric_work_limit:
            raise ValueError("selected precision exceeds numeric work limit")
        budget = numeric_budget(numeric_work_limit)
        # The existing evaluator charges construction, cutoff search and final
        # polylog evaluation to this one shared budget. No retry resets it.
        expression = mzv_dyadic._construct(outside + 1, budget)
        (constant_lower, constant_upper), terms = mzv_dyadic._precision(expression, bits, budget)
        constant_lower = max(constant_lower, Fraction(outside + 1))
        if constant_upper < constant_lower:
            raise ArithmeticError("constant enclosure contradicts its proved positive lower bound")
        scale = Fraction(
            comb(face_size + outside, face_size) ** 2 * prod(range(face_size, face_size + outside)),
            2**outside * factorial(face_size - 2) * (outside + 1),
        )
        leading_lower, leading_upper = scale * constant_lower, scale * constant_upper
        degree_scale = Fraction(degree) ** (face_size - outside - 2)
        lower, upper = leading_lower * (1 - error) * degree_scale, leading_upper * (1 + error) * degree_scale
        certified = (upper - lower) / (upper + lower)
        if not 0 < lower <= upper or certified > relative_target:
            raise ArithmeticError("composed interval does not meet the requested relative accuracy")
        result.update(
            active=True,
            status="certified_relative_accuracy",
            constant_index=outside + 1,
            precision_bits=bits,
            constant_terms=terms,
            numeric_work_units=budget.used,
            constant_interval=_interval_record((constant_lower, constant_upper)),
            constant_relative_width=fraction_record((constant_upper - constant_lower) / (constant_upper + constant_lower)),
            constant_relative_allowance=fraction_record(allowance),
            leading_constant=_interval_record((leading_lower, leading_upper)),
            analytic_relative_error=fraction_record(error),
            coefficient_enclosure=_interval_record((lower, upper)),
            approximation=fraction_record(2 * lower * upper / (lower + upper)),
            certified_relative_error=fraction_record(certified),
        )
    result.update(
        source_hash_policy=HASH_POLICY,
        source_sha256={name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
    )
    return result


def certificate(
    face_size: int,
    outside: int,
    degree: int,
    relative_target: Fraction,
    analytic_work_limit: int = DEFAULT_WORK_LIMIT,
    numeric_work_limit: int = DEFAULT_WORK_LIMIT,
) -> dict[str, object]:
    """Return a complete success record, guarding successful and failed work."""
    sources = _sources()
    root = Path(__file__).resolve().parents[2]
    before = checkout_state(root)
    try:
        return _calculate(face_size, outside, degree, relative_target, analytic_work_limit, numeric_work_limit, sources)
    finally:
        if _sources() != sources or checkout_state(root) != before:
            raise RuntimeError("composition sources, HEAD, tree or checkout status changed during calculation")


def _record_parameters(record: object) -> tuple[int, int, int, Fraction, int, int]:
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported relative-accuracy certificate")
    values = tuple(
        _integer(record.get(name), minimum, name)
        for name, minimum in (
            ("face_size", 2),
            ("outside", 2),
            ("degree", 1),
            ("analytic_work_limit", 1),
            ("numeric_work_limit", 1),
        )
    )
    target = record.get("relative_target")
    if (
        type(target) is not dict
        or set(target) != {"numerator", "denominator"}
        or any(type(value) is not str for value in target.values())
    ):
        raise ValueError("invalid recorded relative target")
    fraction = Fraction(_decimal_integer(target["numerator"]), _decimal_integer(target["denominator"]))
    return values[0], values[1], values[2], fraction, values[3], values[4]


def verify_certificate(record: object) -> None:
    """Regenerate every field with the saved inputs, options and source union."""
    sources = _sources()
    root = Path(__file__).resolve().parents[2]
    before = checkout_state(root)
    try:
        if _canonical(record) != _canonical(certificate(*_record_parameters(record))):
            raise ValueError("saved certificate differs from exact recomputation or sources")
    finally:
        if _sources() != sources or checkout_state(root) != before:
            raise RuntimeError("composition sources, HEAD, tree or checkout status changed during verification")


def _protect_output(path: Path | None, sources: dict[str, bytes], input_path: Path | None) -> None:
    if path is not None:
        root = Path(__file__).resolve().parents[2]
        protected = {(root / name).resolve() for name in sources}
        if input_path is not None:
            protected.add(input_path.resolve())
        if path.resolve() in protected:
            raise ValueError("output must not overwrite evaluator sources, proofs or the verification input")


def main(argv: list[str] | None = None) -> int:
    """Save or replay a complete fixed-degree relative-accuracy certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=_decimal_integer)
    parser.add_argument("--outside", type=_decimal_integer)
    parser.add_argument("--degree", type=_decimal_integer)
    parser.add_argument("--relative-target", type=_target_argument)
    parser.add_argument("--analytic-work-limit", type=_decimal_integer)
    parser.add_argument("--numeric-work-limit", type=_decimal_integer)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    root = Path(__file__).resolve().parents[2]
    before = checkout_state(root)
    input_bytes = None
    try:
        _protect_output(args.output, sources, args.verify)
        if args.verify is not None:
            input_bytes = args.verify.read_bytes()
            record = json.loads(input_bytes, object_pairs_hook=_unique, parse_int=_decimal_integer)
            saved = _record_parameters(record)
            supplied = (
                args.face_size,
                args.outside,
                args.degree,
                args.relative_target,
                args.analytic_work_limit,
                args.numeric_work_limit,
            )
            if any(value is not None and value != old for value, old in zip(supplied, saved, strict=True)):
                raise ValueError("generation override differs from the saved certificate")
            verify_certificate(record)
        else:
            if any(value is None for value in (args.face_size, args.outside, args.degree, args.relative_target, args.output)):
                raise ValueError("generation requires --face-size, --outside, --degree, --relative-target and --output")
            record = certificate(
                args.face_size,
                args.outside,
                args.degree,
                args.relative_target,
                DEFAULT_WORK_LIMIT if args.analytic_work_limit is None else args.analytic_work_limit,
                DEFAULT_WORK_LIMIT if args.numeric_work_limit is None else args.numeric_work_limit,
            )
        if _sources() != sources or checkout_state(root) != before:
            raise RuntimeError("composition sources, HEAD, tree or checkout status changed during the transaction")
        if args.verify is not None and args.verify.read_bytes() != input_bytes:
            raise RuntimeError("verification input changed during the transaction")
        if args.output is None:
            print("verified fixed-degree relative-accuracy certificate")
        else:
            output = args.output.parent.resolve() / args.output.name
            _protect_output(output, sources, args.verify)
            atomic_json(output, record)
    except (OSError, ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        if _sources() != sources or checkout_state(root) != before:
            parser.error("composition sources, HEAD, tree or checkout status changed during the transaction")
        if input_bytes is not None and args.verify.read_bytes() != input_bytes:
            parser.error("verification input changed during the transaction")
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
