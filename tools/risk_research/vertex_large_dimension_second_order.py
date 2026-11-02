"""Certify the first large-vertex correction with a quadratic remainder.

For every integer M>=258 the proof establishes
abs(C_M/B_M-1-beta/(M-2))<=84/(M-2)**2, with an explicit beta in log(2).
The positive-integral oracle and leading normalization remain independent.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from importlib import import_module
from importlib.metadata import version
from math import isqrt
from pathlib import Path
from typing import Any

from .exact import atomic_json
from .saved_records import _unique_object, canonical_json
from .vertex_large_dimension import (
    _integer,
    _rational_ball,
    elementary_enclosures,
    integral_normalized_ball,
    leading_normalized_enclosure,
)

MINIMUM_VERTICES = 258
REMAINDER_NUMERATOR = 84
SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"
SCHEMA = "skeleton-risk-second-order-large-vertex-constant-v1"


def _decimal(value: int) -> str:
    """Encode arbitrary integers without changing process-wide digit limits."""
    if value < 0:
        return "-" + _decimal(-value)
    parts = []
    while value >= 10**500:
        value, remainder = divmod(value, 10**500)
        parts.append(f"{remainder:0500d}")
    return str(value) + "".join(reversed(parts))


def _decimal_integer(text: str) -> int:
    """Read ordinary signed decimal integers using bounded-size chunks."""
    sign = -1 if text.startswith("-") else 1
    digits = text.removeprefix("-").removeprefix("+")
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise ValueError("invalid decimal integer")
    result = 0
    for offset in range(0, len(digits), 500):
        part = digits[offset : offset + 500]
        result = result * 10 ** len(part) + int(part)
    return sign * result


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def _target(text: str) -> Fraction:
    parts = text.split("/")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("target must be a positive rational number")
    try:
        value = Fraction(_decimal_integer(parts[0]), 1 if len(parts) == 1 else _decimal_integer(parts[1]))
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("target must be a positive rational number") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("target must be a positive rational number")
    return value


def beta_enclosure(*, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose beta=(28L**2-18L+1)/(8*(2L-1)**2), L=log(2).

    This rational function increases on 2/3<L<7/10, which contains
    both elementary logarithm endpoints at every allowed precision.
    """
    logarithm, _ = elementary_enclosures(bits)

    def evaluate(value: Fraction) -> Fraction:
        return (28 * value**2 - 18 * value + 1) / (8 * (2 * value - 1) ** 2)

    return evaluate(logarithm[0]), evaluate(logarithm[1])


def relative_remainder_bound(vertices: int) -> Fraction:
    """Bound |C_M/B_M-1-beta/(M-2)| for every integer M>=258."""
    _integer(vertices, MINIMUM_VERTICES, "vertices")
    return Fraction(REMAINDER_NUMERATOR, (vertices - 2) ** 2)


def relative_error_threshold(target: Fraction = Fraction(1, 100)) -> int:
    """Guarantee error at most target relative to B_M*(1+beta/(M-2)).

    Since beta>0, the displayed quadratic radius relative to B_M is
    also a conservative relative radius for the corrected leading term.
    The result applies at every larger integer vertex count.
    """
    if not isinstance(target, Fraction) or target <= 0:
        raise ValueError("target must be a strictly positive Fraction")
    numerator = REMAINDER_NUMERATOR * target.denominator
    required = isqrt(numerator // target.numerator)
    if required**2 * target.numerator < numerator:
        required += 1
    return max(MINIMUM_VERTICES, required + 2)


def corrected_normalized_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose B_M*(1+beta/(M-2)) after removing the usual scale."""
    _integer(vertices, MINIMUM_VERTICES, "vertices")
    lower, upper = leading_normalized_enclosure(vertices, bits=bits)
    beta_lower, beta_upper = beta_enclosure(bits=bits)
    return lower * (1 + beta_lower / (vertices - 2)), upper * (1 + beta_upper / (vertices - 2))


def normalized_constant_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose C_M/(M**2*(M-1)!*(2*log(2))**(M-1)) exactly."""
    radius = relative_remainder_bound(vertices)
    lower, upper = leading_normalized_enclosure(vertices, bits=bits)
    beta_lower, beta_upper = beta_enclosure(bits=bits)
    return (1 + beta_lower / (vertices - 2) - radius) * lower, (1 + beta_upper / (vertices - 2) + radius) * upper


def constant_ball(vertices: int, *, bits: int = 128) -> Any:
    """Restore the factorial/exponential scale with outward Arb arithmetic."""
    lower, upper = normalized_constant_enclosure(vertices, bits=bits)
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        scale = vertices**2 * flint.arb(vertices).gamma() * (2 * flint.arb(2).log()) ** (vertices - 1)
        result = scale * _rational_ball(lower, upper, flint)
        if not result.is_finite():
            raise RuntimeError("constant enclosure is not finite")
    return result


def _source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "tools/risk_research/vertex_large_dimension_second_order.py",
        "tools/risk_research/vertex_large_dimension.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        "docs/advanced/skeleton_risk_vertex_large_dimension_second_order.rst",
    )
    return {name: root / name for name in names}


def _sources() -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in _source_paths().items()}


def build_record(
    vertices: int,
    *,
    bits: int = 128,
    target: Fraction | None = None,
    with_arb: bool = False,
    integral_check: bool = False,
) -> dict[str, object]:
    """Recompute a source-bound record using only its recorded options."""
    sources = _sources()
    _integer(vertices, MINIMUM_VERTICES, "vertices")
    _integer(bits, 32, "bits")
    if type(with_arb) is not bool or type(integral_check) is not bool:
        raise ValueError("backend flags must be booleans")
    if target is not None:
        relative_error_threshold(target)
    lower, upper = normalized_constant_enclosure(vertices, bits=bits)
    leading_lower, leading_upper = leading_normalized_enclosure(vertices, bits=bits)
    beta_lower, beta_upper = beta_enclosure(bits=bits)
    corrected_lower, corrected_upper = corrected_normalized_enclosure(vertices, bits=bits)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "vertices": _decimal(vertices),
        "bits": bits,
        "with_arb": with_arb,
        "integral_check": integral_check,
        "normalization": "C_M/(M^2*(M-1)!*(2*log(2))^(M-1))",
        "normalized_lower": _fraction(lower),
        "normalized_upper": _fraction(upper),
        "leading_normalized_lower": _fraction(leading_lower),
        "leading_normalized_upper": _fraction(leading_upper),
        "beta_lower": _fraction(beta_lower),
        "beta_upper": _fraction(beta_upper),
        "beta_formula": "(28*L^2-18*L+1)/(8*(2*L-1)^2); L=log(2)",
        "corrected_normalized_lower": _fraction(corrected_lower),
        "corrected_normalized_upper": _fraction(corrected_upper),
        "quadratic_remainder_relative_to_B_M": _fraction(relative_remainder_bound(vertices)),
        "corrected_relative_error_bound": _fraction(relative_remainder_bound(vertices)),
        "simple_remainder_start_vertices": MINIMUM_VERTICES,
        "simple_remainder_numerator": REMAINDER_NUMERATOR,
        "simple_remainder_power": 2,
        "proof": "docs/advanced/skeleton_risk_vertex_large_dimension_second_order.rst",
        "source_hash_policy": SOURCE_HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": "leading constant as M tends to infinity; no joint M,D limit",
    }
    if target is not None:
        record["relative_error_target"] = _fraction(target)
        record["relative_error_threshold"] = _decimal(relative_error_threshold(target))
        record["target_scope"] = "relative error against exact B_M*(1+beta/(M-2))"
    if with_arb or integral_check:
        flint = import_module("flint")
        with flint.ctx.workprec(bits):
            digits = bits * 30103 // 100000 + 5
            if with_arb:
                record["constant_ball"] = constant_ball(vertices, bits=bits).str(digits, more=True)
            if integral_check:
                direct = integral_normalized_ball(vertices, bits=bits)
                if not direct.overlaps(_rational_ball(lower, upper, flint)):
                    raise RuntimeError("independent integral contradicts the second-order rational enclosure")
                record["normalized_integral_ball"] = direct.str(digits, more=True)
        record["python_flint"] = version("python-flint")
    if _sources() != sources:
        raise RuntimeError("a second-order constant enclosure source changed during calculation")
    return record


def verify_record(record: object) -> None:
    """Verify strict scalar types and all data by recorded-option replay."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported second-order large-vertex enclosure schema")
    vertices, bits = record.get("vertices"), record.get("bits")
    with_arb, integral_check = record.get("with_arb"), record.get("integral_check")
    if type(vertices) is not str or type(bits) is not int:
        raise ValueError("recorded vertices must be a decimal string and bits an integer")
    vertex_count = _decimal_integer(vertices)
    if _decimal(vertex_count) != vertices:
        raise ValueError("recorded vertices must be canonical")
    if type(with_arb) is not bool or type(integral_check) is not bool:
        raise ValueError("recorded backend flags must be booleans")
    supplied = canonical_json(record)
    target = None
    if "relative_error_target" in record:
        value = record["relative_error_target"]
        if type(value) is not dict or set(value) != {"numerator", "denominator"}:
            raise ValueError("invalid recorded relative error target")
        if any(type(item) is not str for item in value.values()):
            raise ValueError("relative error target must contain decimal strings")
        try:
            target = Fraction(_decimal_integer(value["numerator"]), _decimal_integer(value["denominator"]))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("invalid recorded relative error target") from error
    expected = build_record(
        vertex_count,
        bits=bits,
        target=target,
        with_arb=with_arb,
        integral_check=integral_check,
    )
    if supplied != canonical_json(expected):
        raise ValueError("saved enclosure differs from its recorded options, source or recomputation")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a second-order constant record, preserving old output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=_decimal_integer)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--relative-target", type=_target)
    parser.add_argument("--with-arb", action="store_true")
    parser.add_argument("--integral-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify is not None:
            if (
                any(
                    value is not None
                    for value in (arguments.vertices, arguments.bits, arguments.relative_target, arguments.output)
                )
                or arguments.with_arb
                or arguments.integral_check
            ):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(arguments.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            print("verified second-order large-vertex enclosure")
            return 0
        if arguments.vertices is None or arguments.output is None:
            raise ValueError("generation requires --vertices and --output")
        if arguments.output.resolve() in {path.resolve() for path in _source_paths().values()}:
            raise ValueError("output must not replace an evaluator or proof source")
        record = build_record(
            arguments.vertices,
            bits=128 if arguments.bits is None else arguments.bits,
            target=arguments.relative_target,
            with_arb=arguments.with_arb,
            integral_check=arguments.integral_check,
        )
        output = arguments.output.parent.resolve() / arguments.output.name
        if output.resolve() in {path.resolve() for path in _source_paths().values()}:
            raise ValueError("output must not replace an evaluator or proof source")
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ModuleNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
