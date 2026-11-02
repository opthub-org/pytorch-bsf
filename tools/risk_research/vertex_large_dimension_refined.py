"""Certify a sharper elementary large-vertex constant remainder.

For every integer M>=162 the proof in
``skeleton_risk_vertex_large_dimension_refined.rst`` establishes
abs(C_M/B_M-1)<=6/(M-2). This module keeps the old leading normalization
and the independent positive-integral evaluator. It adds no runtime API.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .exact import atomic_json, fraction_record
from .saved_records import _unique_object, canonical_json
from .vertex_large_dimension import (
    _integer,
    _rational_ball,
    _sqrt_enclosure,
    _target,
    integral_normalized_ball,
    leading_normalized_enclosure,
)

MINIMUM_VERTICES = 162
SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"
SCHEMA = "skeleton-risk-refined-large-vertex-constant-v1"


def relative_remainder_bound(vertices: int) -> Fraction:
    """Bound |C_M/B_M-1| with a simple 6/(M-2) radius for M>=162.

    Below the simple threshold the same proof gives an elementary,
    sometimes wide, bound. No small-vertex claim uses the 6/(M-2) rule.
    """
    _integer(vertices, 4, "vertices")
    n = vertices - 2
    if vertices >= MINIMUM_VERTICES:
        return Fraction(6, n)
    root_lower, root_upper = _sqrt_enclosure(Fraction(n), 64)
    natural = (
        Fraction(17, 4 * n)
        + 4 * root_upper * Fraction(15, 16) ** (n - 2)
        + Fraction(8, 3) / root_lower * Fraction(16, 17) ** n
    )
    return (1 + Fraction(1, 2 * n)) * natural + Fraction(1, 2 * n)


def relative_error_threshold(target: Fraction = Fraction(1, 100)) -> int:
    """Return a vertex count valid for every larger integer vertex count."""
    if not isinstance(target, Fraction) or target <= 0:
        raise ValueError("target must be a strictly positive Fraction")
    required = (6 * target.denominator + target.numerator - 1) // target.numerator
    return max(MINIMUM_VERTICES, required + 2)


def normalized_constant_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose C_M/(M**2*(M-1)!*(2*log(2))**(M-1)) exactly."""
    lower, upper = leading_normalized_enclosure(vertices, bits=bits)
    radius = relative_remainder_bound(vertices)
    return max(Fraction(0), 1 - radius) * lower, (1 + radius) * upper


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
        "tools/risk_research/vertex_large_dimension_refined.py",
        "tools/risk_research/vertex_large_dimension.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        "docs/advanced/skeleton_risk_vertex_large_dimension_refined.rst",
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
    _integer(vertices, 4, "vertices")
    _integer(bits, 32, "bits")
    if type(with_arb) is not bool or type(integral_check) is not bool:
        raise ValueError("backend flags must be booleans")
    threshold = None if target is None else relative_error_threshold(target)
    lower, upper = normalized_constant_enclosure(vertices, bits=bits)
    leading_lower, leading_upper = leading_normalized_enclosure(vertices, bits=bits)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "vertices": vertices,
        "bits": bits,
        "with_arb": with_arb,
        "integral_check": integral_check,
        "normalization": "C_M/(M^2*(M-1)!*(2*log(2))^(M-1))",
        "normalized_lower": fraction_record(lower),
        "normalized_upper": fraction_record(upper),
        "leading_normalized_lower": fraction_record(leading_lower),
        "leading_normalized_upper": fraction_record(leading_upper),
        "relative_error_bound": fraction_record(relative_remainder_bound(vertices)),
        "simple_remainder_start_vertices": MINIMUM_VERTICES,
        "simple_remainder_numerator": 6,
        "proof": "docs/advanced/skeleton_risk_vertex_large_dimension_refined.rst",
        "source_hash_policy": SOURCE_HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": "leading constant as M tends to infinity; no joint M,D limit",
    }
    if target is not None:
        record["relative_error_target"] = fraction_record(target)
        record["relative_error_threshold"] = str(threshold)
    if with_arb or integral_check:
        flint = import_module("flint")
        with flint.ctx.workprec(bits):
            digits = bits * 30103 // 100000 + 5
            if with_arb:
                record["constant_ball"] = constant_ball(vertices, bits=bits).str(digits, more=True)
            if integral_check:
                direct = integral_normalized_ball(vertices, bits=bits)
                if not direct.overlaps(_rational_ball(lower, upper, flint)):
                    raise RuntimeError("independent integral contradicts the refined rational enclosure")
                record["normalized_integral_ball"] = direct.str(digits, more=True)
        record["python_flint"] = version("python-flint")
    if _sources() != sources:
        raise RuntimeError("a refined constant enclosure source changed during calculation")
    return record


def verify_record(record: object) -> None:
    """Verify strict scalar types and all data by recorded-option replay."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported refined large-vertex enclosure schema")
    supplied = canonical_json(record)
    target = None
    if "relative_error_target" in record:
        value = record["relative_error_target"]
        if type(value) is not dict or set(value) != {"numerator", "denominator"}:
            raise ValueError("invalid recorded relative error target")
        if any(type(item) is not str for item in value.values()):
            raise ValueError("relative error target must contain decimal strings")
        try:
            target = Fraction(int(value["numerator"]), int(value["denominator"]))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("invalid recorded relative error target") from error
    expected = build_record(
        record.get("vertices"),
        bits=record.get("bits"),
        target=target,
        with_arb=record.get("with_arb"),
        integral_check=record.get("integral_check"),
    )
    if supplied != canonical_json(expected):
        raise ValueError("saved enclosure differs from its recorded options, source or recomputation")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a refined constant record, preserving old output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int)
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
            print("verified refined large-vertex enclosure")
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
        atomic_json(arguments.output, record)
    except (OSError, ValueError, TypeError, ModuleNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
