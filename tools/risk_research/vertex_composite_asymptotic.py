"""Certify computable vertex centers on a larger single-log joint domain.

The proof keeps every M-dependent degree constant before allowing M to
grow. A normalized elementary center and three separated error budgets
avoid unknown C_M values and enormous degree or factorial construction.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .vertex_joint_growth_refined import _canonical, _parse_integer, _protect_output, _unique, atomic_json
from .vertex_large_dimension import _integer, leading_normalized_enclosure
from .vertex_large_dimension import relative_remainder_bound as local_constant_radius
from .vertex_large_dimension_refined import relative_remainder_bound as first_order_radius
from .vertex_large_dimension_second_order import _decimal_integer, _fraction, _target, corrected_normalized_enclosure
from .vertex_large_dimension_second_order import relative_remainder_bound as second_order_radius

MINIMUM_DEGREE_FLOOR = 256
SCHEMA = "skeleton-risk-computable-composite-vertex-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_vertex_composite_asymptotics.rst"


def maximum_vertices(degree_floor_exponent: int) -> int:
    """Return the proved ceiling on the full block 2**N<=D<2**(N+1)."""
    _integer(degree_floor_exponent, MINIMUM_DEGREE_FLOOR, "degree_floor_exponent")
    logarithm = (degree_floor_exponent + 1).bit_length()
    return 1 + max(3, (degree_floor_exponent - 4 * logarithm) // (8 * logarithm))


def large_vertex_gain(vertices: int) -> int:
    """Return the extra inverse power of two from the positive lower bound."""
    _integer(vertices, 4, "vertices")
    outside = vertices - 1
    return 0 if outside <= 160 else outside // 3 + 2 - (outside - 1).bit_length()


def degree_error_exponent(vertices: int, degree_floor_exponent: int) -> int:
    """Bound the degree error relative to C_M by 2**(-returned value)."""
    _integer(vertices, 4, "vertices")
    if vertices > maximum_vertices(degree_floor_exponent):
        raise ValueError("vertices exceeds the certified single-log ceiling")
    logarithm = (degree_floor_exponent + 1).bit_length()
    if vertices == 4:
        return degree_floor_exponent - 37 - 2 * logarithm
    return degree_floor_exponent - (2 * vertices - 1) * logarithm + large_vertex_gain(vertices)


def center_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose T_M after removing M**2*(M-1)!*(2*log(2))**(M-1)."""
    _integer(vertices, 4, "vertices")
    _integer(bits, 32, "bits")
    if vertices >= 258:
        return corrected_normalized_enclosure(vertices, bits=bits)
    return leading_normalized_enclosure(vertices, bits=bits)


def constant_relative_error(vertices: int) -> Fraction:
    """Bound |C_M/T_M-1| using gap-free old and corrected center branches."""
    _integer(vertices, 4, "vertices")
    if vertices >= 258:
        return second_order_radius(vertices)
    if vertices >= 162:
        return first_order_radius(vertices)
    return local_constant_radius(vertices)


def _affine_target(base: Fraction, coefficient: Fraction, exponent: int, target: Fraction) -> bool:
    """Compare base+coefficient/2**exponent<=target with a bounded shift."""
    if target <= base:
        return False
    gap = target - base
    left = coefficient.numerator * gap.denominator
    right = coefficient.denominator * gap.numerator
    difference = left.bit_length() - right.bit_length()
    if exponent != difference:
        return exponent > difference
    return left <= right << exponent


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "tools/risk_research/vertex_composite_asymptotic.py",
        "tools/risk_research/vertex_joint_growth_refined.py",
        "tools/risk_research/vertex_large_dimension.py",
        "tools/risk_research/vertex_large_dimension_refined.py",
        "tools/risk_research/vertex_large_dimension_second_order.py",
        "tools/risk_research/vertex_refined_asymptotic.py",
        "tools/risk_research/vertex_general_asymptotic.py",
        "tools/risk_research/vertex_general.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        PROOF,
        "docs/advanced/skeleton_risk_vertex_joint_growth_refined.rst",
        "docs/advanced/skeleton_risk_vertex_joint_growth.rst",
        "docs/advanced/skeleton_risk_vertex_refined_asymptotics.rst",
        "docs/advanced/skeleton_risk_vertex_general_asymptotics.rst",
        "docs/advanced/skeleton_risk_vertex_large_dimension.rst",
        "docs/advanced/skeleton_risk_vertex_large_dimension_refined.rst",
        "docs/advanced/skeleton_risk_vertex_large_dimension_second_order.rst",
    )
    return {name: (root / name).read_bytes() for name in names}


def build_record(
    vertices: int,
    degree_floor_exponent: int,
    *,
    bits: int = 128,
    target: Fraction | None = None,
) -> dict[str, Any]:
    """Build a source-bound full-block certificate with separated budgets."""
    sources = _sources()
    exponent = degree_error_exponent(vertices, degree_floor_exponent)
    _integer(bits, 32, "bits")
    if target is not None and (not isinstance(target, Fraction) or target <= 0):
        raise ValueError("target must be a strictly positive Fraction")
    lower, upper = center_enclosure(vertices, bits=bits)
    center = (lower + upper) / 2
    numerical = (upper - lower) / (upper + lower)
    constant = constant_relative_error(vertices)
    base = (1 + numerical) * (1 + constant) - 1
    outside, logarithm = vertices - 1, (degree_floor_exponent + 1).bit_length()
    method = "second_order_corrected" if vertices >= 258 else ("refined_leading" if vertices >= 162 else "local_leading")
    record = {
        "schema": SCHEMA,
        "proof": PROOF,
        "proof_status": "proved_uniform_domain_and_computable_center",
        "vertices": vertices,
        "degree_floor_exponent": degree_floor_exponent,
        "bits": bits,
        "relative_target": None if target is None else _fraction(target),
        "degree_block": {
            "base": 2,
            "lower_exponent": degree_floor_exponent,
            "exclusive_upper_exponent": degree_floor_exponent + 1,
        },
        "allowed_vertices": {"minimum": 4, "maximum": maximum_vertices(degree_floor_exponent)},
        "logarithm_ceiling": logarithm,
        "degree_proof_branch": "critical_squared_log" if vertices == 4 else "single_log",
        "degree_budget_margin": degree_floor_exponent
        - (4 * (37 + 2 * logarithm) if vertices == 4 else 4 * (2 * outside + 1) * logarithm),
        "quadratic_base_margin": (8 * logarithm) ** 2 - 512,
        "large_vertex_gain": large_vertex_gain(vertices),
        "degree_error_relative_to_C_M": {"numerator": 1, "denominator_power_of_two": exponent},
        "degree_pointwise_bound": {
            "degree_power_numerator": -3,
            "degree_power_denominator": 4,
            "additional_denominator_power_of_two": large_vertex_gain(vertices),
        },
        "center": {
            "method": method,
            "removed_scale": "M**2*(M-1)!*(2*log(2))**(M-1)",
            "exact_formula": "B_M*(1+beta/(M-2))" if vertices >= 258 else "B_M",
            "normalized_lower": _fraction(lower),
            "normalized_upper": _fraction(upper),
            "normalized_rational_midpoint": _fraction(center),
        },
        "constant_error_relative_to_exact_center": _fraction(constant),
        "elementary_error_relative_to_rational_midpoint": _fraction(numerical),
        "analytic_composite_relative_to_exact_center": {
            "base": _fraction(constant),
            "tail_coefficient": _fraction(1 + constant),
            "tail_denominator_power_of_two": exponent,
        },
        "composite_relative_to_computable_center": {
            "base": _fraction(base),
            "tail_coefficient": _fraction(1 + base),
            "tail_denominator_power_of_two": exponent,
        },
        "target_met_on_entire_block": None if target is None else _affine_target(base, 1 + base, exponent, target),
        "previous_joint_ceiling": 1 + degree_floor_exponent // (12 * logarithm) if degree_floor_exponent >= 384 else None,
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() for name, data in sources.items()},
        "scope": (
            "all degrees in this dyadic block; center, constant error, "
            "finite-degree error and numerical interval width are distinct"
        ),
    }
    if _sources() != sources:
        raise RuntimeError("composite evaluator sources changed during calculation")
    return record


def _saved_target(value: object) -> Fraction | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("invalid saved relative target")
    numerator, denominator = value["numerator"], value["denominator"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise ValueError("saved target fields must be decimal strings")
    result = Fraction(_decimal_integer(numerator), _decimal_integer(denominator))
    if result <= 0 or _fraction(result) != value:
        raise ValueError("saved relative target is not canonical and positive")
    return result


def verify_record(record: object) -> None:
    """Restore saved options and reject any altered field or source."""
    if not isinstance(record, dict):
        raise ValueError("saved composite certificate must be an object")
    try:
        expected = build_record(
            record["vertices"],
            record["degree_floor_exponent"],
            bits=record["bits"],
            target=_saved_target(record["relative_target"]),
        )
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError("invalid composite certificate options") from error
    if _canonical(record) != _canonical(expected):
        raise ValueError("saved composite certificate differs from recomputation")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a compressed certificate without changing input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=_parse_integer)
    parser.add_argument("--degree-floor-exponent", type=_parse_integer)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--relative-target", type=_target)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify is not None:
            if any(
                value is not None
                for value in (
                    arguments.vertices,
                    arguments.degree_floor_exponent,
                    arguments.bits,
                    arguments.relative_target,
                    arguments.output,
                )
            ):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(
                json.loads(arguments.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique, parse_int=_parse_integer)
            )
            print("verified computable composite vertex certificate")
            return 0
        if arguments.vertices is None or arguments.degree_floor_exponent is None or arguments.output is None:
            raise ValueError("generation requires --vertices, --degree-floor-exponent and --output")
        sources = _sources()
        _protect_output(arguments.output, sources)
        output = arguments.output.parent.resolve() / arguments.output.name
        _protect_output(output, sources)
        record = build_record(
            arguments.vertices,
            arguments.degree_floor_exponent,
            bits=128 if arguments.bits is None else arguments.bits,
            target=arguments.relative_target,
        )
        _protect_output(output, sources)
        if _sources() != sources:
            raise RuntimeError("composite evaluator sources changed before output")
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ZeroDivisionError, RuntimeError) as error:
        parser.error(str(error))
    print(_canonical(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
