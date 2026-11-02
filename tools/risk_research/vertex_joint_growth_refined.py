"""Certify a broader simultaneous vertex-count and degree approximation.

For N=floor(log2(D))>=384 and ell=ceil(log2(N+2)), the range
4<=M<=1+floor(N/(12*ell)) has relative error at most D**(-3/4).
The refined large-M lower bound supplies an additional dyadic gain.
Only compressed integer witnesses are built, never D or a risk value.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MINIMUM_DEGREE_FLOOR = 384
SCHEMA = "skeleton-risk-refined-joint-vertex-growth-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_vertex_joint_growth_refined.rst"


def _integer(value: object, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def maximum_vertices(degree_floor_exponent: int) -> int:
    """Return the proved vertex ceiling for a full block 2**N<=D<2**(N+1)."""
    exponent = _integer(degree_floor_exponent, MINIMUM_DEGREE_FLOOR, "degree_floor_exponent")
    return 1 + exponent // (12 * (exponent + 1).bit_length())


def large_vertex_gain(vertices: int) -> int:
    """Return the extra denominator exponent supplied by the large-M bound."""
    outside = _integer(vertices, 4, "vertices") - 1
    return 0 if outside <= 160 else outside // 3 + 2 - (outside - 1).bit_length()


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "tools/risk_research/vertex_joint_growth_refined.py",
        "tools/risk_research/vertex_joint_growth.py",
        "tools/risk_research/vertex_general_asymptotic.py",
        "tools/risk_research/vertex_general.py",
        PROOF,
        "docs/advanced/skeleton_risk_vertex_joint_growth.rst",
        "docs/advanced/skeleton_risk_vertex_general_asymptotics.rst",
        "docs/advanced/skeleton_risk_vertex_large_dimension_refined.rst",
        "docs/advanced/skeleton_risk_vertex_large_dimension.rst",
    )
    return {name: (root / name).read_bytes() for name in names}


def _payload(vertices: int, degree_floor_exponent: int, sources: dict[str, bytes]) -> dict[str, Any]:
    vertices = _integer(vertices, 4, "vertices")
    ceiling = maximum_vertices(degree_floor_exponent)
    if vertices > ceiling:
        raise ValueError("vertices exceeds the certified ceiling")
    exponent, outside = degree_floor_exponent, vertices - 1
    logarithm = (exponent + 1).bit_length()
    gain = large_vertex_gain(vertices)
    large = outside >= 161
    return {
        "schema": SCHEMA,
        "proof_status": "proved_uniform_growing_vertex_relative_error",
        "proof": PROOF,
        "vertices": vertices,
        "outside_coordinates": outside,
        "degree_floor_exponent": exponent,
        "degree_block": {"base": 2, "lower_exponent": exponent, "exclusive_upper_exponent": exponent + 1},
        "logarithm_ceiling": logarithm,
        "allowed_vertices": {"minimum": 4, "maximum": ceiling},
        "constant_lower_method": "refined_large_vertex" if large else "positive_arc",
        "constant_lower_after_dividing_by_M_squared_q_factorial": {
            "numerator": 1 if large else outside,
            "denominator": 1 if large else 4,
            "power_of_two": outside // 3 if large else 0,
        },
        "margins": {
            "binary_exponent_budget": exponent - 12 * logarithm * outside,
            "quadratic_base_budget": (12 * logarithm) ** 2 - 512,
            "outside_order": exponent + 2 - outside,
        },
        "large_vertex_gain": gain,
        "relative_error_pointwise": {
            "degree_power_numerator": -3,
            "degree_power_denominator": 4,
            "additional_denominator_power_of_two": gain,
        },
        "relative_error_block_upper": {"numerator": 1, "denominator_power_of_two": 3 * exponent // 4 + gain},
        "budget_block_upper": {
            "numerator": 1,
            "denominator_power_of_two": exponent - 3 * outside * logarithm + gain,
        },
        "old_domain_included": exponent >= 1024,
        "old_allowed_vertices_maximum": 1 + exponent // (16 * logarithm) if exponent >= 1024 else None,
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() for name, data in sources.items()},
        "scope": "all integer degrees in the stated block; the vertex approximation is separate from risk comparison",
    }


def build_certificate(vertices: int, degree_floor_exponent: int) -> dict[str, Any]:
    """Specialize the proved bound with compressed exact domain witnesses."""
    sources = _sources()
    report = _payload(vertices, degree_floor_exponent, sources)
    if _sources() != sources:
        raise RuntimeError("joint-growth sources changed during calculation")
    return report


def _decimal(value: int) -> str:
    if not value:
        return "0"
    sign, value = ("-", -value) if value < 0 else ("", value)
    parts = []
    while value:
        value, remainder = divmod(value, 10**500)
        parts.append(f"{remainder:0500d}" if value else str(remainder))
    return sign + "".join(reversed(parts))


def _parse_integer(text: str) -> int:
    text = text.strip()
    sign = -1 if text.startswith("-") else 1
    if text.startswith(("+", "-")):
        text = text[1:]
    if not all(part and all("0" <= digit <= "9" for digit in part) for part in text.split("_")):
        raise ValueError("invalid integer")
    text, result = text.replace("_", ""), 0
    for start in range(0, len(text), 500):
        chunk = text[start : start + 500]
        result = result * 10 ** len(chunk) + int(chunk)
    return sign * result


def _canonical(value: object) -> str:
    if type(value) is int:
        return _decimal(value)
    if value is None or type(value) in (str, bool, float):
        return json.dumps(value, allow_nan=False)
    if type(value) is list:
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    if type(value) is dict and all(type(key) is str for key in value):
        return "{" + ",".join(json.dumps(key) + ":" + _canonical(value[key]) for key in sorted(value)) + "}"
    raise ValueError("certificate contains an unsupported JSON value")


def verify_certificate(record: object) -> None:
    """Recompute every field, rejecting changed types, sources and domains."""
    if type(record) is not dict:
        raise ValueError("certificate must be a JSON object")
    vertices = _integer(record.get("vertices"), 4, "vertices")
    exponent = _integer(record.get("degree_floor_exponent"), MINIMUM_DEGREE_FLOOR, "degree_floor_exponent")
    sources = _sources()
    expected = _payload(vertices, exponent, sources)
    if _canonical(record) != _canonical(expected):
        raise ValueError("certificate does not match the exact domain, witnesses or sources")
    if _sources() != sources:
        raise RuntimeError("joint-growth sources changed during verification")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not overwrite evaluator or proof sources")


def atomic_json(path: Path, record: object) -> None:
    """Atomically save a canonical certificate without global integer limits."""
    text = _canonical(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Save or verify a source-bound certificate for an entire degree block."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--degree-floor-exponent", type=_parse_integer)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--vertices", type=_parse_integer)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        sources = _sources()
        if arguments.verify is not None:
            if arguments.vertices is not None or arguments.output is not None:
                raise ValueError("--verify does not accept generation overrides")
            record = json.loads(arguments.verify.read_bytes(), object_pairs_hook=_unique, parse_int=_parse_integer)
            verify_certificate(record)
            if _sources() != sources:
                raise RuntimeError("joint-growth sources changed during verification")
            print("verified refined joint-growth certificate")
        else:
            if arguments.vertices is None or arguments.output is None:
                raise ValueError("generation requires --vertices and --output")
            _protect_output(arguments.output, sources)
            record = build_certificate(arguments.vertices, arguments.degree_floor_exponent)
            output = arguments.output.parent.resolve() / arguments.output.name
            _protect_output(output, sources)
            if _sources() != sources:
                raise RuntimeError("joint-growth sources changed before saving")
            atomic_json(output, record)
    except (ValueError, RuntimeError, ArithmeticError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
