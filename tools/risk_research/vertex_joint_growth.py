"""Certify a simultaneous growing-vertex and growing-degree risk asymptotic.

For D>=2**1024, set N=floor(log2(D)), ell=ceil(log2(N+2)). If
4<=M<=1+floor(N/(16*ell)), the relative vertex error is at most D**(-3/4).
Certificates describe a full degree block without constructing 2**N.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .exact import atomic_json

MINIMUM_DEGREE_FLOOR = 1024
HASH_POLICY = "sha256-crlf-to-lf-v1"


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def maximum_vertices(degree_floor_exponent: int) -> int:
    """Return the theorem's vertex ceiling for 2**N<=D<2**(N+1)."""
    _integer(degree_floor_exponent, MINIMUM_DEGREE_FLOOR, "degree_floor_exponent")
    logarithm_ceiling = (degree_floor_exponent + 1).bit_length()
    return 1 + degree_floor_exponent // (16 * logarithm_ceiling)


def _raw_sources() -> dict[str, bytes]:
    directory = Path(__file__).parent
    return {
        name: (directory / name).read_bytes()
        for name in (Path(__file__).name, "vertex_general_asymptotic.py", "vertex_general.py", "exact.py")
    }


def _source_hashes(sources: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() for name, data in sources.items()}


def _payload(vertices: int, degree_floor_exponent: int, source_hashes: dict[str, str]) -> dict[str, Any]:
    _integer(vertices, 4, "vertices")
    ceiling = maximum_vertices(degree_floor_exponent)
    if vertices > ceiling:
        raise ValueError(f"vertices exceeds the certified ceiling {ceiling}")
    n, q = degree_floor_exponent, vertices - 1
    logarithm_ceiling = (n + 1).bit_length()
    return {
        "schema": "skeleton-risk-joint-vertex-growth-v1",
        "vertices": vertices,
        "degree_floor_exponent": n,
        "degree_block": {"base": 2, "lower_exponent": n, "exclusive_upper_exponent": n + 1},
        "outside_coordinates": q,
        "logarithm_ceiling": logarithm_ceiling,
        "allowed_vertices": {"minimum": 4, "maximum": ceiling},
        "margins": {
            "binary_exponent_budget": n - 16 * logarithm_ceiling * q,
            "coefficient_base": n + 2 - 512,
            "outside_order": n + 2 - q,
        },
        "budget_base_multipliers": {
            "phi": 1,
            "coefficient": 4,
            "cross": 2,
            "difference": 10,
            "diagonal": 96,
            "remainder_ratio": 512,
        },
        "relative_error_pointwise": {"degree_power_numerator": -3, "degree_power_denominator": 4},
        "relative_error_block_upper": {"numerator": 1, "denominator_power_of_two": 3 * n // 4},
        "source_hash_policy": HASH_POLICY,
        "source_sha256": source_hashes,
        "proof": "docs/advanced/skeleton_risk_vertex_joint_growth.rst",
        "scope": "all integer degrees in the stated block, for the stated integer vertex count",
    }


def build_certificate(vertices: int, degree_floor_exponent: int) -> dict[str, Any]:
    """Build exact integer witnesses; omit exponentially large powers of two."""
    sources = _raw_sources()
    document = _payload(vertices, degree_floor_exponent, _source_hashes(sources))
    if _raw_sources() != sources:
        raise RuntimeError("a joint-growth source changed during calculation")
    return document


def check_certificate(document: object) -> None:
    """Recompute every domain, margin, constant and portable source binding.

    Canonical JSON comparison also rejects bool/float replacements of exact
    integers, unexpected fields and altered compressed error exponents.
    """
    if type(document) is not dict:
        raise ValueError("certificate must be a JSON object")
    sources = _raw_sources()
    if "vertices" not in document or "degree_floor_exponent" not in document:
        raise ValueError("certificate is missing its vertex or degree domain")
    expected = _payload(document["vertices"], document["degree_floor_exponent"], _source_hashes(sources))
    try:
        actual_text = json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":"))
        expected_text = json.dumps(expected, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("certificate contains an invalid JSON value") from error
    if actual_text != expected_text:
        raise ValueError("certificate does not match its exact domain, witnesses or sources")
    if _raw_sources() != sources:
        raise RuntimeError("a joint-growth source changed during verification")


def main(argv: list[str] | None = None) -> int:
    """Write or verify a source-bound certificate for a complete degree block."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--degree-floor-exponent", type=int)
    action.add_argument("--check", type=Path)
    parser.add_argument("--vertices", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    sources = _raw_sources()
    try:
        if arguments.check is not None:
            if arguments.vertices is not None or arguments.output is not None:
                raise ValueError("--check does not accept --vertices or --output")
            document = json.loads(arguments.check.read_text())
            check_certificate(document)
            result: dict[str, object] = {"valid": True, "certificate": str(arguments.check)}
        else:
            if arguments.output is None:
                raise ValueError("--output is required when building a certificate")
            document = build_certificate(arguments.vertices, arguments.degree_floor_exponent)
            check_certificate(document)
            if _raw_sources() != sources:
                raise RuntimeError("a joint-growth source changed before saving")
            atomic_json(arguments.output, document)
            result = document
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
