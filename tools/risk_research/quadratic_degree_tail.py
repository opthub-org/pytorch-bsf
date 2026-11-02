"""Certify inductive victory for every M>=4 and D>=8*M**2.

This specializes a proved uniform sufficient region, not an exact
crossover or a simultaneous asymptotic expansion. Only standard-library
arithmetic is needed; generating a certificate never evaluates a risk.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from .exact import atomic_json
from .fixed_vertex_tail import CoefficientMajorant, _decimal, _fraction, _integer

SCHEMA = "skeleton-risk-quadratic-degree-tail-v1"
SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"
NORMALIZED_MARGIN = Fraction(239, 3200)


def degree_threshold(vertices: int) -> int:
    """Return 8*M**2, sufficient simultaneously for every integer M>=4."""
    _integer(vertices, 4, "vertices")
    return 8 * vertices**2


def coefficient_majorants(vertices: int) -> tuple[CoefficientMajorant, ...]:
    """Bound each lower skeleton using positive diagonals and a sharp facet."""
    _integer(vertices, 4, "vertices")
    bounds = []
    for size in range(1, vertices - 1):
        outside = vertices - size
        coefficient = Fraction(
            math.comb(vertices, size) ** 2 * math.factorial(vertices - 1) * 2**outside,
            math.factorial(size - 1) ** 2,
        )
        bounds.append(CoefficientMajorant(size, coefficient, size - outside - 1))
    size = vertices - 1
    facet = Fraction(size * vertices**2, 2 * math.factorial(size - 2))
    bounds.append(CoefficientMajorant(size, facet, size - 3, True))
    return tuple(bounds)


def lower_skeleton_upper_bounds(vertices: int, degree: int) -> tuple[Fraction, ...]:
    """Return all-degree rational majorants, with exact zero on inactive faces.

    The facet logarithm is bounded by 1+ceil(log2 D), computed with integer
    bit lengths. These bounds also apply below the sufficient threshold;
    they do not decide victory there.
    """
    _integer(degree, 1, "degree")
    logarithm = 1 + (degree - 1).bit_length()
    return tuple(
        bound.coefficient * Fraction(degree) ** bound.exponent * (logarithm if bound.logarithm else 1)
        if degree >= bound.size
        else Fraction(0)
        for bound in coefficient_majorants(vertices)
    )


def normalized_risk_margin(vertices: int, degree: int) -> Fraction:
    """Lower-bound (R_all-R_ind)/(D**(M-1)/(M-1)!) in the proved region."""
    threshold = degree_threshold(vertices)
    _integer(degree, threshold, "degree")
    return NORMALIZED_MARGIN * Fraction(vertices * (vertices - 1), degree)


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "tools/risk_research/quadratic_degree_tail.py",
        "tools/risk_research/fixed_vertex_tail.py",
        "tools/risk_research/exact.py",
        "docs/advanced/skeleton_risk_quadratic_degree_tail.rst",
        "docs/advanced/skeleton_risk_fixed_vertex_tail.rst",
        "docs/advanced/skeleton_risk_codimension_three_exact.rst",
        "docs/advanced/skeleton_risk_vertex_general.rst",
        "docs/advanced/skeleton_risk_theory.rst",
    )
    return {name: (root / name).read_bytes() for name in names}


def certify_vertices(vertices: int) -> dict[str, object]:
    """Record a specialization of the all-M quadratic sufficient theorem."""
    threshold = degree_threshold(vertices)
    p = Fraction(vertices * (vertices - 1), threshold)
    return {
        "schema": SCHEMA,
        "vertices": vertices,
        "from_degree": _decimal(threshold),
        "decision": "inductive_wins",
        "proof_status": "specialization_of_proved_uniform_quadratic_degree_region",
        "proof": "docs/advanced/skeleton_risk_quadratic_degree_tail.rst",
        "all_vertex_domain": "integers M>=4",
        "all_degree_domain": "integers D>=8*M**2",
        "lower_skeleton_majorants": [
            {
                "size": bound.size,
                "coefficient": _fraction(bound.coefficient),
                "degree_exponent": bound.exponent,
                "logarithm": bound.logarithm,
            }
            for bound in coefficient_majorants(vertices)
        ],
        "p_upper_bound": _fraction(Fraction(1, 8)),
        "lower_skeleton_root_sum_over_p_sqrt_b0": _fraction(Fraction(9, 20)),
        "risk_excess_over_p_b0_upper_bound": _fraction(Fraction(2961, 3200)),
        "strict_margin_over_p_b0_lower_bound": _fraction(NORMALIZED_MARGIN),
        "threshold_p": _fraction(p),
        "threshold_normalized_margin": _fraction(normalized_risk_margin(vertices, threshold)),
        "source_hash_policy": SOURCE_HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest() for name, value in _sources().items()
        },
    }


def verify_certificate(record: object) -> None:
    """Reject modified domains, constants, JSON types, sources or extra fields."""
    if not isinstance(record, dict) or type(record.get("vertices")) is not int:
        raise ValueError("certificate must contain an integer vertex count")
    expected = certify_vertices(record["vertices"])
    try:
        matches = json.dumps(record, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("certificate does not match the quadratic theorem and sources") from error
    if not matches:
        raise ValueError("certificate does not match the quadratic theorem and sources")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are not allowed")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    """Create an atomic proof certificate or verify a saved certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--vertices", type=int)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        before = _sources()
        if arguments.vertices is not None:
            if arguments.output is None:
                parser.error("--vertices requires --output")
            root = Path(__file__).resolve().parents[2]
            if arguments.output.resolve() in {(root / name).resolve() for name in before}:
                raise ValueError("output must not overwrite evaluator or proof sources")
            report = certify_vertices(arguments.vertices)
            if _sources() != before:
                raise RuntimeError("proof sources changed during certification")
            atomic_json(arguments.output, report)
        else:
            if arguments.output is not None:
                parser.error("--output is only supported with --vertices")
            verify_certificate(json.loads(arguments.verify.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs))
            if _sources() != before:
                raise RuntimeError("proof sources changed during verification")
            print("verified quadratic-degree-region certificate")
    except (OSError, ValueError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
