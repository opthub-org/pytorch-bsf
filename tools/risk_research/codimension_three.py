"""Evaluate proved boundary-lift weights and a five-vertex edge upper bound.

The transfer inequality uses a trace-correct inclusion-exclusion lift.  The
five-vertex bound is an inequality, not a new exact coefficient formula or
an asymptotic equivalent.  All evaluation uses standard-library fractions.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, prod
from pathlib import Path

from .exact import atomic_json, fraction_record

EDGE_BOUND_NUMERATOR = 30000


def _integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def transfer_squared_weights(face_size: int, outside: int, degree: int) -> tuple[Fraction, ...]:
    """Return weights w_r with sqrt(a_M) <= sum_r sqrt(w_r*a_(s+r)).

    Here r ranges from zero through outside-1, M=face_size+outside and the
    coefficients on the right have the same source-face size.  Their top
    coefficient at r=0 is binom(degree-1,face_size-1), including zero on
    inactive degrees.  No coefficient or square root is evaluated here.
    """
    _integer(face_size, "face_size", 2)
    _integer(outside, "outside", 1)
    _integer(degree, "degree", 1)
    vertices = face_size + outside
    allocation = comb(vertices, face_size) ** 2
    result = []
    for selected in range(outside):
        smaller = face_size + selected
        radius = Fraction(prod(range(smaller, vertices)), prod(2 * degree + h for h in range(smaller, vertices)))
        result.append(allocation * Fraction(comb(outside, selected) ** 2, comb(smaller, face_size) ** 2) * radius)
    return tuple(result)


def five_vertex_edge_upper_bound(degree: int) -> Fraction:
    """Return a proved rational upper bound for a_1(5,degree).

    The result is zero at degree one.  At every integer degree>=2 the
    coefficient is strictly smaller than both candidates in this minimum.
    The operation count is independent of the degree, excluding integer
    bit complexity.  This does not evaluate the exact edge coefficient.
    """
    _integer(degree, "degree", 1)
    if degree == 1:
        return Fraction(0)
    d = degree
    rational = 300 * (
        Fraction(24 * (d - 1), (2 * d + 2) * (2 * d + 3) * (2 * d + 4))
        + Fraction(108, (2 * d + 3) * (2 * d + 4))
        + Fraction(864, (d + 4) * (d + 5) * (2 * d + 4))
    )
    return min(Fraction(EDGE_BOUND_NUMERATOR, d * d), rational)


def main(argv: list[str] | None = None) -> int:
    """Write exact rational bounds without Sage, SymPy or floating point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        _integer(arguments.max_degree, "max_degree", 1)
        source = Path(__file__).read_bytes()
        records = {
            str(degree): fraction_record(five_vertex_edge_upper_bound(degree)) for degree in range(1, arguments.max_degree + 1)
        }
        if Path(__file__).read_bytes() != source:
            raise RuntimeError("bound source changed during evaluation")
        report = {
            "schema": "skeleton-risk-codimension-three-bound-v1",
            "proof_status": "proved_analytic_upper_bound",
            "vertices": 5,
            "face_size": 2,
            "active_domain_start": 2,
            "inactive_degrees": [1],
            "strict_on_active_domain": True,
            "power_bound": {"numerator": EDGE_BOUND_NUMERATOR, "degree_power": -2},
            "bounds": records,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "proof": "docs/advanced/skeleton_risk_codimension_three.rst",
        }
        atomic_json(arguments.output, report)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
