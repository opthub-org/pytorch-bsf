"""Certify an explicit winning degree tail at every fixed vertex count.

For every integer M>=4 the analytic theorem constructs H(M) such that
optimal inductive asymptotic risk is smaller for all degrees D>=H(M).
The threshold is deliberately conservative; no individual risks or
degree-dependent polynomial tables are needed to construct it.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .exact import atomic_json

SCHEMA = "skeleton-risk-fixed-vertex-tail-v1"
SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _ceil_root(value: Fraction, order: int) -> int:
    """Round a positive rational root upward using integer comparisons."""
    lower, upper = 0, 1
    while upper**order * value.denominator < value.numerator:
        upper *= 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if middle**order * value.denominator >= value.numerator:
            upper = middle
        else:
            lower = middle
    return upper


def transfer_constant(size: int, outside: int) -> Fraction:
    """Return C with a_(s-1)(s+q,D)<=C D**(s-q-1) for D>=s.

    This is a coarse bound obtained from the trace-correct boundary
    transfer and Cauchy--Schwarz over all proper outside subsets.
    At q=0 it is the top-dimensional binomial bound.
    """
    _integer(size, 2, "size")
    _integer(outside, 0, "outside")
    normalized = [Fraction(1, math.factorial(size - 1))]
    for q in range(1, outside + 1):
        normalized.append(
            (2**q - 1)
            * sum(
                (
                    math.comb(q, r)
                    * Fraction(math.factorial(size + q - 1), 2 ** (q - r) * math.factorial(size + r - 1))
                    * normalized[r]
                    for r in range(q)
                ),
                Fraction(0),
            )
        )
    return math.comb(size + outside, size) ** 2 * normalized[-1]


@dataclass(frozen=True)
class CoefficientMajorant:
    """An all-active-degree bound C*D**exponent*(1+log D)**logarithm."""

    size: int
    coefficient: Fraction
    exponent: int
    logarithm: bool = False


def coefficient_majorants(vertices: int) -> tuple[CoefficientMajorant, ...]:
    """Bound every lower skeleton under the fixed-vertex theorem."""
    _integer(vertices, 4, "vertices")
    q = vertices - 1
    bounds = [CoefficientMajorant(1, Fraction(vertices**2 * math.factorial(q) * 2**q), -q)]
    bounds.extend(
        CoefficientMajorant(size, transfer_constant(size, vertices - size), 2 * size - vertices - 1)
        for size in range(2, vertices - 1)
    )
    size = vertices - 1
    bounds.append(CoefficientMajorant(size, Fraction(size * vertices**2, 2 * math.factorial(size - 2)), size - 3, True))
    return tuple(bounds)


def lower_skeleton_upper_bounds(vertices: int, degree: int) -> tuple[Fraction, ...]:
    """Return rational bounds, using 1+ceil(log2 D) for 1+log D.

    Each inactive source face is returned as exact zero. The bounds
    do not include the top-dimensional coefficient.
    """
    _integer(degree, 1, "degree")
    logarithm = 1 + (degree - 1).bit_length()
    return tuple(
        bound.coefficient * Fraction(degree) ** bound.exponent * (logarithm if bound.logarithm else 1)
        if degree >= bound.size
        else Fraction(0)
        for bound in coefficient_majorants(vertices)
    )


def _threshold_data(vertices: int) -> tuple[int, int, int, Fraction, Fraction]:
    bounds = coefficient_majorants(vertices)
    root_sum = sum(_ceil_root(bound.coefficient, 2) for bound in bounds)
    gap = Fraction(vertices, math.factorial(vertices - 2))
    linear = Fraction(12 * root_sum, gap)
    t = max(1, -(-linear.numerator // linear.denominator), _ceil_root(Fraction(8 * root_sum**2, gap), 6))
    threshold = max(vertices, t**4)
    upper = Fraction(3 * root_sum, t) + Fraction(2 * root_sum**2, t**6)
    if upper >= gap:
        raise ArithmeticError("the analytic degree tail has no positive margin")
    return threshold, root_sum, t, upper, gap - upper


def degree_threshold(vertices: int) -> int:
    """Return H(M), a sufficient degree threshold for every fixed M>=4."""
    return _threshold_data(vertices)[0]


def _decimal(value: int) -> str:
    """Serialize large integers without altering Python's global digit limit."""
    if value == 0:
        return "0"
    sign, value = ("-", -value) if value < 0 else ("", value)
    chunks = []
    while value:
        value, remainder = divmod(value, 10**500)
        chunks.append(f"{remainder:0500d}" if value else str(remainder))
    return sign + "".join(reversed(chunks))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "tools/risk_research/fixed_vertex_tail.py",
        "tools/risk_research/exact.py",
        "docs/advanced/skeleton_risk_fixed_vertex_tail.rst",
        "docs/advanced/skeleton_risk_codimension_three.rst",
        "docs/advanced/skeleton_risk_vertex_general.rst",
        "docs/advanced/skeleton_risk_theory.rst",
    ]
    return {name: (root / name).read_bytes() for name in names}


def certify_vertices(vertices: int) -> dict[str, object]:
    """Specialize the general theorem, recording exact constants and margin."""
    threshold, root_sum, t, upper, margin = _threshold_data(vertices)
    return {
        "schema": SCHEMA,
        "vertices": vertices,
        "from_degree": _decimal(threshold),
        "decision": "inductive_wins",
        "proof_status": "specialization_of_proved_fixed_vertex_all_degree_tail",
        "proof": "docs/advanced/skeleton_risk_fixed_vertex_tail.rst",
        "lower_skeleton_majorants": [
            {
                "size": bound.size,
                "coefficient": _fraction(bound.coefficient),
                "degree_exponent": bound.exponent,
                "logarithm": bound.logarithm,
            }
            for bound in coefficient_majorants(vertices)
        ],
        "integer_root_sum_majorant": _decimal(root_sum),
        "fourth_root_threshold": _decimal(t),
        "normalized_risk_excess_upper_bound": _fraction(upper),
        "normalized_all_at_once_gap_lower_bound": _fraction(Fraction(vertices, math.factorial(vertices - 2))),
        "strict_normalized_margin": _fraction(margin),
        "source_hash_policy": SOURCE_HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in _sources().items()
        },
    }


def verify_certificate(record: object) -> None:
    """Reject altered fields, domains, constants, thresholds and sources."""
    if not isinstance(record, dict) or type(record.get("vertices")) is not int:
        raise ValueError("certificate must have an integer vertex count")
    if record != certify_vertices(record["vertices"]):
        raise ValueError("certificate does not match the proof specialization and sources")


def main(argv: list[str] | None = None) -> int:
    """Create an atomic proof specialization or verify a saved one."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--vertices", type=int)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.vertices is not None:
            if arguments.output is None:
                parser.error("--vertices requires --output")
            sources = _sources()
            report = certify_vertices(arguments.vertices)
            if _sources() != sources:
                raise RuntimeError("proof sources changed during certification")
            atomic_json(arguments.output, report)
        else:
            if arguments.output is not None:
                parser.error("--output is only supported with --vertices")
            verify_certificate(json.loads(arguments.verify.read_text(encoding="utf-8")))
            print("verified fixed-vertex degree-tail certificate")
    except (OSError, ValueError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
