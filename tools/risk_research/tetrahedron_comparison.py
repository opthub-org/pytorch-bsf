"""Certify the tetrahedron optimal-risk comparison at every positive degree.

The exact finite comparisons cover D=1,...,624. The analytic proof in
``skeleton_risk_tetrahedron_comparison.rst`` covers D>=625. This module uses
the proved vertex, edge, facet and top-dimensional coefficient formulas.
"""

import argparse
import hashlib
import math
from fractions import Fraction
from pathlib import Path

from .comparison import radical_comparison
from .edge_closed import tetrahedron_edge_coefficients
from .exact import atomic_json, fraction_record
from .vertex_single import tetrahedron_vertex_coefficient

TAIL_START = 625


def tetrahedron_coefficients(maximum: int) -> dict[int, tuple[Fraction, ...]]:
    """Return all four exact coefficients at degrees 1 through ``maximum``."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    edges = tetrahedron_edge_coefficients(maximum)
    harmonics = [Fraction(0)]
    for index in range(1, 2 * maximum + 6):
        harmonics.append(harmonics[-1] + Fraction(1, index))
    result: dict[int, tuple[Fraction, ...]] = {}
    for degree in range(1, maximum + 1):
        facet = (
            Fraction(0)
            if degree < 3
            else Fraction(24, degree + 2)
            * ((degree - 1) * harmonics[degree - 1] - 1 - (degree + 5) * (harmonics[2 * degree + 3] - harmonics[degree + 5]))
        )
        result[degree] = (
            tetrahedron_vertex_coefficient(degree),
            edges[degree],
            facet,
            Fraction(math.comb(degree - 1, 3)) if degree >= 4 else Fraction(0),
        )
    return result


def certify_tetrahedron_comparison(*, max_bits: int = 4096) -> dict[str, object]:
    """Certify the finite range and record the separate all-degree tail proof.

    Strict integer square-root enclosures determine every finite sign. A
    possible equality or insufficient precision raises ``ArithmeticError``.
    The documented analytic bounds cover the remaining infinite range.
    """
    coefficients = tetrahedron_coefficients(TAIL_START - 1)
    points = []
    losses = []
    for degree, values in coefficients.items():
        point = radical_comparison(values, math.comb(degree + 3, 3), max_bits=max_bits)
        point["degree"] = degree
        point["coefficients"] = [fraction_record(value) for value in values]
        points.append(point)
        if point["decision"] == "all_at_once_wins":
            losses.append(degree)
    if losses != [3, 4, 5, 6, 7]:
        raise ArithmeticError(f"finite comparisons disagree with the stated classification: {losses}")
    tail_bound = Fraction(19, 10) + Fraction(361, 62500)
    if tail_bound >= 2:
        raise ArithmeticError("analytic tail constant is not below two")
    return {
        "schema": "skeleton-risk-tetrahedron-comparison-v1",
        "proof_status": "strict_comparisons_for_all_integer_degrees_ge_1",
        "vertices": 4,
        "all_at_once_wins_degrees": losses,
        "finite_comparisons": points,
        "tail": {
            "from_degree": TAIL_START,
            "proof": "docs/advanced/skeleton_risk_tetrahedron_comparison.rst",
            "coefficient_bounds": ["a0<=72/D^3", "a1<108", "a2<=48*sqrt(D)"],
            "top_coefficient": "binom(D-1,3)",
            "all_at_once_minus_top": "2*D^2+2",
            "risk_excess_over_top_divided_by_degree_squared_upper_bound": fraction_record(tail_bound),
            "strict_margin_below_two": fraction_record(2 - tail_bound),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write a source-bound certificate atomically after all checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bits", type=int, default=4096)
    arguments = parser.parse_args(argv)
    names = ["tetrahedron_comparison.py", "comparison.py", "edge_closed.py", "vertex_single.py", "exact.py"]
    paths = [Path(__file__).with_name(name) for name in names]
    sources = {path.name: path.read_bytes() for path in paths}
    try:
        report = certify_tetrahedron_comparison(max_bits=arguments.max_bits)
        if any(path.read_bytes() != sources[path.name] for path in paths):
            raise RuntimeError("comparison sources changed during certification")
        report["source_sha256"] = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
        atomic_json(arguments.output, report)
    except (ValueError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
