"""Certify the five-vertex optimal-risk comparison at every positive degree.

Exact finite comparisons cover degrees1..63. The separate analytic proof
in ``skeleton_risk_five_vertex_comparison.rst`` covers every degree>=64.
The default integer-kernel implementation uses only the standard library.
"""

import argparse
import hashlib
import math
from fractions import Fraction
from pathlib import Path

from .comparison import radical_comparison
from .edge_closed import codimension_two_coefficient
from .exact import atomic_json, fraction_record
from .research_exact import integer_kernel, scalar_coefficient
from .vertex_five import five_vertex_coefficient

TAIL_START = 64


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def five_vertex_coefficients(maximum: int, *, backend: str = "python") -> dict[int, tuple[Fraction, ...]]:
    """Return all five exact coefficients at degrees1..maximum.

    The edge uses the previously proved scalar formula with an integer
    kernel. The remaining entries use the harmonic vertex, single angular
    codimension-two, codimension-one, and top-dimensional formulas.
    """
    _positive_integer(maximum, "maximum")
    if backend not in ("python", "flint"):
        raise ValueError("backend must be python or flint")
    kernel = integer_kernel(3, maximum - 2, backend=backend) if maximum >= 2 else None
    result: dict[int, tuple[Fraction, ...]] = {}
    for degree in range(1, maximum + 1):
        edge = scalar_coefficient(5, degree, 1, kernel) if kernel is not None and degree >= 2 else Fraction(0)
        facet = 100 * sum(
            (Fraction(math.comb(k + 2, 2), (degree - 3 - k) * (degree + k + 8)) for k in range(max(0, degree - 3))),
            Fraction(0),
        )
        result[degree] = (
            five_vertex_coefficient(degree),
            edge,
            codimension_two_coefficient(3, degree),
            facet,
            Fraction(math.comb(degree - 1, 4)) if degree >= 5 else Fraction(0),
        )
    return result


def tail_certificate() -> dict[str, object]:
    """Record the exact rational margin in the separate all-degree proof."""
    start = TAIL_START
    # At D>=64, D^(3/8)>19/4 and sqrt(D)>=8.
    if start**3 * 4**8 <= 19**8:
        raise ArithmeticError("the fractional-power tail bound does not hold")
    q = Fraction(30, 19) + Fraction(174, start**2) + Fraction(70, start * 8) + Fraction(140, start**3)
    upper = Fraction(5, 12) * q + q * q / start
    margin = Fraction(5, 6) - upper
    if margin <= 0:
        raise ArithmeticError("the analytic tail is not separated")
    return {
        "from_degree": start,
        "proof": "docs/advanced/skeleton_risk_five_vertex_comparison.rst",
        "edge_bound_proof": "docs/advanced/skeleton_risk_codimension_three.rst",
        "coefficient_bounds": [
            "sqrt(a0)<140/D^2",
            "sqrt(a1)<174/D",
            "sqrt(a2)<70/sqrt(D)",
            "sqrt(a3)<(15/2)*D^(5/8)",
        ],
        "top_coefficient": "binom(D-1,4)",
        "all_at_once_minus_top": "(5/6)*D^3+(25/6)*D",
        "lower_skeleton_sqrt_sum_divided_by_degree_upper_bound": fraction_record(q),
        "risk_excess_over_top_divided_by_degree_cubed_upper_bound": fraction_record(upper),
        "strict_margin_below_five_sixths": fraction_record(margin),
    }


def certify_five_vertex_comparison(*, backend: str = "python", max_bits: int = 4096) -> dict[str, object]:
    """Certify every finite sign and attach the independent analytic tail.

    An unresolved radical interval, a possible equality, or disagreement
    with the stated loss set raises an error rather than a partial theorem.
    """
    _positive_integer(max_bits, "max_bits")
    coefficients = five_vertex_coefficients(TAIL_START - 1, backend=backend)
    points = []
    losses = []
    for degree, values in coefficients.items():
        point = radical_comparison(values, math.comb(degree + 4, 4), max_bits=max_bits)
        point["degree"] = degree
        point["coefficients"] = [fraction_record(value) for value in values]
        points.append(point)
        if point["decision"] == "all_at_once_wins":
            losses.append(degree)
    if losses != list(range(3, 12)):
        raise ArithmeticError(f"finite comparisons disagree with the stated classification: {losses}")
    return {
        "schema": "skeleton-risk-five-vertex-comparison-v1",
        "proof_status": "strict_comparisons_for_all_integer_degrees_ge_1",
        "vertices": 5,
        "all_at_once_wins_degrees": losses,
        "finite_comparisons": points,
        "tail": tail_certificate(),
    }


def main(argv: list[str] | None = None) -> int:
    """Write a source-bound all-degree comparison certificate atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "flint"), default="python")
    parser.add_argument("--max-bits", type=int, default=4096)
    arguments = parser.parse_args(argv)
    names = ["five_vertex_comparison.py", "comparison.py", "research_exact.py", "vertex_five.py", "edge_closed.py", "exact.py"]
    paths = [Path(__file__).with_name(name) for name in names]
    sources = {path.name: path.read_bytes() for path in paths}
    try:
        report = certify_five_vertex_comparison(backend=arguments.backend, max_bits=arguments.max_bits)
        if any(path.read_bytes() != sources[path.name] for path in paths):
            raise RuntimeError("comparison sources changed during certification")
        report["source_sha256"] = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
        atomic_json(arguments.output, report)
    except (ValueError, ArithmeticError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
