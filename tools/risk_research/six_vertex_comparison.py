"""Certify optimal skeleton-risk comparisons at every six-vertex degree.

The analytic proof covers all D>=128. Exact coefficients and integer
square-root enclosures classify every smaller positive degree.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from .codimension_three_exact import nonvertex_coefficient
from .comparison import radical_comparison
from .edge_closed import codimension_two_coefficient
from .exact import atomic_json, fraction_record
from .research_exact import integer_kernel, scalar_coefficient
from .vertex_general import transformed_diagonals

TAIL_START = 128
SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def six_vertex_coefficients(maximum: int, *, backend: str = "python") -> dict[int, tuple[Fraction, ...]]:
    """Evaluate every six-vertex coefficient at degrees1..maximum exactly.

    The default path uses positive univariate diagonals for the vertex,
    edge and triangle. The optional FLINT path independently evaluates
    the original scalar formula with integer kernels F**4 and F**3.
    The remaining coefficients use established reduced formulas.
    """
    _positive_integer(maximum, "maximum")
    if backend not in ("python", "flint"):
        raise ValueError("backend must be python or flint")
    kernels = (
        {size: integer_kernel(6 - size, maximum - size, backend="flint") for size in (2, 3) if maximum >= size}
        if backend == "flint"
        else {}
    )
    diagonals = transformed_diagonals(5, maximum)
    rows = {}
    for degree in range(1, maximum + 1):
        weight, total = 1, Fraction(0)
        for j in range(degree):
            total += weight * diagonals[j]
            weight = weight * (degree - 1 - j) * (degree + 1 - j) // (j + 1) ** 2
        vertex = Fraction(4320, math.comb(2 * degree, degree - 1) * math.prod(range(2 * degree + 1, 2 * degree + 6))) * total
        if backend == "flint":
            nonvertices = tuple(
                scalar_coefficient(6, degree, size - 1, kernels[size]) if degree >= size else Fraction(0) for size in (2, 3)
            )
        else:
            nonvertices = tuple(nonvertex_coefficient(6, degree, size) for size in (2, 3))
        facet = 180 * sum(
            (Fraction(math.comb(k + 3, 3), (degree - 4 - k) * (degree + k + 10)) for k in range(max(0, degree - 4))),
            Fraction(0),
        )
        rows[degree] = (
            vertex,
            *nonvertices,
            codimension_two_coefficient(4, degree),
            facet,
            Fraction(math.comb(degree - 1, 5)) if degree >= 6 else Fraction(0),
        )
    return rows


def tail_certificate() -> dict[str, object]:
    """Specialize the independently proved entire degree tail."""
    start = TAIL_START
    # L(128)<25/4, sqrt(128)>11, sqrt(15*L(128)/128)<7/8.
    if start != 128 or start <= 11**2 or Fraction(15 * 25, 4 * start) >= Fraction(49, 64):
        raise ArithmeticError("tail endpoint inequalities failed")
    q = (
        Fraction(7, 8)
        + Fraction(95, start * 11)
        + Fraction(39, start**2)
        + Fraction(165 * 5, 2 * start**2 * 11)
        + Fraction(658, start**2 * 11)
        + Fraction(1860, start**3)
        + Fraction(990, start**3 * 11)
        + Fraction(372, start**4)
    )
    upper = Fraction(11, 60) * q + q**2 / start
    margin = Fraction(1, 4) - upper
    if margin <= 0:
        raise ArithmeticError("the analytic tail is not strictly separated")
    return {
        "from_degree": start,
        "proof": "docs/advanced/skeleton_risk_six_vertex_comparison.rst",
        "coefficient_bounds": [
            "sqrt(a0)<372/D^(5/2)",
            "sqrt(a1)<1860/D^(3/2)+990/D^2",
            "sqrt(a2)<39/sqrt(D)+165*sqrt(1+log D)/D+658/D",
            "sqrt(a3)<95",
            "a4<=15*D^2*(1+log D)",
        ],
        "top_coefficient": "binom(D-1,5)",
        "all_at_once_minus_top": "D^4/4+15*D^2/4+2",
        "lower_skeleton_sqrt_sum_divided_by_degree_to_three_halves_upper_bound": fraction_record(q),
        "risk_excess_over_top_divided_by_degree_four_upper_bound": fraction_record(upper),
        "strict_margin_below_one_quarter": fraction_record(margin),
    }


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "tools/risk_research/six_vertex_comparison.py",
        "tools/risk_research/comparison.py",
        "tools/risk_research/codimension_three_exact.py",
        "tools/risk_research/research_exact.py",
        "tools/risk_research/vertex_general.py",
        "tools/risk_research/edge_closed.py",
        "tools/risk_research/exact.py",
        "docs/advanced/skeleton_risk_six_vertex_comparison.rst",
        "docs/advanced/skeleton_risk_codimension_three.rst",
        "docs/advanced/skeleton_risk_codimension_three_exact.rst",
    ]
    return {name: (root / name).read_bytes() for name in names}


def certify_six_vertex_comparison(*, backend: str = "python", max_bits: int = 4096) -> dict[str, object]:
    """Certify every finite sign and attach a strict analytic tail margin."""
    _positive_integer(max_bits, "max_bits")
    coefficients = six_vertex_coefficients(TAIL_START - 1, backend=backend)
    if tuple(coefficients) != tuple(range(1, TAIL_START)) or any(len(row) != 6 for row in coefficients.values()):
        raise ArithmeticError("finite prefix is incomplete")
    points, losses = [], []
    for degree, values in coefficients.items():
        point = radical_comparison(values, math.comb(degree + 5, 5), max_bits=max_bits)
        point.update(degree=degree, coefficients=[fraction_record(value) for value in values])
        points.append(point)
        if point["decision"] == "all_at_once_wins":
            losses.append(degree)
    if losses != list(range(3, 17)):
        raise ArithmeticError(f"finite comparisons disagree with the stated classification: {losses}")
    return {
        "schema": "skeleton-risk-six-vertex-comparison-v1",
        "proof_status": "strict_comparisons_for_all_integer_degrees_ge_1",
        "vertices": 6,
        "all_at_once_wins_degrees": losses,
        "finite_comparisons": points,
        "tail": tail_certificate(),
        "source_hash_policy": SOURCE_HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in _sources().items()
        },
    }


def verify_certificate(record: object) -> None:
    """Recompute every degree and reject altered or partial certificates."""
    if not isinstance(record, dict) or record != certify_six_vertex_comparison():
        raise ValueError("certificate does not match the full comparison and sources")


def main(argv: list[str] | None = None) -> int:
    """Write a complete source-bound certificate or verify a saved one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--backend", choices=("python", "flint"), default="python")
    parser.add_argument("--max-bits", type=int, default=4096)
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify is not None:
            if arguments.output is not None or arguments.backend != "python" or arguments.max_bits != 4096:
                parser.error("--verify cannot be combined with generation options")
            verify_certificate(json.loads(arguments.verify.read_text(encoding="utf-8")))
            print("verified six-vertex all-degree comparison")
        else:
            if arguments.output is None:
                parser.error("generation requires --output")
            sources = _sources()
            report = certify_six_vertex_comparison(backend=arguments.backend, max_bits=arguments.max_bits)
            if _sources() != sources:
                raise RuntimeError("comparison sources changed during certification")
            atomic_json(arguments.output, report)
    except (OSError, ValueError, ArithmeticError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
