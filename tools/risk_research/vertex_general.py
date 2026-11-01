"""Evaluate transformed vertex sums and enclose every fixed-M leading constant.

The all-degree identity and the fixed-M asymptotic theorem are proved in
``skeleton_risk_vertex_general.rst``. Everything here uses exact fractions;
neither SageMath nor an optional numerical backend is required.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

from .exact import atomic_json, fraction_record


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _phi_rows(power: int, count: int) -> list[list[Fraction]]:
    """Evaluate [t^j]g(t)^r/(1-t) through elementary symmetric polynomials."""
    rows = [[Fraction(1)] * count]
    for order in range(1, power + 1):
        elementary = [Fraction(1)] + [Fraction(0)] * (order - 1)
        for k in range(1, order):
            for index in range(order - 1, 0, -1):
                elementary[index] += elementary[index - 1] / k
        prefactor = Fraction(factorial(order - 1))
        row = []
        for j in range(count):
            if j:
                for index in range(order - 1, 0, -1):
                    elementary[index] -= elementary[index - 1] / j
                prefactor *= Fraction(j, j + order)
            row.append(prefactor * elementary[-1])
        rows.append(row)
    return rows


def _cross_diagonal(rows: list[list[Fraction]], order: int) -> list[Fraction]:
    """Return the interior terms of (g(x)+g(y))^order/((1-x)(1-y))."""
    return [
        sum((comb(order, a) * rows[a][j] * rows[order - a][j] for a in range(1, order)), Fraction(0))
        for j in range(len(rows[0]))
    ]


def transformed_diagonals(power: int, count: int) -> list[Fraction]:
    """Return positive signed-binomial diagonals of the outside kernel power.

    This is a univariate convolution construction, not a bounded-length
    classical-special-function formula. The number of rational operations is
    O(power**2*count + power*count**2); integer bit lengths also grow.
    """
    _integer(power, 1, "power")
    _integer(count, 0, "count")
    rows = _phi_rows(power, count)
    result = [(-1) ** (power + 1) * value for value in rows[power]]
    for k in range(power - 1):
        cross = _cross_diagonal(rows, power - k)
        if k == 0:
            result = [left + right for left, right in zip(result, cross, strict=True)]
            continue
        coefficients = [value - (rows[k][j - 1] if j else 0) for j, value in enumerate(rows[k])]
        multiplier = (-1) ** k * comb(power, k)
        for j in range(count):
            result[j] += multiplier * sum((coefficients[h] * cross[j - h] for h in range(j + 1)), Fraction(0))
    return result


def vertex_coefficient(vertices: int, degree: int) -> Fraction:
    """Evaluate a_0(vertices, degree) exactly for every vertices>=2, degree>=1."""
    _integer(vertices, 2, "vertices")
    _integer(degree, 1, "degree")
    total = Fraction(0)
    weight = 1
    for j, diagonal in enumerate(transformed_diagonals(vertices - 1, degree)):
        total += weight * diagonal
        weight = weight * (degree - 1 - j) * (degree + 1 - j) // (j + 1) ** 2
    prefactor = vertices**2 * factorial(vertices - 1)
    denominator = comb(2 * degree, degree - 1) * prod(range(2 * degree + 1, 2 * degree + vertices))
    return prefactor * total / denominator


def _cross_majorant(order: int) -> int:
    def bound(a: int) -> int:
        return int(factorial(a) * a ** (a - 1))

    return sum(comb(order, a) * bound(a) * bound(order - a) for a in range(1, order))


def leading_constant_enclosure(vertices: int, terms: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose C_M in a_0(M,D)~C_M/D**(M+1), for each fixed M>=4.

    The first ``terms`` summands are exact. An integrated coefficient majorant
    bounds the infinite tail; an integer logarithm majorant avoids rounding.
    Positivity of the integral representation permits a zero lower endpoint.
    The theorem is not uniform when M grows with D.
    """
    _integer(vertices, 4, "vertices")
    _integer(terms, 1, "terms")
    q, order = vertices - 1, vertices - 2
    rows = _phi_rows(order, terms)
    partial = q * sum(_cross_diagonal(rows, order), Fraction(0))
    logarithm = 1 + (terms - 1).bit_length()
    exponent = order - 2
    integral = sum(
        (
            Fraction(factorial(exponent) * logarithm ** (exponent - k), factorial(exponent - k) * (order - 1) ** (k + 1))
            for k in range(exponent + 1)
        ),
        Fraction(0),
    )
    tail = Fraction(q * _cross_majorant(order), terms ** (order - 1)) * integral
    scale = Fraction(vertices**2 * factorial(vertices - 1), 2 ** (vertices - 3))
    return scale * max(Fraction(0), partial - tail), scale * (partial + tail)


def main(argv: list[str] | None = None) -> int:
    """Save an exact vertex coefficient or a certified leading-constant interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--degree", type=int)
    action.add_argument("--constant-terms", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    original = Path(__file__).read_bytes()
    try:
        record: dict[str, object] = {
            "schema": "skeleton-risk-general-vertex-v1",
            "vertices": arguments.vertices,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "proof": "docs/advanced/skeleton_risk_vertex_general.rst",
        }
        if arguments.degree is not None:
            record.update(
                {
                    "degree": arguments.degree,
                    "coefficient": fraction_record(vertex_coefficient(arguments.vertices, arguments.degree)),
                }
            )
        else:
            lower, upper = leading_constant_enclosure(arguments.vertices, arguments.constant_terms)
            record.update(
                {
                    "terms": arguments.constant_terms,
                    "constant_lower": fraction_record(lower),
                    "constant_upper": fraction_record(upper),
                    "degree_exponent": arguments.vertices + 1,
                    "scope": "fixed vertex count M>=4 as D tends to infinity",
                }
            )
        if Path(__file__).read_bytes() != original:
            raise RuntimeError("general vertex source changed during calculation")
        atomic_json(arguments.output, record)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
