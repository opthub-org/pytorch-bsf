"""Evaluate proved codimension-two reductions using exact rational arithmetic.

The tetrahedron edge formula has a fixed-order recurrence representation.
It uses harmonic numbers and two scalar auxiliary recurrences rather than
the original degree-dependent multiple sum. It is not an elementary
harmonic-number closed form: the two auxiliary sequences remain explicit.
"""

import argparse
import json
from fractions import Fraction
from math import comb
from pathlib import Path


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _tables(maximum: int) -> tuple[list[Fraction], list[Fraction], list[Fraction], list[Fraction]]:
    """Build H, H^(2), F and G from their nonsingular recurrences at zero."""
    harmonic = [Fraction(0)]
    harmonic2 = [Fraction(0)]
    auxiliary_f = [Fraction(0)]
    auxiliary_g = [Fraction(0)]
    for n in range(1, maximum + 1):
        harmonic.append(harmonic[-1] + Fraction(1, n))
        harmonic2.append(harmonic2[-1] + Fraction(1, n * n))
        auxiliary_f.append(auxiliary_f[-1] - auxiliary_g[-1] / n + Fraction(1, n * n))
        auxiliary_g.append(auxiliary_g[-1] / 2 + Fraction(1, n))
    return harmonic, harmonic2, auxiliary_f, auxiliary_g


def auxiliary_sequences(maximum: int) -> tuple[list[Fraction], list[Fraction]]:
    """Return F[0..maximum], G[0..maximum], with F[0]=G[0]=0.

    For n>=1, G[n]=G[n-1]/2+1/n and
    F[n]=F[n-1]-G[n-1]/n+1/n^2. Equivalently,
    F[n]=sum(1/(j^2*binom(n,j)),j=1..n) and
    G[n]=sum(1/(j*binom(n,j)),j=1..n).
    """
    if type(maximum) is not int or maximum < 0:
        raise ValueError("maximum must be a nonnegative integer")
    _, _, auxiliary_f, auxiliary_g = _tables(maximum)
    return auxiliary_f, auxiliary_g


def _half_harmonic_range(n: int, harmonic: list[Fraction], harmonic2: list[Fraction]) -> Fraction:
    h, h2 = harmonic[n], harmonic2[n]
    return (
        h * h / (n * (n + 1) * (n + 2))
        + Fraction(n * n + 2, n * n * (n + 1) * (n + 2)) * h2
        - Fraction(2 * (n - 2), n**3 * (n + 2)) * h
        - Fraction(n * n - 3 * n + 6, n**4 * (n + 2))
    )


def _full_reciprocal_range(n: int, auxiliary_f: list[Fraction], auxiliary_g: list[Fraction]) -> Fraction:
    return (
        Fraction(2 * (n * n + n + 2), n * n * (n + 1) * (n + 2)) * auxiliary_f[n]
        - Fraction(4 * (n - 2), n**3 * (n + 2)) * auxiliary_g[n]
        - Fraction(2 * (n**3 - 2 * n * n + 3 * n + 6), n**4 * (n + 1) * (n + 2))
    )


def _central_correction(n: int, j: int, harmonic: list[Fraction]) -> Fraction:
    denominator = j * (j + 1) * (n - j) * (n - j + 1)
    c = Fraction((n + 1) * (j * (n - j) - 1), denominator)
    f = (harmonic[j + 1] + harmonic[n - j + 1] - harmonic[n] - c) / denominator
    v = Fraction(1, j * j * (j + 1) * (n - j) ** 2 * (n - j + 1) * comb(n, j))
    return f - v


def tetrahedron_edge_coefficients(max_degree: int) -> dict[int, Fraction]:
    """Return a_1(4,D), 1<=D<=max_degree, from the proved recurrence formula.

    All divisions in the formula and auxiliary recurrences are nonsingular
    on this range. Degree one returns zero by the same formula.
    """
    _positive_integer(max_degree, "max_degree")
    harmonic, harmonic2, auxiliary_f, auxiliary_g = _tables(2 * max_degree + 3)
    result = {}
    for degree in range(1, max_degree + 1):
        n = 2 * degree + 3
        result[degree] = 432 * (
            _half_harmonic_range(n, harmonic, harmonic2)
            - _full_reciprocal_range(n, auxiliary_f, auxiliary_g) / 2
            - _central_correction(n, degree, harmonic)
            - _central_correction(n, degree + 1, harmonic)
        )
    return result


def tetrahedron_edge_coefficient(degree: int) -> Fraction:
    """Return the exact tetrahedron edge coefficient at a positive degree."""
    _positive_integer(degree, "degree")
    return tetrahedron_edge_coefficients(degree)[degree]


def codimension_two_coefficient(size: int, degree: int) -> Fraction:
    """Return a_(size-1)(size+2,D) using the single angular-mode sum.

    The general codimension-two reduction is proved, but its angular sum
    remains degree-dependent unless size=1 or size=2 is separately reduced.
    """
    _positive_integer(size, "size")
    _positive_integer(degree, "degree")
    if degree < size:
        return Fraction(0)
    n = degree - size
    maximum = 2 * degree + size + 1
    harmonic, _, _, _ = _tables(maximum)
    total = Fraction(0)
    for k in range(n + 1):
        multiplicity = comb(k + size - 2, size - 2) if size >= 2 else int(k == 0)
        if not multiplicity:
            continue
        length = n - k
        beta = 3 * size + 2 * k - 1
        denominator = (length + 1) * (length + 2) * (length + beta + 1) * (length + beta + 2)
        c = Fraction((maximum + 1) * ((length + 1) * (length + beta + 1) - 1), denominator)
        boundary = Fraction(1, (length + beta + 1) * (length + beta + 2) * comb(maximum, length))
        total += (
            multiplicity
            * Fraction(2 * size * (size + 1), denominator)
            * (harmonic[length + 2] + harmonic[length + beta + 2] - harmonic[maximum] - c - boundary)
        )
    return comb(size + 2, size) ** 2 * total


def main(argv: list[str] | None = None) -> None:
    """Write reproducible exact sequence data from the proved representation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    values = tetrahedron_edge_coefficients(args.max_degree)
    record = {
        "algorithm": "proved_codimension_two_reduction_with_two_auxiliary_recurrences",
        "proof_status": "fixed_order_recurrence_representation",
        "records": [
            {"degree": degree, "numerator": str(value.numerator), "denominator": str(value.denominator)}
            for degree, value in values.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
