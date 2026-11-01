"""Evaluate five-vertex vertex risk with a single harmonic sum.

The exact formula uses four harmonic states, including one depth-two finite
harmonic sum. Its number of terms grows with the degree; no bounded-length
classical-special-function expression or degree recurrence is claimed.
"""

from collections.abc import Iterator
from fractions import Fraction
from math import comb, prod


def _quartic_diagonals(count: int) -> Iterator[Fraction]:
    """Yield the first ``count`` signed-binomial diagonals of ``F**4``."""
    harmonic = Fraction(0)
    harmonic2 = Fraction(0)
    harmonic3 = Fraction(0)
    depth_two = Fraction(0)
    for n in range(1, count + 1):
        if n > 1:
            k = n - 1
            # The depth-two sum uses H_(k-1), before updating H_k.
            depth_two += harmonic / (k * k)
            harmonic += Fraction(1, k)
            harmonic2 += Fraction(1, k * k)
            harmonic3 += Fraction(1, k * k * k)
        numerator = (
            2 * n * (n + 1) * (n + 2) * harmonic**3
            - (5 * n**3 + 7 * n**2 - 6 * n - 12) * harmonic**2
            + 2 * n * (n + 1) * (n + 2) * (2 * n + 3) * harmonic * harmonic2
            - 2 * (2 * n**4 + 15 * n**3 + 44 * n**2 + 47 * n + 24) * harmonic
            + (n + 1) * (4 * n**4 + 22 * n**3 + 41 * n**2 + 26 * n + 12) * harmonic2
            - 4 * n * (n + 1) * (n + 2) ** 2 * harmonic3
            - 4 * n * (n + 1) ** 2 * (n + 2) * (n + 3) * depth_two
            + 12
        )
        yield 12 * numerator / (n * n * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3))


def five_vertex_coefficient(degree: int) -> Fraction:
    """Return exact ``a_0(5, degree)`` using four harmonic states.

    The uniform-simplex identity holds at every positive integer degree.
    It needs ``degree`` rational summands and a fixed number of rational
    states, avoiding the bivariate outside-kernel convolution. Exact rational
    arithmetic preserves the cancellations within each diagonal weight.
    The leading asymptotic term is
    ``7200*(zeta(2)-zeta(3))/degree**6``.
    """
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    total = Fraction(0)
    weight = 1
    for j, diagonal in enumerate(_quartic_diagonals(degree)):
        total += weight * diagonal
        weight = weight * (degree - 1 - j) * (degree + 1 - j) // ((j + 1) * (j + 1))
    denominator = comb(2 * degree, degree - 1) * prod(range(2 * degree + 1, 2 * degree + 5))
    return 600 * total / denominator
