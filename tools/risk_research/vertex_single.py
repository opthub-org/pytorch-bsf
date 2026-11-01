"""Evaluate the proved positive single sum for tetrahedron vertex risk.

The number of terms grows with the degree. This formula avoids bivariate
kernel convolution; it is not a fixed-order recurrence or a bounded-length
classical-special-function expression.
"""

from fractions import Fraction
from math import comb


def tetrahedron_vertex_coefficient(degree: int) -> Fraction:
    """Return exact ``a_0(4, degree)`` from one positive harmonic sum.

    The identity holds for every positive integer degree under the existing
    uniform-simplex assumptions. Its asymptotic leading term is
    ``48*pi**2/degree**5``. The computation uses ``degree`` rational terms and
    stores no bivariate coefficient table.
    """
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    harmonic = Fraction(0)
    harmonic2 = Fraction(0)
    total = Fraction(0)
    weight = 1
    for j in range(degree):
        if j:
            harmonic += Fraction(1, j)
            harmonic2 += Fraction(1, j * j)
        a, b, c = j + 1, j + 2, j + 3
        # This grouping makes each term's positivity explicit, including j=0.
        bracket = harmonic * (harmonic - Fraction(j, b)) / (a * b * c) + harmonic2 / (a * c) + Fraction(2, a * a * b * b * c)
        total += weight * bracket
        weight = weight * (degree - 1 - j) * (degree + 1 - j) // ((j + 1) * (j + 1))
    denominator = comb(2 * degree, degree - 1) * (2 * degree + 1) * (2 * degree + 2) * (2 * degree + 3)
    return 576 * total / denominator
