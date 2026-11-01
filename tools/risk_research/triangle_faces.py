"""Evaluate five-vertex triangular-face risk using a fixed rational state.

The formula is proved in ``docs/advanced/skeleton_risk_extensions.rst``.
It replaces the degree-dependent angular sum by harmonic numbers, the two
edge auxiliary sequences, and one central inverse-binomial sequence.
No symbolic-algebra package is needed to evaluate it.
"""

from fractions import Fraction
from math import comb

from .edge_closed import (
    _central_correction,
    _full_reciprocal_range,
    _half_harmonic_range,
    _positive_integer,
    _tables,
)


def triangle_face_auxiliary_sequence(maximum: int) -> list[Fraction]:
    """Return J_n for 1 <= n <= maximum; index 0 is an unused zero.

    Let T_n = sum(1/(j*j*comb(2*n,j)), j=1..n-1). Then
    J_n = H_n^(2) - T_n + (F_(2n) - H_(2n)^(2))/2.
    The update below removes every degree-dependent sum from evaluation.
    """
    _positive_integer(maximum, "maximum")
    result = [Fraction(0)] * (maximum + 1)
    result[1] = Fraction(3, 4)
    for n in range(1, maximum):
        denominator = (2 * n + 1) * (2 * n + 2)
        result[n + 1] = (
            result[n] + Fraction(1, (n + 1) ** 2) - Fraction((3 * n + 2) ** 2, 2 * n * n * denominator * comb(2 * n, n))
        )
    return result


def triangle_face_coefficients(max_degree: int) -> dict[int, Fraction]:
    """Return exact a_2(5,D) for every degree from 1 through max_degree.

    The same nonsingular formula gives zero at degrees 1 and 2, when
    triangular faces have no proper controls.
    """
    _positive_integer(max_degree, "max_degree")
    harmonic, harmonic2, auxiliary_f, auxiliary_g = _tables(2 * max_degree + 4)
    auxiliary_j = triangle_face_auxiliary_sequence(max_degree + 2)
    result = {}
    for degree in range(1, max_degree + 1):
        n = degree + 2
        total = 2 * n
        reciprocal_binomial = Fraction(1, comb(total, n))
        middle_harmonic = harmonic[n] + harmonic[n + 2] - harmonic[total]
        harmonic_antisymmetric = Fraction(n + 1, 2 * n * (n + 2)) - middle_harmonic / (2 * n * (n + 1))
        antisymmetric = (
            harmonic_antisymmetric
            + auxiliary_j[n] / (2 * n)
            - Fraction(3 * n + 1, 4 * n**3 * (n + 1)) * reciprocal_binomial
            - Fraction(2 * n**4 + 2 * n**3 - 3 * n * n + 1, 4 * n**3 * (n + 1) ** 2)
        )
        half_range = (
            _half_harmonic_range(total, harmonic, harmonic2)
            - _full_reciprocal_range(total, auxiliary_f, auxiliary_g) / 2
            - _central_correction(total, n, harmonic) / 2
        )
        result[degree] = 2400 * (
            -3 * half_range
            + antisymmetric
            + _central_correction(total, n - 2, harmonic)
            + 2 * _central_correction(total, n - 1, harmonic)
        )
    return result


def triangle_face_coefficient(degree: int) -> Fraction:
    """Return exact a_2(5,D) at one positive degree."""
    _positive_integer(degree, "degree")
    return triangle_face_coefficients(degree)[degree]
