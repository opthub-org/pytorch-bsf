"""Evaluate the two-outside-coordinate kernel without polynomial convolution."""

from fractions import Fraction
from functools import cache
from math import factorial


@cache
def harmonic(index: int) -> Fraction:
    """Return the exact harmonic number, including H_0 = 0."""
    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    return sum((Fraction(1, value) for value in range(1, index + 1)), Fraction(0))


def two_coordinate_kernel(h: int, ell: int) -> Fraction:
    """Evaluate [x^h y^ell] F(x,y)^2 by its proved harmonic identity."""
    if any(type(value) is not int or value < 0 for value in (h, ell)):
        raise ValueError("kernel indices must be nonnegative integers")
    coefficient = Fraction(factorial(h + ell + 1), factorial(h + 2) * factorial(ell + 2))
    bracket = (h + ell + 2) * harmonic(h + ell + 1) - 1 - h * harmonic(h) - ell * harmonic(ell)
    return 2 * coefficient * bracket + Fraction(2, (h + 1) * (h + 2) * (ell + 1) * (ell + 2))
