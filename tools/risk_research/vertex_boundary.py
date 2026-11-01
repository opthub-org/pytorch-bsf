"""Evaluate the proved vertex boundary-pair reduction in exact arithmetic.

For ``M >= 3``, this small-degree reference reduces the outside kernel from
``F**(M-1)`` to powers at most ``F**(M-3)``. The curve case uses only the
univariate axis function. The result remains a finite scalar sum, rather
than a new bounded-length closed form in the degree.
"""

import math
from fractions import Fraction

PROOF_STATUS = "reduced_finite_sum"
Kernel = dict[tuple[int, int], Fraction]


def _truncated_product(left: Kernel, right: Kernel, degree: int) -> Kernel:
    result: Kernel = {}
    for (i, j), a in left.items():
        for (k, ell), b in right.items():
            if i + k <= degree and j + ell <= degree:
                key = (i + k, j + ell)
                result[key] = result.get(key, Fraction(0)) + a * b
    return {key: value for key, value in result.items() if value}


def _axis_powers(maximum: int, degree: int) -> list[list[Fraction]]:
    """Return truncated powers of ``-log(1-x)/x`` as univariate arrays."""
    powers = [[Fraction(1)] + [Fraction(0)] * degree]
    base = [Fraction(1, h + 1) for h in range(degree + 1)]
    for _ in range(maximum):
        previous = powers[-1]
        powers.append([sum((previous[i] * base[h - i] for i in range(h + 1)), Fraction(0)) for h in range(degree + 1)])
    return powers


def _outside_powers(maximum: int, degree: int) -> list[Kernel]:
    base = {
        (h, ell): Fraction(math.comb(h + ell, h), (h + 1) * (ell + 1)) for h in range(degree + 1) for ell in range(degree + 1)
    }
    powers = [{(0, 0): Fraction(1)}]
    for _ in range(maximum):
        powers.append(_truncated_product(powers[-1], base, degree))
    return powers


def _boundary_kernel(outside_coordinates: int, degree: int) -> Kernel:
    q = outside_coordinates
    axis = _axis_powers(q, degree)
    outside = _outside_powers(max(q - 2, 0), degree)
    result = {(h, 0): (-1) ** (q + 1) * value for h, value in enumerate(axis[q])}
    for a in range(1, q):
        for b in range(1, q - a + 1):
            shared = q - a - b
            multiplier = (
                (-1) ** (a + b) * math.factorial(q) // (math.factorial(a) * math.factorial(b) * math.factorial(shared))
            )
            first = {(h, 0): value for h, value in enumerate(axis[a])}
            second = {(0, ell): value for ell, value in enumerate(axis[b])}
            kernel = _truncated_product(_truncated_product(outside[shared], first, degree), second, degree)
            for key, value in kernel.items():
                result[key] = result.get(key, Fraction(0)) + multiplier * value
    return result


def vertex_boundary_risk(vertices: int, degree: int) -> Fraction:
    """Evaluate the reduced uniform-simplex vertex coefficient exactly.

    The identity is valid for every integer ``vertices >= 2`` and ``degree >= 1``.
    This implementation is an independent small-degree reference: its rational
    polynomial convolutions are not intended for large research batches.
    """
    if type(vertices) is not int or vertices < 2 or type(degree) is not int or degree < 1:
        raise ValueError("vertices >= 2 and degree >= 1 must be integers")
    kernel = _boundary_kernel(vertices - 1, degree - 1)
    weights = [(-1) ** h * math.comb(degree - 1, h) * math.comb(degree + 1, h) * math.factorial(h) for h in range(degree)]
    numerator = sum(
        (weights[h] * weights[ell] * math.factorial(2 * degree - h - ell) * value for (h, ell), value in kernel.items()),
        Fraction(0),
    )
    return vertices**2 * numerator / math.prod(range(vertices, vertices + 2 * degree))
