"""Independent scalar, Gram and symbolic checks of the triangular-face state."""

from fractions import Fraction
from math import comb

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research.edge_closed import codimension_two_coefficient
from tools.risk_research.triangle_faces import (
    triangle_face_auxiliary_sequence,
    triangle_face_coefficient,
    triangle_face_coefficients,
)


def test_auxiliary_state_matches_independent_defining_sums():
    values = triangle_face_auxiliary_sequence(127)
    for n in [*range(1, 19), 64, 127]:
        half = sum((Fraction(1, j * j * comb(2 * n, j)) for j in range(1, n)), Fraction(0))
        full = sum((Fraction(1, j * j * comb(2 * n, j)) for j in range(1, 2 * n + 1)), Fraction(0))
        harmonic2 = sum((Fraction(1, j * j) for j in range(1, n + 1)), Fraction(0))
        full_harmonic2 = sum((Fraction(1, j * j) for j in range(1, 2 * n + 1)), Fraction(0))
        assert values[n] == harmonic2 - half + (full - full_harmonic2) / 2


@pytest.mark.parametrize("degree", range(1, 17))
def test_triangle_face_state_matches_original_scalar_oracle(degree):
    assert triangle_face_coefficient(degree) == _scalar_risk_sum(5, degree, 3)


@pytest.mark.parametrize("degree", [3, 4])
def test_triangle_face_state_matches_independent_gram_propagation(degree):
    assert triangle_face_coefficient(degree) == _exact_coefficients(5, degree)[0][2]


def test_triangle_face_state_matches_mode_sum_in_fresh_large_degrees():
    values = triangle_face_coefficients(300)
    for degree in [1, 2, 31, 64, 121, 241, 257, 300]:
        assert values[degree] == codimension_two_coefficient(3, degree)


def test_triangle_face_state_initial_values_include_inactive_degrees():
    assert triangle_face_coefficients(5) == {
        1: Fraction(0),
        2: Fraction(0),
        3: Fraction(40, 3),
        4: Fraction(255, 11),
        5: Fraction(2675, 91),
    }


def test_weighted_reciprocal_telescope_is_an_exact_identity():
    total, j = sp.symbols("N j")
    certificate = 1 / (2 * j**2 * (total - j + 1))
    integrand = (total / 2 - j) / (j**2 * (j + 1) * (total - j) ** 2 * (total - j + 1))
    residual = (
        integrand
        - (j + 1) * certificate.subs(j, j + 1) / (total - j)
        + certificate
        - (1 / j**2 - 1 / (total - j) ** 2) / (2 * total)
    )
    assert sp.factor(residual) == 0


def test_weighted_harmonic_summation_by_parts_certificates():
    total, j = sp.symbols("N j")
    denominator = j * (j + 1) * (total - j) * (total - j + 1)
    r = 1 / (j * (total - j + 1))
    t = 1 / (j * (total - j + 2))
    assert sp.factor((total / 2 - j) / denominator - (r - r.subs(j, j + 1)) / 2) == 0
    assert sp.factor(r * (1 / (j + 1) - 1 / (total - j + 2)) - t + t.subs(j, j + 1)) == 0


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "3"])
@pytest.mark.parametrize("function", [triangle_face_auxiliary_sequence, triangle_face_coefficient, triangle_face_coefficients])
def test_triangle_face_state_validates_positive_integer_inputs(function, invalid):
    with pytest.raises(ValueError, match="positive integer"):
        function(invalid)
