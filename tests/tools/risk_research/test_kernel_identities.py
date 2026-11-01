"""Check the new kernel identity against independent exact convolution."""

from fractions import Fraction

import pytest

from tests.sampling import _vertex_kernel
from tools.risk_research.kernel_identities import harmonic, two_coordinate_kernel


def test_two_coordinate_kernel_matches_independent_convolution():
    for h in range(13):
        for ell in range(13):
            assert two_coordinate_kernel(h, ell) == _vertex_kernel(2, h, ell)


def test_axes_include_the_constant_and_high_degree_boundaries():
    assert two_coordinate_kernel(0, 0) == 1
    for h in (0, 1, 2, 5, 20, 100):
        expected = 2 * harmonic(h + 1) / (h + 2)
        assert two_coordinate_kernel(h, 0) == two_coordinate_kernel(0, h) == expected


@pytest.mark.parametrize("indices", [(-1, 0), (0, -1), (True, 0), (1.0, 0)])
def test_invalid_kernel_indices(indices):
    with pytest.raises(ValueError):
        two_coordinate_kernel(*indices)


def test_harmonic_empty_sum_and_invalid_indices():
    assert harmonic(0) == Fraction(0)
    assert harmonic(3) == Fraction(11, 6)
    with pytest.raises(ValueError):
        harmonic(-1)
