"""Check the boundary-pair vertex identity against independent references."""

from fractions import Fraction

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research.vertex_boundary import vertex_boundary_risk


@pytest.mark.parametrize(("vertices", "degree"), [(m, d) for m in range(2, 8) for d in range(1, 8)])
def test_boundary_pair_risk_matches_original_kernel_scalar_sum(vertices, degree):
    assert vertex_boundary_risk(vertices, degree) == _scalar_risk_sum(vertices, degree, 1)


@pytest.mark.parametrize(("vertices", "degree"), [(3, 3), (4, 3), (4, 4)])
def test_boundary_pair_risk_matches_independent_gram_propagation(vertices, degree):
    assert vertex_boundary_risk(vertices, degree) == _exact_coefficients(vertices, degree)[0][0]


@pytest.mark.parametrize("vertices", [2, 3, 4, 7, 12])
def test_linear_vertex_boundary_value(vertices):
    assert vertex_boundary_risk(vertices, 1) == Fraction(2 * vertices, vertices + 1)


@pytest.mark.parametrize(("vertices", "degree"), [(1, 2), (4, 0), (True, 2), (4, False), (4.0, 2), (4, 2.0)])
def test_invalid_vertex_parameters(vertices, degree):
    with pytest.raises(ValueError, match="integers"):
        vertex_boundary_risk(vertices, degree)
