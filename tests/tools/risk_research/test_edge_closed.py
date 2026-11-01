"""Independent exact checks of the codimension-two recurrence representation."""

import json
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research.edge_closed import (
    auxiliary_sequences,
    codimension_two_coefficient,
    tetrahedron_edge_coefficient,
    tetrahedron_edge_coefficients,
)
from tools.risk_research.edge_recurrence import _integer_rows

EXAMPLES = Path(__file__).resolve().parents[3] / "tools/risk_research/examples"


def test_auxiliary_recurrences_match_independent_binomial_sums():
    auxiliary_f, auxiliary_g = auxiliary_sequences(40)
    assert auxiliary_f[0] == auxiliary_g[0] == 0
    for n in range(1, 41):
        assert auxiliary_f[n] == sum((Fraction(1, j * j * comb(n, j)) for j in range(1, n + 1)), Fraction(0))
        assert auxiliary_g[n] == sum((Fraction(1, j * comb(n, j)) for j in range(1, n + 1)), Fraction(0))


@pytest.mark.parametrize("degree", range(1, 11))
def test_tetrahedron_edge_recurrence_matches_scalar_oracle(degree):
    assert tetrahedron_edge_coefficient(degree) == _scalar_risk_sum(4, degree, 2)


@pytest.mark.parametrize("degree", range(1, 5))
def test_tetrahedron_edge_recurrence_matches_gram_propagation(degree):
    assert tetrahedron_edge_coefficient(degree) == _exact_coefficients(4, degree)[0][1]


@pytest.mark.parametrize("size", range(1, 5))
@pytest.mark.parametrize("offset", range(5))
def test_codimension_two_modes_match_scalar_oracle(size, offset):
    degree = size + offset
    assert codimension_two_coefficient(size, degree) == _scalar_risk_sum(size + 2, degree, size)


def test_degree_one_uses_the_same_nonsingular_formula():
    values = tetrahedron_edge_coefficients(5)
    assert values == {1: Fraction(0), 2: Fraction(36, 7), 3: Fraction(33, 7), 4: Fraction(169, 44), 5: Fraction(34, 11)}


def test_reciprocal_range_certificate_is_an_identity():
    n, j = sp.symbols("n j")
    polynomial = (
        1 / (n + 2)
        - (6 * n + 4) * j / (n**3 * (n + 1) * (n + 2))
        - 6 * (n - 2) * j**2 / (n**3 * (n + 2))
        + 4 * (n - 2) * j**3 / (n**3 * (n + 1) * (n + 2))
    )
    certificate = polynomial / (j**2 * (n - j + 1))
    c0 = -8 * (n - 2) / (n**3 * (n + 1) * (n + 2))
    c1 = (n**2 + n + 2) / (n**2 * (n + 1) * (n + 2))
    residual = (
        1 / (j**2 * (j + 1) * (n - j) ** 2 * (n - j + 1))
        - (j + 1) * certificate.subs(j, j + 1) / (n - j)
        + certificate
        - c0
        - c1 * (1 / j**2 + 1 / (n - j) ** 2)
    )
    assert sp.factor(residual) == 0


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_degree_validation(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        tetrahedron_edge_coefficient(invalid)


def test_absent_codimension_two_controls():
    assert codimension_two_coefficient(3, 2) == 0


@pytest.fixture(scope="module")
def independent_edge_values():
    return {degree: _scalar_risk_sum(4, degree, 2) for degree in range(1, 17)}


@pytest.fixture(params=[7, 8])
def recurrence(request):
    return json.loads((EXAMPLES / f"tetrahedron_edge_recurrence_order{request.param}.json").read_text(encoding="utf-8"))


def _polynomial_value(coefficients, degree):
    value = 0
    for coefficient in reversed(coefficients):
        value = value * degree + int(coefficient)
    return value


def test_certified_operator_initials_and_residuals(recurrence, independent_edge_values):
    assert recurrence["status"] == "certified_by_state_identity"
    assert recurrence["proof_status"] == "exact_rational_state_identity"
    assert recurrence["domain_start"] == 1
    assert recurrence["symbolic_state_residuals"] == ["0"] * 8
    assert recurrence["required_initial_indices"] == list(range(1, recurrence["order"] + 1))
    for degree, value in recurrence["initial_values"].items():
        assert Fraction(value) == independent_edge_values[int(degree)]
    for degree in range(1, 17 - recurrence["order"]):
        assert (
            sum(
                _polynomial_value(row, degree) * independent_edge_values[degree + shift]
                for shift, row in enumerate(recurrence["coefficients"])
            )
            == 0
        )


def test_certified_operator_has_no_missing_integer_restart(recurrence):
    degree = sp.Symbol("D")
    leading = sp.Poly(sum(int(c) * degree**i for i, c in enumerate(recurrence["coefficients"][-1])), degree)
    positive_integer_roots = sorted(int(root) for root in leading.ground_roots() if root.is_Integer and root >= 1)
    assert positive_integer_roots == recurrence["exceptional_indices"] == []
    assert recurrence["polynomial_degree"] == max(len(row) - 1 for row in recurrence["coefficients"])


@pytest.mark.parametrize("coefficient", ["__import__('os')", "1/2", "+1", "01", "-0", 1, True, 1.0])
def test_operator_coefficient_parser_accepts_only_canonical_integers(coefficient):
    with pytest.raises(ValueError):
        _integer_rows([["1"], [coefficient]], 1)
