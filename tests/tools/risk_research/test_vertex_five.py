"""Check the five-vertex harmonic formula against independent risk oracles."""

import json
from fractions import Fraction
from functools import lru_cache
from math import comb, pi
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research.research_exact import integer_kernel
from tools.risk_research.vertex_five import _quartic_diagonals, five_vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("degree", range(1, 21))
def test_five_vertex_formula_matches_original_scalar_kernel_sum(degree):
    assert five_vertex_coefficient(degree) == _scalar_risk_sum(5, degree, 1)


@pytest.mark.parametrize("degree", range(1, 6))
def test_five_vertex_formula_matches_independent_gram_propagation(degree):
    assert five_vertex_coefficient(degree) == _exact_coefficients(5, degree)[0][0]


@lru_cache(maxsize=1)
def _original_quartic_kernel():
    return integer_kernel(4, 8, backend="python")


@pytest.mark.parametrize("index", range(9))
def test_harmonic_diagonals_match_direct_binomial_transform_of_original_kernel(index):
    kernel = _original_quartic_kernel()
    direct = sum(
        (
            (-1) ** (h + ell) * comb(index, h) * comb(index, ell) * kernel.value(h, ell)
            for h in range(index + 1)
            for ell in range(index + 1)
        ),
        Fraction(0),
    )
    assert list(_quartic_diagonals(index + 1))[index] == direct


def test_five_vertex_formula_matches_fresh_high_degree_integer_kernel_holdouts():
    # F^4 was computed independently after the candidate evaluator was frozen.
    document = json.loads((ROOT / "tools/risk_research/examples/five_vertex_vertex_holdouts.json").read_bytes())
    assert [case["degree"] for case in document["cases"]] == [193, 227, 251]
    for case in document["cases"]:
        assert five_vertex_coefficient(case["degree"]) == Fraction(int(case["numerator"]), int(case["denominator"]))


def test_harmonic_diagonals_are_positive_and_start_with_one():
    diagonals = list(_quartic_diagonals(200))
    assert diagonals[0] == 1
    assert all(value > 0 for value in diagonals)
    assert five_vertex_coefficient(1) == Fraction(5, 3)


def test_scaled_coefficient_approaches_the_independent_zeta_constant():
    # A convergence check supplements the proof; it does not establish the
    # asymptotic assertion. This also distinguishes the symmetric-term factor.
    zeta3 = 1.2020569031595942854
    leading = 7200 * (pi * pi / 6 - zeta3)
    scaled = float(1024**6 * five_vertex_coefficient(1024))
    assert scaled == pytest.approx(leading, rel=0.04, abs=0)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "1", None, Fraction(1)])
def test_five_vertex_degree_is_an_exact_positive_integer(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        five_vertex_coefficient(invalid)
