"""Check exact fixed-degree comparisons against independent mathematical oracles."""

import hashlib
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import (
    _exact_coefficients,
    _large_dimension_limit,
    _risk_polynomial_coefficients,
    _scalar_risk_sum,
    _triangle_edge_risk,
    _triangle_vertex_risk,
)
from tools.risk_research import comparison
from tools.risk_research.comparison import certify_degree, coefficient_values, dimension_polynomials, radical_comparison
from tools.risk_research.exact import record_fraction

ROOT = Path(__file__).resolve().parents[3]
LOSS_INTERVALS = {
    9: (5, 70, 115),
    10: (5, 80, 131),
    11: (5, 91, 146),
    12: (6, 102, 162),
    13: (6, 113, 178),
    14: (6, 125, 193),
    15: (6, 136, 209),
    16: (6, 147, 225),
    17: (7, 159, 240),
    18: (7, 170, 256),
    19: (7, 181, 272),
    20: (7, 193, 288),
}


@pytest.mark.parametrize("degree", range(1, 9))
def test_direct_polynomials_match_independently_counted_tables(degree):
    for skeleton, row in enumerate(dimension_polynomials(degree)):
        assert row == tuple(reversed(_risk_polynomial_coefficients(degree, skeleton)))


@pytest.mark.parametrize("vertices", [2, 4, 9, 71])
def test_degree_nine_matches_original_scalar_reference(vertices):
    values = coefficient_values(dimension_polynomials(9), vertices)
    for size in range(1, min(9, vertices) + 1):
        assert values[size - 1] == _scalar_risk_sum(vertices, 9, size)
    assert values[vertices:] == (Fraction(0),) * max(9 - vertices, 0)


def test_degree_nine_matches_independent_gram_propagation():
    expected, _, _ = _exact_coefficients(3, 9)
    assert coefficient_values(dimension_polynomials(9), 3)[:3] == tuple(expected)


def test_degree_twenty_cancellations_match_all_triangle_closed_forms():
    degree = 20
    values = coefficient_values(dimension_polynomials(degree), 3)
    assert values[:3] == (_triangle_vertex_risk(degree), _triangle_edge_risk(degree), Fraction(math.comb(degree - 1, 2)))
    assert values[3:] == (Fraction(0),) * (degree - 3)


@pytest.mark.parametrize("degree", [9, 12, 20])
def test_leading_coefficients_give_the_independent_large_dimension_limits(degree):
    for skeleton, row in enumerate(dimension_polynomials(degree)):
        assert row[-1] / math.factorial(skeleton + 1) ** 2 == _large_dimension_limit(degree, skeleton)


@pytest.mark.parametrize("degree", LOSS_INTERVALS)
def test_all_dimension_classification_has_a_strict_finite_and_infinite_certificate(degree):
    first, last, tail_start = LOSS_INTERVALS[degree]
    result = certify_degree(degree)
    assert result["all_at_once_wins_vertices"] == list(range(first, last + 1))
    tail = result["tail"]
    assert isinstance(tail, dict)
    assert tail["from_vertices"] == tail_start >= degree
    bounds = [record_fraction(value) for value in tail["coefficient_upper_bounds"]]
    assert record_fraction(tail["risk_upper_bound"]) == degree * sum(bounds)
    assert record_fraction(tail["risk_upper_bound"]) < int(tail["all_at_once_lower_bound"])
    assert record_fraction(tail["strict_margin"]) > 0
    points = result["finite_comparisons"]
    assert isinstance(points, list)
    assert [point["vertices"] for point in points] == list(range(2, tail_start))
    assert all(int(point["strict_margin_numerator"]) > 0 for point in points)
    assert all(point["decision"] in {"inductive_wins", "all_at_once_wins"} for point in points)


def test_low_degrees_are_also_strictly_classified():
    assert certify_degree(1)["all_at_once_wins_vertices"] == []
    assert certify_degree(2)["all_at_once_wins_vertices"] == []
    assert certify_degree(3)["all_at_once_wins_vertices"] == list(range(3, 11))


def test_radicals_require_increased_precision_for_a_close_strict_sign():
    values = (Fraction(1) + Fraction(1, 2**100),)
    with pytest.raises(ArithmeticError, match="unresolved"):
        radical_comparison(values, 1, max_bits=64)
    result = radical_comparison(values, 1, max_bits=128)
    assert result["decision"] == "all_at_once_wins"
    assert result["bits"] == 128


@pytest.mark.parametrize("values", [(Fraction(1),), (Fraction(2),), (Fraction(4, 9), Fraction(1, 9))])
def test_exact_equality_is_never_a_strict_comparison(values):
    target = 2 if values == (Fraction(2),) else 1
    with pytest.raises(ArithmeticError, match="equality is possible"):
        radical_comparison(values, target, max_bits=64)


@pytest.mark.parametrize("value", [-1, 0, True, 1.5, "9"])
def test_degree_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="degree"):
        dimension_polynomials(value)
    with pytest.raises(ValueError, match="degree"):
        certify_degree(value)


@pytest.mark.parametrize("vertices", [1, True, 2.0])
def test_vertex_count_is_not_coerced(vertices):
    with pytest.raises(ValueError, match="vertices"):
        coefficient_values(((Fraction(2),),), vertices)


@pytest.mark.parametrize("polynomials", [(), ((Fraction(0), Fraction(1)),), ((2.0,),)])
def test_malformed_or_inexact_dimension_polynomials_are_rejected(polynomials):
    with pytest.raises(ValueError, match="polynomial"):
        coefficient_values(polynomials, 2)


def test_negative_coefficients_are_rejected_and_absent_faces_are_not_evaluated():
    with pytest.raises(ValueError, match="negative"):
        coefficient_values(((Fraction(-1),),), 2)
    polynomials = ((Fraction(1),) * 5, (Fraction(1),) * 3, (Fraction(-1),))
    assert coefficient_values(polynomials, 2)[2] == 0


@pytest.mark.parametrize("values", [(), (Fraction(-1),), (1.0,), (1,)])
def test_radicals_require_exact_nonnegative_input(values):
    with pytest.raises(ValueError, match="Fractions"):
        radical_comparison(values, 1)


@pytest.mark.parametrize(("keyword", "value"), [("max_tail", 8), ("max_tail", True), ("max_bits", 0), ("max_bits", 1.5)])
def test_certification_limits_are_validated(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        certify_degree(9, **{keyword: value})


def test_insufficient_tail_search_does_not_claim_an_all_dimension_result():
    with pytest.raises(ArithmeticError, match="infinite-tail bound"):
        certify_degree(9, max_tail=9)


def test_module_cli_writes_exact_certificates_and_preserves_output_on_failure(tmp_path):
    output = tmp_path / "nested" / "degree-nine.json"
    command = [sys.executable, "-m", "tools.risk_research.comparison", "--degrees", "9", "--output", str(output)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["source_sha256"] == hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest()
    assert report["records"][0]["all_at_once_wins_vertices"] == list(range(5, 71))
    result = subprocess.run(command + ["--max-tail", "9"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no sufficient infinite-tail bound" in result.stderr
    assert output.read_bytes() == original


def test_source_change_during_certification_preserves_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"original source")
    output = tmp_path / "comparison.json"
    output.write_bytes(b"existing certificate")

    def changing_certificate(degree, **kwargs):
        source.write_bytes(b"modified source")
        return {"degree": degree}

    monkeypatch.setattr(comparison, "__file__", str(source))
    monkeypatch.setattr(comparison, "certify_degree", changing_certificate)
    with pytest.raises(SystemExit):
        comparison.main(["--degrees", "9", "--output", str(output)])
    assert output.read_bytes() == b"existing certificate"
