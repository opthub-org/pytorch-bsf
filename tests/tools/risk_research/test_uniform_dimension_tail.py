"""Independently check the all-degree dimension-tail bound and certificates."""

import copy
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _scalar_risk_sum, _tabulated_coefficient, _vertex_kernel
from tools.risk_research import uniform_dimension_tail as tail
from tools.risk_research.comparison import radical_comparison

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


@pytest.mark.parametrize("degree", range(1, 9))
def test_scalar_and_independently_counted_coefficients_satisfy_uniform_bounds(degree):
    bounds = tail.coefficient_upper_bounds(degree)
    for vertices in [4 * degree, tail.dimension_threshold(degree), tail.dimension_threshold(degree) + 1]:
        values = tuple(_tabulated_coefficient(degree, size - 1, vertices) for size in range(1, degree + 1))
        assert all(0 <= value <= bound for value, bound in zip(values, bounds, strict=True))
        if degree <= 5:
            assert values == tuple(_scalar_risk_sum(vertices, degree, size) for size in range(1, degree + 1))
        if vertices >= tail.dimension_threshold(degree):
            comparison = radical_comparison(tuple(5 * value for value in values), 4 * math.comb(vertices + degree - 1, degree))
            assert comparison["decision"] == "inductive_wins"


@pytest.mark.parametrize("power", [1, 2, 5, 31])
def test_independent_kernel_is_coefficientwise_bounded_by_rising_factorial(power):
    for h in range(5):
        for ell in range(5):
            majorant = Fraction(math.prod(range(power, power + h + ell)), math.factorial(h) * math.factorial(ell))
            assert 0 <= _vertex_kernel(power, h, ell) <= majorant


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 9, 17, 31, 97])
def test_exact_bound_chain_and_all_vertex_domain(degree):
    bounds = tail.coefficient_upper_bounds(degree)
    report = tail.certify_degree(degree)
    tail.verify_certificate(report)
    upper = tail.risk_upper_bound(degree)
    threshold = tail.dimension_threshold(degree)
    target_lower = Fraction(threshold**degree, math.factorial(degree))
    assert degree * sum(bounds) <= upper < Fraction(4, 5) * target_lower
    assert target_lower <= math.comb(threshold + degree - 1, degree)
    assert _fraction(report["risk_upper_bound"]) == upper
    assert _fraction(report["all_at_once_lower_bound"]) == target_lower
    assert _fraction(report["strict_margin_below_ratio_bound"]) == Fraction(4, 5) * target_lower - upper
    assert report["coefficient_bound_from_vertices"] == str(4 * degree)
    assert 4 * degree <= threshold


def test_degree31_holdout_checks_new_high_dimensional_scalar_modes():
    degree = 31
    vertices = 4 * degree
    bounds = tail.coefficient_upper_bounds(degree)
    # These sizes were not used to choose the degree-independent proof constants.
    # The original scalar reference uses the exact angular sum and kernel.
    for size in [28, 29, 30, 31]:
        assert 0 < _scalar_risk_sum(vertices, degree, size) <= bounds[size - 1]
    for power in [vertices - 1, vertices - 28]:
        for h, ell in [(0, 30), (3, 4), (4, 3), (30, 0)]:
            majorant = Fraction(math.prod(range(power, power + h + ell)), math.factorial(h) * math.factorial(ell))
            assert 0 < _vertex_kernel(power, h, ell) <= majorant


@pytest.mark.parametrize("degree", [0, -1, True, False, 1.5, "3", None])
@pytest.mark.parametrize(
    "function", [tail.dimension_threshold, tail.coefficient_upper_bounds, tail.risk_upper_bound, tail.certify_degree]
)
def test_invalid_degree_rejected_before_arithmetic(function, degree):
    with pytest.raises(ValueError, match="positive integer"):
        function(degree)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("from_vertices", "1"),
        ("coefficient_bound_from_vertices", "1"),
        ("risk_upper_bound", {"numerator": "0", "denominator": "1"}),
        ("all_at_once_lower_bound", {"numerator": "999999", "denominator": "1"}),
        ("strict_ratio_upper_bound", {"numerator": "1", "denominator": "1"}),
        ("strict_margin_below_ratio_bound", {"numerator": "-1", "denominator": "1"}),
        ("proof_status", "proved_by_finite_checks"),
        ("proof", "unrelated-proof.rst"),
        ("source_sha256", {}),
        ("degree", True),
        ("degree", 0),
        ("unexpected", "field"),
    ],
)
def test_altered_certificate_rejected(field, replacement):
    report = copy.deepcopy(tail.certify_degree(7))
    report[field] = replacement
    with pytest.raises(ValueError):
        tail.verify_certificate(report)


@pytest.mark.parametrize("record", [None, [], 3, {}, {"degree": "3"}])
def test_malformed_certificate_rejected(record):
    with pytest.raises(ValueError, match="integer degree"):
        tail.verify_certificate(record)


def test_cli_round_trip_and_failure_preserve_existing_artifact(tmp_path):
    output = tmp_path / "tail.json"
    command = [sys.executable, "-m", "tools.risk_research.uniform_dimension_tail"]
    subprocess.run([*command, "--degree", "31", "--output", str(output)], cwd=ROOT, check=True)
    original = output.read_bytes()
    subprocess.run([*command, "--verify", str(output)], cwd=ROOT, check=True)
    assert output.read_bytes() == original
    report = json.loads(original)
    tail.verify_certificate(report)
    for arguments in [["--degree", "0", "--output", str(output)], ["--verify", str(output), "--output", str(output)]]:
        result = subprocess.run([*command, *arguments], cwd=ROOT, check=False, capture_output=True)
        assert result.returncode != 0
        assert output.read_bytes() == original
    report["from_vertices"] = "2"
    output.write_text(json.dumps(report))
    invalid = output.read_bytes()
    result = subprocess.run([*command, "--verify", str(output)], cwd=ROOT, check=False, capture_output=True)
    assert result.returncode != 0
    assert output.read_bytes() == invalid


def test_cli_requires_a_destination_for_certification():
    result = subprocess.run(
        [sys.executable, "-m", "tools.risk_research.uniform_dimension_tail", "--degree", "3"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"requires --output" in result.stderr


def test_committed_degree31_certificate_matches_current_proof():
    path = ROOT / "tools/risk_research/examples/uniform_dimension_tail_degree31.json"
    tail.verify_certificate(json.loads(path.read_text()))


def test_large_certificate_round_trip_preserves_python_integer_digit_limit():
    before = sys.get_int_max_str_digits()
    record = json.loads(json.dumps(tail.certify_degree(1000)))
    assert len(record["all_at_once_lower_bound"]["numerator"]) > 4300
    tail.verify_certificate(record)
    assert sys.get_int_max_str_digits() == before
