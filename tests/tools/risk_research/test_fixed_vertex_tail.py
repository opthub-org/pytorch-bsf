"""Independently check the fixed-M tail bounds and proof specializations."""

import copy
import hashlib
import json
import math
import sys
from fractions import Fraction

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import fixed_vertex_tail as tail
from tools.risk_research.vertex_general import transformed_diagonals


@pytest.mark.parametrize("vertices,degree", [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 2), (5, 3), (6, 2), (6, 3)])
def test_all_majorants_bound_independent_fraction_gram_risks(vertices, degree):
    actual = _exact_coefficients(vertices, degree)[0]
    actual += [Fraction(0)] * (vertices - len(actual))
    bounds = tail.lower_skeleton_upper_bounds(vertices, degree)
    assert len(bounds) == vertices - 1
    assert all(Fraction(0) <= value <= upper for value, upper in zip(actual, bounds, strict=False))
    assert all(upper == 0 for size, upper in enumerate(bounds, 1) if size > degree)


@pytest.mark.parametrize("vertices,degree", [(4, 11), (5, 9), (6, 7), (7, 5)])
def test_majorants_bound_original_scalar_formula(vertices, degree):
    bounds = tail.lower_skeleton_upper_bounds(vertices, degree)
    for size, upper in enumerate(bounds, 1):
        assert _scalar_risk_sum(vertices, degree, size) <= upper


@pytest.mark.parametrize("power", range(1, 9))
def test_vertex_diagonal_l1_majorant(power):
    assert all(0 < value <= 4**power for value in transformed_diagonals(power, 13))


def test_transfer_normalizations_at_zero_and_one_outside_coordinate():
    for size in range(2, 9):
        assert tail.transfer_constant(size, 0) == Fraction(1, math.factorial(size - 1))
        assert tail.transfer_constant(size, 1) == Fraction(size * (size + 1) ** 2, 2 * math.factorial(size - 1))
    assert tail.transfer_constant(2, 2) == 486
    assert tail.transfer_constant(2, 3) == 65100


@pytest.mark.parametrize("vertices", [4, 5, 6, 7, 12, 20])
def test_threshold_specialization_has_strict_all_tail_margin(vertices):
    report = json.loads(json.dumps(tail.certify_vertices(vertices)))
    threshold = int(report["from_degree"])
    parameter = int(report["fourth_root_threshold"])
    assert threshold >= vertices and threshold >= parameter**4
    actual_gap = Fraction(vertices, math.factorial(vertices - 2))
    excess = Fraction(**{k: int(v) for k, v in report["normalized_risk_excess_upper_bound"].items()})
    margin = Fraction(**{k: int(v) for k, v in report["strict_normalized_margin"].items()})
    assert actual_gap - excess == margin >= actual_gap / 2 > 0
    for bound in tail.coefficient_majorants(vertices):
        assert tail._ceil_root(bound.coefficient, 2) ** 2 >= bound.coefficient
    tail.verify_certificate(json.loads(json.dumps(report)))


@pytest.mark.parametrize("value,order", [(Fraction(9, 4), 2), (Fraction(16), 2), (Fraction(65), 6), (Fraction(1, 10000), 6)])
def test_rational_root_rounds_outward(value, order):
    observed = tail._ceil_root(value, order)
    assert Fraction((observed - 1) ** order) < value <= Fraction(observed**order)


@pytest.mark.parametrize("invalid", [True, False, 3, 0, -1, 4.0, "4", None])
def test_invalid_vertex_counts_are_rejected(invalid):
    with pytest.raises(ValueError, match="vertices"):
        tail.degree_threshold(invalid)
    with pytest.raises(ValueError, match="vertices"):
        tail.lower_skeleton_upper_bounds(invalid, 1)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "1", None])
def test_invalid_degrees_are_rejected(invalid):
    with pytest.raises(ValueError, match="degree"):
        tail.lower_skeleton_upper_bounds(4, invalid)


@pytest.mark.parametrize("size,outside", [(1, 1), (True, 1), (2, -1), (2, True), (2, 1.5)])
def test_transfer_domain_is_validated(size, outside):
    with pytest.raises(ValueError):
        tail.transfer_constant(size, outside)


@pytest.mark.parametrize(
    "field", ["from_degree", "strict_normalized_margin", "lower_skeleton_majorants", "source_sha256", "proof_status"]
)
def test_certificate_changes_are_detected(field):
    report = copy.deepcopy(tail.certify_vertices(6))
    report[field] = "altered"
    with pytest.raises(ValueError, match="does not match"):
        tail.verify_certificate(report)


@pytest.mark.parametrize("record", [None, [], {}, {"vertices": True}, {"vertices": 6.0}])
def test_malformed_certificate_is_rejected(record):
    with pytest.raises(ValueError):
        tail.verify_certificate(record)


def test_cli_round_trip_and_canonical_source_hashes(tmp_path):
    output = tmp_path / "tail.json"
    assert tail.main(["--vertices", "6", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert tail.main(["--verify", str(output)]) == 0
    assert report["source_hash_policy"] == "sha256-crlf-to-lf-v1"
    assert report["source_sha256"] == {
        name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in tail._sources().items()
    }


def test_crlf_hashes_are_portable_but_concurrent_raw_edit_is_rejected(tmp_path, monkeypatch):
    original_sources = tail._sources()
    crlf = {name: source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for name, source in original_sources.items()}
    expected = tail.certify_vertices(6)
    monkeypatch.setattr(tail, "_sources", lambda: crlf)
    assert tail.certify_vertices(6) == expected
    output = tmp_path / "tail.json"
    output.write_bytes(b"previous complete certificate\n")
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return original_sources if calls == 1 else crlf

    monkeypatch.setattr(tail, "_sources", changing_sources)
    with pytest.raises(SystemExit) as error:
        tail.main(["--vertices", "6", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous complete certificate\n"


def test_failed_generation_and_bad_verify_preserve_output(tmp_path):
    output = tmp_path / "tail.json"
    output.write_bytes(b"previous complete certificate\n")
    with pytest.raises(SystemExit):
        tail.main(["--vertices", "3", "--output", str(output)])
    assert output.read_bytes() == b"previous complete certificate\n"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    with pytest.raises(SystemExit):
        tail.main(["--verify", str(malformed)])
    assert output.read_bytes() == b"previous complete certificate\n"


def test_large_decimal_does_not_change_interpreter_digit_limit():
    before = sys.get_int_max_str_digits()
    value = 10**5000 + 7
    observed = tail._decimal(value)
    assert len(observed) == 5001 and observed.startswith("1") and observed.endswith("0007")
    assert sys.get_int_max_str_digits() == before
