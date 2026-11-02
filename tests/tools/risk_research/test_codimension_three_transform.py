"""Independent identities, risk oracles and strict saved-record contracts."""

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import codimension_three_transform as transform
from tools.risk_research.codimension_three_exact import angular_mode_ratio, codimension_three_coefficient

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(("first", "second"), [(0, 9), (1, 24), (16, 35), (17, 17), (17, 64), (33, 19), (65, 67)])
def test_integer_polynomial_multiplier_matches_direct_coefficients(first, second):
    left = [(-1) ** index * (index**2 + 1) for index in range(first)]
    right = [(index % 5) - 2 for index in range(second)]
    expected = []
    if left and right:
        expected = [
            sum(left[j] * right[index - j] for j in range(first) if 0 <= index - j < second)
            for index in range(first + second - 1)
        ]
    assert transform._multiply(left, right) == expected


@pytest.mark.parametrize("last", range(7))
def test_arbitrary_diagonal_basis_vectors_satisfy_the_coefficient_identity(last):
    total = 2 * last + 5
    for j in range(last + 1):
        values = [Fraction(int(index == j), j + 3) for index in range(last + 1)]
        expected = [
            Fraction(math.comb(length, j) * math.comb(total - length, j), (j + 3) * math.comb(total, length))
            if j <= length
            else Fraction(0)
            for length in range(last + 1)
        ]
        assert transform._averages(values, total) == expected


@pytest.mark.parametrize(("size", "degree"), [(2, 2), (2, 19), (2, 48), (3, 3), (3, 24), (5, 35)])
def test_every_transformed_mode_matches_the_existing_positive_average(size, degree):
    scale = Fraction(size * (size + 1) * (size + 2), (2 * degree + size) * (2 * degree + size + 1) * (2 * degree + size + 2))
    actual = transform.mode_averages(size, degree)
    assert len(actual) == degree - size + 1
    for length, value in enumerate(actual):
        angular = degree - size - length
        assert value > 0
        assert scale * value == angular_mode_ratio(size, 3, degree, angular)


@pytest.mark.parametrize("size", [2, 3, 4, 7])
@pytest.mark.parametrize("offset", [-1, 0, 1, 7, 29])
def test_full_coefficient_matches_the_original_double_sum(size, offset):
    degree = size + offset
    assert transform.coefficient(size, degree) == codimension_three_coefficient(size, degree)


@pytest.mark.parametrize(("size", "degree"), [(2, 2), (2, 4), (2, 7), (3, 3), (3, 6), (4, 7)])
def test_original_scalar_kernel_checks_the_risk_normalization(size, degree):
    assert transform.coefficient(size, degree) == _scalar_risk_sum(size + 3, degree, size)


@pytest.mark.parametrize(("size", "degree"), [(2, 2), (2, 3), (3, 3)])
def test_independent_gram_propagation_checks_the_risk_normalization(size, degree):
    assert transform.coefficient(size, degree) == _exact_coefficients(size + 3, degree)[0][size - 1]


def test_frozen_original_kernel_values_and_new_post_construction_holdouts():
    for name in ("codimension_three_exact_holdouts.json", "codimension_three_transform_holdouts.json"):
        report = json.loads((ROOT / "tools/risk_research/examples" / name).read_bytes())
        assert report["method"] == "original_integer_outside_kernel_scalar_sum"
        for row in report["cases"]:
            if row["outside"] == 3:
                expected = Fraction(int(row["numerator"]), int(row["denominator"]))
                assert transform.coefficient(row["face_size"], row["degree"]) == expected


@pytest.mark.parametrize("function", [transform.mode_averages, transform.coefficient])
@pytest.mark.parametrize(("size", "degree"), [(True, 5), (1, 5), (2.0, 5), (2, True), (2, 0), (2, 3.0)])
def test_integer_domains_reject_coercion_and_invalid_degrees(function, size, degree):
    with pytest.raises(ValueError):
        function(size, degree)


def test_inactive_degrees_are_zero_and_have_no_modes():
    for size in (2, 3, 8):
        for degree in range(1, size):
            assert transform.mode_averages(size, degree) == []
            assert transform.coefficient(size, degree) == 0


def test_cli_rechecks_exact_records_and_preserves_the_input(tmp_path):
    source, checked = tmp_path / "source.json", tmp_path / "checked.json"
    args = ["--face-size", "3", "--degree", "41"]
    assert transform.main([*args, "--output", str(source)]) == 0
    before = source.read_bytes()
    assert transform.main([*args, "--verify", str(source), "--output", str(checked)]) == 0
    assert checked.read_bytes() == before == source.read_bytes()
    report = json.loads(before)
    assert report["diagonal_terms"] == 39
    assert report["reference_double_sum_terms"] == 780
    assert report["algorithm"] == "homogeneous_karatsuba_binomial_transform"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("degree", 8),
        ("outside", True),
        ("diagonal_terms", 8.0),
        ("extra", 1),
        ("coefficient", {"numerator": "0", "denominator": "1"}),
        ("source_hash_policy", "unknown"),
    ],
)
def test_typed_record_changes_fail_and_preserve_the_existing_destination(tmp_path, key, value):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    args = ["--face-size", "2", "--degree", "9"]
    assert transform.main([*args, "--output", str(source)]) == 0
    report = json.loads(source.read_bytes())
    report[key] = value
    source.write_text(json.dumps(report), encoding="utf-8")
    before = source.read_bytes()
    output.write_bytes(b"preserve destination")
    with pytest.raises(SystemExit) as error:
        transform.main([*args, "--verify", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert source.read_bytes() == before and output.read_bytes() == b"preserve destination"


@pytest.mark.parametrize("document", ['{"degree":9,"degree":9}', '{"degree":NaN}', "[]", "null"])
def test_malformed_duplicate_and_nonfinite_records_fail_without_output_loss(tmp_path, document):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    source.write_text(document, encoding="utf-8")
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        transform.main(["--face-size", "2", "--degree", "9", "--verify", str(source), "--output", str(output)])
    assert output.read_bytes() == b"keep"


def test_raw_source_edit_is_rejected_and_existing_output_survives(tmp_path, monkeypatch):
    calls = 0
    original = transform._raw_sources

    def changed():
        nonlocal calls
        calls += 1
        sources = original()
        if calls == 2:
            name = next(iter(sources))
            sources[name] = sources[name].replace(b"\n", b"\r\n")
        return sources

    monkeypatch.setattr(transform, "_raw_sources", changed)
    output = tmp_path / "output.json"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        transform.main(["--face-size", "2", "--degree", "9", "--output", str(output)])
    assert output.read_bytes() == b"keep"


def test_normalized_fingerprints_are_line_ending_portable():
    lf = {"example": b"line\nsecond\n"}
    crlf = {name: value.replace(b"\n", b"\r\n") for name, value in lf.items()}
    assert transform._record(2, 4, lf) == transform._record(2, 4, crlf)
    assert transform._record(2, 4, lf)["source_sha256"] == {"example": hashlib.sha256(lf["example"]).hexdigest()}


def test_sources_and_saved_input_cannot_be_used_as_output(tmp_path):
    source = tmp_path / "saved.json"
    args = ["--face-size", "2", "--degree", "4"]
    assert transform.main([*args, "--output", str(source)]) == 0
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        transform.main([*args, "--verify", str(source), "--output", str(source)])
    assert source.read_bytes() == before
    for name in transform._raw_sources():
        path = ROOT / name
        before_source = path.read_bytes()
        with pytest.raises(SystemExit):
            transform.main([*args, "--output", str(path)])
        assert path.read_bytes() == before_source


def test_large_decimal_serialization_preserves_the_process_limit():
    before = sys.get_int_max_str_digits()
    value = 10**5000 + 123
    expected = "1" + "0" * 4997 + "123"
    assert transform._decimal(value) == expected
    assert transform._decimal(-value) == "-" + expected
    assert transform._decimal(0) == "0"
    assert sys.get_int_max_str_digits() == before
