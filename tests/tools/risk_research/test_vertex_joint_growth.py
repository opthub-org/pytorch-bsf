"""Independent error budgets and exact, portable degree-block certificates."""

import copy
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import factorial
from pathlib import Path

import pytest
import sympy as sp

from tools.risk_research import vertex_joint_growth as joint
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_general_asymptotic import _majorants, error_constants

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("q", [3, 4, 5, 7, 11, 19, 31, 53])
def test_uniform_budget_bounds_cover_original_exact_constants(q):
    phi, coefficient, cross, difference = _majorants(q)
    for r in range(1, q + 1):
        assert phi[r] <= q ** (2 * r)
        assert coefficient[r] <= (4 * q**2) ** r
        if r >= 2:
            assert cross[r] <= (2 * q**2) ** r
            assert difference[r] <= (10 * q**2) ** r
    constants = error_constants(q + 1)
    assert constants.limiting <= (2 * q**2) ** q
    assert constants.diagonal <= (96 * q**2) ** q
    constant_lower = Fraction((q + 1) ** 2 * factorial(q) * q, 4)
    assert constants.remainder / constant_lower <= (512 * q**2) ** q


@pytest.mark.parametrize("vertices", [4, 5, 6, 8, 12, 17])
def test_integral_lower_bound_is_below_independent_rational_series_interval(vertices):
    q = vertices - 1
    lower, upper = leading_constant_enclosure(vertices, 256 if vertices == 17 else 128)
    new_lower = Fraction(vertices**2 * factorial(q) * q, 4)
    assert 0 < new_lower < lower < upper


def test_remaining_constant_inequalities_have_exact_positive_margins():
    t = sp.Symbol("t", nonnegative=True)
    q = t + 3
    polynomial = sp.Poly(32 * q**2 - 273 - q * (q + 1), t)
    assert all(coefficient >= 0 for coefficient in polynomial.all_coeffs())
    assert sp.expand(4 * q**2 - (q + 1) ** 2) == sp.expand((q - 1) * (3 * q + 1))
    assert 96 <= 5**3 and 32 <= 4**3 and 29 <= 4**3
    assert 4 * 3**2 <= 4**3


@pytest.mark.parametrize(("exponent", "ceiling"), [(1024, 6), (2046, 12), (2047, 11), (10**9, 2083334)])
def test_integer_vertex_ceiling_includes_logarithm_jump_boundaries(exponent, ceiling):
    assert joint.maximum_vertices(exponent) == ceiling
    certificate = joint.build_certificate(ceiling, exponent)
    joint.check_certificate(certificate)
    ell = certificate["logarithm_ceiling"]
    assert 1 << (ell - 1) < exponent + 2 <= 1 << ell
    assert all(value >= 0 for value in certificate["margins"].values())
    with pytest.raises(ValueError, match="ceiling"):
        joint.build_certificate(ceiling + 1, exponent)


@pytest.mark.parametrize("degree", [2**1024, 2**1024 + 1, 2**1025 - 1, 2**1025, 2**1031 - 1])
def test_certificate_domain_and_compressed_error_cover_actual_integer_degrees(degree):
    n = degree.bit_length() - 1
    record = joint.build_certificate(6, n)
    assert 2**n <= degree < 2 ** (n + 1)
    q = record["outside_coordinates"]
    ell = record["logarithm_ceiling"]
    assert 512 * q**2 <= (n + 2) ** 3
    assert 4 * q * ell <= n // 4
    budget = (512 * q**2) ** q * (n + 2) ** (q - 1)
    assert budget**4 <= degree
    dyadic_exponent = record["relative_error_block_upper"]["denominator_power_of_two"]
    assert Fraction(budget, degree) <= Fraction(1, 1 << dyadic_exponent)


@pytest.mark.parametrize("exponent", [10**9, 10**100])
def test_enormous_degree_blocks_have_small_certificates_without_giant_powers(exponent):
    record = joint.build_certificate(joint.maximum_vertices(exponent), exponent)
    serialized = json.dumps(record)
    assert len(serialized) < 5000
    assert record["relative_error_block_upper"]["denominator_power_of_two"] == 3 * exponent // 4
    joint.check_certificate(json.loads(serialized))


@pytest.mark.parametrize("vertices", [True, 3, 6.0, None])
def test_invalid_vertex_types_are_rejected(vertices):
    with pytest.raises(ValueError, match="vertices"):
        joint.build_certificate(vertices, 1024)


@pytest.mark.parametrize("exponent", [True, 1023, 1024.0, None])
def test_invalid_degree_floor_types_are_rejected(exponent):
    with pytest.raises(ValueError, match="degree_floor_exponent"):
        joint.maximum_vertices(exponent)


@pytest.mark.parametrize("field", ["vertices", "degree_floor_exponent", "outside_coordinates", "logarithm_ceiling"])
def test_changed_primary_certificate_fields_are_rejected(field):
    record = joint.build_certificate(6, 1024)
    record[field] += 1
    with pytest.raises(ValueError):
        joint.check_certificate(record)


@pytest.mark.parametrize(
    "section", ["margins", "budget_base_multipliers", "degree_block", "relative_error_pointwise", "relative_error_block_upper"]
)
def test_every_numeric_witness_and_bound_is_bound_to_the_certificate(section):
    original = joint.build_certificate(6, 1024)
    for field in original[section]:
        record = copy.deepcopy(original)
        record[section][field] += 1
        with pytest.raises(ValueError, match="does not match"):
            joint.check_certificate(record)


@pytest.mark.parametrize("replacement", [True, 1.0, "1"])
def test_bool_float_and_string_replacements_of_exact_integer_are_rejected(replacement):
    record = joint.build_certificate(6, 1024)
    record["relative_error_block_upper"]["numerator"] = replacement
    with pytest.raises(ValueError, match="does not match"):
        joint.check_certificate(record)


def test_changed_source_or_hash_policy_unknown_fields_and_missing_fields_are_rejected():
    original = joint.build_certificate(6, 1024)
    mutations = []
    for field in ("schema", "scope", "proof", "source_hash_policy"):
        record = copy.deepcopy(original)
        record[field] += "changed"
        mutations.append(record)
    for name in original["source_sha256"]:
        record = copy.deepcopy(original)
        record["source_sha256"][name] = "0" * 64
        mutations.append(record)
    record = copy.deepcopy(original)
    record["unknown"] = "extra"
    mutations.append(record)
    record = copy.deepcopy(original)
    del record["margins"]
    mutations.append(record)
    for record in mutations:
        with pytest.raises(ValueError, match="does not match"):
            joint.check_certificate(record)


@pytest.mark.parametrize("document", [None, [], 1024, "certificate"])
def test_nonobject_documents_are_rejected(document):
    with pytest.raises(ValueError, match="JSON object"):
        joint.check_certificate(document)


def test_raw_line_ending_change_during_build_is_rejected_despite_same_portable_hash(monkeypatch):
    first = {"source": b"line\n"}
    second = {"source": b"line\r\n"}
    assert joint._source_hashes(first) == joint._source_hashes(second)
    sequence = iter([first, second])
    monkeypatch.setattr(joint, "_raw_sources", lambda: next(sequence))
    with pytest.raises(RuntimeError, match="changed"):
        joint.build_certificate(6, 1024)


def test_raw_line_ending_change_during_verification_is_rejected(monkeypatch):
    record = joint.build_certificate(6, 1024)
    first = joint._raw_sources()
    second = {name: data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for name, data in first.items()}
    sequence = iter([first, second])
    monkeypatch.setattr(joint, "_raw_sources", lambda: next(sequence))
    with pytest.raises(RuntimeError, match="changed"):
        joint.check_certificate(record)


def test_cli_roundtrip_and_tampered_certificate_rejection(tmp_path, capsys):
    output = tmp_path / "joint.json"
    assert joint.main(["--vertices", "6", "--degree-floor-exponent", "1024", "--output", str(output)]) == 0
    document = json.loads(output.read_bytes())
    assert document == json.loads(capsys.readouterr().out)
    assert joint.main(["--check", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    document["margins"]["binary_exponent_budget"] += 1
    output.write_text(json.dumps(document))
    with pytest.raises(SystemExit) as error:
        joint.main(["--check", str(output)])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--vertices", "7", "--degree-floor-exponent", "1024"],
        ["--vertices", "6", "--degree-floor-exponent", "1023"],
        ["--degree-floor-exponent", "1024"],
        ["--check", "missing.json"],
    ],
)
def test_invalid_cli_or_check_does_not_overwrite_previous_output(tmp_path, arguments):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous\n")
    with pytest.raises(SystemExit) as error:
        joint.main([*arguments, "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


def test_cli_missing_output_is_rejected():
    with pytest.raises(SystemExit) as error:
        joint.main(["--vertices", "6", "--degree-floor-exponent", "1024"])
    assert error.value.code == 2


def test_output_failure_keeps_previous_certificate(tmp_path, monkeypatch):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous\n")

    def fail(path, record):
        raise OSError("cannot replace output")

    monkeypatch.setattr(joint, "atomic_json", fail)
    with pytest.raises(SystemExit) as error:
        joint.main(["--vertices", "6", "--degree-floor-exponent", "1024", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


def test_real_crlf_checkout_verifies_lf_certificate_and_rejects_genuine_source_edit(tmp_path):
    source = Path(joint.__file__).parent
    copies = {}
    for style in ("lf", "crlf"):
        checkout = tmp_path / style
        directory = checkout / "tools/risk_research"
        directory.mkdir(parents=True)
        for name in joint._raw_sources():
            data = (source / name).read_bytes().replace(b"\r\n", b"\n")
            (directory / name).write_bytes(data if style == "lf" else data.replace(b"\n", b"\r\n"))
        copies[style] = checkout
    output = tmp_path / "portable.json"
    command = [sys.executable, "-S", "-m", "tools.risk_research.vertex_joint_growth"]
    for style, arguments in (
        ("lf", ["--vertices", "6", "--degree-floor-exponent", "1024", "--output", str(output)]),
        ("crlf", ["--check", str(output)]),
    ):
        result = subprocess.run(
            [*command, *arguments],
            cwd=copies[style],
            env={**os.environ, "PYTHONPATH": str(copies[style])},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    changed = copies["crlf"] / "tools/risk_research/vertex_general.py"
    changed.write_bytes(changed.read_bytes() + b"\r\n# changed content\r\n")
    result = subprocess.run(
        [*command, "--check", str(output)],
        cwd=copies["crlf"],
        env={**os.environ, "PYTHONPATH": str(copies["crlf"])},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2 and "sources" in result.stderr
