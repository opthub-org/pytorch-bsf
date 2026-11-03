"""Check all-index height certificates and their independent saved contracts."""

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction
from math import factorial
from pathlib import Path

import pytest

from tools.risk_research import leading_constant_height as height
from tools.risk_research import leading_constant_mzv as compiler

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("index", [3, 5, 8, 11])
def test_compiled_final_coefficients_have_the_proved_common_denominator(index):
    bounds = height.height_bounds(index)
    expression = compiler.leading_constant_expression(index, 1_000_000)
    common = factorial(index) ** bounds["denominator_power"]
    for value in expression.values():
        assert common % value.denominator == 0
        assert abs(value.numerator).bit_length() <= bounds["stored_numerator_or_denominator_bits"]
        assert value.denominator.bit_length() <= bounds["stored_numerator_or_denominator_bits"]


@pytest.mark.parametrize("order", range(1, 13))
def test_integer_phi_coefficients_have_exact_factorial_l1_norm(order):
    polynomial = compiler._phi_polynomial(order, compiler._budget(10000))
    assert all(value.denominator == 1 for value in polynomial)
    assert sum(map(abs, polynomial)) == factorial(order) ** 2


@pytest.mark.parametrize("first,second", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("shift", [1, 5, 17])
def test_two_pole_multipliers_have_cubic_denominator_and_bounded_size(first, second, shift):
    poles = compiler._partial_fractions({0: first, shift: second}, compiler._budget(1000))
    assert all(shift**3 % value.denominator == 0 and abs(value) <= 2 for value in poles.values())
    for n in (1, 3, 19):
        actual = sum((value / (n + offset) ** power for (offset, power), value in poles.items()), Fraction(0))
        assert actual == Fraction(1, n**first * (n + shift) ** second)


@pytest.mark.parametrize("bad", [None, True, False, 2, 0, -1, 3.0, "3"])
def test_index_validation_is_strict(bad):
    with pytest.raises(ValueError, match="constant index"):
        height.height_bounds(bad)


def test_bound_only_api_does_not_allocate_factorial_or_old_exponential(monkeypatch):
    monkeypatch.setattr(height, "leading_constant_expression", lambda *args: pytest.fail("compiler called"))
    index = 10**5000
    bounds = height.height_bounds(index)
    assert bounds["denominator_factorial_base"] == index
    assert bounds["denominator_power"] > index
    assert bounds["stored_numerator_or_denominator_bits"].bit_length() < 3 * index.bit_length()


def test_saved_record_restores_nondefault_options_and_preserves_input():
    record = height.build_record(8, 180000)
    before = deepcopy(record)
    height.verify_record(record)
    assert record == before
    assert record["work_limit"] == 180000
    assert record["observations_include_intermediates"] is False
    assert record["observed_final_coefficients"]["denominators_divide_factorial_power"] is True


@pytest.mark.parametrize("field", ["bounds", "observed_final_coefficients", "mzv_terms", "scope", "constant_index"])
def test_record_tampering_is_rejected(field):
    record = height.build_record(5, 10000)
    record[field] = True
    with pytest.raises(ValueError):
        height.verify_record(record)


def test_record_type_aliases_and_extra_fields_are_rejected():
    record = height.build_record(3)
    record["observed_final_coefficients"]["denominators_divide_factorial_power"] = 1
    with pytest.raises(ValueError, match="differs"):
        height.verify_record(record)
    record = height.build_record(3)
    record["unrecorded"] = 0
    with pytest.raises(ValueError, match="differs"):
        height.verify_record(record)


@pytest.mark.parametrize("operation", ["build", "verify"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_direct_operations_reject_raw_source_changes(monkeypatch, operation, change):
    record = height.build_record(5)
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in height._sources().items()}
    if change == "line-endings":
        after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
        hashes = {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in after.items()}
        assert hashes == record["source_sha256"]
    else:
        after = dict(before)
        after[next(iter(after))] += b"\n# virtual source change\n"
    assert before != after
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(height, "_sources", sources)
    with pytest.raises(RuntimeError, match="source changed during calculation"):
        height.build_record(5) if operation == "build" else height.verify_record(record)
    assert calls == 2


@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_outer_cli_guard_is_independent_and_preserves_output(tmp_path, monkeypatch, change):
    record = height.build_record(5)
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in height._sources().items()}
    after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
    if change == "content":
        after[next(iter(after))] += b"# virtual change\n"
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(height, "_sources", sources)
    monkeypatch.setattr(height, "build_record", lambda *args: record)
    output = tmp_path / "record.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        height.main(["--output", str(output)])
    assert output.read_text() == "keep"


def test_work_budget_failure_preserves_existing_output(tmp_path):
    output = tmp_path / "keep.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        height.main(["--constant-index", "7", "--work-limit", "1", "--output", str(output)])
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="work limit"):
        height.build_record(10**5000)


def test_generation_cannot_replace_bound_sources():
    sources = height._sources()
    for name in sources:
        with pytest.raises(SystemExit):
            height.main(["--output", str(ROOT / name)])


def test_verify_cli_rejects_duplicate_keys_and_generation_options(tmp_path):
    output = tmp_path / "bad.json"
    output.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(SystemExit):
        height.main(["--verify", str(output)])
    with pytest.raises(SystemExit):
        height.main(["--verify", str(output), "--constant-index", "5"])


def test_crlf_records_match_and_replay_without_site_packages(tmp_path):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, raw in height._sources().items():
            dest = checkout / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
        output = tmp_path / (checkout.name + ".json")
        for args in (["--constant-index", "5", "--work-limit", "12345", "--output", str(output)], ["--verify", str(output)]):
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-m", "tools.risk_research.leading_constant_height", *args],
                cwd=checkout,
                env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]
