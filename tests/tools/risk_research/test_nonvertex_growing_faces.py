"""Check the moving-domain corollary, exact tails and strict saved replay."""

import copy
import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from tools.risk_research import nonvertex_growing_faces as growing

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


@pytest.mark.parametrize("root", [2, 3, 5, 19, 1 << 127])
def test_fourth_power_boundaries_without_rounding(root):
    for degree in (root**4 - 1, root**4, root**4 + 1):
        if degree < 16:
            continue
        actual = growing.face_limit(degree)
        assert actual**4 <= degree < (actual + 1) ** 4
        assert actual == root - (degree < root**4)


@pytest.mark.parametrize("outside", [2, 3, 4, 7, 11])
@pytest.mark.parametrize("degree", [16, 17, 80, 81, 82, 2400, 2401, 2402, (1 << 73) + 29])
def test_original_analytic_inequality_implies_new_uniform_bound(outside, degree):
    # Independent high-precision algebra check. This is not used as a proof
    # or as a substitute for an interval enclosure of an actual coefficient.
    with localcontext() as context:
        context.prec = 90
        d = Decimal(degree)
        logarithm = 1 + d.ln()
        root = d.sqrt()
        h = growing.face_limit(degree)
        q = growing.general.diagonal_majorant(outside)[0]
        majorant = Decimal(q.numerator) / Decimal(q.denominator)
        factor, tail = (logarithm, logarithm + 1) if outside == 2 else (Decimal(1), Decimal(1))
        new = (
            88 * logarithm / root
            if outside == 2
            else (Decimal(39) / 2 * majorant + Decimal(outside * (outside + 1)) / 4) / root
        )
        for size in {2, h, max(2, h - 1)}:
            assert degree >= max(4, 2 * (size - 1))
            prefactor = Decimal(outside * (2 * size + outside - 1)) / 4
            original = Decimal(29) / 2 * majorant * factor / root
            original += (prefactor + majorant * (2 * tail + factor * ((size - 2) * logarithm + (size - 1) * (size - 2)))) / d
            assert original <= new
        rational = growing.relative_error_bound(outside, degree)
        assert new <= Decimal(rational.numerator) / Decimal(rational.denominator)


@pytest.mark.parametrize("outside", [2, 3, 6, 9])
def test_all_tail_threshold_parities_and_moving_face_range(outside):
    for target in (Fraction(1000), Fraction(1, 37), Fraction(1, 10**19)):
        threshold = growing.relative_error_threshold(outside, target)
        exponent = threshold.bit_length() - 1
        assert exponent >= 4 and threshold == 1 << exponent
        assert growing.dyadic_tail_bound(outside, exponent) <= target
        if exponent > 4:
            assert growing.dyadic_tail_bound(outside, exponent - 1) > target
        for offset in range(5):
            assert growing.dyadic_tail_bound(outside, exponent + offset + 1) < growing.dyadic_tail_bound(
                outside, exponent + offset
            )
        for degree in (threshold, 2 * threshold - 1, 2 * threshold, 16 * threshold + 11):
            assert growing.relative_error_bound(outside, degree) <= target
        assert growing.face_limit(16 * threshold) == 2 * growing.face_limit(threshold) or (
            growing.face_limit(16 * threshold) == 2 * growing.face_limit(threshold) + 1
        )


@pytest.mark.parametrize("bad", [True, False, None, "2", 2.0, -1, 0, 1])
def test_strict_outside_types_and_resource_refusal_precede_majorant(bad, monkeypatch):
    monkeypatch.setattr(growing.general, "diagonal_majorant", lambda *a, **k: pytest.fail("majorant before refusal"))
    with pytest.raises(ValueError):
        growing.certificate(bad, 16)


@pytest.mark.parametrize("bad", [True, False, None, "16", 16.0, -1, 0, 15])
def test_strict_degree_domain(bad):
    with pytest.raises(ValueError):
        growing.face_limit(bad)
    with pytest.raises(ValueError):
        growing.certificate(2, bad)


@pytest.mark.parametrize("bad", [None, True, 0.1, "1/2", Fraction(0), Fraction(-1)])
def test_threshold_target_validation(bad, monkeypatch):
    monkeypatch.setattr(growing.general, "diagonal_majorant", lambda *a, **k: pytest.fail("majorant before target validation"))
    with pytest.raises(ValueError, match="target"):
        growing.relative_error_threshold(2, bad)


@pytest.mark.parametrize(
    "options",
    [{"outside": 1 << 100}, {"degree": 1 << 100}, {"target": Fraction(1, 1 << 100)}, {"work_limit": 1_000_001}],
)
def test_selected_work_before_majorant(options, monkeypatch):
    monkeypatch.setattr(growing.general, "diagonal_majorant", lambda *a, **k: pytest.fail("majorant before work refusal"))
    kwargs = {"outside": 2, "degree": 16, "work_limit": 79, **options}
    with pytest.raises(ValueError, match="work"):
        growing.certificate(**kwargs)


def test_exact_selected_work_boundary():
    # q2: 27+4+6*(degreebits+targetbits); no target uses 5 bits at D16.
    with pytest.raises(ValueError, match="selected work"):
        growing.certificate(2, 16, work_limit=60)
    record = growing.certificate(2, 16, work_limit=61)
    assert record["selected_input_work"] == 61


def test_record_roundtrip_and_exact_json_types():
    record = growing.certificate(7, (1 << 91) + 17, Fraction(1, 37), 190_000)
    growing.verify_certificate(record)
    all_tail = record["all_tail"]
    assert isinstance(all_tail, dict)
    assert _fraction(all_tail["relative_error_bound"]) <= Fraction(1, 37)
    maximum_face_size, degree = record["maximum_face_size"], record["degree"]
    assert type(maximum_face_size) is int and type(degree) is int
    assert maximum_face_size**4 <= degree
    for key, value in (("outside", True), ("extra", 1), ("maximum_face_size", 2), ("source_hash_policy", "different")):
        altered = copy.deepcopy(record)
        altered[key] = value
        with pytest.raises(ValueError):
            growing.verify_certificate(altered)


def test_raw_source_transaction_ignores_hash_normalization_only_for_record(monkeypatch):
    original = growing._sources()
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        return original if calls == 1 else {**original, growing.PROOF: original[growing.PROOF].replace(b"\n", b"\r\n")}

    monkeypatch.setattr(growing, "_sources", changed)
    with pytest.raises(RuntimeError, match="sources changed"):
        growing.certificate(2, 16)


@pytest.mark.parametrize("initial_line_ending", ["lf", "crlf"])
def test_outer_verifier_detects_raw_change_accepted_by_inner_builder(monkeypatch, initial_line_ending):
    sources = growing._sources()
    proof_lf = sources[growing.PROOF].replace(b"\r\n", b"\n")
    proof_crlf = proof_lf.replace(b"\n", b"\r\n")
    before, after = (proof_lf, proof_crlf) if initial_line_ending == "lf" else (proof_crlf, proof_lf)
    original = {**sources, growing.PROOF: before}
    changed_sources = {**sources, growing.PROOF: after}
    monkeypatch.setattr(growing, "_sources", lambda: original)
    record = growing.certificate(2, 16)
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        return original if calls == 1 else changed_sources

    monkeypatch.setattr(growing, "_sources", changed)
    with pytest.raises(RuntimeError, match="sources changed during verification"):
        growing.verify_certificate(record)
    # The inner builder saw stable alternate bytes and reproduced the normalized
    # record. The matched error identifies the outer transaction's rejection.


def test_cli_stdlib_saved_options_and_preserve_output(tmp_path):
    output = tmp_path / "saved.json"
    command = [sys.executable, "-S", "-m", "tools.risk_research.nonvertex_growing_faces"]
    environment = {**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(arguments):
        return subprocess.run(command + arguments, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)

    assert (
        run(
            [
                "--outside",
                "7",
                "--degree",
                str((1 << 91) + 17),
                "--relative-target",
                "1/37",
                "--work-limit",
                "190000",
                "--output",
                str(output),
            ]
        ).returncode
        == 0
    )
    original = output.read_bytes()
    record = json.loads(original)
    assert record["work_limit"] == 190_000
    assert run(["--verify", str(output)]).returncode == 0
    assert run(["--verify", str(output), "--work-limit", "190001", "--output", str(output)]).returncode == 2
    assert run(["--outside", "2", "--degree", "15", "--output", str(output)]).returncode == 2
    assert output.read_bytes() == original


def test_cli_duplicate_input_source_protection_and_changed_input(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    record = growing.certificate(2, 16)
    path.write_text(json.dumps(record))
    original = growing.verify_certificate

    def changed(value):
        original(value)
        path.write_text(json.dumps(record) + " ")

    monkeypatch.setattr(growing, "verify_certificate", changed)
    with pytest.raises(SystemExit):
        growing.main(["--verify", str(path)])
    with pytest.raises(SystemExit):
        growing.main(["--outside", "2", "--degree", "16", "--output", str(ROOT / growing.PROOF)])
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(SystemExit):
        growing.main(["--verify", str(path)])
