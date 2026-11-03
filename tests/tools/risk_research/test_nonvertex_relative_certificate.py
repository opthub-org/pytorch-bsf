"""Check true-value accuracy composition, refusals and complete replay."""

import copy
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _scalar_risk_sum
from tools.risk_research import nonvertex_relative_certificate as compose
from tools.risk_research.codimension_three_exact import nonvertex_leading_constant_enclosure

ROOT = Path(__file__).resolve().parents[3]
DEGREE = 2**80 + 17


def _fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


@pytest.mark.parametrize("size,outside", [(2, 2), (3, 4), (4, 5)])
def test_two_errors_meet_true_value_accuracy_and_original_constant_overlap(size, outside):
    record = compose.certificate(size, outside, DEGREE, Fraction(1, 100), 190000, 180000)
    interval = record["coefficient_enclosure"]
    assert isinstance(interval, dict)
    lower, upper = _fraction(interval["lower"]), _fraction(interval["upper"])
    estimate = _fraction(record["approximation"])
    certified = _fraction(record["certified_relative_error"])
    assert 0 < lower <= upper and certified <= Fraction(1, 100)
    # Independently maximize abs(estimate/value-1): monotonicity places its
    # extrema at the endpoints. The true coefficient is not computed here.
    assert estimate / lower - 1 == 1 - estimate / upper == certified
    assert lower < estimate < upper
    for value in (lower, (2 * lower + upper) / 3, (lower + upper) / 2, upper):
        assert abs(estimate / value - 1) <= certified
    error = _fraction(record["analytic_relative_error"])
    delta = _fraction(record["constant_relative_width"])
    assert certified == (delta + error) / (1 + delta * error)
    assert delta <= _fraction(record["constant_relative_allowance"])
    constant = record["constant_interval"]
    assert isinstance(constant, dict)
    bits = record["precision_bits"]
    assert type(bits) is int
    assert _fraction(constant["lower"]) >= outside + 1
    assert _fraction(constant["upper"]) - _fraction(constant["lower"]) <= Fraction(1, 2**bits)
    # This original positive-series interval has an independent truncation
    # argument. Overlap is a check, not proof of nesting in a broad oracle.
    old_lower, old_upper = nonvertex_leading_constant_enclosure(size, outside, 37)
    leading = record["leading_constant"]
    assert isinstance(leading, dict)
    assert max(old_lower, _fraction(leading["lower"])) <= min(old_upper, _fraction(leading["upper"]))
    work = record["numeric_work_units"]
    assert type(work) is int and 0 < work <= 180000
    compose.verify_certificate(record)


@pytest.mark.parametrize("size,outside,degree", [(2, 2, 2), (3, 4, 5), (4, 3, 7)])
def test_small_original_finite_coefficients_are_positive_but_bound_refuses(size, outside, degree, monkeypatch):
    assert _scalar_risk_sum(size + outside, degree, size) > 0
    monkeypatch.setattr(compose.mzv_dyadic, "_construct", lambda *a: pytest.fail("MZV work before analytic refusal"))
    with pytest.raises(ValueError, match="current analytic bound cannot certify"):
        compose.certificate(size, outside, degree, Fraction(99, 100))


@pytest.mark.parametrize("size,outside,degree", [(3, 2, 1), (5, 4, 3)])
def test_inactive_face_is_exact_zero_before_analytic_or_mzv_work(size, outside, degree, monkeypatch):
    assert _scalar_risk_sum(size + outside, degree, size) == 0
    monkeypatch.setattr(
        compose.nonvertex_general_error, "relative_error_bound", lambda *a, **k: pytest.fail("inactive analytic work")
    )
    monkeypatch.setattr(compose.mzv_dyadic, "_construct", lambda *a: pytest.fail("inactive MZV work"))
    record = compose.certificate(size, outside, degree, Fraction(1, 100), 190000, 180000)
    assert record["status"] == "exact_zero_inactive_face" and record["active"] is False
    assert _fraction(record["approximation"]) == 0
    interval = record["coefficient_enclosure"]
    assert isinstance(interval, dict)
    assert all(_fraction(value) == 0 for value in interval.values())
    assert record["certified_relative_error"] is None and record["analytic_relative_error"] is None
    assert record["constant_interval"] is None and record["numeric_work_units"] == 0
    compose.verify_certificate(record)


@pytest.mark.parametrize("target", [True, False, 0, 0.01, "1/100", None, Fraction(0), Fraction(-1), Fraction(1), Fraction(2)])
def test_invalid_target_rejected_before_component_construction(target, monkeypatch):
    monkeypatch.setattr(
        compose.nonvertex_general_error, "relative_error_bound", lambda *a, **k: pytest.fail("analytic work before validation")
    )
    monkeypatch.setattr(compose.mzv_dyadic, "_construct", lambda *a: pytest.fail("MZV work before validation"))
    with pytest.raises(ValueError, match="relative_target"):
        compose.certificate(2, 2, DEGREE, target)


@pytest.mark.parametrize("bad", [True, False, None, 1.5, "3", 0, -1])
def test_strict_input_and_resource_types(bad):
    with pytest.raises(ValueError):
        compose.certificate(bad, 2, DEGREE, Fraction(1, 100))
    with pytest.raises(ValueError):
        compose.certificate(2, bad, DEGREE, Fraction(1, 100))
    with pytest.raises(ValueError):
        compose.certificate(2, 2, bad, Fraction(1, 100))
    for kwargs in ({"analytic_work_limit": bad}, {"numeric_work_limit": bad}):
        with pytest.raises(ValueError):
            compose.certificate(2, 2, DEGREE, Fraction(1, 100), **kwargs)


@pytest.mark.parametrize(
    "target,analytic,numeric",
    [(Fraction(1, 2**300), 100, 200000), (Fraction(1, 100), 1000001, 200000), (Fraction(1, 100), 200000, 1000001)],
)
def test_resource_and_target_bit_refusal_precedes_components(target, analytic, numeric, monkeypatch):
    monkeypatch.setattr(
        compose.nonvertex_general_error, "relative_error_bound", lambda *a, **k: pytest.fail("analytic work before validation")
    )
    monkeypatch.setattr(compose.mzv_dyadic, "_construct", lambda *a: pytest.fail("MZV work before validation"))
    with pytest.raises(ValueError, match="work"):
        compose.certificate(2, 2, DEGREE, target, analytic, numeric)


def test_analytic_target_equality_conservatively_refuses_without_constant_work(monkeypatch):
    target = compose.nonvertex_general_error.relative_error_bound(2, 2, DEGREE)
    monkeypatch.setattr(compose.mzv_dyadic, "_construct", lambda *a: pytest.fail("MZV work before equality refusal"))
    with pytest.raises(ValueError, match="bound < target"):
        compose.certificate(2, 2, DEGREE, target)


def test_numeric_refusal_has_no_precision_retry(monkeypatch):
    original = compose.mzv_dyadic._construct
    calls = []

    def track(index, budget):
        calls.append(budget)
        return original(index, budget)

    monkeypatch.setattr(compose.mzv_dyadic, "_construct", track)
    with pytest.raises(ValueError, match="work limit"):
        compose.certificate(3, 4, DEGREE, Fraction(1, 100), numeric_work_limit=30)
    assert len(calls) == 1 and calls[0].used > 30


@pytest.mark.parametrize("width", [Fraction(3), Fraction(1, 2), Fraction(1, 8), Fraction(3, 19), Fraction(1, 2**177)])
def test_preselected_absolute_precision(width):
    bits = compose._precision_bits(width)
    assert bits >= 1 and Fraction(1, 2**bits) <= width
    if bits > 1:
        assert Fraction(1, 2 ** (bits - 1)) > width


@pytest.mark.parametrize("failure", [False, True])
def test_whole_composition_raw_source_guard_covers_failures(monkeypatch, failure):
    state = {"source": b"before"}
    monkeypatch.setattr(compose, "_sources", lambda: dict(state))

    def changed(*args, **kwargs):
        state["source"] = b"after\r\n"
        if failure:
            raise ValueError("component refused")
        return Fraction(0)

    monkeypatch.setattr(compose.nonvertex_general_error, "relative_error_bound", changed)
    with pytest.raises(RuntimeError, match="composition sources"):
        compose.certificate(2, 2, DEGREE, Fraction(1, 100))


def test_whole_composition_head_tree_status_guard(monkeypatch):
    counter = 0

    def snapshot(*args):
        nonlocal counter
        counter += 1
        return counter

    monkeypatch.setattr(compose, "checkout_state", snapshot)
    with pytest.raises(RuntimeError, match="HEAD, tree or checkout status"):
        compose.certificate(3, 2, 1, Fraction(1, 100))


@pytest.mark.parametrize("field", ["degree", "numeric_work_limit", "approximation", "source_sha256", "extra"])
def test_full_typed_replay_rejects_mutation(field):
    record = compose.certificate(3, 2, 1, Fraction(1, 91), 180000, 170000)
    changed = copy.deepcopy(record)
    changed[field] = True
    with pytest.raises((ValueError, TypeError)):
        compose.verify_certificate(changed)


def test_cli_restores_options_and_rejects_overrides_preserving_output(tmp_path):
    output = tmp_path / "success.json"
    argv = [
        "--face-size",
        "3",
        "--outside",
        "4",
        "--degree",
        str(DEGREE),
        "--relative-target",
        "1/100",
        "--analytic-work-limit",
        "190000",
        "--numeric-work-limit",
        "180000",
        "--output",
        str(output),
    ]
    assert compose.main(argv) == 0
    saved = output.read_bytes()
    assert compose.main(["--verify", str(output)]) == 0
    with pytest.raises(SystemExit) as error:
        compose.main(["--verify", str(output), "--numeric-work-limit", "200000"])
    assert error.value.code != 0 and output.read_bytes() == saved
    with pytest.raises(SystemExit):
        compose.main(
            ["--face-size", "2", "--outside", "2", "--degree", "4", "--relative-target", "1/100", "--output", str(output)]
        )
    assert output.read_bytes() == saved
    with pytest.raises(SystemExit):
        compose.main(["--verify", str(output), "--output", str(output)])
    assert output.read_bytes() == saved


def test_cli_input_mutation_and_duplicate_keys_fail_without_output(tmp_path, monkeypatch):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    record = compose.certificate(3, 2, 1, Fraction(1, 91), 180000, 170000)
    source.write_text(json.dumps(record))

    def mutate(*args):
        source.write_text("{}")

    monkeypatch.setattr(compose, "verify_certificate", mutate)
    with pytest.raises(SystemExit):
        compose.main(["--verify", str(source), "--output", str(output)])
    assert not output.exists()
    source.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(SystemExit):
        compose.main(["--verify", str(source)])


def test_stdlib_cli_and_protected_sources(tmp_path):
    output = tmp_path / "record.json"
    env = {**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-S", "-m", "tools.risk_research.nonvertex_relative_certificate"]
    result = subprocess.run(
        [
            *command,
            "--face-size",
            "3",
            "--outside",
            "4",
            "--degree",
            str(DEGREE),
            "--relative-target",
            "1/100",
            "--analytic-work-limit",
            "190000",
            "--numeric-work-limit",
            "180000",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [*command, "--verify", str(output)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    sources = compose._sources()
    assert len(sources) >= 19 and compose.PROOF in sources
    for path in sources:
        with pytest.raises(ValueError, match="output must not overwrite"):
            compose._protect_output(ROOT / path, sources, None)
