"""Exact proof arithmetic, saved-option replay and independent constants."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from tools.risk_research import vertex_large_dimension as previous
from tools.risk_research import vertex_large_dimension_refined as refined
from tools.risk_research.checkout_harness import checkout_state, run_python

ROOT = Path(__file__).resolve().parents[3]


def test_signed_fourth_derivative_bounds_and_gaussian_constants_are_exact():
    x = sp.Symbol("x", real=True)
    lower = x**2 - x**4 / 6 + 2 * (1 - x**2 / 2) * (sp.Rational(2, 3) - x**2 / 6)
    assert sp.expand(lower) == sp.Rational(4, 3)
    upper = (
        x**2
        - x**4 / 6
        + x**6 / 120
        + 2 * (1 - x**2 / 2 + x**4 / 24) * (sp.Rational(7, 10) - x**2 / 8)
        + sp.Rational(1, 2)
        - x**2 / 8
        + x**4 / 96
        - 2
    )
    assert sp.expand(upper) == -sp.Rational(1, 10) - 3 * x**2 / 40 + 13 * x**4 / 480 - x**6 / 480
    assert Fraction(842, 1215) > Fraction(29, 42)
    assert Fraction(1, 14) + Fraction(1, 192) < Fraction(1, 13)
    assert Fraction(1, 192) + Fraction(1, 312) == Fraction(7, 832)
    assert Fraction(5887, 3328) + Fraction(29, 12) < Fraction(17, 4)
    assert Fraction(23, 4) + Fraction(21, 8 * 160) < 6
    assert Fraction(127, 10) ** 2 > 160
    assert 8 * 160 * Fraction(127, 10) < 2**14
    assert 14 * Fraction(7, 10) < Fraction(158, 16) < 10


@pytest.mark.parametrize("vertices", [4, 5, 17, 161, 162, 163, 601, 602, 1001, 100000])
@pytest.mark.parametrize("bits", [32, 47, 128])
def test_refined_enclosures_are_ordered_and_improve_the_old_bound(vertices, bits):
    lower, upper = refined.normalized_constant_enclosure(vertices, bits=bits)
    old_lower, old_upper = previous.normalized_constant_enclosure(vertices, bits=bits)
    assert isinstance(lower, Fraction) and isinstance(upper, Fraction)
    assert 0 <= old_lower <= lower < upper <= old_upper
    if vertices >= 162:
        assert lower > 0
        assert refined.relative_remainder_bound(vertices) == Fraction(6, vertices - 2)


@pytest.mark.parametrize("target", [Fraction(1, 100), Fraction(1, 1000), Fraction(2), Fraction(7, 300)])
def test_threshold_guarantees_the_target_at_every_larger_vertex_count(target):
    threshold = refined.relative_error_threshold(target)
    assert threshold >= 162 and threshold < previous.relative_error_threshold(target)
    for vertices in (threshold, threshold + 1, 2 * threshold, threshold**2):
        assert refined.relative_remainder_bound(vertices) <= target
    if threshold > 162:
        assert refined.relative_remainder_bound(threshold - 1) > target
    assert refined.relative_error_threshold() == 602


@pytest.mark.parametrize("vertices", [True, 3, 0, 6.0, None])
@pytest.mark.parametrize("function", [refined.relative_remainder_bound, refined.normalized_constant_enclosure])
def test_invalid_vertex_domains_are_rejected(function, vertices):
    with pytest.raises(ValueError, match="vertices"):
        function(vertices)


@pytest.mark.parametrize("target", [0, True, Fraction(0), Fraction(-1), 0.1])
def test_invalid_relative_targets_are_rejected(target):
    with pytest.raises(ValueError, match="target"):
        refined.relative_error_threshold(target)


@pytest.mark.parametrize("bits", [True, 31, 0, 64.0])
def test_precision_is_rejected_before_optional_backend_import(bits, monkeypatch):
    def forbidden(name):
        raise AssertionError("optional import occurred before precision validation")

    monkeypatch.setattr(refined, "import_module", forbidden)
    with pytest.raises(ValueError, match="bits"):
        refined.constant_ball(602, bits=bits)


@pytest.mark.parametrize("field,value", [("with_arb", 1), ("integral_check", "false")])
def test_backend_flags_have_strict_boolean_types(field, value):
    with pytest.raises(ValueError, match="booleans"):
        refined.build_record(602, **{field: value})


def test_recorded_nondefault_precision_and_target_are_replayed_exactly():
    record = refined.build_record(602, bits=47, target=Fraction(7, 3000))
    snapshot = copy.deepcopy(record)
    refined.verify_record(record)
    assert record == snapshot
    assert record["bits"] == 47 and record["relative_error_threshold"] == "2574"
    assert record["source_sha256"] == {
        name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in refined._sources().items()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("vertices", 602.0),
        ("bits", True),
        ("with_arb", 0),
        ("integral_check", 0),
        ("simple_remainder_numerator", 6.0),
        ("relative_error_bound", {"numerator": "5", "denominator": "600"}),
        ("extra", "unrecognized"),
        ("source_hash_policy", "unrecognized"),
        ("relative_error_target", {"numerator": 1, "denominator": "100"}),
        ("relative_error_target", {"numerator": "1", "denominator": "0"}),
        ("relative_error_target", {"numerator": "1", "denominator": "100", "extra": "field"}),
    ],
)
def test_tampered_records_and_json_scalar_type_changes_are_rejected(field, value):
    record = refined.build_record(602, bits=47)
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        refined.verify_record(record)


@pytest.fixture
def isolated_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    names = list(refined._sources()) + ["tools/risk_research/checkout_harness.py", "tools/risk_research/vertex_general.py"]
    for name in names:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    for arguments in (["init", "-q"], ["add", "."], ["commit", "-qm", "Fixture source"]):
        subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Harness Test",
                "-c",
                "user.email=harness@example.invalid",
                *arguments,
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
    return checkout


def _run(checkout, arguments, *, executable=sys.executable, no_site=False):
    state = checkout_state(checkout)
    return run_python(
        checkout,
        arguments,
        modules=("tools.risk_research.vertex_large_dimension_refined",),
        executable=executable,
        expected_head=state.head,
        expected_tree=state.tree,
        no_site=no_site,
        timeout=60,
    )


def test_stdlib_cli_roundtrip_preserves_input_and_rejects_duplicates(isolated_checkout, tmp_path):
    output = tmp_path / "record.json"
    generated = _run(
        isolated_checkout,
        [
            "-m",
            "tools.risk_research.vertex_large_dimension_refined",
            "--vertices",
            "602",
            "--bits",
            "47",
            "--relative-target",
            "7/3000",
            "--output",
            str(output),
        ],
        no_site=True,
    )
    assert generated.returncode == 0, generated.stderr
    before = output.read_bytes()
    for _ in range(2):
        verified = _run(
            isolated_checkout,
            ["-m", "tools.risk_research.vertex_large_dimension_refined", "--verify", str(output)],
            no_site=True,
        )
        assert verified.returncode == 0, verified.stderr
        assert output.read_bytes() == before
    output.write_text(before.decode().replace('"bits": 47', '"bits": 47, "bits": 47'))
    duplicate = output.read_bytes()
    result = _run(
        isolated_checkout, ["-m", "tools.risk_research.vertex_large_dimension_refined", "--verify", str(output)], no_site=True
    )
    assert result.returncode == 2 and "duplicate JSON key" in result.stderr
    assert output.read_bytes() == duplicate


def test_invalid_generation_and_verify_overrides_preserve_previous_output(tmp_path):
    output = tmp_path / "record.json"
    output.write_bytes(b"previous\n")
    for arguments in (
        ["--vertices", "3", "--output", str(output)],
        ["--vertices", "602", "--bits", "31", "--output", str(output)],
        ["--verify", str(output), "--bits", "128"],
        ["--verify", str(output), "--output", str(output)],
    ):
        with pytest.raises(SystemExit) as error:
            refined.main(arguments)
        assert error.value.code == 2 and output.read_bytes() == b"previous\n"


@pytest.mark.parametrize(
    "name",
    [
        "tools/risk_research/vertex_large_dimension_refined.py",
        "tools/risk_research/vertex_large_dimension.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        "docs/advanced/skeleton_risk_vertex_large_dimension_refined.rst",
    ],
)
@pytest.mark.parametrize("alias_kind", ["direct", "leaf-symlink", "parent-symlink"])
def test_output_source_destinations_are_rejected_before_computation_or_write(name, alias_kind, tmp_path, monkeypatch):
    source = ROOT / name
    assert name in refined._sources()
    before = source.read_bytes()
    output = source
    if alias_kind == "leaf-symlink":
        output = tmp_path / "aliased-source"
        try:
            output.symlink_to(source)
        except OSError as error:
            pytest.skip(f"symbolic links are unavailable: {error}")
    elif alias_kind == "parent-symlink":
        parent = tmp_path / "aliased-directory"
        try:
            parent.symlink_to(source.parent, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symbolic links are unavailable: {error}")
        output = parent / source.name
    events = []

    def build_without_side_effects(*args, **kwargs):
        events.append("build")
        return {"patched_record": "safe red regression"}

    def write_without_side_effects(*args, **kwargs):
        events.append("write")

    monkeypatch.setattr(refined, "build_record", build_without_side_effects)
    monkeypatch.setattr(refined, "atomic_json", write_without_side_effects)
    try:
        with pytest.raises(SystemExit) as error:
            refined.main(["--vertices", "602", "--bits", "32", "--output", str(output)])
        assert error.value.code == 2
        assert events == []
    finally:
        # Even against the unsafe implementation, both production side-effect
        # points are patched and the protected source is never written.
        assert source.read_bytes() == before


def test_generation_can_replace_an_unrelated_output_and_verify_it(tmp_path, capsys):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous\n")
    assert refined.main(["--vertices", "602", "--bits", "32", "--output", str(output)]) == 0
    record = json.loads(output.read_text())
    assert record["vertices"] == 602 and record["bits"] == 32
    before = output.read_bytes()
    assert refined.main(["--verify", str(output)]) == 0
    assert output.read_bytes() == before
    assert "verified" in capsys.readouterr().out


def test_concurrent_source_changes_and_backend_failures_preserve_output(tmp_path, monkeypatch):
    output = tmp_path / "record.json"
    output.write_bytes(b"previous\n")
    actual = refined._sources()
    calls = iter([actual, {**actual, "changed": b"source"}])
    monkeypatch.setattr(refined, "_sources", lambda: next(calls))
    with pytest.raises(SystemExit) as error:
        refined.main(["--vertices", "602", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"
    monkeypatch.setattr(refined, "_sources", lambda: actual)

    def fail(name):
        raise ModuleNotFoundError("simulated missing optional backend")

    monkeypatch.setattr(refined, "import_module", fail)
    with pytest.raises(SystemExit) as error:
        refined.main(["--vertices", "602", "--with-arb", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


def test_crlf_fingerprints_match_and_real_edits_fail_replay(monkeypatch):
    original = refined._sources()
    monkeypatch.setattr(refined, "_sources", lambda: original)
    record = refined.build_record(602, bits=47)
    crlf = {name: raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for name, raw in original.items()}
    monkeypatch.setattr(refined, "_sources", lambda: crlf)
    refined.verify_record(record)
    changed = dict(crlf)
    name = next(iter(changed))
    changed[name] += b"# Genuine source edit\r\n"
    monkeypatch.setattr(refined, "_sources", lambda: changed)
    with pytest.raises(ValueError, match="differs"):
        refined.verify_record(record)


def test_real_crlf_checkout_replays_lf_record_and_detects_committed_edit(isolated_checkout, tmp_path):
    output = tmp_path / "line-endings.json"
    module = "tools.risk_research.vertex_large_dimension_refined"
    generated = _run(isolated_checkout, ["-m", module, "--vertices", "602", "--bits", "47", "--output", str(output)])
    assert generated.returncode == 0, generated.stderr
    before = output.read_bytes()
    for name in refined._sources():
        path = isolated_checkout / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    def commit_fixture():
        for arguments in (["add", "."], ["commit", "-qm", "Fixture line endings"]):
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "core.autocrlf=false",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "user.name=Harness Test",
                    "-c",
                    "user.email=harness@example.invalid",
                    *arguments,
                ],
                cwd=isolated_checkout,
                check=True,
                capture_output=True,
            )

    commit_fixture()
    result = _run(isolated_checkout, ["-m", module, "--verify", str(output)], no_site=True)
    assert result.returncode == 0, result.stderr
    changed = isolated_checkout / "tools/risk_research/exact.py"
    changed.write_bytes(changed.read_bytes() + b"\r\n# Genuine source edit\r\n")
    commit_fixture()
    result = _run(isolated_checkout, ["-m", module, "--verify", str(output)], no_site=True)
    assert result.returncode == 2 and "differs" in result.stderr
    assert output.read_bytes() == before


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("optional python-flint is unavailable")
        executable = sys.executable
    return executable


def test_native_frozen_holdouts_and_series_use_independent_integrals(flint_python, isolated_checkout):
    script = """
from flint import arb,ctx
import flint
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball,_rational_ball
from tools.risk_research.vertex_large_dimension_refined import normalized_constant_enclosure,constant_ball
ctx.prec=73
for m in (7,13,37,163,257,601,1001):
    direct=integral_normalized_ball(m,bits=96)
    assert ctx.prec==73
    low,high=normalized_constant_enclosure(m,bits=96)
    with ctx.workprec(128):
        bound=_rational_ball(low,high,flint)
        assert bound.contains(direct),(m,bound,direct)
        if m in (7,13):
            low,high=leading_constant_enclosure(m,192)
            scale=m*m*arb(m).gamma()*(2*arb(2).log())**(m-1)
            assert (_rational_ball(low,high,flint)/scale).overlaps(direct)
assert constant_ball(4,bits=96).contains(48*arb.pi()**2)
assert constant_ball(5,bits=96).contains(7200*(arb(2).zeta()-arb(3).zeta()))
assert ctx.prec==73
print('frozen holdouts, independent series and normalization passed')
"""
    result = _run(isolated_checkout, ["-c", script], executable=flint_python)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [32, 47, 128])
def test_native_saved_balls_round_outward_and_replay_nondefault_options(flint_python, isolated_checkout, tmp_path, bits):
    output = tmp_path / "native.json"
    script = f"""
import json
from flint import arb,ctx
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
from tools.risk_research.vertex_large_dimension_refined import build_record,verify_record
ctx.prec=73
record=build_record(602,bits={bits},with_arb=True,integral_check=True)
assert ctx.prec==73
verify_record(record)
assert ctx.prec==73
with open({str(output)!r},'w') as stream:
    json.dump(record,stream)
record=json.loads(open({str(output)!r}).read())
with ctx.workprec(160):
    direct=integral_normalized_ball(602,bits=128)
    original=integral_normalized_ball(602,bits={bits})
    assert arb(record['normalized_integral_ball']).contains(original)
    assert arb(record['normalized_integral_ball']).overlaps(direct)
    scale=602**2*arb(602).gamma()*(2*arb(2).log())**601
    assert arb(record['constant_ball']).contains(scale*direct)
assert ctx.prec==73
print('outward saved balls and native recorded-option replay passed')
"""
    result = _run(isolated_checkout, ["-c", script], executable=flint_python)
    assert result.returncode == 0, result.stdout + result.stderr
