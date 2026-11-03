"""Check the c6 finite telescope, classical reduction and strict records."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from math import factorial
from pathlib import Path

import pytest

from tools.risk_research import c6_closed_form as closed
from tools.risk_research.vertex_general import _phi_rows

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def _boundary(n, h, g, j):
    numerator = (
        -(120 * n**3 + 420 * n**2 + 440 * n + 144) * h**3
        + (144 * n**3 + 504 * n**2 + 528 * n + 144) * h * g
        - (24 * n**3 + 84 * n**2 + 88 * n) * j
        + (-78 * n**3 - 183 * n**2 + 54 * n + 144) * h**2
        + (42 * n**3 + 111 * n**2 + 18 * n) * g
        - (30 * n**3 + 84 * n**2 + 72 * n) * h
    )
    return numerator / (36 * n**2 * (n + 1) * (n + 2))


def _finite_zeta(indices, maximum):
    if not indices:
        return Fraction(1)
    return sum(
        (Fraction(1, n ** indices[0]) * _finite_zeta(indices[1:], n - 1) for n in range(1, maximum + 1)),
        Fraction(0),
    )


def test_finite_telescope_matches_original_elementary_symmetric_rows():
    rows = _phi_rows(4, 43)
    h, g, j = Fraction(0), Fraction(0), Fraction(0)
    total, residual = Fraction(0), Fraction(0)
    assert _boundary(1, h, g, j) == 0
    for n in range(1, 44):
        original = 10 * rows[1][n - 1] * rows[4][n - 1] + 20 * rows[2][n - 1] * rows[3][n - 1]
        term = (-20 * h**3 + 24 * h * g - 4 * j + 47 * h**2 - 17 * g + 21 * h + 5) / (6 * n**2)
        new_h, new_g, new_j = h + Fraction(1, n), g + Fraction(1, n**2), j + Fraction(1, n**3)
        assert original / 120 == _boundary(n + 1, new_h, new_g, new_j) - _boundary(n, h, g, j) + term
        total += original / 120
        residual += term
        assert total == _boundary(n + 1, new_h, new_g, new_j) + residual
        h, g, j = new_h, new_g, new_j


@pytest.mark.parametrize("count", [1, 2, 3, 7, 13, 23])
def test_descending_mzv_coefficients_match_exact_finite_series(count):
    rows = _phi_rows(4, count)
    original = 6 * sum(
        (10 * rows[1][j] * rows[4][j] + 20 * rows[2][j] * rows[3][j] for j in range(count)),
        Fraction(0),
    )
    h = sum((Fraction(1, n) for n in range(1, count + 1)), Fraction(0))
    g = sum((Fraction(1, n**2) for n in range(1, count + 1)), Fraction(0))
    j = sum((Fraction(1, n**3) for n in range(1, count + 1)), Fraction(0))
    finite_mzv = sum(
        (coefficient * _finite_zeta(indices, count) for indices, coefficient in closed.MZV_COEFFICIENTS),
        Fraction(0),
    )
    assert original == 720 * _boundary(count + 1, h, g, j) + finite_mzv


@pytest.mark.parametrize(
    ("indices", "dual"),
    [((2, 1, 1, 1), (5,)), ((2, 1, 2), (2, 3)), ((2, 2, 1), (3, 2)), ((2, 1, 1), (4,)), ((2, 1), (3,))],
)
def test_binary_word_duality_uses_the_documented_descending_convention(indices, dual):
    word = "".join("0" * (index - 1) + "1" for index in indices)
    reversed_complement = "".join("1" if value == "0" else "0" for value in reversed(word))
    expected_word = "".join("0" * (index - 1) + "1" for index in dual)
    assert reversed_complement == expected_word


def test_signed_product_bounds_enclose_all_endpoint_choices(monkeypatch):
    intervals = {
        2: (Fraction(1), Fraction(2)),
        3: (Fraction(1), Fraction(2)),
        4: (Fraction(10), Fraction(11)),
        5: (Fraction(1), Fraction(2)),
    }
    monkeypatch.setattr(closed, "zeta_enclosure", lambda order, terms: intervals[order])
    lo, hi = closed.c6_enclosure(17)
    assert 0 < lo < hi
    for z2, z3, z4, z5 in product(*intervals.values()):
        value = 600 * z2 + 2520 * z3 + 13980 * z4 - 4320 * z2 * z3 - 10080 * z5
        assert lo <= value <= hi


def test_intervals_shrink_despite_signed_cancellation_and_scale_to_C7():
    lo, hi = closed.c6_enclosure(1)
    assert lo == 0 and hi > 152
    for terms in (2, 5, 11, 29, 65):
        left, right = closed.c6_enclosure(terms)
        assert lo <= left <= right <= hi
        lo, hi = left, right
    assert Fraction(152836, 1000) < lo < hi < Fraction(152837, 1000)
    record = json.loads(json.dumps(closed.build_record(29)))
    scale = Fraction(7**2 * factorial(6), 2**4)
    assert scale == 2205
    assert _fraction(record["C7"]["lower"]) == scale * _fraction(record["c6"]["lower"])
    assert _fraction(record["C7"]["upper"]) == scale * _fraction(record["c6"]["upper"])


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 2.0, "2"])
def test_invalid_terms_are_rejected(value):
    with pytest.raises(ValueError):
        closed.c6_enclosure(value)
    with pytest.raises(ValueError):
        closed.build_record(value)


def test_cli_saved_nondefault_terms_round_trip(tmp_path):
    out = tmp_path / "record.json"
    assert closed.main(["--terms", "17", "--output", str(out)]) == 0
    original = out.read_bytes()
    assert closed.main(["--verify", str(out)]) == 0
    assert out.read_bytes() == original
    assert json.loads(original)["terms"] == 17
    for options in (["--terms", "17"], ["--output", str(out)]):
        with pytest.raises(SystemExit):
            closed.main(["--verify", str(out), *options])
        assert out.read_bytes() == original


@pytest.mark.parametrize(
    "field",
    [
        "terms",
        "vertices",
        "c6",
        "C7",
        "zeta_coefficients",
        "zeta_product_coefficients",
        "mzv_coefficients_descending",
        "source_sha256",
        "extra",
    ],
)
def test_record_tampering_is_rejected(field):
    record = json.loads(json.dumps(closed.build_record(8)))
    if field in ("terms", "vertices"):
        record[field] = float(record[field])
    elif field in ("c6", "C7"):
        record[field]["lower"]["numerator"] = "123"
    elif field == "extra":
        record[field] = "unexpected"
    else:
        record[field] = {}
    with pytest.raises(ValueError):
        closed.verify_record(record)


@pytest.mark.parametrize("record", [None, [], {}, {"schema": "wrong"}, {"schema": closed.SCHEMA}])
def test_unsupported_or_incomplete_records_fail(record):
    with pytest.raises(ValueError):
        closed.verify_record(record)


def test_duplicate_keys_nan_and_invalid_inputs_preserve_files(tmp_path):
    out = tmp_path / "keep.json"
    out.write_text("keep")
    for args in (["--terms", "0"], ["--terms", "-1"], ["--terms", "bad"]):
        with pytest.raises(SystemExit):
            closed.main([*args, "--output", str(out)])
        assert out.read_text() == "keep"
    for text in ('{"schema":"a","schema":"b"}', "NaN", "[]"):
        saved = tmp_path / "invalid.json"
        saved.write_text(text)
        with pytest.raises(SystemExit):
            closed.main(["--verify", str(saved)])
        assert saved.read_text() == text
    with pytest.raises(SystemExit):
        closed.main([])


def test_protected_sources_and_mid_calculation_edit_preserve_output(tmp_path, monkeypatch):
    for relative in closed._sources():
        with pytest.raises(SystemExit):
            closed.main(["--output", str(ROOT / relative)])
    out = tmp_path / "keep.json"
    out.write_text("keep")
    original = closed._sources()
    calls = 0

    def changed_sources():
        nonlocal calls
        calls += 1
        return original if calls == 1 else {k: v.replace(b"\n", b"\r\n") for k, v in original.items()}

    monkeypatch.setattr(closed, "_sources", changed_sources)
    with pytest.raises(SystemExit):
        closed.main(["--output", str(out)])
    assert out.read_text() == "keep"


@pytest.mark.parametrize("operation", ["build", "verify"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_direct_record_operations_reject_raw_source_changes(monkeypatch, operation, change):
    record = closed.build_record(7)
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in closed._sources().items()}
    if change == "line-endings":
        after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
        for snapshot in (before, after):
            hashes = {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in snapshot.items()}
            assert hashes == record["source_sha256"]
    else:
        after = dict(before)
        first = next(iter(after))
        after[first] += b"\n# virtual source change\n"
    assert before != after
    calls = 0

    def fake_sources():
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(closed, "_sources", fake_sources)
    with pytest.raises(RuntimeError, match="a closed-form source changed during calculation"):
        if operation == "build":
            closed.build_record(7)
        else:
            closed.verify_record(record)
    assert calls == 2


def test_canonical_parent_is_used_even_after_alias_retargets(tmp_path, monkeypatch):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    protect = closed._protect_output
    calls, written = 0, []

    def retarget_after_check(path, sources):
        nonlocal calls
        protect(path, sources)
        calls += 1
        if calls == 2:
            alias.unlink()
            alias.symlink_to(ROOT / "tools/risk_research", target_is_directory=True)

    monkeypatch.setattr(closed, "_protect_output", retarget_after_check)
    monkeypatch.setattr(closed, "atomic_json", lambda path, value: written.append(path))
    assert closed.main(["--output", str(alias / "c6_closed_form.py")]) == 0
    assert written == [actual / "c6_closed_form.py"]


def test_crlf_provenance_is_portable_and_stdlib_only(tmp_path):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, raw in closed._sources().items():
            dest = checkout / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
        output = tmp_path / (checkout.name + ".json")
        result = subprocess.run(
            [sys.executable, "-S", "-m", "tools.risk_research.c6_closed_form", "--terms", "13", "--output", str(output)],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]


def test_native_original_series_integral_and_classical_expression_agree():
    executable = os.environ.get("BSF_FLINT_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for original positive-integral checks")
        executable = sys.executable
    script = r"""
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import c6_closed_form as closed
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(closed.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(x): return arb(fmpq(x.numerator,x.denominator))
before=ctx.prec
for bits in (73,117,181):
 with ctx.workprec(bits):
  exact=600*arb(2).zeta()+2520*arb(3).zeta()+13980*arb(4).zeta()-4320*arb(2).zeta()*arb(3).zeta()-10080*arb(5).zeta()
  integral=16*(2*arb(2).log())**6*integral_normalized_ball(7,bits=bits)
  lo,hi=closed.c6_enclosure(bits//2)
  assert integral.overlaps(exact) and exact>ball(lo) and exact<ball(hi)
  for terms in (47,109,173):
   left,right=leading_constant_enclosure(7,terms)
   assert integral>ball(left/2205) and integral<ball(right/2205)
assert ctx.prec==before
"""
    result = subprocess.run(
        [executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
