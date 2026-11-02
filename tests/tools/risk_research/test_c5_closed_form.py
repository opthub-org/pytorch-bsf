"""Check the finite c5 reduction and its independent rational evaluator."""

import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

import pytest

from tools.risk_research import c5_closed_form as closed
from tools.risk_research.vertex_general import _phi_rows

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


@pytest.mark.parametrize("terms", [1, 2, 3, 9, 23, 47])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_complete_homogeneous_transform_matches_binomial_definition(order, terms):
    expected = sum(
        (
            sum(((-1) ** j * comb(m, j) * Fraction(1, (j + 1) ** order) for j in range(m + 1)), Fraction(0)) / 2 ** (m + 1)
            for m in range(terms)
        ),
        Fraction(0),
    ) / (1 - Fraction(1, 2 ** (order - 1)))
    left, right = closed.zeta_enclosure(order, terms)
    assert left == expected
    assert right - left == Fraction(1, 2**terms) / (1 - Fraction(1, 2 ** (order - 1)))


def test_finite_telescope_matches_original_elementary_symmetric_rows():
    rows = _phi_rows(3, 39)
    h, g = Fraction(0), Fraction(0)
    total, residual = Fraction(0), Fraction(0)

    def boundary(n, harmonic, square):
        return (
            Fraction(6 * n**2 + 9 * n + 2, 2 * n**2 * (n + 1)) * harmonic**2
            - Fraction(2 * n + 3, 2 * n * (n + 1)) * square
            + harmonic / (n + 1)
        )

    for n in range(1, 40):
        old_e4 = 8 * rows[1][n - 1] * rows[3][n - 1] + 6 * rows[2][n - 1] ** 2
        term = (3 * h**2 - g - 5 * h - 1) / n**2
        next_h, next_g = h + Fraction(1, n), g + Fraction(1, n**2)
        assert old_e4 / 24 == boundary(n + 1, next_h, next_g) - boundary(n, h, g) + term
        total += old_e4 / 24
        residual += term
        assert total == boundary(n + 1, next_h, next_g) + residual
        h, g = next_h, next_g


@pytest.mark.parametrize("face_size", [2, 3, 4, 7, 13])
def test_q4_normalization_matches_existing_general_theorem(face_size):
    scale = Fraction(
        comb(face_size + 4, face_size) ** 2 * prod(range(face_size, face_size + 4)),
        16 * factorial(face_size - 2) * 5,
    )
    assert closed.q4_scale(face_size) == scale
    lo, hi = closed.c5_enclosure(23)
    assert closed.nonvertex_leading_constant_enclosure(face_size, 23) == (lo * scale, hi * scale)


def test_signed_cancellation_bounds_shrink_and_records_use_540_scale():
    lo, hi = closed.c5_enclosure(1)
    assert lo == 0 and hi > 55
    for terms in (2, 5, 11, 29, 65):
        left, right = closed.c5_enclosure(terms)
        assert lo <= left <= right <= hi
        lo, hi = left, right
    assert Fraction(55464, 1000) < lo < hi < Fraction(55465, 1000)
    record = json.loads(json.dumps(closed.build_record(29, 7)))
    assert _fraction(record["C6"]["lower"]) == 540 * _fraction(record["c5"]["lower"])
    assert _fraction(record["C6"]["upper"]) == 540 * _fraction(record["c5"]["upper"])


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 2.0, "2"])
def test_invalid_scalar_parameters_are_rejected(value):
    with pytest.raises(ValueError):
        closed.zeta_enclosure(2, value)
    with pytest.raises(ValueError):
        closed.q4_scale(value)
    with pytest.raises(ValueError):
        closed.zeta_enclosure(value)


def test_large_fraction_encoding_does_not_change_process_integer_limits():
    before = sys.get_int_max_str_digits()
    encoded = closed._fraction(Fraction(10**5000 + 17, 10**5001 + 19))
    assert len(encoded["numerator"]) == 5001 and encoded["numerator"].endswith("17")
    assert len(encoded["denominator"]) == 5002 and encoded["denominator"].endswith("19")
    assert sys.get_int_max_str_digits() == before


def test_cli_saved_nondefault_options_round_trip(tmp_path):
    out = tmp_path / "record.json"
    assert closed.main(["--terms", "17", "--face-size", "5", "--output", str(out)]) == 0
    original = out.read_bytes()
    assert closed.main(["--verify", str(out)]) == 0
    assert out.read_bytes() == original
    assert json.loads(original)["face_size"] == 5
    for options in (["--terms", "17"], ["--face-size", "5"], ["--output", str(out)]):
        with pytest.raises(SystemExit):
            closed.main(["--verify", str(out), *options])
        assert out.read_bytes() == original


@pytest.mark.parametrize("field", ["terms", "face_size", "outside", "c5", "C6", "q4_scale", "source_sha256", "extra"])
def test_record_tampering_is_rejected(field):
    record = json.loads(json.dumps(closed.build_record(8, 3)))
    if field in ("terms", "face_size", "outside"):
        record[field] = float(record[field])
    elif field in ("c5", "C6"):
        record[field]["lower"]["numerator"] = "0"
    elif field == "q4_scale":
        record[field]["denominator"] = "0"
    elif field == "source_sha256":
        record[field] = {}
    else:
        record[field] = "unexpected"
    with pytest.raises(ValueError):
        closed.verify_record(record)


@pytest.mark.parametrize("record", [None, [], {}, {"schema": "wrong"}, {"schema": closed.SCHEMA}])
def test_unsupported_or_incomplete_records_fail(record):
    with pytest.raises(ValueError):
        closed.verify_record(record)


def test_duplicate_keys_nan_and_invalid_inputs_preserve_files(tmp_path):
    out = tmp_path / "keep.json"
    out.write_text("keep")
    for args in (["--terms", "0"], ["--face-size", "1"], ["--terms", "bad"]):
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


def test_canonical_parent_is_passed_to_writer_even_after_alias_retargets(tmp_path, monkeypatch):
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
    assert closed.main(["--output", str(alias / "c5_closed_form.py")]) == 0
    assert written == [actual / "c5_closed_form.py"]


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
            [sys.executable, "-S", "-m", "tools.risk_research.c5_closed_form", "--terms", "13", "--output", str(output)],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]


def test_native_original_series_integral_and_zeta_agree():
    executable = os.environ.get("BSF_FLINT_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for original positive-integral checks")
        executable = sys.executable
    script = r"""
from fractions import Fraction
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import c5_closed_form as closed
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(closed.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(x): return arb(fmpq(x.numerator,x.denominator))
before=ctx.prec
for bits in (64,111,192):
 with ctx.workprec(bits):
  exact=900*arb(4).zeta()-600*arb(3).zeta()-120*arb(2).zeta()
  integral=8*(2*arb(2).log())**5*integral_normalized_ball(6,bits=bits)
  lo,hi=closed.c5_enclosure(bits//2)
  assert integral.overlaps(exact) and exact>ball(lo) and exact<ball(hi)
  for terms in (47,97,161):
   left,right=leading_constant_enclosure(6,terms)
   assert integral>ball(left/540) and integral<ball(right/540)
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
