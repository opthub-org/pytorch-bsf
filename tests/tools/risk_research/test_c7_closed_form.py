"""Check the weight-six c7 reduction, rational intervals and source-bound records."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from math import comb, factorial
from pathlib import Path

import pytest

from tools.risk_research import c7_closed_form as closed
from tools.risk_research import leading_constant_mzv as compiler
from tools.risk_research.vertex_general import _phi_rows

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def _harmonics(words, count):
    suffixes = {word[i:] for word in words for i in range(len(word) + 1)}
    values = {word: Fraction(0) for word in suffixes}
    values[()] = Fraction(1)
    for n in range(1, count + 1):
        previous = values.copy()
        for word in suffixes:
            if word:
                values[word] = previous[word] + previous[word[1:]] / n ** word[0]
    return values


def _upper_boundaries():
    budget = compiler._budget(200_000)
    nodes = []

    def descend(poles, word):
        _, antiderivative = compiler._telescope(poles, budget)
        nodes.append((word, antiderivative))
        if word:
            descend(compiler._shift_divide(antiderivative, word[0], budget), word[1:])

    for word, poles in compiler._initial(7, budget).items():
        descend(poles, word)
    return nodes


def test_mzv_table_is_the_exact_general_construction_specialization():
    expression = compiler.leading_constant_expression(7)
    assert expression == dict(closed.MZV_COEFFICIENTS)
    assert () not in expression
    assert len(expression) == 12 and max(map(sum, expression)) == 6


@pytest.mark.parametrize("count", [1, 2, 5, 11])
def test_finite_original_series_equals_mzv_prefix_plus_every_boundary(count):
    rows = _phi_rows(5, count)
    original = 7 * sum(
        (sum((comb(6, a) * rows[a][j] * rows[6 - a][j] for a in range(1, 6)), Fraction(0)) for j in range(count)),
        Fraction(0),
    )
    nodes = _upper_boundaries()
    words = set(dict(closed.MZV_COEFFICIENTS)) | {word for word, _ in nodes}
    values = _harmonics(words, count)
    compiled = sum((coefficient * values[word] for word, coefficient in closed.MZV_COEFFICIENTS), Fraction(0))
    boundary = sum(
        (
            sum((coefficient / (count + 1 + shift) ** power for (shift, power), coefficient in poles.items()), Fraction(0))
            * values[word]
            for word, poles in nodes
        ),
        Fraction(0),
    )
    assert original == compiled + boundary


@pytest.mark.parametrize("count", [1, 3, 8])
def test_finite_stuffle_partitions_include_equal_indices(count):
    words = {(2,), (3,), (4,), (5,), (6,), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 2, 2)}
    h = _harmonics(words, count)
    assert h[2,] * h[3,] == h[2, 3] + h[3, 2] + h[5,]
    assert h[2,] * h[4,] == h[2, 4] + h[4, 2] + h[6,]
    assert h[3,] ** 2 == 2 * h[3, 3] + h[6,]
    assert h[2,] ** 3 == 6 * h[2, 2, 2] + 3 * h[2, 4] + 3 * h[4, 2] + h[6,]


@pytest.mark.parametrize(
    ("indices", "dual"),
    [
        ((2, 1), (3,)),
        ((2, 1, 1), (4,)),
        ((2, 1, 1, 1), (5,)),
        ((2, 1, 1, 1, 1), (6,)),
        ((2, 1, 2), (2, 3)),
        ((2, 2, 1), (3, 2)),
        ((2, 1, 1, 2), (2, 4)),
        ((2, 1, 2, 1), (3, 3)),
        ((2, 2, 1, 1), (4, 2)),
    ],
)
def test_duality_binary_words_use_the_descending_convention(indices, dual):
    word = "".join("0" * (index - 1) + "1" for index in indices)
    transformed = "".join("1" if value == "0" else "0" for value in reversed(word))
    assert transformed == "".join("0" * (index - 1) + "1" for index in dual)


def test_signed_and_repeated_monomials_enclose_every_endpoint_choice(monkeypatch):
    intervals = {order: (Fraction(1), Fraction(2)) for order in range(2, 6)}
    intervals[6] = (Fraction(10), Fraction(11))
    monkeypatch.setattr(closed, "zeta_enclosure", lambda order, terms: intervals[order])
    lower, upper = closed.c7_enclosure(13)
    assert 0 < lower < upper
    for z2, z3, z4, z5, z6 in product(*intervals.values()):
        exact = (
            211680 * z6
            - 189000 * z5
            - 75250 * z4
            - 18900 * z3
            - 3920 * z2
            - 78960 * z2 * z3
            + 85680 * z2 * z4
            + 50400 * z3**2
            + 5040 * z2**3
        )
        assert lower <= exact <= upper


def test_nested_rational_intervals_and_existing_C8_normalization():
    lower, upper = closed.c7_enclosure(1)
    assert lower == 0 < upper
    for terms in (2, 5, 13, 31, 67):
        new_lower, new_upper = closed.c7_enclosure(terms)
        assert lower <= new_lower <= new_upper <= upper
        lower, upper = new_lower, new_upper
    assert lower > 0 and upper - lower < Fraction(1, 10**10)
    scale = Fraction(8**2 * factorial(7), 2**5)
    assert scale == 10080
    record = json.loads(json.dumps(closed.build_record(23)))
    assert _fraction(record["C8"]["lower"]) == scale * _fraction(record["c7"]["lower"])
    assert _fraction(record["C8"]["upper"]) == scale * _fraction(record["c7"]["upper"])


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 2.0, "2"])
def test_invalid_terms_are_rejected(value):
    with pytest.raises(ValueError):
        closed.c7_enclosure(value)
    with pytest.raises(ValueError):
        closed.build_record(value)


def test_nondefault_record_round_trip_and_conflicting_options(tmp_path):
    out = tmp_path / "record.json"
    assert closed.main(["--terms", "19", "--output", str(out)]) == 0
    before = out.read_bytes()
    assert json.loads(before)["terms"] == 19
    assert closed.main(["--verify", str(out)]) == 0
    assert out.read_bytes() == before
    for options in (["--terms", "19"], ["--output", str(out)]):
        with pytest.raises(SystemExit):
            closed.main(["--verify", str(out), *options])
        assert out.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    [
        "terms",
        "vertices",
        "c7",
        "C8",
        "zeta_coefficients",
        "zeta_monomial_coefficients",
        "mzv_coefficients_descending",
        "source_sha256",
        "extra",
    ],
)
def test_changed_types_coefficients_and_unknown_record_fields_fail(field):
    record = json.loads(json.dumps(closed.build_record(5)))
    if field in ("terms", "vertices"):
        record[field] = float(record[field])
    elif field in ("c7", "C8"):
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


def _source_snapshots(change):
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in closed._sources().items()}
    after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
    if change == "content":
        after = dict(before)
        first = next(iter(after))
        after[first] += b"\n# virtual edit\n"
    else:
        for name in before:
            assert hashlib.sha256(before[name]).digest() == hashlib.sha256(after[name].replace(b"\r\n", b"\n")).digest()
    assert before != after
    return before, after


@pytest.mark.parametrize("operation", ["build", "verify"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_direct_operations_reject_raw_source_changes(monkeypatch, operation, change):
    record = closed.build_record(7)
    before, after = _source_snapshots(change)
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


@pytest.mark.parametrize("operation", ["generate", "verify"])
@pytest.mark.parametrize("guard", ["inner", "outer"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_cli_inflight_changes_preserve_files(tmp_path, monkeypatch, capsys, operation, guard, change):
    before, after = _source_snapshots(change)
    with monkeypatch.context() as prepare:
        prepare.setattr(closed, "_sources", lambda: after if guard == "outer" else before)
        record = closed.build_record(7)
    out = tmp_path / "keep.json"
    out.write_text(json.dumps(record) if operation == "verify" else "keep")
    original = out.read_bytes()
    calls = 0

    def fake_sources():
        nonlocal calls
        calls += 1
        return before if calls <= (2 if guard == "inner" else 1) else after

    monkeypatch.setattr(closed, "_sources", fake_sources)
    options = ["--verify", str(out)] if operation == "verify" else ["--terms", "7", "--output", str(out)]
    with pytest.raises(SystemExit) as raised:
        closed.main(options)
    assert raised.value.code == 2
    assert "a closed-form source changed during calculation" in capsys.readouterr().err
    assert calls == (3 if guard == "inner" else 4)
    assert out.read_bytes() == original


def test_invalid_cli_inputs_duplicate_keys_and_nan_preserve_files(tmp_path):
    out = tmp_path / "keep.json"
    out.write_text("keep")
    for terms in ("0", "-1", "invalid"):
        with pytest.raises(SystemExit):
            closed.main(["--terms", terms, "--output", str(out)])
        assert out.read_text() == "keep"
    for value in ('{"schema":"a","schema":"b"}', "NaN", "[]"):
        out.write_text(value)
        with pytest.raises(SystemExit):
            closed.main(["--verify", str(out)])
        assert out.read_text() == value
    with pytest.raises(SystemExit):
        closed.main([])


def test_every_bound_source_and_its_alias_are_protected(tmp_path):
    for relative in closed._sources():
        with pytest.raises(SystemExit):
            closed.main(["--output", str(ROOT / relative)])
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(ROOT / closed.PROOF)
    except OSError:
        pytest.skip("file symlinks unavailable")
    before = alias.read_bytes()
    with pytest.raises(SystemExit):
        closed.main(["--output", str(alias)])
    assert alias.read_bytes() == before


def test_canonical_parent_survives_alias_retargeting(tmp_path, monkeypatch):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    protect = closed._protect_output
    calls, written = 0, []

    def retarget(path, sources):
        nonlocal calls
        protect(path, sources)
        calls += 1
        if calls == 2:
            alias.unlink()
            alias.symlink_to(ROOT / "tools/risk_research", target_is_directory=True)

    monkeypatch.setattr(closed, "_protect_output", retarget)
    monkeypatch.setattr(closed, "atomic_json", lambda path, record: written.append(path))
    assert closed.main(["--terms", "5", "--output", str(alias / "c7_closed_form.py")]) == 0
    assert written == [actual / "c7_closed_form.py"]


def test_actual_lf_crlf_sources_are_portable_and_stdlib_only(tmp_path):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, raw in closed._sources().items():
            dest = checkout / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
        output = tmp_path / (checkout.name + ".json")
        result = subprocess.run(
            [sys.executable, "-S", "-m", "tools.risk_research.c7_closed_form", "--terms", "13", "--output", str(output)],
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
            pytest.skip("set BSF_FLINT_PYTHON for independent original-integral checks")
        executable = sys.executable
    script = r"""
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import c7_closed_form as closed
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(closed.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(x): return arb(fmpq(x.numerator,x.denominator))
before=ctx.prec
for bits in (79,139):
 with ctx.workprec(bits):
  z2,z3,z4,z5,z6=[arb(k).zeta() for k in range(2,7)]
  classical=211680*z6-189000*z5-75250*z4-18900*z3-3920*z2-78960*z2*z3+85680*z2*z4+50400*z3**2+5040*z2**3
  integral=32*(2*arb(2).log())**7*integral_normalized_ball(8,bits=bits)
  lower,upper=closed.c7_enclosure(bits//2)
  assert integral.overlaps(classical) and classical>ball(lower) and classical<ball(upper)
  for terms in (43,101):
   left,right=leading_constant_enclosure(8,terms)
   assert integral>ball(left/10080) and integral<ball(right/10080)
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
