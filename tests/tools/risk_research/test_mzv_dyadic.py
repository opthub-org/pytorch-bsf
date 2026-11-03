"""Check dyadic integral cuts, geometric tails, precision and strict records."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from tools.risk_research import leading_constant_mzv as original
from tools.risk_research import mzv_dyadic as dyadic

ROOT = Path(__file__).resolve().parents[3]


def _fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
def test_counting_tail_matches_independent_finite_chain_prefix(depth):
    for terms in (1, 2, 3, 7, 13):
        prefix = sum((Fraction(comb(n - 1, depth - 1), 2**n) for n in range(depth, terms + 1)), Fraction(0))
        assert (terms + 1) ** 2 * dyadic._tail((2,) * depth, terms) == 1 - prefix


def test_split_includes_both_boundaries_and_reverses_reflected_prefix():
    cuts = dyadic._split((2, 1, 2), original._budget(10000))
    assert cuts == (
        ((), (2, 1, 2)),
        ((1,), (1, 1, 2)),
        ((2,), (1, 2)),
        ((3,), (2,)),
        ((1, 3), (1,)),
        ((2, 3), ()),
    )
    assert dyadic.mzv_enclosure((2, 1), 19) == dyadic.mzv_enclosure((3,), 19)


@pytest.mark.parametrize("word", [(2,), (2, 1), (2, 2), (2, 1, 1), (3, 2), (2, 1, 2)])
def test_dyadic_enclosures_nest_and_refine_original_definition_intervals(word):
    previous: tuple[Fraction, Fraction] | None = None
    for terms in (1, 2, 5, 17, 37):
        lo, hi = dyadic.mzv_enclosure(word, terms)
        assert 0 <= lo < hi
        if previous is not None:
            assert previous[0] <= lo <= hi <= previous[1]
        previous = lo, hi
    assert previous is not None
    old = original.mzv_enclosure(word, 73)
    assert old[0] < previous[0] < previous[1] < old[1]
    assert previous[1] - previous[0] < old[1] - old[0]


@pytest.mark.parametrize("bits", [1, 9, 23, 41])
def test_actual_width_meets_requested_absolute_target(bits):
    lo, hi = dyadic.mzv_precision_enclosure((2, 1, 2), bits)
    assert 0 < hi - lo <= Fraction(1, 2**bits)
    assert lo > 0


@pytest.mark.parametrize("index", [3, 4, 5, 6, 7, 8])
def test_signed_compiled_constant_has_absolute_width_and_original_overlap(index):
    lo, hi = dyadic.constant_enclosure(index, 25)
    assert 0 < hi - lo <= Fraction(1, 2**25)
    old = original.constant_enclosure(index, 61)
    assert old[0] <= lo <= hi <= old[1]
    record = dyadic.build_constant_record(index, 25)
    scale = _fraction(record["vertex_scale"])
    scaled = record["vertex_leading_constant"]
    assert isinstance(scaled, dict)
    assert _fraction(scaled["lower"]) == scale * lo
    assert _fraction(scaled["upper"]) == scale * hi
    assert _fraction(record["width"]) == hi - lo


def test_signed_endpoint_selection_handles_cancellation(monkeypatch):
    values = {(2,): (Fraction(1, 2), Fraction(3, 4)), (1,): (Fraction(1, 3), Fraction(2, 3)), (): (Fraction(1), Fraction(1))}
    monkeypatch.setattr(dyadic, "_polylogs", lambda *args: values)
    expression = {(): Fraction(7), (2,): Fraction(-5), (3,): Fraction(2)}
    cuts: dict[dyadic.Word, tuple[tuple[dyadic.Word, dyadic.Word], ...]] = {(2,): (((2,), ()),), (3,): (((1,), (1,)),)}
    lo, hi = dyadic._enclose(expression, cuts, 1, original._budget(1000))
    for x in values[(2,)]:
        for y in values[(1,)]:
            assert lo <= 7 - 5 * x + 2 * y * y <= hi


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 2.0, "2"])
def test_invalid_numeric_options_are_rejected(value):
    for function in (dyadic.mzv_enclosure, dyadic.polylog_enclosure, dyadic.mzv_precision_enclosure):
        with pytest.raises(ValueError):
            function((2,), value)
    with pytest.raises(ValueError):
        dyadic.constant_enclosure(value)
    with pytest.raises(ValueError):
        dyadic.build_record((2,), 12, value)


@pytest.mark.parametrize("word", [(), [2], (1,), (0,), (2, True), (2, 1.0), (2, 0), "2,1"])
def test_invalid_mzv_words_are_rejected(word):
    for function in (dyadic.mzv_enclosure, dyadic.mzv_precision_enclosure, dyadic.build_record):
        with pytest.raises(ValueError):
            function(word)


def test_giant_requests_fail_before_expansion_or_power_allocation():
    huge = 10**100
    calls = [
        lambda: dyadic.mzv_enclosure((huge,), 1),
        lambda: dyadic.mzv_enclosure((2,), huge),
        lambda: dyadic.polylog_enclosure((huge,), 1),
        lambda: dyadic.mzv_precision_enclosure((2,), huge),
        lambda: dyadic.constant_enclosure(huge),
        lambda: dyadic.build_record((2,), 1, 1_000_001),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="work limit"):
            call()


@pytest.mark.parametrize("constant", [False, True])
def test_cli_replays_all_nondefault_options_and_preserves_record(tmp_path, constant):
    out = tmp_path / "record.json"
    kind = ["--constant-index", "6"] if constant else ["--indices", "2,1,2"]
    args = [*kind, "--bits", "19", "--work-limit", "170001", "--output", str(out)]
    assert dyadic.main(args) == 0
    raw = out.read_bytes()
    record = json.loads(raw)
    assert record["bits"] == 19 and record["work_limit"] == 170001
    assert dyadic.main(["--verify", str(out)]) == 0
    assert out.read_bytes() == raw
    for options in (
        ["--bits", "19"],
        ["--indices", "2"],
        ["--constant-index", "6"],
        ["--work-limit", "170001"],
        ["--output", str(out)],
    ):
        with pytest.raises(SystemExit):
            dyadic.main(["--verify", str(out), *options])
        assert out.read_bytes() == raw


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    "field", ["bits", "terms", "value", "width", "work_limit", "selected_work_units", "source_sha256", "kind", "extra"]
)
def test_whole_record_tampering_is_rejected(constant, field):
    original_record = dyadic.build_constant_record(5, 7) if constant else dyadic.build_record((2, 1), 7)
    record = json.loads(json.dumps(original_record))
    if type(record.get(field)) is int:
        record[field] = float(record[field])
    elif field == "kind":
        record[field] = "unknown"
    else:
        record[field] = None
    with pytest.raises((ValueError, TypeError)):
        dyadic.verify_record(record)


def test_invalid_json_budget_and_input_failures_preserve_output(tmp_path):
    out = tmp_path / "keep.json"
    out.write_text("keep")
    for args in (
        ["--indices", "2,1,2", "--work-limit", "1"],
        ["--bits", "0"],
        ["--constant-index", "2"],
        ["--indices", "2,,1"],
    ):
        with pytest.raises(SystemExit):
            dyadic.main([*args, "--output", str(out)])
        assert out.read_text() == "keep"
    for text in ('{"schema":"a","schema":"b"}', "NaN", "[]"):
        invalid = tmp_path / "invalid.json"
        invalid.write_text(text)
        with pytest.raises(SystemExit):
            dyadic.main(["--verify", str(invalid)])
        assert invalid.read_text() == text
    with pytest.raises(SystemExit):
        dyadic.main([])


@pytest.mark.parametrize("mode", ["word", "constant", "verify-word", "verify-constant"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_direct_operations_reject_mid_calculation_raw_source_changes(monkeypatch, mode, change):
    record = dyadic.build_constant_record(5, 7) if "constant" in mode else dyadic.build_record((2, 1), 7)
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in dyadic._sources().items()}
    after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
    if change == "content":
        after = dict(before)
        after[next(iter(after))] += b"\n# virtual source change\n"
    else:
        assert {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in after.items()} == record[
            "source_sha256"
        ]
    assert after != before
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(dyadic, "_sources", sources)
    with pytest.raises(RuntimeError, match="source changed during"):
        if mode.startswith("verify"):
            dyadic.verify_record(record)
        elif mode == "constant":
            dyadic.build_constant_record(5, 7)
        else:
            dyadic.build_record((2, 1), 7)
    assert calls == (4 if mode.startswith("verify") else 2)


@pytest.mark.parametrize("mode", ["generate", "verify"])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_outer_cli_guard_is_independent_of_inner_transaction(tmp_path, monkeypatch, mode, change):
    record = dyadic.build_record((2, 1), 7)
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in dyadic._sources().items()}
    after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
    if change == "content":
        after = {**before, "virtual": b"source change"}
    else:
        assert {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in after.items()} == record[
            "source_sha256"
        ]
    assert before != after
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return before if calls == 1 else after

    monkeypatch.setattr(dyadic, "_sources", sources)
    monkeypatch.setattr(dyadic, "build_record", lambda *args: record)
    monkeypatch.setattr(dyadic, "verify_record", lambda *args: None)
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(record))
    saved_bytes = saved.read_bytes()
    out = tmp_path / "keep.json"
    out.write_text("keep")
    args = ["--output", str(out)] if mode == "generate" else ["--verify", str(saved)]
    with pytest.raises(SystemExit):
        dyadic.main(args)
    assert out.read_text() == "keep"
    assert saved.read_bytes() == saved_bytes
    assert calls == 2


def test_protected_sources_and_canonical_output_parent(tmp_path, monkeypatch):
    for relative in dyadic._sources():
        with pytest.raises(SystemExit):
            dyadic.main(["--output", str(ROOT / relative)])
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    original_protect = dyadic._protect_output
    calls, written = 0, []

    def retarget(path, sources):
        nonlocal calls
        original_protect(path, sources)
        calls += 1
        if calls == 2:
            alias.unlink()
            alias.symlink_to(ROOT / "tools/risk_research", target_is_directory=True)

    monkeypatch.setattr(dyadic, "_protect_output", retarget)
    monkeypatch.setattr(dyadic, "atomic_json", lambda path, value: written.append(path))
    assert dyadic.main(["--bits", "7", "--output", str(alias / "mzv_dyadic.py")]) == 0
    assert written == [actual / "mzv_dyadic.py"]


@pytest.mark.parametrize("constant", [False, True])
def test_stdlib_only_lf_crlf_record_generation_and_verification(tmp_path, constant):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, raw in dyadic._sources().items():
            dest = checkout / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", ending))
        out = tmp_path / (checkout.name + ".json")
        mode = ["--constant-index", "5"] if constant else ["--indices", "2,1,2"]
        env = {**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"}
        for args in ([*mode, "--bits", "13", "--output", str(out)], ["--verify", str(out)]):
            result = subprocess.run(
                [sys.executable, "-S", "-m", "tools.risk_research.mzv_dyadic", *args],
                cwd=checkout,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        records.append(json.loads(out.read_text()))
    assert records[0] == records[1]


def test_native_polylog_identity_and_original_vertex_integrals():
    executable = os.environ.get("BSF_FLINT_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for independent native checks")
        executable = sys.executable
    script = r"""
from fractions import Fraction
from math import factorial
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import mzv_dyadic as new
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(new.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(x):return arb(fmpq(x.numerator,x.denominator))
before=ctx.prec
for bits in (91,139):
 with ctx.workprec(bits):
  for order in (1,2,3,5):
   lo,hi=new.polylog_enclosure((order,),29)
   value=(arb(1)/2).polylog(order)
   assert value>ball(lo) and value<ball(hi)
  for word,order in [((2,),2),((2,1),3),((2,1,1),4)]:
   lo,hi=new.mzv_precision_enclosure(word,37)
   value=arb(order).zeta()
   assert value>ball(lo) and value<ball(hi)
  for index in (5,6,7):
   lo,hi=new.constant_enclosure(index,35)
   integral=2**(index-2)*(2*arb(2).log())**index*integral_normalized_ball(index+1,bits=bits)
   assert integral.is_finite() and integral.overlaps(ball(lo).union(ball(hi)))
   left,right=leading_constant_enclosure(index+1,89)
   scale=Fraction((index+1)**2*factorial(index),2**(index-2))
   assert ball(lo)>ball(left/scale) and ball(hi)<ball(right/scale)
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


def test_large_rational_record_scaling_does_not_reparse_decimal_strings(monkeypatch):
    # The exact endpoints can have >4300 decimal digits at accepted precision.
    # Keep arithmetic cheap while exercising the full serializer/scaling path.
    lower = Fraction(2**20000 + 1, 2**20000)
    upper = lower + Fraction(1, 2**20001)
    monkeypatch.setattr(dyadic, "_precision", lambda *args: ((lower, upper), 1))
    record = dyadic.build_constant_record(5, 13)
    value = record["value"]
    assert isinstance(value, dict)
    lower_record = value["lower"]
    assert isinstance(lower_record, dict)
    numerator = lower_record["numerator"]
    assert isinstance(numerator, str) and len(numerator) > 4300
    scale = Fraction(6**2 * 120, 8)
    scaled = record["vertex_leading_constant"]
    assert isinstance(scaled, dict)
    assert scaled["lower"] == original._fraction(scale * lower)
    assert scaled["upper"] == original._fraction(scale * upper)
    dyadic.verify_record(record)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("change", ["content", "line-endings"])
def test_direct_verifier_guards_canonical_comparison_after_stable_builder(monkeypatch, constant, mismatch, change):
    record = dyadic.build_constant_record(5, 7) if constant else dyadic.build_record((2, 1), 7)
    if mismatch:
        record["extra"] = True
    before = {name: raw.replace(b"\r\n", b"\n") for name, raw in dyadic._sources().items()}
    after = {name: raw.replace(b"\n", b"\r\n") for name, raw in before.items()}
    if change == "content":
        after = {**before, "virtual": b"source change"}
    else:
        assert {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in after.items()} == record[
            "source_sha256"
        ]
    assert before != after
    changed = False
    snapshots = []
    comparisons = 0
    canonical = dyadic.canonical_json

    def sources():
        snapshots.append(changed)
        return after if changed else before

    def compare(value):
        nonlocal changed, comparisons
        comparisons += 1
        changed = True
        return canonical(value)

    monkeypatch.setattr(dyadic, "_sources", sources)
    monkeypatch.setattr(dyadic, "canonical_json", compare)
    with pytest.raises(RuntimeError, match="source changed during verification"):
        dyadic.verify_record(record)
    # The inner builder's two snapshots were stable. Only the verifier's
    # transaction can detect the edit made during the later comparison.
    assert snapshots == [False, False, False, True]
    assert comparisons == 2
