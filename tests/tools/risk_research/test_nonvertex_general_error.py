"""Check general diagonal majorants, finite risks and replay contracts."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import nonvertex_general_error as error
from tools.risk_research.vertex_general import transformed_diagonals

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("outside", range(2, 10))
def test_global_majorant_including_small_indices_and_critical_q2(outside):
    majorant, logarithmic = error.diagonal_majorant(outside)
    values = transformed_diagonals(outside, 47)
    harmonic = Fraction(0)
    assert values[0] == 1 and logarithmic == (outside == 2)
    for j, value in enumerate(values):
        harmonic += Fraction(1, j + 1)
        # H_N <= 1+log(N); the stronger harmonic check is independent
        # of the evaluator's integer-bit-length logarithm majorant.
        factor = harmonic if logarithmic else 1
        assert 0 < value <= majorant * factor / (j + 1) ** 2
    if outside == 2:
        assert values[-1] * len(values) ** 2 > majorant


@pytest.mark.parametrize("size,outside,degree", [(2, 2, 9), (3, 2, 13), (2, 5, 11), (6, 7, 13)])
@pytest.mark.parametrize("sparse", [False, True])
def test_sequence_independent_averaging_bound(size, outside, degree, sparse):
    majorant, logarithmic = error.diagonal_majorant(outside)
    values = []
    harmonic = Fraction(0)
    for j in range(degree - size + 1):
        harmonic += Fraction(1, j + 1)
        values.append(majorant * (harmonic if logarithmic else 1) * int(not sparse or j % 3 == 1) / (j + 1) ** 2)
    actual = Fraction(0)
    for length, value in enumerate(values):
        other = 2 * degree + size - length - 1
        average = sum((comb(length, j) * comb(other, j) * values[j] for j in range(length + 1)), Fraction(0)) / comb(
            length + other, length
        )
        actual += abs(average - value)
    assert actual <= error.averaging_error_bound(size, outside, degree)


@pytest.mark.parametrize("size,outside,degree", [(2, 2, 2), (2, 2, 4), (2, 5, 2), (2, 5, 3), (3, 3, 3)])
def test_finite_coefficient_enclosures_against_independent_gram(size, outside, degree):
    actual = _exact_coefficients(size + outside, degree)[0][size - 1]
    lower, upper = error.coefficient_enclosure(size, outside, degree, terms=13)
    assert lower <= actual <= upper


@pytest.mark.parametrize("size,outside,degree", [(2, 5, 11), (3, 6, 9), (5, 2, 12)])
def test_original_scalar_coefficients(size, outside, degree):
    actual = _scalar_risk_sum(size + outside, degree, size)
    lower, upper = error.coefficient_enclosure(size, outside, degree, terms=17)
    assert lower <= actual <= upper


def test_post_freeze_original_integer_kernel_holdouts():
    fixture = json.loads((ROOT / "tools/risk_research/examples/nonvertex_general_error_holdouts.json").read_text())
    assert fixture["method"] == "original_integer_outside_kernel_scalar_sum"
    assert [(row["face_size"], row["outside"], row["degree"]) for row in fixture["cases"]] == [
        (2, 5, 41),
        (3, 7, 31),
        (5, 8, 23),
        (9, 2, 19),
        (4, 3, 37),
    ]
    for row in fixture["cases"]:
        exact = Fraction(int(row["coefficient"]["numerator"]), int(row["coefficient"]["denominator"]))
        lower, upper = error.coefficient_enclosure(row["face_size"], row["outside"], row["degree"], terms=29)
        assert lower <= exact <= upper


@pytest.mark.parametrize("size,outside", [(2, 2), (3, 2), (9, 2), (2, 3), (3, 5), (9, 8)])
def test_all_tail_threshold_and_odd_even_blocks(size, outside):
    for target in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 10**20)):
        threshold = error.relative_error_threshold(size, outside, target)
        exponent = threshold.bit_length() - 1
        assert threshold == 2**exponent and threshold >= max(4, 2 * (size - 1))
        assert error.dyadic_tail_bound(size, outside, exponent) <= target
        assert error.dyadic_tail_bound(size, outside, exponent - 1) > target
        previous = error.dyadic_tail_bound(size, outside, exponent)
        for offset in range(1, 6):
            current = error.dyadic_tail_bound(size, outside, exponent + offset)
            assert current <= previous
            previous = current
        for degree in (threshold, threshold + 1, 2 * threshold - 1, 2 * threshold, 3 * threshold):
            assert error.relative_error_bound(size, outside, degree) <= target


@pytest.mark.parametrize("size,outside", [(2, 2), (3, 5), (9, 7)])
def test_inactive_prefix_and_start_boundaries(size, outside):
    for degree in range(1, size):
        assert error.relative_error_bound(size, outside, degree) == 1
        assert error.coefficient_enclosure(size, outside, degree, terms=7) == (0, 0)
        with pytest.raises(ValueError, match="averaging requires"):
            error.averaging_error_bound(size, outside, degree)
    assert error.relative_error_bound(size, outside, size) > 0
    with pytest.raises(ValueError):
        error.dyadic_tail_bound(size, outside, 1)
    if size > 3:
        with pytest.raises(ValueError, match="dyadic degree"):
            error.dyadic_tail_bound(size, outside, 2)


@pytest.mark.parametrize("bad", [True, False, None, 1.0, "2", -1, 0])
def test_strict_types_and_ranges(bad):
    for parameters in ((bad, 5, 8), (2, bad, 8), (2, 5, bad)):
        with pytest.raises(ValueError):
            error.relative_error_bound(*parameters)
    with pytest.raises(ValueError):
        error.leading_constant_enclosure(2, 5, bad)
    with pytest.raises(ValueError):
        error.certificate(2, 5, 8, work_limit=bad)


@pytest.mark.parametrize("target", [0, 0.01, True, Fraction(0), Fraction(-1), "1/100"])
def test_target_requires_positive_exact_fraction(target):
    with pytest.raises(ValueError, match="target"):
        error.relative_error_threshold(2, 5, target)


def test_resource_refusal_precedes_large_integer_or_constant_work(monkeypatch):
    monkeypatch.setattr(error, "general_error_constants", lambda *a: pytest.fail("expensive constants"))
    monkeypatch.setattr(error, "refined_error_constants", lambda *a: pytest.fail("expensive constants"))
    monkeypatch.setattr(error, "nonvertex_leading_constant_enclosure", lambda *a: pytest.fail("constant series"))
    for action in (
        lambda: error.diagonal_majorant(10**100),
        lambda: error.leading_constant_enclosure(2, 5, 10**100),
        lambda: error.coefficient_enclosure(10**100, 5, 8),
        lambda: error.dyadic_tail_bound(2, 5, 10**100),
        lambda: error.relative_error_bound(2, 5, 8, work_limit=error.MAX_WORK_LIMIT + 1),
    ):
        with pytest.raises(ValueError, match="work"):
            action()


def test_huge_degree_needs_no_degree_table_or_global_digit_limit_change(monkeypatch):
    monkeypatch.setattr("tools.risk_research.vertex_general.transformed_diagonals", lambda *a: pytest.fail("degree table"))
    before = sys.get_int_max_str_digits()
    degree = 2**15000
    record = error.certificate(2, 5, degree, terms=3)
    assert record["degree"] == degree and record["active"] is True
    assert len(record["coefficient_enclosure"]["upper"]["denominator"]) > 4300
    assert sys.get_int_max_str_digits() == before


def test_cli_replays_all_nondefault_options_and_conflicting_overrides(tmp_path):
    path = tmp_path / "record.json"
    output = tmp_path / "copy.json"
    arguments = [
        "--face-size",
        "3",
        "--outside",
        "5",
        "--degree",
        str(2**81 + 7),
        "--constant-terms",
        "19",
        "--relative-target",
        "1/100",
        "--work-limit",
        "190000",
        "--output",
        str(path),
    ]
    assert error.main(arguments) == 0
    assert error.main(["--verify", str(path), "--output", str(output)]) == 0
    assert path.read_bytes() == output.read_bytes()
    for option, value in [
        ("--face-size", "2"),
        ("--outside", "6"),
        ("--degree", "9"),
        ("--constant-terms", "20"),
        ("--relative-target", "1/10"),
        ("--work-limit", "180000"),
    ]:
        output.write_bytes(b"keep")
        with pytest.raises(SystemExit):
            error.main(["--verify", str(path), option, value, "--output", str(output)])
        assert output.read_bytes() == b"keep"
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tools.risk_research.nonvertex_general_error", "--verify", str(path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("face_size", True),
        ("outside", 5.0),
        ("active", 1),
        ("unknown", 0),
        ("work_limit", True),
        ("source_hash_policy", "wrong"),
    ],
)
def test_tampered_scalar_types_and_fields_rejected(field, value):
    record = error.certificate(3, 5, 9, terms=7, work_limit=190000)
    record[field] = value
    with pytest.raises(ValueError):
        error.verify_certificate(record)


@pytest.mark.parametrize("operation", ["build", "verify"])
@pytest.mark.parametrize("initial_crlf", [False, True])
def test_direct_raw_source_guard_even_when_canonical_hash_does_not_change(monkeypatch, operation, initial_crlf):
    record = error.certificate(2, 5, 9, terms=7)
    original = error._sources()
    name = "tools/risk_research/nonvertex_general_error.py"
    lf = original[name].replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    before, after = (crlf, lf) if initial_crlf else (lf, crlf)
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return {**original, name: before if calls == 1 else after}

    monkeypatch.setattr(error, "_sources", changing_sources)
    with pytest.raises(RuntimeError, match="sources changed"):
        if operation == "build":
            error.certificate(2, 5, 9, terms=7)
        else:
            error.verify_certificate(record)


def test_outer_cli_guard_is_independent_of_builder_guard(tmp_path, monkeypatch):
    record = error.certificate(2, 5, 9, terms=7)
    sources = error._sources()
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return sources if calls == 1 else {**sources, "changed": b"x"}

    monkeypatch.setattr(error, "certificate", lambda *a: copy.deepcopy(record))
    monkeypatch.setattr(error, "_sources", changing_sources)
    output = tmp_path / "record.json"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--face-size", "2", "--outside", "5", "--degree", "9", "--output", str(output)])
    assert output.read_bytes() == b"keep"


def test_duplicate_keys_preserve_previous_output(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"schema":"' + error.SCHEMA + '","schema":"' + error.SCHEMA + '"}')
    output = tmp_path / "record.json"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(source), "--output", str(output)])
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("name", list(error._sources()))
def test_bound_source_output_rejected_before_calculation(name, monkeypatch):
    monkeypatch.setattr(error, "certificate", lambda *a: pytest.fail("source collision computed"))
    with pytest.raises(SystemExit):
        error.main(["--face-size", "2", "--outside", "5", "--degree", "9", "--output", str(ROOT / name)])


def test_output_parent_is_pinned_before_atomic_writer(tmp_path, monkeypatch):
    safe = tmp_path / "safe"
    safe.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(safe, target_is_directory=True)
    protected = ROOT / error.PROOF
    old = protected.read_bytes()
    destination = safe / protected.name
    writer = error.atomic_json

    def retarget(path, record):
        alias.unlink()
        alias.symlink_to(protected.parent, target_is_directory=True)
        assert path == destination and path.resolve() == destination
        writer(path, record)

    monkeypatch.setattr(error, "atomic_json", retarget)
    assert (
        error.main(
            [
                "--face-size",
                "2",
                "--outside",
                "5",
                "--degree",
                "9",
                "--constant-terms",
                "7",
                "--output",
                str(alias / protected.name),
            ]
        )
        == 0
    )
    assert destination.is_file() and protected.read_bytes() == old


def test_native_original_positive_integrals_with_separate_precision():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for independent Arb integral checks")
        executable = sys.executable
    script = r"""
from fractions import Fraction
from math import comb,factorial,prod
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import nonvertex_general_error as error
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(error.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(q): return arb(fmpq(q.numerator,q.denominator))
for q in (2,5,7):
    m=q+2
    for bits in (51,97):
        before=ctx.prec
        normalized=integral_normalized_ball(m,bits=bits)
        assert ctx.prec==before
        with ctx.workprec(bits):
            C=normalized*m*m*factorial(q+1)*(2*arb(2).log())**(q+1)
            if q==2:
                # The old M4 positive-integral tail is broad at every
                # precision. Check its known exact value before using
                # that separate classical oracle for the tight check.
                exact=48*arb.pi()**2
                assert C.contains(exact)
                C=exact
            for s in (2,4):
                scale=Fraction(comb(s+q,s)**2*prod(range(s,s+q)),2*factorial(s-2)*(q+1)*(q+2)**2*factorial(q+1))
                actual=C*ball(scale)
                lo,hi=error.leading_constant_enclosure(s,q,23)
                assert actual>ball(lo) and actual<ball(hi)
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
