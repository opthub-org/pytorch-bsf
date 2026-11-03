"""Check q4 coefficient norms, logarithmic tails and source-bound records."""

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
from tools.risk_research import nonvertex_q4_refined as error
from tools.risk_research.nonvertex_q4_error import relative_error_threshold as old_threshold
from tools.risk_research.vertex_general import _phi_rows, transformed_diagonals

ROOT = Path(__file__).resolve().parents[3]


def test_q4_coefficient_norms_include_index_zero_and_signed_terms():
    rows = _phi_rows(4, 50)
    harmonic = Fraction(0)
    b1 = [Fraction(1)] + [Fraction(-1, j * (j + 1)) for j in range(1, 50)]
    b2 = [sum((b1[k] * b1[j - k] for k in range(j + 1)), Fraction(0)) for j in range(50)]
    for j in range(50):
        if j:
            assert b2[j] == Fraction(2 * (2 * harmonic - 3), j * (j + 1) * (j + 2))
            harmonic += Fraction(1, j)
        assert b2[j] == rows[2][j] - (rows[2][j - 1] if j else 0)
    assert b2[:4] == [1, -1, Fraction(-1, 12), 0]
    assert rows[2][2] < 0
    sequences = {
        "b1": (b1, 2, 3),
        "b2": (b2, 6, 12),
        "e2": ([2 * value**2 for value in rows[1]], 2, 4),
        "e3": ([6 * a * b for a, b in zip(rows[1], rows[2], strict=True)], 24, 84),
        "e4": ([8 * a * c + 6 * b**2 for a, b, c in zip(rows[1], rows[2], rows[3], strict=True)], 528, 2496),
        "phi4": (rows[4], 648, 3240),
    }
    for values, size, derivative in sequences.values():
        for j, value in enumerate(values):
            assert abs(value) <= Fraction(size, (j + 1) ** 2)
            if j + 1 < len(values):
                assert abs(values[j + 1] - value) <= Fraction(derivative, (j + 1) ** 3)


@pytest.mark.parametrize("left", ["square", "signed_cube", "unit"])
@pytest.mark.parametrize("right", ["square", "signed_cube", "unit"])
def test_signed_convolution_lemma_and_boundary_term(left, right):
    def sequence(name):
        if name == "unit":
            return [Fraction(int(j == 0)) for j in range(22)], 1, 1
        if name == "signed_cube":
            return [Fraction((-1) ** j, (j + 1) ** 3) for j in range(22)], 1, 2
        return [Fraction(1, (j + 1) ** 2) for j in range(22)], 1, 2

    a, size_a, delta_a = sequence(left)
    b, size_b, delta_b = sequence(right)
    c = [sum((a[j] * b[n - j] for j in range(n + 1)), Fraction(0)) for n in range(22)]
    for n in range(21):
        half = n // 2
        split = sum((a[j] * (b[n + 1 - j] - b[n - j]) for j in range(half + 1)), Fraction(0))
        split += sum((b[j] * (a[n + 1 - j] - a[n - j]) for j in range(n - half)), Fraction(0))
        split += a[half + 1] * b[n - half]
        assert c[n + 1] - c[n] == split
        assert abs(c[n]) <= Fraction(16 * size_a * size_b, (n + 1) ** 2)
        assert abs(c[n + 1] - c[n]) <= Fraction(16 * (size_a * delta_b + size_b * delta_a + size_a * size_b), (n + 1) ** 3)


def test_new_q4_discrete_derivative_and_positive_mass():
    diagonal = transformed_diagonals(4, 81)
    assert diagonal[:3] == [1, 2, Fraction(37, 24)]
    assert sum(diagonal[:3]) > 4
    assert error.DIAGONAL_MAJORANT == 5400
    assert error.DIFFERENCE_MAJORANT == 179568 == 6 * 29928
    for j, value in enumerate(diagonal):
        assert 0 < value <= Fraction(error.DIAGONAL_MAJORANT, (j + 1) ** 2)
        if j + 1 < len(diagonal):
            assert abs(diagonal[j + 1] - value) <= Fraction(error.DIFFERENCE_MAJORANT, (j + 1) * (j + 2) * (j + 3))


@pytest.mark.parametrize("length", range(10))
@pytest.mark.parametrize("extra", [0, 1, 4, 17])
def test_inverse_factorial_deficit_moment_without_a_diagonal_assumption(length, extra):
    other = length + extra
    total = length + other
    actual = sum(
        (
            Fraction(comb(length, j) * comb(other, j) * (length - j), comb(total, length) * (j + 1) * (j + 2) * (j + 3))
            for j in range(length + 1)
        ),
        Fraction(0),
    )
    assert actual == Fraction(
        length * (total + 1) * (total + 2),
        (length + 1) * (length + 2) * (other + 1) * (other + 2) * (other + 3),
    )


@pytest.mark.parametrize("size,degree", [(2, 2), (2, 8), (3, 3), (3, 9), (5, 13), (9, 15)])
def test_new_averaging_error_bounds_original_positive_diagonals(size, degree):
    values = transformed_diagonals(4, degree - size + 1)
    actual = Fraction(0)
    for length, value in enumerate(values):
        other = 2 * degree + size - length - 1
        average = sum((comb(length, j) * comb(other, j) * values[j] for j in range(length + 1)), Fraction(0)) / comb(
            length + other, length
        )
        actual += abs(average - value)
    assert actual <= error.averaging_error_bound(size, degree)
    if degree == size:
        assert actual == error.averaging_error_bound(size, degree) == 0


@pytest.mark.parametrize("size,degree", [(2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 2)])
def test_refined_enclosure_contains_independent_gram_coefficients(size, degree):
    actual = _exact_coefficients(size + 4, degree)[0][size - 1] if degree >= size else Fraction(0)
    lower, upper = error.coefficient_enclosure(size, degree, terms=17)
    assert lower <= actual <= upper
    if degree < size:
        assert lower == actual == upper == 0
        assert error.relative_error_bound(size, degree) == 1


@pytest.mark.parametrize("size,degree", [(2, 10), (3, 9), (4, 11), (6, 12)])
def test_refined_enclosure_contains_original_scalar_coefficients(size, degree):
    actual = _scalar_risk_sum(size + 4, degree, size)
    lower, upper = error.coefficient_enclosure(size, degree)
    assert lower <= actual <= upper


def test_post_freeze_original_f4_holdouts():
    fixture = json.loads((ROOT / "tools/risk_research/examples/nonvertex_q4_refined_holdouts.json").read_text())
    assert fixture["method"] == "original_integer_outside_kernel_scalar_sum" and fixture["outside"] == 4
    assert [(row["face_size"], row["degree"]) for row in fixture["cases"]] == [(2, 53), (3, 61), (5, 37), (9, 29)]
    for row in fixture["cases"]:
        lower, upper = error.coefficient_enclosure(row["face_size"], row["degree"])
        assert lower <= Fraction(int(row["numerator"]), int(row["denominator"])) <= upper


@pytest.mark.parametrize("size", [2, 3, 4, 8, 16])
def test_all_tail_threshold_has_strict_improvement_and_covers_larger_degrees(size):
    for target in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 10**20)):
        threshold = error.relative_error_threshold(size, target)
        exponent = threshold.bit_length() - 1
        assert threshold == 2**exponent and threshold >= 2 * (size - 1)
        assert error.dyadic_tail_bound(size, exponent) <= target
        assert error.dyadic_tail_bound(size, exponent - 1) > target
        assert threshold < old_threshold(size, target)
        for offset in range(5):
            assert error.dyadic_tail_bound(size, exponent + offset) <= target
        for degree in (threshold, threshold + 1, 2 * threshold - 1, 2 * threshold, 3 * threshold):
            assert error.relative_error_bound(size, degree) <= target
    if size in (2, 3):
        assert error.relative_error_threshold(size, Fraction(1, 100)) == 2**30
        logarithmic, constant = error.bound_constants(size)
        numerator = 5569315 if size == 2 else 5613867
        assert logarithmic * 31 + constant == numerator
        assert Fraction(numerator, 2**30) < Fraction(1, 100)


@pytest.mark.parametrize("value", [True, False, None, 1.0, "2", -1, 0])
def test_integer_types_and_ranges_are_strict(value):
    with pytest.raises(ValueError):
        error.relative_error_bound(value, 8)
    with pytest.raises(ValueError):
        error.relative_error_bound(2, value)
    with pytest.raises(ValueError):
        error.leading_constant_enclosure(2, value)
    with pytest.raises(ValueError):
        error.bound_constants(value)
    with pytest.raises(ValueError):
        error.dyadic_tail_bound(2, value)


def test_active_domain_and_dyadic_domain_boundaries():
    with pytest.raises(ValueError, match="degree >= face_size"):
        error.averaging_error_bound(3, 2)
    with pytest.raises(ValueError, match="dyadic degree"):
        error.dyadic_tail_bound(4, 2)
    assert error.dyadic_tail_bound(2, 1) > 0


@pytest.mark.parametrize("target", [0, -1, 0.01, True, "1/100", Fraction(0), Fraction(-1, 100)])
def test_target_requires_a_positive_exact_fraction(target):
    with pytest.raises(ValueError):
        error.relative_error_threshold(2, target)


def test_large_degree_uses_no_risk_table_or_process_digit_limit_change(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.risk_research.vertex_general.transformed_diagonals", lambda *args: pytest.fail("degree table"))
    previous = sys.get_int_max_str_digits()
    degree = 1 << 17000
    record = error.certificate(3, degree, terms=1, target=Fraction(1, 100))
    path = tmp_path / "huge.json"
    error.atomic_json(path, record)
    saved = json.loads(path.read_bytes(), parse_int=error._decimal_integer)
    assert type(saved["degree"]) is int and saved["degree"] == degree
    error.verify_certificate(saved)
    assert sys.get_int_max_str_digits() == previous


def test_cli_replays_saved_nondefault_options_and_preserves_conflicting_output(tmp_path):
    path, other = tmp_path / "record.json", tmp_path / "checked.json"
    assert (
        error.main(
            [
                "--face-size",
                "4",
                "--degree",
                "125",
                "--constant-terms",
                "13",
                "--relative-target",
                "1/9",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    assert error.main(["--verify", str(path), "--output", str(other)]) == 0
    assert path.read_bytes() == other.read_bytes()
    assert error.main(["--verify", str(path)]) == 0
    assert json.loads(path.read_bytes())["constant_terms"] == 13
    other.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(path), "--constant-terms", "12", "--output", str(other)])
    assert other.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "field,value",
    [
        ("face_size", True),
        ("degree", 9.0),
        ("constant_terms", 0),
        ("outside", 3),
        ("active", 1),
        ("unknown", 0),
        ("diagonal_majorant", 5401),
        ("difference_majorant", 179569),
    ],
)
def test_changed_fields_are_rejected_before_replacing_previous_output(tmp_path, field, value):
    record = json.loads(error._canonical(error.certificate(2, 9, terms=8)))
    record[field] = value
    path, output = tmp_path / "input.json", tmp_path / "old.json"
    path.write_text(json.dumps(record))
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(path), "--output", str(output)])
    assert output.read_bytes() == b"keep"


def test_changed_nested_target_and_source_hash_are_rejected():
    record = json.loads(error._canonical(error.certificate(3, 81, terms=11, target=Fraction(1, 7))))
    record["relative_target"]["numerator"] = True
    with pytest.raises(ValueError):
        error.verify_certificate(record)
    record = json.loads(error._canonical(error.certificate(3, 81, terms=11)))
    record["source_sha256"][error.PROOF] = "0" * 64
    with pytest.raises(ValueError):
        error.verify_certificate(record)


def _simulate_inflight_source_change(monkeypatch, stage, change):
    sources = {name: source.replace(b"\r\n", b"\n") for name, source in error._sources().items()}
    proof = sources[error.PROOF]
    replacement = proof.replace(b"\n", b"\r\n") if change == "line_endings" else proof + b"\nvirtual source change\n"
    assert replacement != proof
    if change == "line_endings":
        assert replacement.replace(b"\r\n", b"\n") == proof
    changed = {**sources, error.PROOF: replacement}
    state = {"changed": False}
    evaluator = getattr(error, stage)

    def evaluate(*args, **kwargs):
        result = evaluator(*args, **kwargs)
        state["changed"] = True
        return result

    monkeypatch.setattr(error, "_sources", lambda: changed if state["changed"] else sources)
    monkeypatch.setattr(error, stage, evaluate)
    return state


@pytest.mark.parametrize("operation", ["certificate", "verify_certificate"])
@pytest.mark.parametrize(
    "stage,target",
    [
        ("leading_constant_enclosure", None),
        ("leading_constant_enclosure", Fraction(1, 100)),
        ("relative_error_threshold", Fraction(1, 100)),
    ],
)
@pytest.mark.parametrize("change", ["content", "line_endings"])
def test_direct_api_rejects_inflight_raw_source_changes(operation, stage, target, change, monkeypatch):
    record = error.certificate(2, 8, terms=7, target=target)
    state = _simulate_inflight_source_change(monkeypatch, stage, change)
    with pytest.raises(RuntimeError, match="sources changed during calculation"):
        if operation == "certificate":
            error.certificate(2, 8, terms=7, target=target)
        else:
            error.verify_certificate(record)
    assert state["changed"]


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("stage", ["leading_constant_enclosure", "relative_error_threshold"])
@pytest.mark.parametrize("change", ["content", "line_endings"])
def test_cli_inflight_raw_source_changes_preserve_previous_output(replay, stage, change, tmp_path, monkeypatch):
    saved, output = tmp_path / "saved.json", tmp_path / "old.json"
    error.atomic_json(saved, error.certificate(2, 8, terms=7, target=Fraction(1, 100)))
    saved_bytes = saved.read_bytes()
    output.write_bytes(b"keep previous certificate")
    state = _simulate_inflight_source_change(monkeypatch, stage, change)
    arguments = (
        ["--verify", str(saved)]
        if replay
        else ["--face-size", "2", "--degree", "8", "--constant-terms", "7", "--relative-target", "1/100"]
    )
    with pytest.raises(SystemExit) as failure:
        error.main([*arguments, "--output", str(output)])
    assert failure.value.code == 2 and state["changed"]
    assert output.read_bytes() == b"keep previous certificate"
    assert saved.read_bytes() == saved_bytes


@pytest.mark.parametrize("starting_line_ending", [b"\n", b"\r\n"])
def test_stable_crlf_sources_keep_normalized_hashes(starting_line_ending, monkeypatch):
    sources = {
        name: source.replace(b"\r\n", b"\n").replace(b"\n", starting_line_ending) for name, source in error._sources().items()
    }
    monkeypatch.setattr(error, "_sources", lambda: sources)
    original = error.certificate(2, 8, terms=7)
    crlf_sources = {name: source.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for name, source in sources.items()}
    monkeypatch.setattr(error, "_sources", lambda: crlf_sources)
    changed = error.certificate(2, 8, terms=7)
    assert changed == original
    error.verify_certificate(original)


def test_cli_outer_source_guard_preserves_output_after_api_returns(tmp_path, monkeypatch):
    output = tmp_path / "old.json"
    output.write_bytes(b"keep")
    sources = error._sources()
    state = {"changed": False}
    evaluator = error.certificate

    def evaluate(*args, **kwargs):
        result = evaluator(*args, **kwargs)
        state["changed"] = True
        return result

    monkeypatch.setattr(error, "_sources", lambda: {**sources, "changed": b"x"} if state["changed"] else sources)
    monkeypatch.setattr(error, "certificate", evaluate)
    with pytest.raises(SystemExit) as failure:
        error.main(["--face-size", "2", "--degree", "8", "--output", str(output)])
    assert failure.value.code == 2 and state["changed"]
    assert output.read_bytes() == b"keep"


def test_duplicate_keys_and_concurrent_source_change_preserve_output(tmp_path, monkeypatch):
    path, output = tmp_path / "bad.json", tmp_path / "old.json"
    path.write_text('{"schema":"a","schema":"b"}')
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(path), "--output", str(output)])
    sources = error._sources()
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return sources if calls <= 2 else {**sources, "changed": b"x"}

    monkeypatch.setattr(error, "_sources", changing_sources)
    with pytest.raises(SystemExit):
        error.main(["--face-size", "2", "--degree", "8", "--output", str(output)])
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("name", list(error._sources()))
def test_bound_source_destinations_are_rejected_before_computation(name, monkeypatch):
    monkeypatch.setattr(error, "certificate", lambda *args: pytest.fail("source collision computed"))
    with pytest.raises(SystemExit):
        error.main(["--face-size", "2", "--degree", "8", "--output", str(ROOT / name)])


def test_validated_canonical_parent_is_used_after_alias_retargeting(tmp_path, monkeypatch):
    protected = ROOT / error.PROOF
    original = protected.read_bytes()
    safe = tmp_path / "safe"
    safe.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(safe, target_is_directory=True)
    except (OSError, NotImplementedError) as link_error:
        pytest.skip(str(link_error))
    destination = safe / protected.name
    writer = error.atomic_json

    def retarget(path, record):
        alias.unlink()
        alias.symlink_to(protected.parent, target_is_directory=True)
        assert path == destination and path.resolve() == destination
        writer(path, record)

    monkeypatch.setattr(error, "atomic_json", retarget)
    assert error.main(["--face-size", "2", "--degree", "8", "--output", str(alias / protected.name)]) == 0
    assert destination.is_file() and protected.read_bytes() == original


def test_native_positive_integral_remains_separate_from_finite_degree_error():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for independent Arb integral checks")
        executable = sys.executable
    script = r"""
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path
from flint import arb, ctx, fmpq
from tools.risk_research import nonvertex_q4_refined as error
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(error.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(q):
    return arb(fmpq(q.numerator,q.denominator))
ctx.prec=73
for bits in (53,137):
    normalized=integral_normalized_ball(6,bits=bits)
    assert ctx.prec==73
    with ctx.workprec(bits):
        C6=normalized*36*factorial(5)*(2*arb(2).log())**5
        for s in (2,3,9):
            scale=Fraction(comb(s+4,s)**2*prod(range(s,s+4)),43200*factorial(s-2))
            actual=C6*ball(scale)
            for terms in (11,47,83):
                lower,upper=error.leading_constant_enclosure(s,terms)
                assert actual>ball(lower) and actual<ball(upper)
    assert ctx.prec==73
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
