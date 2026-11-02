"""Check the q4 cutoff proof, independent risks and complete saved records."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb, factorial, isqrt, prod
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import nonvertex_q4_error as error
from tools.risk_research.codimension_three_exact import nonvertex_coefficient
from tools.risk_research.vertex_general import transformed_diagonals
from tools.risk_research.vertex_refined_asymptotic import error_constants

ROOT = Path(__file__).resolve().parents[3]


def _fraction(value):
    return Fraction(error._decimal_integer(value["numerator"]), error._decimal_integer(value["denominator"]))


def test_proved_global_majorant_and_positive_mass_include_small_indices():
    constants = error_constants(5)
    assert constants.diagonal == 341752 and constants.limiting == 192
    assert constants.diagonal + constants.limiting == error.DIAGONAL_MAJORANT
    diagonal = transformed_diagonals(4, 65)
    assert diagonal[:3] == [1, 2, Fraction(37, 24)]
    assert sum(diagonal[:3]) == Fraction(109, 24) > 4
    for j, value in enumerate(diagonal):
        assert 0 < value <= Fraction(error.DIAGONAL_MAJORANT, (j + 1) ** 2)


@pytest.mark.parametrize("length", range(9))
@pytest.mark.parametrize("extra", [0, 1, 3, 15])
def test_both_inverse_factorial_moments_and_deficit_mean(length, extra):
    other = length + extra
    total = length + other
    probability = [Fraction(comb(length, j) * comb(other, j), comb(total, length)) for j in range(length + 1)]
    inverse = sum((p / ((j + 1) * (j + 2)) for j, p in enumerate(probability)), Fraction(0))
    deficit = sum((p * (length - j) / ((j + 1) * (j + 2)) for j, p in enumerate(probability)), Fraction(0))
    mean = sum((p * (length - j) for j, p in enumerate(probability)), Fraction(0))
    assert inverse == Fraction((total + 1) * (total + 2), (length + 1) * (length + 2) * (other + 1) * (other + 2))
    assert deficit == Fraction(length * (total + 1), (length + 1) * (other + 1) * (other + 2))
    assert mean == (Fraction(length**2, total) if total else 0)


@pytest.mark.parametrize("size,degree", [(2, 2), (2, 7), (2, 12), (3, 3), (3, 11), (4, 4), (8, 8)])
@pytest.mark.parametrize("sequence", ["alternating", "one_spike", "inverse_square"])
def test_cutoff_lemma_for_sequences_unrelated_to_the_new_diagonal(size, degree, sequence):
    count = degree - size + 1
    values = [Fraction(int(sequence != "alternating" or j % 2 == 0), (j + 1) ** 2) for j in range(count)]
    if sequence == "one_spike":
        values = [Fraction(int(j == count // 2), (j + 1) ** 2) for j in range(count)]
    actual = Fraction(0)
    for length in range(count):
        other = 2 * degree + size - length - 1
        average = sum((comb(length, j) * comb(other, j) * values[j] for j in range(length + 1)), Fraction(0)) / comb(
            length + other, length
        )
        actual += abs(average - values[length])
    cutoff = isqrt(degree)
    assert actual <= Fraction(9 * cutoff, 2 * degree) + Fraction(10, cutoff + 1)
    assert error.averaging_error_bound(degree) == error.DIAGONAL_MAJORANT * (
        Fraction(9 * cutoff, 2 * degree) + Fraction(10, cutoff + 1)
    )


@pytest.mark.parametrize("size,degree", [(2, 5), (3, 8), (4, 12), (7, 14)])
def test_angular_normalization_and_original_positive_coefficient(size, degree):
    diagonal = transformed_diagonals(4, degree - size + 1)
    weighted = Fraction(0)
    deficit = Fraction(0)
    for length, value in enumerate(diagonal):
        other = 2 * degree + size - length - 1
        average = sum((comb(length, j) * comb(other, j) * diagonal[j] for j in range(length + 1)), Fraction(0)) / comb(
            length + other, length
        )
        weight = prod((Fraction(degree - length - r, degree) for r in range(2, size)), start=Fraction(1))
        weighted += weight * average
        deficit += (1 - weight) * value
    expected = (
        Fraction(
            comb(size + 4, size) ** 2 * prod(range(size, size + 4)), prod(range(2 * degree + size, 2 * degree + size + 4))
        )
        * Fraction(degree ** (size - 2), factorial(size - 2))
        * weighted
    )
    assert expected == nonvertex_coefficient(size + 4, degree, size)
    harmonic = sum((Fraction(1, j) for j in range(1, len(diagonal) + 1)), Fraction(0))
    assert 0 <= deficit <= Fraction(error.DIAGONAL_MAJORANT, degree) * ((size - 2) * harmonic + (size - 1) * (size - 2))


@pytest.mark.parametrize("size,degree", [(2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (3, 3), (3, 5)])
def test_enclosure_contains_independent_gram_coefficients(size, degree):
    coefficients = _exact_coefficients(size + 4, degree)[0]
    actual = coefficients[size - 1] if degree >= size else Fraction(0)
    lower, upper = error.coefficient_enclosure(size, degree, terms=16)
    assert lower <= actual <= upper
    if degree < size:
        assert lower == actual == upper == 0 and error.relative_error_bound(size, degree) == 1


@pytest.mark.parametrize("size,degree", [(2, 9), (2, 13), (3, 11), (4, 8), (6, 10)])
def test_enclosure_contains_original_scalar_coefficients(size, degree):
    actual = _scalar_risk_sum(size + 4, degree, size)
    lower, upper = error.coefficient_enclosure(size, degree)
    assert lower <= actual <= upper


def test_frozen_original_f4_integer_kernel_holdouts():
    fixture = json.loads((ROOT / "tools/risk_research/examples/nonvertex_q4_error_holdouts.json").read_text())
    assert fixture["method"] == "original_integer_outside_kernel_scalar_sum" and fixture["outside"] == 4
    assert [(row["face_size"], row["degree"]) for row in fixture["cases"]] == [(2, 43), (3, 47), (5, 29), (8, 19)]
    for row in fixture["cases"]:
        lower, upper = error.coefficient_enclosure(row["face_size"], row["degree"])
        assert lower <= Fraction(int(row["numerator"]), int(row["denominator"])) <= upper


@pytest.mark.parametrize("size", [2, 3, 4, 8, 16])
def test_all_tail_threshold_covers_odd_blocks_jumps_and_every_larger_degree(size):
    for target in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 10**20)):
        threshold = error.relative_error_threshold(size, target)
        exponent = threshold.bit_length() - 1
        assert threshold == 2**exponent and threshold >= 2 * (size - 1)
        assert error.dyadic_tail_bound(size, exponent) <= target
        assert error.dyadic_tail_bound(size, exponent - 1) > target
        for offset in range(5):
            assert error.dyadic_tail_bound(size, exponent + offset) <= target
        for degree in (threshold, threshold + 1, 2 * threshold - 1, 2 * threshold, 3 * threshold):
            assert error.relative_error_bound(size, degree) <= target
    if size in (2, 3):
        assert error.relative_error_threshold(size, Fraction(1, 100)) == 2**54


@pytest.mark.parametrize("value", [True, False, None, 1.0, "2", -1, 0])
def test_invalid_integer_parameter_types_and_ranges(value):
    with pytest.raises(ValueError):
        error.relative_error_bound(value, 8)
    with pytest.raises(ValueError):
        error.relative_error_bound(2, value)
    with pytest.raises(ValueError):
        error.leading_constant_enclosure(2, value)


@pytest.mark.parametrize("target", [0, -1, 0.01, True, "1/100", Fraction(0), Fraction(-1, 100)])
def test_target_requires_a_positive_exact_fraction(target):
    with pytest.raises(ValueError):
        error.relative_error_threshold(2, target)


def test_large_degree_has_no_degree_sized_evaluation_or_global_digit_limit_change(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.risk_research.vertex_general.transformed_diagonals", lambda *args: pytest.fail("degree table"))
    previous = sys.get_int_max_str_digits()
    degree = 1 << 15000
    record = error.certificate(2, degree, terms=1, target=Fraction(1, 100))
    path = tmp_path / "huge.json"
    error.atomic_json(path, record)
    saved = json.loads(path.read_bytes(), parse_int=error._decimal_integer)
    assert type(saved["degree"]) is int and saved["degree"] == degree
    error.verify_certificate(saved)
    assert sys.get_int_max_str_digits() == previous


def test_cli_replays_nondefault_options_and_writes_identical_record(tmp_path):
    path, other = tmp_path / "record.json", tmp_path / "checked.json"
    assert (
        error.main(
            ["--face-size", "3", "--degree", "81", "--constant-terms", "9", "--relative-target", "1/7", "--output", str(path)]
        )
        == 0
    )
    assert error.main(["--verify", str(path), "--output", str(other)]) == 0
    assert path.read_bytes() == other.read_bytes()
    assert error.main(["--verify", str(path)]) == 0
    assert json.loads(path.read_bytes())["constant_terms"] == 9
    other.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(path), "--constant-terms", "8", "--output", str(other)])
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
        ("diagonal_majorant", 25),
    ],
)
def test_tampering_is_rejected_before_output_replacement(tmp_path, field, value):
    record = copy.deepcopy(error.certificate(2, 9, terms=8))
    record[field] = value
    path, output = tmp_path / "input.json", tmp_path / "old.json"
    path.write_text(json.dumps(record))
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        error.main(["--verify", str(path), "--output", str(output)])
    assert output.read_bytes() == b"keep"


def test_duplicate_keys_and_changed_sources_preserve_existing_output(tmp_path, monkeypatch):
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
def test_protected_source_destinations_rejected_before_computation(name, monkeypatch):
    monkeypatch.setattr(error, "certificate", lambda *args: pytest.fail("source collision computed"))
    with pytest.raises(SystemExit):
        error.main(["--face-size", "2", "--degree", "8", "--output", str(ROOT / name)])


def test_validated_output_parent_is_pinned_before_writer_entry(tmp_path, monkeypatch):
    protected = ROOT / error.PROOF
    old = protected.read_bytes()
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
    assert destination.is_file() and protected.read_bytes() == old


def test_native_positive_integral_leading_constant_with_separate_precision():
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
from tools.risk_research import nonvertex_q4_error as error
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(error.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(q):
    return arb(fmpq(q.numerator,q.denominator))
for bits in (47,128):
    before=ctx.prec
    normalized=integral_normalized_ball(6,bits=bits)
    assert ctx.prec==before
    with ctx.workprec(bits):
        C6=normalized*36*factorial(5)*(2*arb(2).log())**5
        for s in (2,3,7):
            scale=Fraction(comb(s+4,s)**2*prod(range(s,s+4)),43200*factorial(s-2))
            actual=C6*ball(scale)
            for terms in (8,32,64):
                lower,upper=error.leading_constant_enclosure(s,terms)
                assert actual>ball(lower) and actual<ball(upper)
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
