"""Independently check the finite-degree q3 error and complete records."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import nonvertex_q3_error as error
from tools.risk_research.codimension_three_exact import harmonic_diagonals

ROOT = Path(__file__).resolve().parents[3]


def _fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def test_global_diagonal_majorants_include_zero_and_small_indices():
    diagonal = harmonic_diagonals(129)
    assert sum(diagonal[:8]) == Fraction(8568671, 2116800) > 4
    for j, value in enumerate(diagonal):
        assert 0 < value <= Fraction(25, (j + 1) ** 2)
    for j in range(len(diagonal) - 1):
        assert abs(diagonal[j + 1] - diagonal[j]) <= Fraction(102, (j + 1) * (j + 2) * (j + 3))


@pytest.mark.parametrize("length", range(8))
@pytest.mark.parametrize("extra", [0, 1, 4, 15])
def test_deficit_moment_matches_direct_binomial_probability_sum(length, extra):
    other = length + extra
    normalizer = comb(length + other, length)
    actual = (
        sum(
            (
                Fraction(comb(length, j) * comb(other, j) * (length - j), (j + 1) * (j + 2) * (j + 3))
                for j in range(length + 1)
            ),
            Fraction(0),
        )
        / normalizer
    )
    expected = Fraction(
        length * (length + other + 1) * (length + other + 2),
        (length + 1) * (length + 2) * (other + 1) * (other + 2) * (other + 3),
    )
    assert actual == expected


@pytest.mark.parametrize("size,degree", [(2, 2), (2, 7), (2, 18), (3, 3), (3, 8), (3, 19), (4, 4), (4, 12), (7, 15)])
def test_averaging_lipschitz_bound_and_angular_deficit_by_exact_sums(size, degree):
    diagonal = harmonic_diagonals(degree - size + 1)
    total, direct = Fraction(0), Fraction(0)
    for length in range(degree - size + 1):
        other = 2 * degree + size - length - 1
        average = sum((comb(length, j) * comb(other, j) * diagonal[j] for j in range(length + 1)), Fraction(0)) / comb(
            length + other, length
        )
        direct += abs(average - diagonal[length])
        weight = prod((Fraction(degree - length - r, degree) for r in range(2, size)), start=Fraction(1))
        total += weight * average
    count = degree - size + 1
    harmonic = sum((Fraction(1, j) for j in range(1, count + 1)), Fraction(0))
    averaging_bound = Fraction(408, degree + 2 * size + 2) * (harmonic + Fraction(2, count + 1) - 2)
    assert direct <= averaging_bound
    deficit = sum(diagonal) - sum(
        (
            value * prod((Fraction(degree - length - r, degree) for r in range(2, size)), start=Fraction(1))
            for length, value in enumerate(diagonal)
        ),
        Fraction(0),
    )
    assert 0 <= deficit <= Fraction(25 * ((size - 2) * harmonic + (size - 1) * (size - 2)), degree)
    assert abs(total - sum(diagonal)) <= averaging_bound + deficit


@pytest.mark.parametrize("size,degree", [(2, 1), (2, 2), (2, 5), (3, 1), (3, 2), (3, 3), (3, 5)])
def test_finite_enclosures_contain_independent_gram_coefficients(size, degree):
    lower, upper = error.coefficient_enclosure(size, degree, terms=9)
    exact = _exact_coefficients(size + 3, degree)[0]
    actual = exact[size - 1] if size <= degree else Fraction(0)
    assert lower <= actual <= upper
    if degree < size:
        assert lower == actual == upper == 0
        assert error.relative_error_bound(size, degree) == 1


@pytest.mark.parametrize("size,degree", [(2, 9), (2, 13), (3, 11), (3, 14), (4, 6), (5, 7), (7, 9)])
def test_finite_enclosures_contain_original_scalar_coefficients(size, degree):
    actual = _scalar_risk_sum(size + 3, degree, size)
    lower, upper = error.coefficient_enclosure(size, degree)
    assert lower <= actual <= upper


def test_new_original_integer_kernel_holdouts_after_proof_freeze():
    fixture = json.loads((ROOT / "tools/risk_research/examples/nonvertex_q3_error_holdouts.json").read_text())
    assert fixture["method"] == "original_integer_outside_kernel_scalar_sum"
    assert [(row["face_size"], row["degree"]) for row in fixture["cases"]] == [(2, 43), (3, 47), (4, 31), (7, 19)]
    for row in fixture["cases"]:
        lower, upper = error.coefficient_enclosure(row["face_size"], row["degree"])
        actual = Fraction(int(row["numerator"]), int(row["denominator"]))
        assert lower <= actual <= upper


@pytest.mark.parametrize("size", [2, 3, 4, 8, 16])
def test_dyadic_all_tail_thresholds_use_exact_target_and_are_sufficient(size):
    for target in [Fraction(1, 10), Fraction(1, 100), Fraction(1, 10000), Fraction(1, 10**30)]:
        threshold = error.relative_error_threshold(size, target)
        exponent = threshold.bit_length() - 1
        assert threshold == 2**exponent and exponent >= 2 and threshold >= 2 * (size - 1)
        assert error.dyadic_tail_bound(size, exponent) <= target
        if exponent > max(2, (2 * size - 3).bit_length()):
            assert error.dyadic_tail_bound(size, exponent - 1) > target
        for degree in [threshold, threshold + 1, 2 * threshold - 1, 2 * threshold, 3 * threshold]:
            assert error.relative_error_bound(size, degree) <= target
            lower, upper = error.coefficient_enclosure(size, degree, terms=7)
            assert 0 < lower < upper
    if size in (2, 3):
        assert error.relative_error_threshold(size, Fraction(1, 100)) == 262144


def test_large_degree_evaluation_does_not_build_degree_sized_tables():
    degree = 2**256
    for size in (2, 3, 7):
        lower, upper = error.coefficient_enclosure(size, degree, terms=2)
        assert 0 < lower < upper
        assert error.relative_error_bound(size, degree) <= error.dyadic_tail_bound(size, 256)


@pytest.mark.parametrize("size", [True, False, 1, 0, -1, 2.0, "2", None])
def test_unsupported_source_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="face_size"):
        error.coefficient_enclosure(size, 4)


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2", None])
def test_invalid_degree_and_constant_term_types_are_rejected(value):
    with pytest.raises(ValueError):
        error.relative_error_bound(2, value)
    with pytest.raises(ValueError):
        error.leading_constant_enclosure(2, value)


@pytest.mark.parametrize("value", [True, False, 0, -1, Fraction(0), Fraction(-1), 0.01, "1/100", None])
def test_target_must_be_a_positive_exact_fraction(value):
    with pytest.raises(ValueError, match="target"):
        error.relative_error_threshold(2, value)


@pytest.mark.parametrize("size", [2, 3, 4, 5, 7, 12])
def test_all_source_leading_normalizations_match_existing_closed_formula(size):
    factor = Fraction(size * (size + 1) ** 3 * (size + 2) ** 3 * (size + 3) ** 2, 24 * factorial(size - 2))
    baseline = error.leading_constant_enclosure(2, 9)
    assert error.leading_constant_enclosure(size, 9) == tuple(value * factor / 3600 for value in baseline)


@pytest.mark.parametrize("size", [4, 7, 20])
def test_general_active_prefix_and_dyadic_start_are_separate(size):
    assert error.coefficient_enclosure(size, size - 1) == (0, 0)
    for degree in (size, 2 * size - 3, 2 * size - 2, 2 * size):
        lower, upper = error.coefficient_enclosure(size, degree, 3)
        assert 0 <= lower < upper
    with pytest.raises(ValueError, match="dyadic degree"):
        error.dyadic_tail_bound(size, 2)


def test_cli_round_trip_restores_nondefault_terms_and_target(tmp_path):
    output, checked = tmp_path / "record.json", tmp_path / "checked.json"
    arguments = ["--face-size", "3", "--degree", "262144", "--constant-terms", "7", "--relative-target", "1/100"]
    assert error.main([*arguments, "--output", str(output)]) == 0
    original = json.loads(output.read_text())
    assert original["constant_terms"] == 7
    assert original["all_tail"]["requested_degree_in_tail"] is True
    assert _fraction(original["relative_error_bound"]) <= Fraction(1, 100)
    assert error.main(["--verify", str(output), "--output", str(checked)]) == 0
    assert error.main(["--verify", str(output)]) == 0
    assert error.main([*arguments, "--verify", str(output), "--output", str(checked)]) == 0
    assert json.loads(checked.read_text()) == original
    changed = copy.deepcopy(original)
    changed["outside"] = 3.0
    output.write_text(json.dumps(changed))
    previous = checked.read_bytes()
    with pytest.raises(SystemExit):
        error.main([*arguments, "--verify", str(output), "--output", str(checked)])
    assert checked.read_bytes() == previous


def test_verification_rejects_conflicting_overrides_and_invalid_saved_parameters(tmp_path):
    output = tmp_path / "record.json"
    assert error.main(["--face-size", "2", "--degree", "30", "--constant-terms", "9", "--output", str(output)]) == 0
    original = output.read_bytes()
    for option, value in [
        ("--face-size", "3"),
        ("--degree", "31"),
        ("--constant-terms", "10"),
        ("--relative-target", "1/100"),
    ]:
        with pytest.raises(SystemExit):
            error.main(["--verify", str(output), "--output", str(output), option, value])
        assert output.read_bytes() == original
    invalid_records: list[object] = [None, [], {"schema": "unsupported"}, {"schema": error.SCHEMA}]
    for invalid in invalid_records:
        with pytest.raises(ValueError):
            error.verify_certificate(invalid)
    for target in [
        {"numerator": True, "denominator": "1"},
        {"numerator": "1", "denominator": "0"},
        {"numerator": "bad", "denominator": "1"},
        {"extra": "1"},
    ]:
        record = json.loads(original)
        record["relative_target"] = target
        with pytest.raises(ValueError):
            error.verify_certificate(record)


@pytest.mark.parametrize("change", ["unknown", "bound", "source", "target", "false_integer", "duplicate", "nan"])
def test_complete_record_tampering_fails_and_preserves_destination(tmp_path, change):
    output, previous = tmp_path / "record.json", tmp_path / "previous.json"
    arguments = ["--face-size", "2", "--degree", "20", "--relative-target", "1/100", "--output", str(output)]
    assert error.main(arguments) == 0
    record = json.loads(output.read_text())
    if change == "unknown":
        record["extra"] = "not recognized"
    elif change == "bound":
        record["relative_error_bound"]["numerator"] = "0"
    elif change == "source":
        record["source_sha256"] = {}
    elif change == "target":
        record["relative_target"]["denominator"] = "200"
    elif change == "false_integer":
        record["active"] = 1
    elif change == "nan":
        record["degree"] = float("nan")
    text = json.dumps(record)
    if change == "duplicate":
        text = text[:-1] + ',"degree":20}'
    previous.write_text(text)
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        error.main([*arguments, "--verify", str(previous)])
    assert output.read_bytes() == saved


def test_failed_source_guard_and_cli_validation_preserve_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "previous.json"
    output.write_bytes(b"keep this result")
    arguments = ["--face-size", "2", "--degree", "20", "--output", str(output)]
    for extra in [["--constant-terms", "1"], ["--relative-target", "0"], ["--relative-target", "1/0"]]:
        with pytest.raises(SystemExit):
            error.main([*arguments, *extra])
        assert output.read_bytes() == b"keep this result"
    sources = error._sources()
    calls = 0

    def changed_sources():
        nonlocal calls
        calls += 1
        return sources if calls == 1 else {name: value.replace(b"\n", b"\r\n") for name, value in sources.items()}

    monkeypatch.setattr(error, "_sources", changed_sources)
    with pytest.raises(SystemExit):
        error.main(arguments)
    assert output.read_bytes() == b"keep this result"


def test_lf_and_crlf_real_checkouts_reproduce_records_without_site_packages(tmp_path):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, content in error._sources().items():
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content.replace(b"\r\n", b"\n").replace(b"\n", ending))
        output = tmp_path / (checkout.name + ".json")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "tools.risk_research.nonvertex_q3_error",
                "--face-size",
                "3",
                "--degree",
                "100",
                "--constant-terms",
                "9",
                "--output",
                str(output),
            ],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]


def test_native_arb_known_constants_and_large_positive_double_sum():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for native Arb independent q3 checks")
        executable = sys.executable
    script = r"""
from fractions import Fraction
from math import comb,prod
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import nonvertex_q3_error as error
assert Path(error.__file__).resolve().is_relative_to(Path.cwd().resolve())
def ball(q):
    return arb(fmpq(q.numerator,q.denominator))
with ctx.workprec(192):
    for s in (2,3):
        actual=(3600 if s==2 else 36000)*(arb.pi()**2/6-arb(3).zeta())
        for terms in (2,9,64):
            lower,upper=error.leading_constant_enclosure(s,terms)
            assert actual>ball(lower) and actual<ball(upper)
    for s,D in ((2,2048),(3,2049)):
        H,H2=arb(0),arb(0)
        delta=[]
        for j in range(D-s+1):
            if j:
                H+=arb(1)/j
                H2+=arb(1)/(j*j)
            delta.append(6*(H*(H-arb(j)/(j+2))/((j+1)*(j+2)*(j+3))+H2/((j+1)*(j+3))+arb(2)/((j+1)**2*(j+2)**2*(j+3))))
        total=arb(0)
        for k in range(D-s+1):
            L=D-s-k
            B=D+2*s+k-1
            probability=arb(1)/comb(2*D+s-1,L)
            mean=arb(0)
            for j in range(L+1):
                mean+=probability*delta[j]
                probability=probability*(L-j)*(B-j)/((j+1)**2)
            total+=comb(k+s-2,s-2)*mean
        actual=total*comb(s+3,s)**2*prod(range(s,s+3))/prod(range(2*D+s,2*D+s+3))
        lower,upper=error.coefficient_enclosure(s,D,64)
        assert 0<lower<upper
        assert actual>ball(lower) and actual<ball(upper)
        A=(3600 if s==2 else 36000)*(arb.pi()**2/6-arb(3).zeta())
        relative=abs(actual*D**(5-s)/A-1)
        assert relative<ball(error.relative_error_bound(s,D))
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
