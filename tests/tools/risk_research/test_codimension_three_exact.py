"""Independent scalar, Gram, polynomial and saved-record checks."""

import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import codimension_three_exact as diagonal
from tools.risk_research.edge_closed import codimension_two_coefficient
from tools.risk_research.research_exact import integer_kernel
from tools.risk_research.vertex_general import leading_constant_enclosure as vertex_constant_enclosure

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("outside", [1, 2, 3, 4])
@pytest.mark.parametrize("size", [2, 3, 4])
def test_positive_diagonal_coefficients_match_original_scalar_sum(size, outside):
    for degree in [1, size, size + 1, size + 3]:
        assert diagonal.nonvertex_coefficient(size + outside, degree, size) == _scalar_risk_sum(size + outside, degree, size)


@pytest.mark.parametrize(("vertices", "size", "degree"), [(5, 2, 2), (5, 2, 3), (6, 3, 3), (6, 2, 3)])
def test_nonvertex_diagonals_match_independent_gram_propagation(vertices, size, degree):
    assert diagonal.nonvertex_coefficient(vertices, degree, size) == _exact_coefficients(vertices, degree)[0][size - 1]


@pytest.mark.parametrize("size", [2, 3, 5])
@pytest.mark.parametrize("degree", [1, 4, 11])
def test_top_face_and_existing_codimension_two_are_recovered(size, degree):
    top = Fraction(math.comb(degree - 1, size - 1)) if degree >= size else Fraction(0)
    assert diagonal.nonvertex_coefficient(size, degree, size) == top
    assert diagonal.nonvertex_coefficient(size + 2, degree, size) == codimension_two_coefficient(size, degree)


def test_harmonic_q3_diagonals_match_independent_signed_kernel_transform():
    kernel = integer_kernel(3, 8, backend="python")
    actual = diagonal.harmonic_diagonals(9)
    assert actual[0] == 1
    for j, value in enumerate(actual):
        expected = sum(
            (
                (-1) ** (h + ell) * math.comb(j, h) * math.comb(j, ell) * kernel.value(h, ell)
                for h in range(j + 1)
                for ell in range(j + 1)
            ),
            Fraction(0),
        )
        assert value == expected and value > 0
    assert diagonal.harmonic_diagonals(0) == []


@pytest.mark.parametrize(("size", "outside", "degree"), [(2, 1, 5), (2, 3, 7), (3, 3, 7), (4, 4, 8)])
def test_every_angular_mode_matches_its_original_radial_functional(size, outside, degree):
    kernel = integer_kernel(outside, degree - size, backend="python")
    for angular in range(degree - size + 1):
        length = degree - size - angular
        weights = [
            (-1) ** h * math.comb(length, h) * math.comb(degree + angular + 2 * size - 1, h) * math.factorial(h)
            for h in range(length + 1)
        ]
        expected = sum(
            (
                weights[h] * weights[ell] * math.factorial(2 * degree + size - h - ell - 1) * kernel.value(h, ell)
                for h in range(length + 1)
                for ell in range(length + 1)
            ),
            Fraction(0),
        ) / (math.factorial(size - 1) * math.prod(range(size + outside, size + outside + 2 * degree)))
        assert diagonal.angular_mode_ratio(size, outside, degree, angular) == expected


def _compositions(total, size):
    if size == 0:
        if total == 0:
            yield ()
    elif size == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _compositions(total - first, size - 1):
                yield (first, *rest)


def _constant_angular_polynomial(size, degree, subset):
    # Direct homogeneous propagation on s+3 coordinates.  Expand powers
    # of the source radius and uniform-variable moments into monomials;
    # do not call any outside kernel or transformed-diagonal code.
    result: dict[tuple[int, ...], Fraction] = {}
    length = degree - size
    for h in range(length + 1):
        radial = (-1) ** h * math.comb(length, h) * math.comb(degree + 2 * size - 1, h)
        for source in _compositions(length - h, size):
            source_factor = Fraction(math.factorial(length - h), math.prod(math.factorial(a) for a in source))
            for powers in _compositions(h, len(subset)):
                moment = Fraction(math.factorial(h), math.prod(math.factorial(a) * (a + 1) for a in powers))
                outside = [0] * 3
                for index, power in zip(subset, powers, strict=True):
                    outside[index] = power
                monomial = tuple(a + 1 for a in source) + tuple(outside)
                result[monomial] = result.get(monomial, Fraction(0)) + radial * source_factor * moment
    return {powers: value for powers, value in result.items() if value}


def _inner(left, right):
    vertices = len(next(iter(left)))
    result = Fraction(0)
    for a, first in left.items():
        for b, second in right.items():
            moment = Fraction(
                math.factorial(vertices - 1) * math.prod(math.factorial(x + y) for x, y in zip(a, b, strict=True)),
                math.factorial(vertices - 1 + sum(a) + sum(b)),
            )
            result += first * second * moment
    return result


@pytest.mark.parametrize(("size", "degree"), [(2, 5), (2, 6), (3, 6), (3, 7)])
def test_genuine_full_bubble_trace_and_norm_ratio_by_independent_polynomial_moments(size, degree):
    full = _constant_angular_polynomial(size, degree, (0, 1, 2))
    difference = dict(full)
    for count in range(3):
        for subset in itertools.combinations(range(3), count):
            sign = (-1) ** (4 - count)
            for powers, value in _constant_angular_polynomial(size, degree, subset).items():
                difference[powers] = difference.get(powers, Fraction(0)) - sign * value
    difference = {powers: value for powers, value in difference.items() if value}
    assert difference  # A real full bubble, not a vacuous degree<M case.
    assert all(all(power > 0 for power in powers) for powers in difference)
    assert _inner(full, difference) == 0
    source_norm = Fraction(math.factorial(size - 1) * 2**size, math.factorial(3 * size - 1))
    assert _inner(full, full) / source_norm == diagonal.angular_mode_ratio(size, 3, degree, 0)


def test_frozen_larger_cases_match_independent_original_integer_kernel():
    report = json.loads((ROOT / "tools/risk_research/examples/codimension_three_exact_holdouts.json").read_bytes())
    assert report["method"] == "original_integer_outside_kernel_scalar_sum"
    assert [(row["face_size"], row["outside"], row["degree"]) for row in report["cases"]] == [
        (2, 3, 17),
        (2, 3, 37),
        (3, 3, 23),
        (3, 3, 41),
        (4, 3, 31),
        (2, 4, 29),
    ]
    for row in report["cases"]:
        expected = Fraction(int(row["numerator"]), int(row["denominator"]))
        assert diagonal.nonvertex_coefficient(row["face_size"] + row["outside"], row["degree"], row["face_size"]) == expected


@pytest.mark.parametrize(("size", "outside", "ratio"), [(2, 2, Fraction(3, 8)), (2, 3, Fraction(1, 2)), (3, 3, Fraction(5))])
def test_known_constant_relations_recover_existing_codimension_two_and_q3_normalizations(size, outside, ratio):
    for terms in [8, 128]:
        lower, upper = vertex_constant_enclosure(outside + 2, terms)
        assert diagonal.nonvertex_leading_constant_enclosure(size, outside, terms) == (ratio * lower, ratio * upper)


def test_exact_inverse_factorial_moment_used_for_summable_domination():
    for length in range(7):
        for beta in range(9):
            other = length + beta
            normalizer = math.comb(length + other, length)
            actual = (
                sum(
                    (Fraction(math.comb(length, j) * math.comb(other, j), (j + 1) * (j + 2)) for j in range(length + 1)),
                    Fraction(0),
                )
                / normalizer
            )
            assert actual == Fraction(
                (length + other + 1) * (length + other + 2), (length + 1) * (length + 2) * (other + 1) * (other + 2)
            )
            inverse_square = (
                sum(
                    (Fraction(math.comb(length, j) * math.comb(other, j), (j + 1) ** 2) for j in range(length + 1)),
                    Fraction(0),
                )
                / normalizer
            )
            assert inverse_square <= Fraction(8, (length + 1) * (length + 2))


def test_nonvertex_leading_intervals_contain_independent_known_zeta_constants():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for native known-zeta interval checks")
        executable = sys.executable
    script = """
from flint import arb,ctx,fmpq
from tools.risk_research.codimension_three_exact import nonvertex_leading_constant_enclosure
with ctx.workprec(192):
    cases=[(2,2,18*arb.pi()**2),(2,3,3600*(arb.pi()**2/6-arb(3).zeta())),
           (3,3,36000*(arb.pi()**2/6-arb(3).zeta()))]
    for s,q,known in cases:
        for terms in (16,128):
            lower,upper=nonvertex_leading_constant_enclosure(s,q,terms)
            assert known>arb(fmpq(lower.numerator,lower.denominator))
            assert known<arb(fmpq(upper.numerator,upper.denominator))
"""
    result = subprocess.run([executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (diagonal.harmonic_diagonals, (True,)),
        (diagonal.harmonic_diagonals, (-1,)),
        (diagonal.harmonic_diagonals, (2.0,)),
        (diagonal.nonvertex_coefficient, (5, 3, 1)),
        (diagonal.nonvertex_coefficient, (2, 3, 3)),
        (diagonal.nonvertex_coefficient, (5, 0, 2)),
        (diagonal.nonvertex_coefficient, (5.0, 3, 2)),
        (diagonal.nonvertex_coefficient, (5, True, 2)),
        (diagonal.nonvertex_coefficient, (5, 3, True)),
        (diagonal.codimension_three_coefficient, (True, 3)),
        (diagonal.angular_mode_ratio, (2, -1, 3, 0)),
        (diagonal.angular_mode_ratio, (2, True, 3, 0)),
        (diagonal.angular_mode_ratio, (2, 3, 1, 0)),
        (diagonal.angular_mode_ratio, (2, 3, 3, 2)),
        (diagonal.angular_mode_ratio, (2, 3, 3, False)),
        (diagonal.nonvertex_leading_constant_enclosure, (1, 3, 8)),
        (diagonal.nonvertex_leading_constant_enclosure, (2, 1, 8)),
        (diagonal.nonvertex_leading_constant_enclosure, (2, 3, 0)),
        (diagonal.nonvertex_leading_constant_enclosure, (2, 3, True)),
    ],
)
def test_integer_domains_reject_coercion_and_invalid_angular_indices(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


def test_cli_recomputes_complete_records_and_rejects_altered_values_or_domains(tmp_path):
    original, checked = tmp_path / "value.json", tmp_path / "checked.json"
    arguments = ["--face-size", "3", "--degree", "9"]
    assert diagonal.main([*arguments, "--output", str(original)]) == 0
    report = json.loads(original.read_bytes())
    assert report["algorithm"] == "positive_harmonic_double_sum"
    assert report["source_hash_policy"] == "sha256-crlf-to-lf-v1"
    assert diagonal.main([*arguments, "--verify", str(original), "--output", str(checked)]) == 0
    assert checked.read_bytes() == original.read_bytes()
    for key, value in [
        ("degree", 8),
        ("coefficient", {"numerator": "0", "denominator": "1"}),
        ("source_hash_policy", "unknown"),
    ]:
        altered = {**report, key: value}
        original.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            diagonal.main([*arguments, "--verify", str(original), "--output", str(checked)])
        assert error.value.code == 2
        assert json.loads(checked.read_bytes()) == report


def test_cli_validation_and_raw_source_guard_preserve_previous_destination(tmp_path, monkeypatch):
    output = tmp_path / "previous.json"
    output.write_text("previous output", encoding="utf-8")
    with pytest.raises(SystemExit):
        diagonal.main(["--face-size", "1", "--degree", "4", "--output", str(output)])
    original = diagonal._sources()
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        if calls == 1:
            return original
        return {name: source + b"\r\n" for name, source in original.items()}

    monkeypatch.setattr(diagonal, "_sources", changing_sources)
    with pytest.raises(SystemExit) as error:
        diagonal.main(["--face-size", "2", "--degree", "6", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "previous output"


def test_real_crlf_and_lf_source_files_produce_identical_records_without_site_packages(tmp_path):
    paths = []
    for line_endings in ["lf", "crlf"]:
        destination = tmp_path / line_endings / "tools/risk_research"
        destination.mkdir(parents=True)
        for name, source in diagonal._sources().items():
            canonical = source.replace(b"\r\n", b"\n")
            (destination / name).write_bytes(canonical if line_endings == "lf" else canonical.replace(b"\n", b"\r\n"))
        output = tmp_path / f"{line_endings}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "tools.risk_research.codimension_three_exact",
                "--face-size",
                "2",
                "--degree",
                "7",
                "--output",
                str(output),
            ],
            cwd=tmp_path / line_endings,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        paths.append(output)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    report = json.loads(paths[0].read_bytes())
    assert report["source_sha256"] == {
        name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in diagonal._sources().items()
    }
