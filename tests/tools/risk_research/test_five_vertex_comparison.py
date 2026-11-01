"""Verify the all-degree five-vertex comparison and its finite certificate."""

import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import five_vertex_comparison as comparison

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def certificate():
    return comparison.certify_five_vertex_comparison()


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_five_vertex_rows_match_independent_gram_propagation(degree):
    observed = comparison.five_vertex_coefficients(degree)[degree]
    expected = _exact_coefficients(5, degree)[0]
    assert observed == tuple(expected + [Fraction(0)] * (5 - len(expected)))


@pytest.mark.parametrize("degree", [7, 11, 13])
def test_five_vertex_rows_match_original_rational_scalar_sum(degree):
    observed = comparison.five_vertex_coefficients(degree)[degree]
    assert observed == tuple(_scalar_risk_sum(5, degree, size) for size in range(1, 6))


def test_finite_prefix_covers_every_degree_with_strict_margins(certificate):
    assert certificate["all_at_once_wins_degrees"] == list(range(3, 12))
    points = certificate["finite_comparisons"]
    assert [point["degree"] for point in points] == list(range(1, 64))
    assert all(point["bits"] == 32 and int(point["strict_margin_numerator"]) > 0 for point in points)
    for point in points:
        decision = "all_at_once_wins" if 3 <= point["degree"] <= 11 else "inductive_wins"
        assert point["decision"] == decision


def test_tail_has_a_separate_positive_exact_margin(certificate):
    tail = certificate["tail"]
    assert tail["from_degree"] == 64
    q = tail["lower_skeleton_sqrt_sum_divided_by_degree_upper_bound"]
    q = Fraction(int(q["numerator"]), int(q["denominator"]))
    upper = tail["risk_excess_over_top_divided_by_degree_cubed_upper_bound"]
    upper = Fraction(int(upper["numerator"]), int(upper["denominator"]))
    margin = tail["strict_margin_below_five_sixths"]
    margin = Fraction(int(margin["numerator"]), int(margin["denominator"]))
    assert upper == 5 * q / 12 + q**2 / 64
    assert margin == Fraction(5, 6) - upper > 0


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "5"])
def test_ranges_reject_nonpositive_or_noninteger_inputs(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        comparison.five_vertex_coefficients(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        comparison.certify_five_vertex_comparison(max_bits=invalid)


@pytest.mark.parametrize("invalid", ["wrong", None, [], True])
def test_backend_validation_applies_even_at_inactive_degree(invalid):
    with pytest.raises(ValueError, match="backend"):
        comparison.five_vertex_coefficients(1, backend=invalid)


def test_partial_prefix_cannot_be_reported_as_complete(monkeypatch):
    monkeypatch.setattr(comparison, "five_vertex_coefficients", lambda *args, **kwargs: {1: (Fraction(5, 3),)})
    with pytest.raises(ArithmeticError, match="stated classification"):
        comparison.certify_five_vertex_comparison()


def test_cli_writes_complete_source_bound_report(tmp_path):
    output = tmp_path / "comparison.json"
    assert comparison.main(["--output", str(output)]) == 0
    report = json.loads(output.read_bytes())
    assert len(report["finite_comparisons"]) == 63
    assert report["all_at_once_wins_degrees"] == list(range(3, 12))
    for name, digest in report["source_sha256"].items():
        assert digest == hashlib.sha256(Path(comparison.__file__).with_name(name).read_bytes()).hexdigest()


def test_insufficient_precision_preserves_existing_output(tmp_path):
    output = tmp_path / "comparison.json"
    original = b"existing complete certificate\n"
    output.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        comparison.main(["--output", str(output), "--max-bits", "1"])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_changed_dependency_preserves_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "comparison.json"
    original = b"existing complete certificate\n"
    output.write_bytes(original)
    dependency = Path(comparison.__file__).with_name("research_exact.py")
    real_read = Path.read_bytes
    changed = False

    def read_bytes(path):
        data = real_read(path)
        return data + b"\n# changed\n" if changed and path == dependency else data

    def certify(**kwargs):
        nonlocal changed
        changed = True
        return {"proof_status": "test"}

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(comparison, "certify_five_vertex_comparison", certify)
    with pytest.raises(SystemExit) as error:
        comparison.main(["--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_native_kernel_backend_independently_reproduces_every_finite_row(certificate):
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if not executable:
        pytest.skip("set BSF_FLINT_PYTHON for the native kernel cross-check")
    source = (
        "import json; from tools.risk_research.five_vertex_comparison import five_vertex_coefficients; "
        "print(json.dumps({d:[str(v) for v in row] for d,row in five_vertex_coefficients(63,backend='flint').items()}))"
    )
    result = subprocess.run([executable, "-c", source], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    observed = json.loads(result.stdout)
    for point in certificate["finite_comparisons"]:
        expected = [Fraction(int(v["numerator"]), int(v["denominator"])) for v in point["coefficients"]]
        assert [Fraction(v) for v in observed[str(point["degree"])]] == expected
