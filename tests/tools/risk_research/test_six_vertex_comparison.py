"""Certify the six-vertex finite prefix and independent analytic tail."""

import copy
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import six_vertex_comparison as comparison

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def certificate():
    return comparison.certify_six_vertex_comparison()


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_rows_match_independent_gram_propagation(degree):
    observed = comparison.six_vertex_coefficients(degree)[degree]
    expected = _exact_coefficients(6, degree)[0]
    assert observed == tuple(expected + [Fraction(0)] * (6 - len(expected)))


@pytest.mark.parametrize("degree", [7, 11, 17])
def test_rows_match_original_rational_scalar_formula(degree):
    observed = comparison.six_vertex_coefficients(degree)[degree]
    assert observed == tuple(_scalar_risk_sum(6, degree, size) for size in range(1, 7))


def test_every_finite_degree_is_separated_without_shape_assumptions(certificate):
    assert certificate["all_at_once_wins_degrees"] == list(range(3, 17))
    points = certificate["finite_comparisons"]
    assert [p["degree"] for p in points] == list(range(1, 128))
    for point in points:
        assert len(point["coefficients"]) == 6
        assert int(point["strict_margin_numerator"]) > 0
        assert point["decision"] == ("all_at_once_wins" if 3 <= point["degree"] <= 16 else "inductive_wins")


def test_tail_has_exact_positive_rational_margin(certificate):
    tail = certificate["tail"]
    q = Fraction(
        **{k: int(v) for k, v in tail["lower_skeleton_sqrt_sum_divided_by_degree_to_three_halves_upper_bound"].items()}
    )
    upper = Fraction(**{k: int(v) for k, v in tail["risk_excess_over_top_divided_by_degree_four_upper_bound"].items()})
    margin = Fraction(**{k: int(v) for k, v in tail["strict_margin_below_one_quarter"].items()})
    assert tail["from_degree"] == 128
    assert q == Fraction(702559551, 738197504)
    assert upper == Fraction(11, 60) * q + q**2 / 128
    assert margin == Fraction(1, 4) - upper > Fraction(6, 100)
    for degree in [6, 17, 128, 1000]:
        assert math.comb(degree + 5, 5) - math.comb(degree - 1, 5) == Fraction(degree**4 + 15 * degree**2, 4) + 2


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "6", None])
def test_ranges_reject_nonpositive_or_noninteger_values(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        comparison.six_vertex_coefficients(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        comparison.certify_six_vertex_comparison(max_bits=invalid)


@pytest.mark.parametrize("invalid", ["wrong", None, [], True])
def test_backend_validation_even_when_every_nonvertex_is_inactive(invalid):
    with pytest.raises(ValueError, match="backend"):
        comparison.six_vertex_coefficients(1, backend=invalid)


def test_a_partial_prefix_cannot_masquerade_as_complete(monkeypatch):
    monkeypatch.setattr(comparison, "six_vertex_coefficients", lambda *args, **kwargs: {1: (Fraction(12, 7),) * 6})
    with pytest.raises(ArithmeticError, match="incomplete"):
        comparison.certify_six_vertex_comparison()


@pytest.mark.parametrize("field", ["tail", "finite_comparisons", "source_sha256"])
def test_tampered_certificate_is_rejected(certificate, field):
    changed = copy.deepcopy(certificate)
    changed[field] = "altered"
    with pytest.raises(ValueError, match="does not match"):
        comparison.verify_certificate(changed)


def test_cli_round_trip(tmp_path):
    output = tmp_path / "six-vertex.json"
    assert comparison.main(["--output", str(output)]) == 0
    assert comparison.main(["--verify", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(report["finite_comparisons"]) == 127
    assert report["source_hash_policy"] == "sha256-crlf-to-lf-v1"


def test_insufficient_precision_preserves_existing_output(tmp_path):
    output = tmp_path / "six-vertex.json"
    original = b"previous complete certificate\n"
    output.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        comparison.main(["--output", str(output), "--max-bits", "1"])
    assert error.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize("initial_line_ending", ["lf", "crlf"])
def test_portable_crlf_hashes_and_raw_concurrent_edit_guard(certificate, tmp_path, monkeypatch, initial_line_ending):
    initial_sources = {name: source.replace(b"\r\n", b"\n") for name, source in comparison._sources().items()}
    if initial_line_ending == "crlf":
        initial_sources = {name: source.replace(b"\n", b"\r\n") for name, source in initial_sources.items()}
    monkeypatch.setattr(comparison, "_sources", lambda: initial_sources)
    sources = comparison._sources()
    lf = {name: source.replace(b"\r\n", b"\n") for name, source in sources.items()}
    crlf = {name: source.replace(b"\n", b"\r\n") for name, source in lf.items()}
    monkeypatch.setattr(comparison, "_sources", lambda: crlf)
    assert comparison.certify_six_vertex_comparison() == certificate
    calls = 0

    def changing():
        nonlocal calls
        calls += 1
        return lf if calls == 1 else crlf

    monkeypatch.setattr(comparison, "_sources", changing)
    monkeypatch.setattr(comparison, "certify_six_vertex_comparison", lambda **kwargs: {})
    output = tmp_path / "six-vertex.json"
    output.write_bytes(b"previous complete certificate\n")
    with pytest.raises(SystemExit) as error:
        comparison.main(["--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous complete certificate\n"


def test_original_native_kernels_reproduce_all_127_rows(certificate):
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if not executable:
        pytest.skip("set BSF_FLINT_PYTHON for independent original-kernel verification")
    source = (
        "import json; from tools.risk_research.six_vertex_comparison import six_vertex_coefficients; "
        "print(json.dumps({d:[str(v) for v in row] for d,row in six_vertex_coefficients(127,backend='flint').items()}))"
    )
    result = subprocess.run([executable, "-c", source], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    observed = json.loads(result.stdout)
    for point in certificate["finite_comparisons"]:
        expected = [Fraction(**{k: int(v) for k, v in item.items()}) for item in point["coefficients"]]
        assert [Fraction(value) for value in observed[str(point["degree"])]] == expected
