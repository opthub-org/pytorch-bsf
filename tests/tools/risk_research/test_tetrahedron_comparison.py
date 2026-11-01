"""Check the complete tetrahedron comparison with exact finite enclosures."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients
from tools.risk_research import tetrahedron_comparison
from tools.risk_research.tetrahedron_comparison import certify_tetrahedron_comparison, tetrahedron_coefficients


def test_all_tetrahedron_coefficients_match_independent_gram_propagation():
    coefficients = tetrahedron_coefficients(6)
    for degree, observed in coefficients.items():
        expected = _exact_coefficients(4, degree)[0]
        assert observed == tuple(expected + [Fraction(0)] * (4 - len(expected)))


def test_every_finite_degree_and_the_analytic_tail_are_strictly_certified():
    certificate = certify_tetrahedron_comparison()
    assert certificate["all_at_once_wins_degrees"] == [3, 4, 5, 6, 7]
    points = certificate["finite_comparisons"]
    assert isinstance(points, list)
    assert all(isinstance(point, dict) for point in points)
    assert [point["degree"] for point in points] == list(range(1, 625))
    assert all(int(point["strict_margin_numerator"]) > 0 for point in points)
    assert all(point["decision"] == "inductive_wins" for point in points if point["degree"] >= 8)
    tail = certificate["tail"]
    assert isinstance(tail, dict)
    assert tail["from_degree"] == 625
    upper = tail["risk_excess_over_top_divided_by_degree_squared_upper_bound"]
    margin = tail["strict_margin_below_two"]
    assert Fraction(int(upper["numerator"]), int(upper["denominator"])) < 2
    assert Fraction(int(margin["numerator"]), int(margin["denominator"])) > 0


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5])
def test_coefficient_range_requires_positive_integer(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        tetrahedron_coefficients(invalid)


def test_cli_writes_a_source_bound_certificate(tmp_path, monkeypatch):
    monkeypatch.setattr(tetrahedron_comparison, "certify_tetrahedron_comparison", lambda **kwargs: {"proof_status": "test"})
    output = tmp_path / "certificate.json"
    assert tetrahedron_comparison.main(["--output", str(output)]) == 0
    result = json.loads(output.read_bytes())
    assert result["proof_status"] == "test"
    for name, digest in result["source_sha256"].items():
        source = Path(tetrahedron_comparison.__file__).with_name(name)
        assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cli_preserves_existing_output_when_comparison_is_unresolved(tmp_path, monkeypatch):
    output = tmp_path / "certificate.json"
    original = b"existing complete certificate\n"
    output.write_bytes(original)

    def unresolved(**kwargs):
        raise ArithmeticError("overlapping bounds")

    monkeypatch.setattr(tetrahedron_comparison, "certify_tetrahedron_comparison", unresolved)
    with pytest.raises(SystemExit) as error:
        tetrahedron_comparison.main(["--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_cli_preserves_output_if_a_dependency_changes_during_certification(tmp_path, monkeypatch):
    output = tmp_path / "certificate.json"
    original = b"existing complete certificate\n"
    output.write_bytes(original)
    dependency = Path(tetrahedron_comparison.__file__).with_name("vertex_single.py")
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
    monkeypatch.setattr(tetrahedron_comparison, "certify_tetrahedron_comparison", certify)
    with pytest.raises(SystemExit) as error:
        tetrahedron_comparison.main(["--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original
