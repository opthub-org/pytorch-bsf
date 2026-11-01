"""Independent defining sums and oracles for a uniform fifteen-state family."""

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import codimension_two_family as family
from tools.risk_research.edge_closed import _central_correction, _tables, codimension_two_coefficient

ROOT = Path(__file__).resolve().parents[3]


def _state(h, epsilon):
    n = 2 * h + epsilon
    hn = sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))
    hh = sum((Fraction(1, j) for j in range(1, h + 1)), Fraction(0))
    hn2 = sum((Fraction(1, j**2) for j in range(1, n + 1)), Fraction(0))
    hh2 = sum((Fraction(1, j**2) for j in range(1, h + 1)), Fraction(0))
    f = sum((Fraction(1, j**2 * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    g = sum((Fraction(1, j * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    p = sum((Fraction(1, (j + 1) * math.comb(n, j)) for j in range(n + 1)), Fraction(0))
    t1 = sum((Fraction(1, j * math.comb(n, j)) for j in range(1, h + 1)), Fraction(0))
    t2 = sum((Fraction(1, j**2 * math.comb(n, j)) for j in range(1, h + 1)), Fraction(0))
    u = sum((Fraction(1, (j + 1) * math.comb(n, j)) for j in range(1, h + 1)), Fraction(0))
    return [hn**2, hn * hh, hh**2, hn, hh, hn2, hh2, f, g, p, t1, t2, u, Fraction(1, math.comb(n, h)), Fraction(1)]


def _apply(row, state):
    return sum((a * b for a, b in zip(row, state, strict=True)), Fraction(0))


@pytest.mark.parametrize("epsilon", [1, 2])
@pytest.mark.parametrize("h", [1, 2, 3, 7, 19, 47])
def test_all_fifteen_transitions_match_independent_defining_sums(h, epsilon):
    current = _state(h, epsilon)
    assert [_apply(row, current) for row in family._transition_rows(Fraction(h), epsilon)] == _state(h + 1, epsilon)
    if h == 1:
        assert current == family._initial_state(epsilon)


@pytest.mark.parametrize("epsilon", [1, 2])
@pytest.mark.parametrize("h", [2, 7, 16])
def test_arbitrary_polynomial_moments_match_direct_kernel_sum(h, epsilon):
    n = 2 * h + epsilon
    harmonic = _tables(n)[0]
    state = _state(h, epsilon)
    # All monomials beyond the formerly proved quadratic weight are tested.
    for degree in range(9):
        polynomial = [Fraction(0)] * degree + [Fraction(1)]
        row = family._weighted_half_row(polynomial, Fraction(h), epsilon)
        expected = sum((j**degree * _central_correction(n, j, harmonic) for j in range(1, h + 1)), Fraction(0))
        assert _apply(row, state) == expected, (degree, h, epsilon)


@pytest.mark.parametrize("face_size", range(2, 9))
def test_family_matches_original_scalar_reference_in_both_parities(face_size):
    values = family.family_coefficients(face_size, face_size + 4)
    for degree, value in values.items():
        assert value == _scalar_risk_sum(face_size + 2, degree, face_size)
        assert isinstance(value, Fraction)


@pytest.mark.parametrize(("face_size", "degree"), [(2, 2), (3, 3), (4, 4)])
def test_first_activity_matches_independent_gram_propagation(face_size, degree):
    assert family.family_coefficient(face_size, degree) == _exact_coefficients(face_size + 2, degree)[0][face_size - 1]


@pytest.mark.parametrize("face_size", [2, 3, 4, 5, 6, 7, 8, 11, 12])
def test_large_and_fresh_face_sizes_match_original_angular_formula(face_size):
    values = family.family_coefficients(face_size, 79)
    for degree in [face_size, face_size + 1, 23, 41, 67, 79]:
        assert values[degree] == codimension_two_coefficient(face_size, degree)


def test_frozen_high_degree_holdouts_use_new_larger_source_faces():
    artifact = ROOT / "tools/risk_research/examples/codimension_two_family_holdouts.json"
    document = json.loads(artifact.read_bytes())
    assert [case["vertices"] for case in document["cases"]] == [17, 18, 19, 20]
    for case in document["cases"]:
        value = case["exact"]
        assert family.family_coefficient(case["face_size"], case["degree"]) == Fraction(
            int(value["numerator"]), int(value["denominator"])
        )


@pytest.mark.parametrize("face_size", [2, 3, 4, 5, 6, 11])
def test_inactive_prefix_is_explicit_and_degree_keys_are_complete(face_size):
    assert family.family_coefficients(face_size, 1) == {1: Fraction(0)}
    values = family.family_coefficients(face_size, face_size + 2)
    assert list(values) == list(range(1, face_size + 3))
    assert all(values[d] == 0 for d in range(1, face_size))
    assert all(values[d] > 0 for d in range(face_size, face_size + 3))


@pytest.mark.parametrize("invalid", [True, False, -1, 0, 1, 2.0, "2", None])
def test_face_size_is_not_coerced(invalid):
    with pytest.raises(ValueError, match="face_size"):
        family.family_coefficients(invalid, 2)


@pytest.mark.parametrize("invalid", [True, False, -1, 0, 1.0, "2", None])
def test_degree_is_not_coerced(invalid):
    with pytest.raises(ValueError, match="degree"):
        family.family_coefficient(4, invalid)


def test_standard_library_cli_roundtrips_exact_values_and_provenance(tmp_path):
    output = tmp_path / "family.json"
    source = """
import sys
sys.modules['sage'] = None
sys.modules['sympy'] = None
from tools.risk_research.codimension_two_family import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", source, "--face-size", "5", "--max-degree", "7", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_bytes())
    assert record == json.loads(result.stdout)
    assert record["state_dimension"] == 15
    for degree, value in record["values"].items():
        assert Fraction(int(value["numerator"]), int(value["denominator"])) == codimension_two_coefficient(5, int(degree))
    import hashlib

    assert record["source_sha256"] == hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest()


def test_invalid_cli_preserves_existing_artifact(tmp_path):
    output = tmp_path / "family.json"
    output.write_bytes(b"previous valid output")
    with pytest.raises(SystemExit) as error:
        family.main(["--face-size", "1", "--max-degree", "7", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous valid output"
