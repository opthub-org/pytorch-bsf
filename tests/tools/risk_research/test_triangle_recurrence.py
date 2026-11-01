"""Check the triangular-face output row, operator domain and exact residuals."""

import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tools.risk_research import triangle_recurrence
from tools.risk_research.triangle_faces import triangle_face_coefficient, triangle_face_coefficients

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "tools/risk_research/examples/five_vertex_triangle_recurrence_order8.json"


def _state_from_defining_sums(n):
    harmonic = sum((Fraction(1, j) for j in range(1, 2 * n + 1)), Fraction(0))
    harmonic2 = sum((Fraction(1, j * j) for j in range(1, 2 * n + 1)), Fraction(0))
    middle = sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))
    middle2 = sum((Fraction(1, j * j) for j in range(1, n + 1)), Fraction(0))
    auxiliary_f = sum((Fraction(1, j * j * math.comb(2 * n, j)) for j in range(1, 2 * n + 1)), Fraction(0))
    auxiliary_g = sum((Fraction(1, j * math.comb(2 * n, j)) for j in range(1, 2 * n + 1)), Fraction(0))
    half = sum((Fraction(1, j * j * math.comb(2 * n, j)) for j in range(1, n)), Fraction(0))
    auxiliary_j = middle2 - half + (auxiliary_f - harmonic2) / 2
    return [
        harmonic**2 - harmonic2,
        harmonic,
        middle,
        auxiliary_f - harmonic2,
        auxiliary_g,
        auxiliary_j,
        Fraction(1, math.comb(2 * n, n)),
        Fraction(1),
    ]


def _evaluate_polynomial(coefficients, degree):
    value = 0
    for coefficient in reversed(coefficients):
        value = value * degree + coefficient
    return value


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 7, 16, 31, 64])
def test_collected_output_row_matches_independent_exact_representation(degree):
    coefficients = triangle_recurrence._output_coefficients(Fraction(degree + 2))
    state = _state_from_defining_sums(degree + 2)
    assert all(isinstance(value, Fraction) for value in coefficients)
    assert sum((a * b for a, b in zip(coefficients, state, strict=True)), Fraction(0)) == triangle_face_coefficient(degree)


def test_operator_cancels_the_exact_sequence_including_inactive_degrees_and_fresh_indices():
    document, coefficients = triangle_recurrence._read_operator(OPERATOR.read_bytes())
    indices = [*range(1, 49), 121, 241, 300]
    values = triangle_face_coefficients(max(indices) + document["order"])
    for degree in indices:
        assert sum(_evaluate_polynomial(row, degree) * values[degree + shift] for shift, row in enumerate(coefficients)) == 0
    assert document["initial_values"] == {str(degree): str(value) for degree, value in triangle_face_coefficients(8).items()}
    assert document["initial_values"]["1"] == document["initial_values"]["2"] == "0"


def test_leading_polynomial_has_no_zero_anywhere_on_the_positive_degree_domain():
    document, coefficients = triangle_recurrence._read_operator(OPERATOR.read_bytes())
    leading = coefficients[-1]
    # p_8(1+z) has strictly positive coefficients. Consequently p_8(D)>0
    # for every real D>=1, proving there are no exceptional positive integers.
    shifted = [sum(leading[i] * math.comb(i, k) for i in range(k, len(leading))) for k in range(len(leading))]
    assert all(value > 0 for value in shifted)
    assert document["exceptional_indices"] == []
    assert document["domain_start"] == 1
    assert document["order"] == 8
    assert max(len(row) - 1 for row in coefficients) == document["polynomial_degree"] == 76


@pytest.mark.parametrize(("field", "value"), [("order", True), ("order", 0), ("order", 9), ("polynomial_degree", 76.0)])
def test_operator_order_and_degree_are_not_coerced(field, value):
    document = json.loads(OPERATOR.read_bytes())
    document[field] = value
    with pytest.raises(ValueError, match="integer order"):
        triangle_recurrence._read_operator(json.dumps(document).encode())


@pytest.mark.parametrize("start", [True, 0, 2, 1.0, "1"])
def test_operator_domain_is_an_exact_integer_one(start):
    document = json.loads(OPERATOR.read_bytes())
    document["domain_start"] = start
    with pytest.raises(ValueError, match="domain"):
        triangle_recurrence._read_operator(json.dumps(document).encode())


@pytest.mark.parametrize("coefficient", [True, 1, 1.0, "1.0", "+1", "01", " 1", "1 "])
def test_operator_coefficients_are_canonical_decimal_integer_strings(coefficient):
    document = json.loads(OPERATOR.read_bytes())
    document["coefficients"][0][0] = coefficient
    with pytest.raises(ValueError):
        triangle_recurrence._read_operator(json.dumps(document).encode())


@pytest.mark.parametrize("exceptions", [None, {}, [True], [0], [1.0], ["1"]])
def test_operator_exception_metadata_requires_positive_integers(exceptions):
    document = json.loads(OPERATOR.read_bytes())
    document["exceptional_indices"] = exceptions
    with pytest.raises(ValueError, match="exceptional"):
        triangle_recurrence._read_operator(json.dumps(document).encode())


@pytest.mark.parametrize(("field", "value"), [("schema", "wrong"), ("state_basis", []), ("coefficients", None)])
def test_operator_rejects_wrong_problem_or_missing_arrays(field, value):
    document = json.loads(OPERATOR.read_bytes())
    document[field] = value
    with pytest.raises(ValueError):
        triangle_recurrence._read_operator(json.dumps(document).encode())


def test_invalid_operator_is_rejected_before_optional_sage_is_imported(tmp_path, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("Sage must not be imported before validating the operator")

    source = tmp_path / "wrong.json"
    source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(triangle_recurrence, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="triangular-face"):
        triangle_recurrence.verify_operator(source)


def test_recurrence_module_imports_without_sage_or_unix_resource():
    code = """
import sys
sys.modules['sage'] = None
sys.modules['resource'] = None
from tools.risk_research import triangle_recurrence
assert triangle_recurrence.SCHEMA == 'skeleton-risk-triangle-recurrence-v1'
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.fixture
def sage_python():
    executable = os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        pytest.skip("set BSF_SAGE_PYTHON to a native Sage Python executable for symbolic proof regressions")
    return executable


def _native_sage_check(executable, operator, body):
    source = (
        """
import json
import sys
from pathlib import Path
from tools.risk_research import triangle_recurrence
operator = Path(sys.argv[1])
"""
        + body
    )
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).resolve().parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [executable, "-c", source, str(operator)], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_native_sage_verifies_all_eight_symbolic_state_identities(sage_python):
    result = _native_sage_check(
        sage_python,
        OPERATOR,
        "print(json.dumps(triangle_recurrence.verify_operator(operator)))",
    )
    proof = json.loads(result.stdout)
    assert proof["status"] == "certified_by_state_identity"
    assert proof["symbolic_state_residuals"] == ["0"] * 8
    assert proof["state_denominators_positive_on_domain"] is True
    assert proof["exceptional_indices"] == []


@pytest.mark.parametrize(("mutation", "message"), [("coefficient", "state identity"), ("initial", "initial values")])
def test_native_sage_rejects_false_coefficients_and_initial_values(sage_python, tmp_path, mutation, message):
    document = json.loads(OPERATOR.read_bytes())
    if mutation == "coefficient":
        document["coefficients"][0][0] = str(int(document["coefficients"][0][0]) + 1)
    else:
        document["initial_values"]["3"] = "0"
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(document), encoding="utf-8")
    _native_sage_check(
        sage_python,
        modified,
        f"""
try:
    triangle_recurrence.verify_operator(operator)
except ValueError as error:
    assert {message!r} in str(error), str(error)
else:
    raise AssertionError('false operator was accepted')
""",
    )


def test_native_sage_rederives_the_complete_operator_without_fitting(sage_python):
    _native_sage_check(
        sage_python,
        OPERATOR,
        """
recorded = json.loads(operator.read_bytes())
derived = triangle_recurrence.derive_operator()
for key in ('order', 'polynomial_degree', 'coefficients', 'domain_start',
            'exceptional_indices', 'initial_values', 'state_basis'):
    assert derived[key] == recorded[key], key
""",
    )
