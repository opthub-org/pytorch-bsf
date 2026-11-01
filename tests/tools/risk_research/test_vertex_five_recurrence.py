"""Check the five-vertex operator against exact sums and native certificates."""

import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import vertex_five_recurrence as recurrence
from tools.risk_research.vertex_five import five_vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "tools/risk_research/examples/five_vertex_vertex_recurrence_order17.json"


@pytest.mark.parametrize("degree", [1, 3, 17])
def test_five_vertex_normalization_ratios_include_shift_zero(degree):
    normalization = math.factorial(2 * degree + 4) // (math.factorial(degree - 1) * math.factorial(degree + 1))
    for shift in range(18):
        shifted = math.factorial(2 * (degree + shift) + 4) // (
            math.factorial(degree + shift - 1) * math.factorial(degree + shift + 1)
        )
        ratio = recurrence._normalization_ratio(Fraction(degree), shift)
        assert isinstance(ratio, Fraction)
        assert ratio == Fraction(shifted, normalization)


def test_exact_recurrence_matches_harmonic_sum_and_fresh_heldouts():
    values = recurrence.recurrence_coefficients(307)
    for degree in [*range(1, 49), 121, 241, 263, 277, 293, 307]:
        assert values[degree] == five_vertex_coefficient(degree)
    document, _ = recurrence._read_operator(OPERATOR.read_bytes())
    assert document["initial_values"] == {str(i): str(five_vertex_coefficient(i)) for i in range(1, 18)}
    assert values[1] == Fraction(5, 3)
    assert recurrence.recurrence_coefficients(1) == {1: Fraction(5, 3)}


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_recurrence_agrees_with_original_scalar_and_gram_oracles(degree):
    value = recurrence.recurrence_coefficients(degree)[degree]
    assert value == _scalar_risk_sum(5, degree, 1)
    assert value == _exact_coefficients(5, degree)[0][0]


def test_leading_polynomial_is_strictly_positive_on_the_whole_domain():
    document, integers = recurrence._read_operator(OPERATOR.read_bytes())
    assert all(coefficient > 0 for coefficient in integers[-1])
    assert document["leading_domain_certificate"] == recurrence._leading_domain_certificate(integers[-1])
    assert document["order"] == 17
    assert document["polynomial_degree"] == 188
    assert document["remainder_dimensions"] == [2, 3, 2, 2, 2, 3, 3, 4]
    with pytest.raises(ValueError, match="positive polynomial coefficients"):
        recurrence._leading_domain_certificate([1, 0, 1])


@pytest.mark.parametrize("maximum", [True, False, 0, -1, 1.0, "10"])
def test_sequence_evaluator_rejects_invalid_degrees(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        recurrence.recurrence_coefficients(maximum)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "wrong"),
        ("order", True),
        ("order", 18),
        ("domain_start", 1.0),
        ("exceptional_indices", [1]),
        ("harmonic_basis", []),
        ("remainder_dimensions", []),
        ("required_initial_indices", [1]),
        ("coefficients", None),
    ],
)
def test_operator_rejects_invalid_problem_metadata(field, value):
    document = json.loads(OPERATOR.read_bytes())
    document[field] = value
    with pytest.raises(ValueError):
        recurrence._read_operator(json.dumps(document).encode())


def test_invalid_operator_is_rejected_before_sage_import(tmp_path, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("Sage imported before artifact validation")

    monkeypatch.setattr(recurrence, "import_module", forbidden_import)
    source = tmp_path / "wrong.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="five-vertex"):
        recurrence.verify_operator(source)


def test_recorded_evaluator_needs_no_sage_flint_or_unix_resource():
    source = """
import sys
sys.modules['sage'] = None
sys.modules['flint'] = None
sys.modules['resource'] = None
from tools.risk_research.vertex_five_recurrence import recurrence_coefficients
assert recurrence_coefficients(32)[32] > 0
"""
    subprocess.run([sys.executable, "-c", source], cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("arguments", [[], ["--verify", "in.json", "--output", "out.json"]])
def test_cli_requires_one_action_before_optional_import(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "tools.risk_research.vertex_five_recurrence", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.fixture
def sage_python():
    executable = os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        pytest.skip("set BSF_SAGE_PYTHON for native Sage proof regressions")
    return executable


def _native_sage(executable, operator, body):
    source = (
        """
import json, sys
from pathlib import Path
from tools.risk_research import vertex_five_recurrence as recurrence
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


def test_native_sage_verifies_all_eight_identities_and_denominators(sage_python):
    result = _native_sage(
        sage_python,
        OPERATOR,
        """
proof = recurrence.verify_operator(operator)
proof['harmonic_basis'].append('caller annotation')
assert len(recurrence.STATE) == 8
proof['harmonic_basis'].pop()
print(json.dumps(proof))
""",
    )
    proof = json.loads(result.stdout)
    assert proof["status"] == "certified_by_rational_telescoping"
    assert proof["symbolic_harmonic_residuals"] == ["0"] * 8
    assert proof["certificate_polynomial_degrees"] == [32, 34, 33, 33, 34, 35, 36, 35]
    assert proof["positive_parameter_denominator_checks"] == 280
    assert proof["certificate_denominators_nonzero_on_domain"] is True
    assert proof["exceptional_indices"] == []
    assert proof["lower_boundary"] == "j^2 at j=0"
    assert "j=N+1" in proof["upper_boundary"]


@pytest.mark.parametrize("mutation", ["coefficient", "initial", "domain"])
def test_native_sage_rejects_false_identity_initials_and_domain(sage_python, tmp_path, mutation):
    document = json.loads(OPERATOR.read_bytes())
    if mutation == "coefficient":
        document["coefficients"][0][0] = "1"
        message = "telescoping identities"
    elif mutation == "initial":
        document["initial_values"]["3"] = "0"
        message = "initial values"
    else:
        document["leading_domain_certificate"]["coefficient_count"] -= 1
        message = "leading-domain"
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(document), encoding="utf-8")
    _native_sage(
        sage_python,
        modified,
        f"""
try:
    recurrence.verify_operator(operator)
except ValueError as error:
    assert {message!r} in str(error), str(error)
else:
    raise AssertionError('false artifact was accepted')
""",
    )


def test_native_sage_rederives_the_operator_without_fitting(sage_python):
    _native_sage(
        sage_python,
        OPERATOR,
        """
recorded = json.loads(operator.read_bytes())
derived = recurrence.derive_operator()
for key in ('order', 'polynomial_degree', 'coefficients', 'domain_start',
            'exceptional_indices', 'initial_values', 'harmonic_basis',
            'remainder_dimensions', 'leading_domain_certificate'):
    assert derived[key] == recorded[key], key
derived['harmonic_basis'].append('caller annotation')
assert recurrence.STATE == recorded['harmonic_basis']
""",
    )


def test_native_cli_generates_then_verifies_source_bound_reports(sage_python, tmp_path):
    generated = tmp_path / "operator.json"
    report = tmp_path / "proof.json"
    _native_sage(
        sage_python,
        OPERATOR,
        f"""
import hashlib, io
from contextlib import redirect_stdout
generated, report = Path({str(generated)!r}), Path({str(report)!r})
for action in ('--output', '--verify'):
    stream = io.StringIO()
    with redirect_stdout(stream):
        recurrence.main([action, str(generated), '--report', str(report)])
    proof = json.loads(stream.getvalue())
    assert proof == json.loads(report.read_bytes())
    assert proof['operator_sha256'] == hashlib.sha256(generated.read_bytes()).hexdigest()
    assert proof['source_sha256'] == hashlib.sha256(Path(recurrence.__file__).read_bytes()).hexdigest()
    assert json.loads(generated.read_bytes())['coefficients'] == json.loads(operator.read_bytes())['coefficients']
""",
    )
