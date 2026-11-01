"""Exercise exact recurrence values and independent native symbolic certificates."""

import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tools.risk_research import vertex_degree_recurrence as recurrence
from tools.risk_research.vertex_single import tetrahedron_vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "tools/risk_research/examples/tetrahedron_vertex_recurrence_order10.json"


@pytest.mark.parametrize("degree", [1, 3, 17])
def test_normalization_ratios_are_exact_including_shift_zero(degree):
    normalization = math.factorial(2 * degree + 3) // (math.factorial(degree - 1) * math.factorial(degree + 1))
    for shift in range(11):
        shifted = math.factorial(2 * (degree + shift) + 3) // (
            math.factorial(degree + shift - 1) * math.factorial(degree + shift + 1)
        )
        ratio = recurrence._normalization_ratio(Fraction(degree), shift)
        assert isinstance(ratio, Fraction)
        assert ratio == Fraction(shifted, normalization)


def test_recorded_recurrence_matches_the_proved_sum_including_fresh_heldouts():
    values = recurrence.recurrence_coefficients(300)
    indices = [*range(1, 49), 121, 241, 260, 271, 277, 290, 300]
    for degree in indices:
        assert values[degree] == tetrahedron_vertex_coefficient(degree)
    document, _ = recurrence._read_operator(OPERATOR.read_bytes())
    assert document["initial_values"] == {str(degree): str(tetrahedron_vertex_coefficient(degree)) for degree in range(1, 11)}
    assert values[1] == Fraction(8, 5)
    assert values[2] == Fraction(16, 35)
    assert recurrence.recurrence_coefficients(1) == {1: Fraction(8, 5)}


def test_leading_coefficient_has_one_positive_real_root_between_consecutive_integers():
    document, integers = recurrence._read_operator(OPERATOR.read_bytes())
    signs = [coefficient > 0 for coefficient in integers[-1] if coefficient]
    assert sum(first != second for first, second in zip(signs, signs[1:], strict=False)) == 1
    assert recurrence._evaluate_polynomial(integers[-1], 4) < 0 < recurrence._evaluate_polynomial(integers[-1], 5)
    assert document["leading_domain_certificate"] == recurrence._leading_domain_certificate(integers[-1])
    assert document["exceptional_indices"] == []
    assert document["order"] == 10
    assert document["polynomial_degree"] == 69


@pytest.mark.parametrize("maximum", [True, 0, -1, 1.0, "10"])
def test_sequence_evaluator_rejects_invalid_degree(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        recurrence.recurrence_coefficients(maximum)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "wrong"),
        ("order", True),
        ("order", 11),
        ("domain_start", 1.0),
        ("exceptional_indices", [4]),
        ("harmonic_basis", []),
        ("required_initial_indices", [1]),
        ("coefficients", None),
    ],
)
def test_operator_rejects_invalid_domain_and_problem_metadata(field, value):
    document = json.loads(OPERATOR.read_bytes())
    document[field] = value
    with pytest.raises(ValueError):
        recurrence._read_operator(json.dumps(document).encode())


def test_invalid_operator_is_rejected_before_optional_sage_import(tmp_path, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("Sage must not be imported before validating the artifact")

    monkeypatch.setattr(recurrence, "import_module", forbidden_import)
    source = tmp_path / "wrong.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="vertex degree recurrence"):
        recurrence.verify_operator(source)


def test_module_and_evaluator_work_without_sage_or_unix_resource():
    source = """
import sys
from fractions import Fraction
sys.modules['sage'] = None
sys.modules['resource'] = None
from tools.risk_research.vertex_degree_recurrence import recurrence_coefficients
assert recurrence_coefficients(16)[16] > Fraction(0)
"""
    subprocess.run([sys.executable, "-c", source], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("arguments", [[], ["--verify", "input.json", "--output", "output.json"]])
def test_cli_requires_one_action_before_importing_sage(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "tools.risk_research.vertex_degree_recurrence", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


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
from tools.risk_research import vertex_degree_recurrence as recurrence
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


def test_native_sage_verifies_four_rational_identities_and_boundary_domain(sage_python):
    result = _native_sage_check(
        sage_python,
        OPERATOR,
        """
proof = recurrence.verify_operator(operator)
proof['harmonic_basis'].append('caller annotation')
assert recurrence.STATE == ['H_j^2', 'H_j^(2)', 'H_j', '1']
proof['harmonic_basis'].pop()
print(json.dumps(proof))
""",
    )
    proof = json.loads(result.stdout)
    assert proof["status"] == "certified_by_rational_telescoping"
    assert proof["symbolic_harmonic_residuals"] == ["0"] * 4
    assert proof["certificate_polynomial_degrees"] == [18, 19, 19, 19]
    assert proof["certificate_denominators_nonzero_on_domain"] is True
    assert proof["positive_parameter_denominator_checks"] == 79
    assert proof["exceptional_indices"] == []
    assert proof["lower_boundary"] == "j^2 at j=0"
    assert proof["upper_boundary"] == "binom(N,j)=0 at j=N+1; N=D+order-1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("coefficient", "telescoping identity"), ("initial", "initial values"), ("domain", "leading-domain")],
)
def test_native_sage_rejects_false_coefficients_initial_values_and_domain_proof(sage_python, tmp_path, mutation, message):
    document = json.loads(OPERATOR.read_bytes())
    if mutation == "coefficient":
        document["coefficients"][0][0] = str(int(document["coefficients"][0][0]) + 1)
    elif mutation == "initial":
        document["initial_values"]["3"] = "0"
    else:
        document["leading_domain_certificate"]["positive_real_root_bracket"] = [3, 4]
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(document), encoding="utf-8")
    _native_sage_check(
        sage_python,
        modified,
        f"""
try:
    recurrence.verify_operator(operator)
except ValueError as error:
    assert {message!r} in str(error), str(error)
else:
    raise AssertionError('false operator was accepted')
""",
    )


def test_native_sage_rederives_the_operator_without_sequence_fitting(sage_python):
    _native_sage_check(
        sage_python,
        OPERATOR,
        """
recorded = json.loads(operator.read_bytes())
derived = recurrence.derive_operator()
for key in ('order', 'polynomial_degree', 'coefficients', 'domain_start',
            'exceptional_indices', 'initial_values', 'harmonic_basis',
            'leading_domain_certificate'):
    assert derived[key] == recorded[key], key
derived['harmonic_basis'].append('caller annotation')
assert recurrence.STATE == recorded['harmonic_basis']
""",
    )
