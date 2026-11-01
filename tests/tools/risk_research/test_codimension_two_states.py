"""Independent exact oracles and certificates for the quadratic angular case."""

import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import codimension_two_states as states
from tools.risk_research.edge_closed import _central_correction, _tables, codimension_two_coefficient

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "tools/risk_research/examples/six_vertex_tetrahedral_face_recurrence_order8.json"


def _state_from_defining_sums(m):
    n = 2 * m + 1
    h = sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))
    h2 = sum((Fraction(1, j * j) for j in range(1, n + 1)), Fraction(0))
    hm = sum((Fraction(1, j) for j in range(1, m + 1)), Fraction(0))
    hm2 = sum((Fraction(1, j * j) for j in range(1, m + 1)), Fraction(0))
    f = sum((Fraction(1, j * j * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    g = sum((Fraction(1, j * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    t = sum((Fraction(1, j * j * math.comb(n, j)) for j in range(1, m + 1)), Fraction(0))
    return [h * h - h2, h, hm, f - h2, g, hm2 - t + (f - h2) / 2, Fraction(1, math.comb(n, m)), Fraction(1)]


def _polynomial(coefficients, degree):
    value = 0
    for coefficient in reversed(coefficients):
        value = value * degree + coefficient
    return value


@pytest.mark.parametrize("m", [*range(1, 13), 31, 64, 127])
def test_all_eight_state_transitions_match_independent_defining_sums(m):
    current = _state_from_defining_sums(m)
    updated = [
        sum((a * b for a, b in zip(row, current, strict=True)), Fraction(0)) for row in states._transition_rows(Fraction(m))
    ]
    assert updated == _state_from_defining_sums(m + 1)


@pytest.mark.parametrize("m", [3, 4, 5, 9, 16, 31, 64])
def test_three_half_range_moments_and_boundary_rows_match_defining_kernel(m):
    n = 2 * m + 1
    c, h, hm, u, g, jstate, w, _ = _state_from_defining_sums(m)
    harmonic = _tables(n)[0]
    kernel = {j: _central_correction(n, j, harmonic) for j in range(1, m + 1)}
    s = c / (n * (n + 1) * (n + 2)) - (n * n + n + 2) * u / (n * n * (n + 1) * (n + 2))
    s -= 2 * (n - 2) * (h - g) / (n**3 * (n + 2))
    x = (c + 2 * u + 4 * (h - g) / n) / (2 * (n + 2))
    e = h / (2 * (m + 1) ** 2) - hm / (m + 1) ** 2 + jstate / n - w / (2 * (m + 1) ** 3)
    assert s == sum(kernel.values())
    assert x == sum(j * (n - j) * value for j, value in kernel.items())
    assert e == sum((Fraction(n, 2) - j) * value for j, value in kernel.items())
    for offset in range(3):
        row = states._boundary_row(Fraction(m), offset)
        assert sum(a * b for a, b in zip(row, _state_from_defining_sums(m), strict=True)) == kernel[m - offset]


@pytest.mark.parametrize("degree", range(1, 17))
def test_tetrahedral_face_state_matches_original_scalar_oracle(degree):
    assert states.tetrahedral_face_coefficient(degree) == _scalar_risk_sum(6, degree, 4)


def test_tetrahedral_face_state_matches_independent_gram_propagation():
    assert states.tetrahedral_face_coefficient(4) == _exact_coefficients(6, 4)[0][3]


def test_held_out_large_degrees_match_independent_angular_mode_sum():
    values = states.tetrahedral_face_coefficients(300)
    for degree in [31, 64, 121, 241, 257, 300]:
        assert values[degree] == codimension_two_coefficient(4, degree)


def test_initial_values_include_three_inactive_degrees():
    assert states.tetrahedral_face_coefficients(6) == {
        1: Fraction(0),
        2: Fraction(0),
        3: Fraction(0),
        4: Fraction(375, 13),
        5: Fraction(6900, 91),
        6: Fraction(7375, 56),
    }


def test_new_symmetric_weight_reciprocal_gosper_certificate():
    n, j = sp.symbols("N j")
    delta = j * (j + 1) * (n - j) * (n - j + 1)
    p = -(n + 1) / (n + 2) - 2 * j / (n * (n + 2))
    p += 6 * j**2 / (n * (n + 2)) - 4 * j**3 / (n * (n + 1) * (n + 2))
    r = p / (j**2 * (n - j + 1))
    residual = (j + 1) * r.subs(j, j + 1) / (n - j) - r
    residual += 8 / (n * (n + 1) * (n + 2)) - (1 / j**2 + 1 / (n - j) ** 2) / (n + 2) - 1 / delta
    assert sp.factor(residual) == 0


def test_quadratic_weight_and_rational_partial_fractions_are_identities():
    n, j = sp.symbols("N j")
    x = j * (n - j)
    delta = j * (j + 1) * (n - j) * (n - j + 1)
    c = (n + 1) * (x - 1) / delta
    p = n / 2 - j
    symmetric = (1 / (j + 1) ** 2 + 1 / (n - j + 1) ** 2) / (n + 2)
    symmetric += (3 * n + 4) * (1 / (j + 1) + 1 / (n - j + 1)) / ((n + 1) * (n + 2) ** 2)
    symmetric -= (1 / j + 1 / (n - j)) / (n * (n + 1))
    antisymmetric = (n + 3) * (1 / j - 1 / (n - j) - 1 / (j + 1) + 1 / (n - j + 1)) / (2 * (n + 1) ** 2)
    antisymmetric -= (1 / j**2 - 1 / (n - j) ** 2) / (2 * n * (n + 1))
    antisymmetric -= (1 / (j + 1) ** 2 - 1 / (n - j + 1) ** 2) / (2 * (n + 1))
    assert sp.factor(x * c / delta - symmetric) == 0
    assert sp.factor(p * c / delta - antisymmetric) == 0
    degree = (n - 5) / 2
    assert sp.factor((degree - 2 - j) * (degree - 1 - j) / 2 - ((n * n + 63) / 8 - x / 2 - 4 * p)) == 0


def test_odd_half_inverse_square_increment_follows_from_binomial_ratio():
    m, j, g, w = sp.symbols("m j G w")
    n = 2 * m + 1
    denominator = (n + 1) * (n + 2)
    ratio = (n + 2 - j) * (n + 1 - j) / denominator
    assert sp.factor((ratio - 1) / j**2 - (1 - (2 * n + 3) / j) / denominator) == 0
    s0 = (n + 1) * g / 4 - sp.Rational(1, 2)
    s1 = g / 2 - w / (2 * (m + 1))
    next_w = (m + 2) * w / (2 * (2 * m + 3))
    increment = (s0 - (2 * n + 3) * s1) / denominator + next_w / (m + 1) ** 2
    expected = -(3 * n + 5) * g / (4 * denominator) - 1 / (2 * denominator) + 3 * w / (4 * (m + 1) ** 2)
    assert sp.factor(increment - expected) == 0


def test_operator_has_exact_sequence_residuals_including_fresh_indices():
    document, coefficients = states._read_operator(OPERATOR.read_bytes())
    indices = [*range(1, 49), 121, 241, 300]
    values = states.tetrahedral_face_coefficients(max(indices) + document["order"])
    for degree in indices:
        assert sum(_polynomial(row, degree) * values[degree + shift] for shift, row in enumerate(coefficients)) == 0
    assert document["initial_values"] == {
        str(degree): str(value) for degree, value in states.tetrahedral_face_coefficients(8).items()
    }
    assert document["order"] == 8
    assert document["polynomial_degree"] == 83
    assert document["exceptional_indices"] == []


@pytest.mark.parametrize("function", [states.tetrahedral_face_coefficient, states.tetrahedral_face_coefficients])
@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "4"])
def test_positive_integer_inputs_are_not_coerced(function, invalid):
    with pytest.raises(ValueError, match="positive integer"):
        function(invalid)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "wrong"),
        ("order", True),
        ("order", 9),
        ("domain_start", 1.0),
        ("state_basis", []),
        ("exceptional_indices", [True]),
    ],
)
def test_operator_rejects_inconsistent_problem_metadata(key, value):
    document = json.loads(OPERATOR.read_bytes())
    document[key] = value
    with pytest.raises(ValueError):
        states._read_operator(json.dumps(document).encode())


def test_invalid_operator_is_rejected_before_sage_is_imported(tmp_path, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("Sage imported before validating the operator")

    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(states, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="tetrahedral-face"):
        states.verify_operator(path)


@pytest.fixture
def sage_python():
    executable = os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        pytest.skip("set BSF_SAGE_PYTHON for native Sage proof regressions")
    return executable


def _native_sage(executable, operator, body):
    source = "import json, sys\nfrom pathlib import Path\nfrom tools.risk_research import codimension_two_states as states\n"
    source += "operator=Path(sys.argv[1])\n" + body
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).resolve().parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [executable, "-c", source, str(operator)], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_native_sage_proves_operator_and_rederives_without_fitting(sage_python):
    _native_sage(
        sage_python,
        OPERATOR,
        """
proof=states.verify_operator(operator)
assert proof['symbolic_state_residuals']==['0']*8
assert proof['state_denominators_positive_on_domain']
assert proof['exceptional_indices']==[]
proof['state_basis'].append('caller annotation')
recorded=json.loads(operator.read_bytes())
derived=states.derive_operator()
for key in ('order','polynomial_degree','coefficients','domain_start',
            'exceptional_indices','initial_values','state_basis'):
    assert derived[key]==recorded[key], key
assert states.STATE==recorded['state_basis']
""",
    )


@pytest.mark.parametrize("mutation", ["coefficient", "initial"])
def test_native_sage_rejects_false_certificate_data(sage_python, tmp_path, mutation):
    document = json.loads(OPERATOR.read_bytes())
    if mutation == "coefficient":
        document["coefficients"][0][0] = str(int(document["coefficients"][0][0]) + 1)
        expected = "state identity"
    else:
        document["initial_values"]["4"] = "0"
        expected = "initial values"
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    _native_sage(
        sage_python,
        path,
        f"""
try:
    states.verify_operator(operator)
except ValueError as error:
    assert {expected!r} in str(error), str(error)
else:
    raise AssertionError('false certificate accepted')
""",
    )
