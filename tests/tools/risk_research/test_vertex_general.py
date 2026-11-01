"""Independent kernel/oracle checks and exact tails for the fixed-M theorem."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import vertex_general
from tools.risk_research.research_exact import integer_kernel
from tools.risk_research.vertex_five import five_vertex_coefficient
from tools.risk_research.vertex_single import tetrahedron_vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("power", range(1, 6))
def test_elementary_symmetric_rows_match_independent_logarithmic_series(power):
    t = sp.Symbol("t")
    expression = (-(1 - t) * sp.log(1 - t) / t) ** power / (1 - t)
    expected = sp.series(expression, t, 0, 9).removeO().expand()
    row = vertex_general._phi_rows(power, 9)[power]
    for j, value in enumerate(row):
        assert sp.Rational(value.numerator, value.denominator) == expected.coeff(t, j)


@pytest.mark.parametrize("power", range(1, 6))
def test_univariate_convolutions_match_original_bivariate_kernel(power):
    kernel = integer_kernel(power, 6, backend="python")
    diagonals = vertex_general.transformed_diagonals(power, 7)
    for j, diagonal in enumerate(diagonals):
        original = sum(
            (
                (-1) ** (h + ell) * comb(j, h) * comb(j, ell) * kernel.value(h, ell)
                for h in range(j + 1)
                for ell in range(j + 1)
            ),
            Fraction(0),
        )
        assert diagonal == original
        assert diagonal > 0
    assert diagonals[0] == 1


@pytest.mark.parametrize("vertices", range(2, 7))
@pytest.mark.parametrize("degree", [1, 2, 4, 7])
def test_general_vertex_formula_matches_original_scalar_oracle(vertices, degree):
    assert vertex_general.vertex_coefficient(vertices, degree) == _scalar_risk_sum(vertices, degree, 1)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_six_vertex_formula_matches_independent_gram_propagation(degree):
    assert vertex_general.vertex_coefficient(6, degree) == _exact_coefficients(6, degree)[0][0]


@pytest.mark.parametrize("degree", [15, 30])
def test_existing_harmonic_evaluations_are_recovered(degree):
    assert vertex_general.vertex_coefficient(4, degree) == tetrahedron_vertex_coefficient(degree)
    assert vertex_general.vertex_coefficient(5, degree) == five_vertex_coefficient(degree)


def test_six_vertex_formula_matches_frozen_fresh_integer_kernel_holdouts():
    document = json.loads((ROOT / "tools/risk_research/examples/six_vertex_vertex_holdouts.json").read_bytes())
    assert [case["degree"] for case in document["cases"]] == [67, 79, 97]
    assert document["vertices"] == 6
    for case in document["cases"]:
        assert vertex_general.vertex_coefficient(6, case["degree"]) == Fraction(
            int(case["numerator"]), int(case["denominator"])
        )


def test_empty_diagonal_request_and_known_cross_majorant():
    assert vertex_general.transformed_diagonals(4, 0) == []
    assert vertex_general._cross_majorant(2) == 2
    assert vertex_general._cross_majorant(3) == 24
    for terms in [1, 2, 31, 128]:
        rows = vertex_general._phi_rows(2, terms)
        assert vertex_general._cross_diagonal(rows, 2) == [Fraction(2, (j + 1) ** 2) for j in range(terms)]
        partial = 288 * sum((Fraction(1, (j + 1) ** 2) for j in range(terms)), Fraction(0))
        assert vertex_general.leading_constant_enclosure(4, terms) == (
            max(Fraction(0), partial - Fraction(288, terms)),
            partial + Fraction(288, terms),
        )


@pytest.mark.parametrize("vertices", range(4, 9))
def test_general_constant_enclosures_have_exact_positive_endpoints(vertices):
    lower, upper = vertex_general.leading_constant_enclosure(vertices, 128)
    assert isinstance(lower, Fraction) and isinstance(upper, Fraction)
    assert 0 < lower < upper


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (vertex_general.transformed_diagonals, (True, 1)),
        (vertex_general.transformed_diagonals, (0, 1)),
        (vertex_general.transformed_diagonals, (2, -1)),
        (vertex_general.transformed_diagonals, (2, False)),
        (vertex_general.vertex_coefficient, (1, 3)),
        (vertex_general.vertex_coefficient, (4, 0)),
        (vertex_general.vertex_coefficient, (4.0, 3)),
        (vertex_general.leading_constant_enclosure, (3, 128)),
        (vertex_general.leading_constant_enclosure, (4, 0)),
        (vertex_general.leading_constant_enclosure, (4, True)),
        (vertex_general.leading_constant_enclosure, (False, 128)),
    ],
)
def test_integer_domains_are_strict(function, arguments):
    with pytest.raises(ValueError, match="integer >="):
        function(*arguments)


@pytest.mark.parametrize("action", ["--degree", "--constant-terms"])
def test_cli_roundtrip_is_exact_source_bound_and_needs_no_optional_backend(tmp_path, action):
    output = tmp_path / "general.json"
    source = """
import sys
sys.modules['sage'] = None
sys.modules['flint'] = None
from tools.risk_research.vertex_general import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", source, "--vertices", "6", action, "8", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    record = json.loads(output.read_bytes())
    assert record == json.loads(result.stdout)
    assert record["source_sha256"] == hashlib.sha256(Path(vertex_general.__file__).read_bytes()).hexdigest()
    if action == "--degree":
        fraction = record["coefficient"]
        assert Fraction(int(fraction["numerator"]), int(fraction["denominator"])) == _scalar_risk_sum(6, 8, 1)
    else:
        lower = record["constant_lower"]
        upper = record["constant_upper"]
        expected = vertex_general.leading_constant_enclosure(6, 8)
        assert (
            Fraction(int(lower["numerator"]), int(lower["denominator"])),
            Fraction(int(upper["numerator"]), int(upper["denominator"])),
        ) == expected
        assert record["degree_exponent"] == 7


def test_invalid_cli_preserves_previous_output(tmp_path):
    output = tmp_path / "general.json"
    output.write_text("previous output", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        vertex_general.main(["--vertices", "3", "--constant-terms", "8", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "previous output"


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for native known-zeta interval checks")
        executable = sys.executable
    return executable


@pytest.mark.parametrize("vertices", [4, 5])
def test_constant_intervals_contain_independently_known_zeta_constants(flint_python, vertices):
    source = """
import sys
from flint import arb, ctx, fmpq
from tools.risk_research.vertex_general import leading_constant_enclosure
m = int(sys.argv[1])
with ctx.workprec(192):
    known = 48*arb.pi()**2 if m == 4 else 7200*(arb.pi()**2/6-arb(3).zeta())
    for terms in (16, 128, 512):
        lower, upper = leading_constant_enclosure(m, terms)
        assert known > arb(fmpq(lower.numerator, lower.denominator))
        assert known < arb(fmpq(upper.numerator, upper.denominator))
"""
    result = subprocess.run([flint_python, "-c", source, str(vertices)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
