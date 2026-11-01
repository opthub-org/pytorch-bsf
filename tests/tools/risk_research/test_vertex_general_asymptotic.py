"""Independent series, convolution, oracle and serialized-ball checks."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import comb, factorial, log
from pathlib import Path

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import vertex_general_asymptotic as asymptotic
from tools.risk_research.vertex_five import five_vertex_coefficient
from tools.risk_research.vertex_general import _phi_rows, leading_constant_enclosure, transformed_diagonals, vertex_coefficient
from tools.risk_research.vertex_single import tetrahedron_vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("power", range(1, 6))
def test_binomial_product_b_coefficients_match_independent_logarithmic_series(power):
    t = sp.Symbol("t")
    series = sp.series((-(1 - t) * sp.log(1 - t) / t) ** power, t, 0, 9).removeO().expand()
    rows = _phi_rows(power, 9)[power]
    for j in range(1, 9):
        reciprocals = [sp.Rational(1, k) for k in range(1, power + 1)] + [-sp.Rational(1, k) for k in range(1, j)]
        polynomial = sp.prod(1 + value * t for value in reciprocals).expand()
        magnitude = sp.Rational(factorial(power) ** 2 * factorial(j - 1), factorial(j + power))
        assert abs(series.coeff(t, j)) == abs(magnitude * polynomial.coeff(t, power - 1))
        assert series.coeff(t, j) == rows[j] - rows[j - 1]


@pytest.mark.parametrize("power", range(3, 9))
def test_pointwise_majorants_bound_exact_series_and_cross_differences(power):
    rows = _phi_rows(power, 32)
    phi, coefficient, cross, difference = asymptotic._majorants(power)
    for r in range(1, power + 1):
        for j in range(31):
            n, logarithm = j + 1, 1 + log(j + 1)
            value = rows[r][j]
            increment = value - (rows[r][j - 1] if j else 0)
            if r == 1:
                assert value == Fraction(1, n)
            else:
                assert abs(value) <= phi[r] * logarithm ** (r - 1) / n**r
            assert abs(increment) <= coefficient[r] * logarithm ** (r - 1) / n ** (r + 1)
    for r in range(2, power + 1):
        values = [sum((comb(r, a) * rows[a][j] * rows[r - a][j] for a in range(1, r)), Fraction(0)) for j in range(32)]
        for j in range(31):
            n, logarithm = j + 1, 1 + log(j + 1)
            if r == 2:
                assert values[j] == Fraction(2, n**2)
            else:
                assert abs(values[j]) <= cross[r] * logarithm ** (r - 2) / n**r
            assert abs(values[j + 1] - values[j]) <= difference[r] * logarithm ** (r - 2) / n ** (r + 1)


def test_logarithmic_tail_majorant_has_a_positive_exact_derivative_margin():
    x = sp.Symbol("x", positive=True)
    for p in range(7):
        for alpha in (p + 2, p + 3):
            expression = -sp.diff((1 + sp.log(x)) ** p / x ** (alpha - 1), x)
            expression -= (alpha - 1 - p) * (1 + sp.log(x)) ** p / x**alpha
            expected = 0 if p == 0 else p * sp.log(x) * (1 + sp.log(x)) ** (p - 1) / x**alpha
            assert sp.simplify(expression - expected) == 0


def test_reciprocal_convolution_and_normalization_inequalities_hold_on_their_full_domains():
    m, t, d = sp.symbols("m t D", nonnegative=True)
    n = 2 * m + t + 1
    total = n + 1
    margin = 18 * (m + 1) / total**3 - (1 / ((n - m) * (n - m + 1)) - 1 / total**2)
    numerator, denominator = sp.fraction(sp.factor(margin))
    for polynomial in (numerator, denominator):
        assert all(value >= 0 for value in sp.Poly(polynomial, m, t).coeffs())
    for q in range(3, 9):
        normalization = sp.prod(2 * d / (2 * d + h) for h in range(1, q + 1))
        numerator, denominator = sp.fraction(sp.factor(sp.Rational(q * (q + 1), 4) / d - (1 - normalization)))
        for polynomial in (numerator, denominator):
            assert all(value >= 0 for value in sp.Poly(polynomial, d).all_coeffs())


@pytest.mark.parametrize("vertices", range(4, 9))
def test_diagonal_error_budget_contains_the_certified_constant_interval(vertices):
    q = vertices - 1
    constants = asymptotic.error_constants(vertices)
    scale = Fraction(vertices**2 * factorial(q), 2 ** (q - 2))
    lower, upper = leading_constant_enclosure(vertices, 128)
    diagonals = transformed_diagonals(q, 48)
    for j, value in enumerate(diagonals):
        radius = constants.diagonal * (1 + log(j + 1)) ** (q - 1) / (j + 1) ** 3
        assert abs(value - lower / scale / (j + 1) ** 2) <= radius
        assert abs(value - upper / scale / (j + 1) ** 2) <= radius
    assert 0 < lower / scale <= upper / scale <= constants.limiting


@pytest.mark.parametrize(("k", "r"), [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2)])
def test_quantitative_convolution_bounds_cover_both_parities_and_endpoints(k, r):
    rows = _phi_rows(max(k, r), 32)
    _, coefficient, cross, difference = asymptotic._majorants(max(k, r))
    b = [rows[k][j] - (rows[k][j - 1] if j else 0) for j in range(32)]
    e = [sum((comb(r, a) * rows[a][j] * rows[r - a][j] for a in range(1, r)), Fraction(0)) for j in range(32)]
    if k == 1:
        q, vertices = r + 1, r + 2
        scale = Fraction(vertices**2 * factorial(q), 2 ** (q - 2)) * q
        lower, upper = (value / scale for value in leading_constant_enclosure(vertices, 128))
        constant = 2 ** (r + 1) * difference[r] + (20 + 3 ** (r - 1)) * cross[r]
    else:
        lower = upper = Fraction(0)
        constant = coefficient[k] * (2 ** (r + 2) * difference[r] + 5 * 2**k * cross[r])
    for n in (2, 3, 7, 16, 31):
        value = sum((b[index] * e[n - index] for index in range(n + 1)), Fraction(0))
        exponent = r - 1 if k == 1 else k + r - 3
        radius = constant * (1 + log(n + 1)) ** exponent / (n + 1) ** 3
        assert abs(value + lower / (n + 1) ** 2) <= radius
        assert abs(value + upper / (n + 1) ** 2) <= radius


@pytest.mark.parametrize("vertices", range(4, 9))
@pytest.mark.parametrize("degree", [1, 3, 7])
def test_below_domain_enclosures_are_exact_and_match_original_scalar(vertices, degree):
    expected = _scalar_risk_sum(vertices, degree, 1)
    assert asymptotic.rational_enclosure(vertices, degree) == (expected, expected)


def test_six_vertex_low_degree_fallback_matches_independent_gram():
    expected = _exact_coefficients(6, 2)[0][0]
    assert asymptotic.rational_enclosure(6, 2) == (expected, expected)


def test_exact_fallback_at_last_domain_degree_matches_independent_harmonic_sum():
    exact = tetrahedron_vertex_coefficient(255)
    assert asymptotic.rational_enclosure(4, 255) == (exact, exact)


def test_exact_fallback_matches_frozen_independent_six_vertex_kernel_holdout():
    document = json.loads((ROOT / "tools/risk_research/examples/six_vertex_vertex_holdouts.json").read_bytes())
    case = document["cases"][-1]
    exact = Fraction(int(case["numerator"]), int(case["denominator"]))
    assert asymptotic.rational_enclosure(6, case["degree"]) == (exact, exact)


@pytest.fixture(scope="module")
def exact_six_vertex_boundary():
    return vertex_coefficient(6, 256)


@pytest.mark.parametrize("vertices", [4, 5, 6])
def test_first_asymptotic_degree_contains_exact_harmonic_or_convolution_reference(vertices, exact_six_vertex_boundary):
    expected = {4: tetrahedron_vertex_coefficient, 5: five_vertex_coefficient}.get(vertices)
    exact = exact_six_vertex_boundary if expected is None else expected(256)
    lower, upper = asymptotic.rational_enclosure(vertices, 256)
    assert lower <= exact <= upper
    assert lower < upper


@pytest.mark.parametrize("vertices", [4, 5, 6, 8, 12])
def test_relative_threshold_certifies_integer_logarithm_jumps_and_all_larger_degrees(vertices):
    target = Fraction(1, 100)
    threshold = asymptotic.relative_error_threshold(vertices, target)
    exponent = threshold.bit_length() - 1
    assert threshold == 2**exponent and exponent >= max(8, 2 * (vertices - 2))
    constants = asymptotic.error_constants(vertices)
    lower, _ = leading_constant_enclosure(vertices)
    assert constants.remainder * (exponent + 2) ** (vertices - 2) <= target * lower * threshold
    for degree in (threshold, threshold + 1, 2 * threshold - 1, 3 * threshold, 10 * threshold):
        assert asymptotic.remainder_bound(vertices, degree) * degree ** (vertices + 1) <= target * lower


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (asymptotic.error_constants, (3,)),
        (asymptotic.error_constants, (4.0,)),
        (asymptotic.error_constants, (True,)),
        (asymptotic.remainder_bound, (6, 255)),
        (asymptotic.remainder_bound, (6, True)),
        (asymptotic.rational_enclosure, (6, 0)),
        (asymptotic.rational_enclosure, (False, 1)),
        (asymptotic.relative_error_threshold, (6, 0)),
        (asymptotic.relative_error_threshold, (6, Fraction(0))),
    ],
)
def test_domains_reject_nonintegers_and_invalid_targets(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


def test_zero_constant_lower_bound_requires_more_terms_for_a_relative_threshold():
    with pytest.raises(ValueError, match="increase terms"):
        asymptotic.relative_error_threshold(6, terms=1)


@pytest.mark.parametrize("bits", [True, 0, 31, 32.0])
def test_invalid_precision_is_rejected_before_optional_import(bits, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("optional backend imported for invalid precision")

    monkeypatch.setattr(asymptotic, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="bits"):
        asymptotic.coefficient_ball(6, 256, bits=bits)


def test_cli_needs_no_optional_backend_and_records_exact_source_bound_intervals(tmp_path):
    output = tmp_path / "rational.json"
    source = """
import sys
sys.modules['flint'] = None
from tools.risk_research.vertex_general_asymptotic import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            "--vertices",
            "6",
            "--degree",
            "100000",
            "--relative-target",
            "1/100",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_bytes())
    assert record == json.loads(result.stdout)
    assert record["method"] == "asymptotic" and record["domain_start"] == 256
    assert record["relative_error_threshold"] == str(asymptotic.relative_error_threshold(6))
    assert "coefficient_ball" not in record
    for name, digest in record["source_sha256"].items():
        assert digest == hashlib.sha256((Path(asymptotic.__file__).parent / name).read_bytes()).hexdigest()
    for name in ("constant_lower", "constant_upper", "analytic_error_bound", "coefficient_lower", "coefficient_upper"):
        fraction = record[name]
        exact = Fraction(int(fraction["numerator"]), int(fraction["denominator"]))
        assert str(exact.numerator) == fraction["numerator"] and str(exact.denominator) == fraction["denominator"]


def test_cli_exact_fallback_records_no_inapplicable_analytic_error(tmp_path, capsys):
    output = tmp_path / "exact.json"
    assert asymptotic.main(["--vertices", "6", "--degree", "7", "--output", str(output)]) == 0
    record = json.loads(output.read_bytes())
    assert record == json.loads(capsys.readouterr().out)
    assert record["method"] == "exact" and "analytic_error_bound" not in record
    expected = _scalar_risk_sum(6, 7, 1)
    for key in ("coefficient_lower", "coefficient_upper"):
        value = record[key]
        assert Fraction(int(value["numerator"]), int(value["denominator"])) == expected


@pytest.mark.parametrize(
    "option",
    [["--vertices", "3"], ["--degree", "0"], ["--bits", "31"], ["--constant-terms", "0"], ["--relative-target", "1/0"]],
)
def test_invalid_cli_preserves_existing_report(tmp_path, option):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous report\n")
    arguments = ["--vertices", "6", "--degree", "256", "--output", str(output)] + option
    with pytest.raises(SystemExit) as error:
        asymptotic.main(arguments)
    assert error.value.code == 2
    assert output.read_bytes() == b"previous report\n"


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for native real-ball enclosure regressions")
        executable = sys.executable
    return executable


@pytest.mark.parametrize("bits", [32, 256])
def test_native_serialized_ball_contains_exact_coefficient_and_interval_endpoints(
    flint_python, tmp_path, bits, exact_six_vertex_boundary
):
    output = tmp_path / "ball.json"
    source = """
import io,json,sys
from contextlib import redirect_stdout
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import vertex_general_asymptotic as a
output,bits=Path(sys.argv[1]),int(sys.argv[2])
exact=fmpq(sys.argv[3])
with ctx.workprec(79):
    interval=a.coefficient_ball(6,256,bits=bits)
    assert ctx.prec==79
    assert interval.contains(exact)
    stream=io.StringIO()
    with redirect_stdout(stream):
        assert a.main(['--vertices','6','--degree','256','--bits',str(bits),'--with-arb','--output',str(output)])==0
    assert ctx.prec==79
    record=json.loads(output.read_bytes())
    assert record==json.loads(stream.getvalue()) and record['bits']==bits
    with ctx.workprec(bits+64):
        saved=arb(record['coefficient_ball'])
        assert saved.is_finite() and saved.contains(exact)
        for key in ('coefficient_lower','coefficient_upper'):
            value=record[key]
            assert saved.contains(fmpq(value['numerator']+'/'+value['denominator']))
    assert ctx.prec==79
"""
    result = subprocess.run(
        [flint_python, "-c", source, str(output), str(bits), str(exact_six_vertex_boundary)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
