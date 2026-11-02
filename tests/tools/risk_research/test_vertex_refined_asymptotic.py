"""Independent moment, convolution, interval and serialization checks."""

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
from tools.risk_research import vertex_general_asymptotic as previous
from tools.risk_research import vertex_refined_asymptotic as refined
from tools.risk_research.vertex_five import five_vertex_coefficient
from tools.risk_research.vertex_general import _phi_rows, leading_constant_enclosure, transformed_diagonals, vertex_coefficient

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(("power", "decay"), [(0, 2), (1, 2), (2, 3), (3, 3), (5, 2), (6, 7)])
def test_logarithmic_integral_matches_independent_symbolic_integral(power, decay):
    t = sp.Symbol("t", nonnegative=True)
    integral = sp.integrate((1 + t) ** power * sp.exp(-(decay - 1) * t), (t, 0, sp.oo))
    expected = refined.logarithmic_integral(power, decay)
    assert integral == sp.Rational(expected.numerator, expected.denominator)


@pytest.mark.parametrize("power", range(2, 8))
def test_first_absolute_moment_budgets_cover_exact_series(power):
    rows = _phi_rows(power, 64)
    _, coefficient, cross, _ = previous._majorants(power)
    b = [rows[power][j] - (rows[power][j - 1] if j else 0) for j in range(64)]
    moment = sum((j * abs(value) for j, value in enumerate(b)), Fraction(0))
    assert moment < coefficient[power] * (1 + refined.logarithmic_integral(power - 1, power))
    if power >= 3:
        e = [sum((comb(power, a) * rows[a][j] * rows[power - a][j] for a in range(1, power)), Fraction(0)) for j in range(64)]
        moment = sum(((j + 1) * abs(value) for j, value in enumerate(e)), Fraction(0))
        assert moment < cross[power] * (1 + refined.logarithmic_integral(power - 2, power - 1))


def test_logarithm_absorption_has_the_claimed_global_maximum():
    t = sp.Symbol("t", nonnegative=True)
    derivative = sp.diff((1 + t) ** 2 * sp.exp(-t), t)
    assert sp.simplify(derivative - (1 - t) * (1 + t) * sp.exp(-t)) == 0
    assert 4 / sp.E < 2
    for q in range(4, 11):
        expression = (1 + t) ** (q - 2) * sp.exp(-(q - 3) * t)
        expected = (1 + t) ** 2 * sp.exp(-t) * ((1 + t) * sp.exp(-t)) ** (q - 4)
        assert sp.simplify(expression - expected) == 0


@pytest.mark.parametrize(("k", "r"), [(1, 3), (1, 4), (1, 6), (2, 2), (2, 3), (3, 2), (3, 4), (5, 2)])
def test_single_log_convolution_budgets_cover_both_split_parities(k, r):
    rows = _phi_rows(max(k, r), 40)
    _, coefficient, cross, difference = previous._majorants(max(k, r))
    b = [rows[k][j] - (rows[k][j - 1] if j else 0) for j in range(40)]
    e = [sum((comb(r, a) * rows[a][j] * rows[r - a][j] for a in range(1, r)), Fraction(0)) for j in range(40)]
    if k == 1:
        moment = cross[r] * (1 + refined.logarithmic_integral(r - 2, r - 1))
        constant = 2 ** (r + 1) * difference[r] + (2 + 3 ** (r - 1)) * cross[r] + 18 * moment
        q, vertices = r + 1, r + 2
        scale = Fraction(vertices**2 * factorial(q), 2 ** (q - 2)) * q
        lower, upper = (value / scale for value in leading_constant_enclosure(vertices, 128))
    else:
        moment = coefficient[k] * (1 + refined.logarithmic_integral(k - 1, k))
        constant = 2 ** (r + 1) * difference[r] * moment + 5 * 2**k * coefficient[k] * cross[r]
        lower = upper = Fraction(0)
    for n in (2, 3, 8, 17, 39):
        value = sum((b[index] * e[n - index] for index in range(n + 1)), Fraction(0))
        radius = constant * (1 + log(n + 1)) / (n + 1) ** 3
        assert abs(value + lower / (n + 1) ** 2) < radius
        assert abs(value + upper / (n + 1) ** 2) < radius


@pytest.mark.parametrize("vertices", [5, 6, 7, 9])
def test_diagonal_bound_contains_certified_limiting_interval_at_small_indices(vertices):
    q = vertices - 1
    constants = refined.error_constants(vertices)
    scale = Fraction(vertices**2 * factorial(q), 2 ** (q - 2))
    lower, upper = (value / scale for value in leading_constant_enclosure(vertices, 128))
    assert 0 < lower < upper <= constants.limiting
    for n, value in enumerate(transformed_diagonals(q, 48)):
        radius = constants.diagonal * (1 + log(n + 1)) / (n + 1) ** 3
        assert abs(value - lower / (n + 1) ** 2) < radius
        assert abs(value - upper / (n + 1) ** 2) < radius


@pytest.mark.parametrize("vertices", [5, 6, 8])
@pytest.mark.parametrize("degree", [1, 2, 7])
def test_exact_fallback_matches_original_scalar_and_previous_evaluator(vertices, degree):
    expected = _scalar_risk_sum(vertices, degree, 1)
    assert refined.rational_enclosure(vertices, degree) == (expected, expected)
    assert refined.rational_enclosure(vertices, degree) == previous.rational_enclosure(vertices, degree)


def test_exact_fallback_matches_independent_gram():
    expected = _exact_coefficients(6, 3)[0][0]
    assert refined.rational_enclosure(6, 3) == (expected, expected)


@pytest.mark.parametrize(("vertices", "degree"), [(5, 255), (5, 256), (6, 256)])
def test_exact_boundary_and_first_asymptotic_degree_contain_exact_reference(vertices, degree):
    exact = five_vertex_coefficient(degree) if vertices == 5 else vertex_coefficient(vertices, degree)
    lower, upper = refined.rational_enclosure(vertices, degree)
    assert lower <= exact <= upper
    if degree < 256:
        assert lower == upper
    else:
        assert lower < upper


@pytest.mark.parametrize("vertices", [5, 6, 8, 12])
def test_relative_threshold_covers_all_logarithm_jumps_and_improves_previous(vertices):
    target = Fraction(1, 100)
    threshold = refined.relative_error_threshold(vertices, target)
    exponent = threshold.bit_length() - 1
    constants = refined.error_constants(vertices)
    lower, _ = leading_constant_enclosure(vertices)
    assert exponent >= 8 and threshold == 1 << exponent
    assert constants.remainder * (exponent + 2) <= target * lower * threshold
    assert threshold < previous.relative_error_threshold(vertices, target)
    for degree in (threshold, threshold + 1, 2 * threshold - 1, 2 * threshold, 3 * threshold, 10 * threshold):
        assert refined.remainder_bound(vertices, degree) * degree ** (vertices + 1) <= target * lower


def test_six_vertex_constants_and_documented_threshold_are_exact():
    constants = refined.error_constants(6)
    assert constants.diagonal == 107393368
    assert constants.limiting == 5280
    assert constants.remainder == 1392034027680
    assert constants.logarithm_power == 1
    assert refined.relative_error_threshold(6) == 2**38
    assert previous.relative_error_threshold(6) == 2**56


@pytest.mark.parametrize(("vertices", "degree"), [(5, 256), (6, 100000), (8, 1000000)])
def test_intersection_is_never_wider_than_previous_guarantee(vertices, degree):
    old_lower, old_upper = previous.rational_enclosure(vertices, degree)
    lower, upper = refined.rational_enclosure(vertices, degree)
    assert old_lower <= lower <= upper <= old_upper
    assert upper - lower < old_upper - old_lower


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (refined.logarithmic_integral, (-1, 2)),
        (refined.logarithmic_integral, (1, 1)),
        (refined.logarithmic_integral, (True, 3)),
        (refined.error_constants, (4,)),
        (refined.error_constants, (6.0,)),
        (refined.error_constants, (True,)),
        (refined.remainder_bound, (6, 255)),
        (refined.remainder_bound, (6, False)),
        (refined.rational_enclosure, (6, 0)),
        (refined.relative_error_threshold, (6, 0)),
        (refined.relative_error_threshold, (6, Fraction(0))),
    ],
)
def test_domains_are_strict(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize("terms", [True, 0, 2.0])
def test_exact_fallback_rejects_invalid_constant_terms(terms):
    with pytest.raises(ValueError):
        refined.rational_enclosure(6, 3, terms=terms)


def test_zero_constant_lower_endpoint_cannot_certify_a_relative_threshold():
    with pytest.raises(ValueError, match="increase terms"):
        refined.relative_error_threshold(6, terms=1)


@pytest.mark.parametrize("bits", [True, 0, 31, 32.0])
def test_invalid_precision_is_rejected_before_optional_import(bits, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("optional import for invalid precision")

    monkeypatch.setattr(refined, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="bits"):
        refined.coefficient_ball(6, 256, bits=bits)


def _read_fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def test_cli_requires_only_standard_library_and_separates_analytic_budgets(tmp_path):
    output = tmp_path / "refined.json"
    source = """
import sys
sys.modules['flint']=None
sys.modules['sympy']=None
from tools.risk_research.vertex_refined_asymptotic import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
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
    assert record["method"] == "asymptotic-intersection" and record["logarithm_power"] == 1
    assert record["relative_error_threshold"] == str(2**38)
    refined_radius = _read_fraction(record["refined_analytic_error_bound"])
    previous_radius = _read_fraction(record["previous_analytic_error_bound"])
    assert refined_radius < previous_radius
    assert _read_fraction(record["analytic_error_bound"]) == min(refined_radius, previous_radius)
    assert record["source_hash_policy"] == "sha256-crlf-to-lf-v1"
    for name, digest in record["source_sha256"].items():
        content = (ROOT / "tools/risk_research" / name).read_bytes().replace(b"\r\n", b"\n")
        assert digest == hashlib.sha256(content).hexdigest()


def test_cli_exact_fallback_has_no_inapplicable_analytic_budget(tmp_path, capsys):
    output = tmp_path / "exact.json"
    assert refined.main(["--vertices", "6", "--degree", "3", "--output", str(output)]) == 0
    record = json.loads(output.read_bytes())
    assert record == json.loads(capsys.readouterr().out)
    assert record["method"] == "exact"
    assert "analytic_error_bound" not in record
    assert _read_fraction(record["coefficient_lower"]) == _read_fraction(record["coefficient_upper"])


def test_hash_policy_is_portable_but_raw_source_guard_rejects_newline_only_edit(tmp_path, monkeypatch):
    lf = {"source.py": b"first\nsecond\n"}
    crlf = {"source.py": b"first\r\nsecond\r\n"}
    assert refined._source_hashes(lf) == refined._source_hashes(crlf)
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous output\n")
    sources = iter([lf, crlf])
    monkeypatch.setattr(refined, "_sources", lambda: next(sources))
    with pytest.raises(SystemExit) as error:
        refined.main(["--vertices", "6", "--degree", "256", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous output\n"


def _source_checkout(directory, newline):
    package = directory / "tools/risk_research"
    package.mkdir(parents=True)
    for name, content in refined._sources().items():
        (package / name).write_bytes(content.replace(b"\r\n", b"\n").replace(b"\n", newline))
    return directory


def test_real_crlf_source_files_produce_the_same_exact_report_and_hashes(tmp_path):
    records = []
    for name, newline in [("lf", b"\n"), ("crlf", b"\r\n")]:
        checkout = _source_checkout(tmp_path / name, newline)
        output = checkout / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "tools.risk_research.vertex_refined_asymptotic",
                "--vertices",
                "6",
                "--degree",
                "100000",
                "--output",
                str(output),
            ],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        records.append(json.loads(output.read_bytes()))
    assert records[0] == records[1]


def test_real_newline_only_edit_during_calculation_preserves_existing_report(tmp_path):
    checkout = _source_checkout(tmp_path / "checkout", b"\n")
    output = checkout / "report.json"
    output.write_bytes(b"previous report\n")
    source = """
import sys
from pathlib import Path
from tools.risk_research import vertex_refined_asymptotic as a
original=a.leading_constant_enclosure
changed=False
def replace_newlines(*args,**kwargs):
    global changed
    if not changed:
        path=Path(a.__file__)
        path.write_bytes(path.read_bytes().replace(b'\\n',b'\\r\\n'))
        changed=True
    return original(*args,**kwargs)
a.leading_constant_enclosure=replace_newlines
raise SystemExit(a.main(['--vertices','6','--degree','256','--output',sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", source, str(output)], cwd=checkout, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 2 and "source changed" in result.stderr
    assert output.read_bytes() == b"previous report\n"


@pytest.mark.parametrize(
    "option",
    [["--vertices", "4"], ["--degree", "0"], ["--constant-terms", "0"], ["--relative-target", "0"], ["--bits", "31"]],
)
def test_invalid_cli_preserves_existing_output(tmp_path, option):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous output\n")
    arguments = ["--vertices", "6", "--degree", "256", "--output", str(output)] + option
    with pytest.raises(SystemExit) as error:
        refined.main(arguments)
    assert error.value.code == 2
    assert output.read_bytes() == b"previous output\n"


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for native real-ball checks")
        executable = sys.executable
    return executable


@pytest.mark.parametrize("bits", [32, 256])
def test_native_saved_ball_contains_exact_coefficient_and_endpoints(flint_python, tmp_path, bits):
    output = tmp_path / "ball.json"
    source = """
import io,json,sys
from contextlib import redirect_stdout
from pathlib import Path
from flint import arb,ctx,fmpq
from tools.risk_research import vertex_refined_asymptotic as a
from tools.risk_research.vertex_general import vertex_coefficient
output,bits=Path(sys.argv[1]),int(sys.argv[2])
value=vertex_coefficient(6,256)
exact=fmpq(value.numerator,value.denominator)
with ctx.workprec(79):
    ball=a.coefficient_ball(6,256,bits=bits)
    assert ctx.prec==79 and ball.contains(exact)
    stream=io.StringIO()
    with redirect_stdout(stream):
        assert a.main(['--vertices','6','--degree','256','--bits',str(bits),'--with-arb','--output',str(output)])==0
    assert ctx.prec==79
    record=json.loads(output.read_bytes())
    assert record==json.loads(stream.getvalue())
    with ctx.workprec(bits+64):
        saved=arb(record['coefficient_ball'])
        assert saved.is_finite() and saved.contains(exact)
        for key in ('coefficient_lower','coefficient_upper'):
            value=record[key]
            assert saved.contains(fmpq(value['numerator']+'/'+value['denominator']))
    assert ctx.prec==79
"""
    result = subprocess.run(
        [flint_python, "-c", source, str(output), str(bits)], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
