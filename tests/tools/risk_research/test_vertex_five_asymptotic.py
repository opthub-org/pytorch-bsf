"""Independent inequalities and native five-vertex enclosure round trips."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from tools.risk_research import vertex_five_asymptotic as asymptotic

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("degree", [256, 257, 511, 512, 100000, 10**9])
def test_rational_error_majorant_uses_an_exact_logarithm_upper_bound(degree):
    exponent = (degree - 1).bit_length()
    assert 2 ** (exponent - 1) < degree <= 2**exponent
    assert asymptotic.five_vertex_remainder_bound(degree) == Fraction(3_000_000 * (1 + exponent), degree**7)


@pytest.mark.parametrize("degree", [True, False, -1, 0, 1, 255, 256.0, "256", None])
def test_bound_rejects_degrees_outside_the_proved_integer_domain(degree):
    with pytest.raises(ValueError, match="integer >= 256"):
        asymptotic.five_vertex_remainder_bound(degree)


@pytest.mark.parametrize("bits", [True, False, 0, 31, 32.0, "128"])
def test_invalid_precision_is_rejected_before_importing_flint(bits, monkeypatch):
    def forbidden_import(name):
        raise AssertionError("optional Flint must not be imported for invalid precision")

    monkeypatch.setattr(asymptotic, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="bits"):
        asymptotic.five_vertex_leading_ball(256, bits=bits)


def test_rational_factor_inequalities_hold_for_all_real_n_at_least_one():
    n, t = sp.symbols("n t")
    q = (n + 1) ** 2 * (n + 2) ** 2 * (n + 3)
    p2 = 5 * n**3 + 7 * n**2 - 6 * n - 12
    p4 = 2 * n**4 + 15 * n**3 + 44 * n**2 + 47 * n + 24
    p5 = 2 * n**3 + 3 * n**2 - 2 * n - 12
    second_denominator = (n + 1) * (n + 2) ** 2 * (n + 3)
    margins = [
        5 * q - n**2 * p2,
        5 * q + n**2 * p2,
        2 * q - n * p4,
        2 * (n + 1) * (n + 2) * (n + 3) - n**2 * (2 * n + 3),
        2 * second_denominator - n * p5,
        2 * second_denominator + n * p5,
        q - 144 * n,
    ]
    for margin in margins:
        coefficients = sp.Poly(sp.expand(margin.subs(n, t + 1)), t).all_coeffs()
        assert all(coefficient >= 0 for coefficient in coefficients)
        assert any(coefficient > 0 for coefficient in coefficients)


def test_moments_and_rational_taylor_bounds_follow_independent_identities():
    d, t = sp.symbols("D t", positive=True)
    mean = (d - 1) * (d + 1) / (2 * d)
    second_factorial = (d - 1) * (d - 2) * (d + 1) * d / (2 * d * (2 * d - 1))
    centered1 = sp.factor(mean - d / 2)
    centered2 = sp.factor(second_factorial + mean - d * mean + d**2 / 4)
    assert centered1 == -1 / (2 * d)
    assert sp.factor(centered2 - (d**3 - 2 * d + 2) / (4 * d * (2 * d - 1))) == 0
    numerator, denominator = sp.fraction(sp.factor(d / 4 - centered2))
    for polynomial in (numerator, denominator):
        coefficients = sp.Poly(sp.expand(polynomial.subs(d, t + 12)), t).all_coeffs()
        assert all(coefficient >= 0 for coefficient in coefficients)
        assert coefficients[-1] > 0

    r = 1 / ((t + 1) * (t + 3))
    margins = [2 / t**3 + sp.diff(r, t), 6 / t**4 - sp.diff(r, t, 2), 4 / t**3 - (1 / t**2 - r)]
    for margin in margins:
        numerator, denominator = sp.fraction(sp.factor(margin))
        assert all(coefficient >= 0 for coefficient in sp.Poly(numerator, t).all_coeffs())
        assert all(coefficient >= 0 for coefficient in sp.Poly(denominator, t).all_coeffs())


def test_explicit_error_budget_includes_pointwise_tail_and_normalization_errors():
    assert 48 + 96 + 48 == 192
    assert 120 + 48 + 60 + 1 == 229
    assert 192 + 229 + 192 < 625
    assert 32 + 1 + 192 == 225
    assert 625 * 64 == 40000
    assert 48 * 225 == 10800
    assert Fraction(576, 2048) < 1
    expectation_constant = 40000 + 10800 + 1 + 641 * 32
    assert expectation_constant == 71313
    assert Fraction(75, 2) * expectation_constant + 36000 < asymptotic.REMAINDER_CONSTANT


def test_bound_only_cli_needs_no_flint_and_records_its_actual_source(tmp_path):
    output = tmp_path / "bound.json"
    source = """
import sys
sys.modules['flint'] = None
from tools.risk_research.vertex_five_asymptotic import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", source, "--degree", "256", "--bound-only", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_bytes())
    assert record == json.loads(result.stdout)
    fraction = record["error_upper_bound"]
    assert Fraction(int(fraction["numerator"]), int(fraction["denominator"])) == asymptotic.five_vertex_remainder_bound(256)
    assert record["domain_start"] == 256
    assert record["source_sha256"] == hashlib.sha256(Path(asymptotic.__file__).read_bytes()).hexdigest()
    assert "coefficient_enclosure" not in record


def test_invalid_cli_preserves_an_existing_report(tmp_path):
    output = tmp_path / "previous.json"
    output.write_bytes(b"previous report\n")
    with pytest.raises(SystemExit) as error:
        asymptotic.main(["--degree", "255", "--bound-only", "--output", str(output)])
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


@pytest.mark.parametrize("degree", [256, 1024, 4096])
def test_native_leading_ball_contains_independent_exact_single_sum(flint_python, degree):
    source = """
import sys
from flint import arb, ctx, fmpq
from tools.risk_research.vertex_five import five_vertex_coefficient
from tools.risk_research.vertex_five_asymptotic import five_vertex_leading_ball
d = int(sys.argv[1])
with ctx.workprec(79):
    leading, enclosure = five_vertex_leading_ball(d, bits=128)
    assert ctx.prec == 79
    exact = five_vertex_coefficient(d)
    assert enclosure.contains(fmpq(exact.numerator, exact.denominator))
    with ctx.workprec(192):
        independent = 7200 * (arb(2).zeta() - arb(3).zeta()) / arb(d)**6
        assert leading.overlaps(independent)
    assert ctx.prec == 79
"""
    result = subprocess.run([flint_python, "-c", source, str(degree)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [32, 256])
def test_native_cli_serialized_enclosure_contains_exact_coefficient(flint_python, tmp_path, bits):
    output = tmp_path / "enclosure.json"
    source = """
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from flint import arb, ctx, fmpq
from tools.risk_research import vertex_five_asymptotic as asymptotic
from tools.risk_research.vertex_five import five_vertex_coefficient
output = Path(sys.argv[1])
bits = int(sys.argv[2])
with ctx.workprec(79):
    stream = io.StringIO()
    with redirect_stdout(stream):
        assert asymptotic.main(['--degree', '4096', '--bits', str(bits), '--output', str(output)]) == 0
    assert ctx.prec == 79
    record = json.loads(output.read_bytes())
    assert record == json.loads(stream.getvalue())
    assert record['bits'] == bits
    assert record['python_flint'] == version('python-flint')
    assert record['source_sha256'] == hashlib.sha256(Path(asymptotic.__file__).read_bytes()).hexdigest()
    exact = five_vertex_coefficient(4096)
    with ctx.workprec(bits + 64):
        saved = arb(record['coefficient_enclosure'])
        assert saved.is_finite()
        assert saved.contains(fmpq(exact.numerator, exact.denominator))
    assert ctx.prec == 79
    _, positive = asymptotic.five_vertex_leading_ball(100000, bits=bits)
    assert positive > 0
    assert ctx.prec == 79
"""
    result = subprocess.run(
        [flint_python, "-c", source, str(output), str(bits)], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
