"""Independent moment identities and rigorous two-term vertex enclosures."""

import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from tools.risk_research import vertex_asymptotic
from tools.risk_research.vertex_asymptotic import tetrahedron_vertex_expansion_ball, tetrahedron_vertex_remainder_bound

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("degree", [256, 257, 511, 512, 10000, 10**9])
def test_bound_uses_an_exact_upper_bound_for_the_logarithm(degree):
    exponent = (degree - 1).bit_length()
    assert 2 ** (exponent - 1) < degree <= 2**exponent
    bound = tetrahedron_vertex_remainder_bound(degree)
    assert bound * degree**7 == 10_000_000 * (exponent + 1) ** 2
    assert bound > 0


@pytest.mark.parametrize("degree", [True, False, -1, 0, 1, 255, 256.0, "256"])
def test_explicit_bound_rejects_degrees_outside_the_proved_domain(degree):
    with pytest.raises(ValueError, match="integer >= 256"):
        tetrahedron_vertex_remainder_bound(degree)


@pytest.mark.parametrize("bits", [True, 0, 31, 32.0, "128"])
def test_precision_is_validated_before_optional_flint_import(bits):
    with pytest.raises(ValueError, match="bits"):
        tetrahedron_vertex_expansion_ball(256, bits=bits)


def _moments_from_factorial_moments(d):
    falling = {0: sp.Integer(1)}
    for r in range(1, 5):
        falling[r] = (
            sp.prod(d - 1 - i for i in range(r)) * sp.prod(d + 1 - i for i in range(r)) / sp.prod(2 * d - i for i in range(r))
        )
    raw = {
        power: sum(sp.functions.combinatorial.numbers.stirling(power, r, kind=2) * falling[r] for r in range(power + 1))
        for power in range(5)
    }
    return {
        power: sp.factor(sum(sp.binomial(power, r) * (-d / 2) ** (power - r) * raw[r] for r in range(power + 1)))
        for power in range(1, 5)
    }


def test_documented_moments_follow_exactly_from_vandermonde():
    d = sp.Symbol("D")
    moments = _moments_from_factorial_moments(d)
    assert sp.factor(moments[1] + 1 / (2 * d)) == 0
    assert sp.factor(moments[2] - (d**3 - 2 * d + 2) / (4 * d * (2 * d - 1))) == 0
    assert sp.factor(moments[3] + (3 * d**2 + 2 * d - 4) / (8 * d * (2 * d - 1))) == 0
    assert (
        sp.factor(
            moments[4] - (3 * d**5 - 4 * d**4 - 12 * d**3 + 20 * d**2 + 16 * d - 24) / (16 * d * (2 * d - 3) * (2 * d - 1))
        )
        == 0
    )


def test_moment_inequalities_hold_for_every_real_degree_at_least_twelve():
    d, t = sp.symbols("D t")
    moments = _moments_from_factorial_moments(d)
    margins = [
        d / 4 - moments[2],
        sp.Rational(1, 4) - moments[2] + d / 8,
        sp.Rational(1, 4) + moments[2] - d / 8,
        1 - moments[3],
        1 + moments[3],
        d**2 - moments[4],
    ]
    for margin in margins:
        numerator, denominator = sp.fraction(sp.factor(margin))
        for polynomial in (numerator, denominator):
            coefficients = sp.Poly(sp.expand(polynomial.subs(d, t + 12)), t).all_coeffs()
            assert all(coefficient >= 0 for coefficient in coefficients)
            assert coefficients[-1] > 0


def test_next_term_is_derived_from_the_original_harmonic_weight():
    t, h, z, d, ell = sp.symbols("t h z D L")
    j = 1 / t
    # H_j=log(j)+gamma+O(1/j), H_j^(2)=zeta(2)-1/j+O(1/j^2).
    g = (
        h**2 / ((j + 1) * (j + 2) * (j + 3))
        + (z - 1 / j) / ((j + 1) * (j + 3))
        - j * h / ((j + 1) * (j + 2) ** 2 * (j + 3))
        + 2 / ((j + 1) ** 2 * (j + 2) ** 2 * (j + 3))
    )
    coefficient = sp.expand(sp.series(g, t, 0, 4).removeO()).coeff(t, 3)
    assert coefficient == h**2 - h - 4 * z - 1
    expectation = z * (4 / d**2 + 6 / d**3) + coefficient.subs(h, ell) * 8 / d**3
    normalized = sp.expand(72 / d**3 * (1 - 3 / d) * expectation)
    assert sp.expand(normalized).coeff(d, -5) == 288 * z
    assert sp.expand(normalized).coeff(d, -6) == 576 * (ell**2 - ell - 1) - 2736 * z


def test_explicit_constants_dominate_every_error_contribution():
    assert 23 + 73 + 25 * 2 + 11 < 160
    assert 8 + 12 + 11 + 20480 == 20511
    assert 23 + 17536 == 17559
    assert 40960 + 2 * 20511 + 17559 < 100000
    tail_at_start = 72 * Fraction(256**4 + 80 * 256**2 + 360 * 256, 2**19)
    assert tail_at_start < 1_000_000
    assert vertex_asymptotic.REMAINDER_CONSTANT > 7_200_000 + 25_920 + 1_000_000


def test_bound_only_cli_is_source_bound_and_needs_no_flint(tmp_path):
    output = tmp_path / "bound.json"
    code = """
import sys
sys.modules['flint'] = None
from tools.risk_research.vertex_asymptotic import main
raise SystemExit(main(sys.argv[1:]))
"""
    subprocess.run(
        [sys.executable, "-c", code, "--degree", "256", "--bound-only", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_bytes())
    bound = record["error_upper_bound"]
    assert Fraction(int(bound["numerator"]), int(bound["denominator"])) == tetrahedron_vertex_remainder_bound(256)
    assert record["domain_start"] == 256
    assert "coefficient_enclosure" not in record
    assert len(record["source_sha256"]) == 64


def test_invalid_cli_preserves_existing_output(tmp_path):
    output = tmp_path / "certificate.json"
    output.write_bytes(b"previous certificate\n")
    with pytest.raises(SystemExit) as error:
        vertex_asymptotic.main(["--degree", "255", "--bound-only", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous certificate\n"


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON to a Python with python-flint for native real-ball regressions")
        executable = sys.executable
    return executable


@pytest.mark.parametrize("degree", [256, 1024, 4096])
def test_native_balls_contain_independent_exact_single_sum(flint_python, degree):
    source = """
import sys
from flint import arb, ctx, fmpq
from tools.risk_research.vertex_asymptotic import tetrahedron_vertex_expansion_ball
from tools.risk_research.vertex_single import tetrahedron_vertex_coefficient
d = int(sys.argv[1])
before = ctx.prec
approximation, enclosure = tetrahedron_vertex_expansion_ball(d, bits=128)
assert ctx.prec == before
exact = tetrahedron_vertex_coefficient(d)
assert enclosure.contains(fmpq(exact.numerator, exact.denominator))
with ctx.workprec(192):
    dd = arb(d)
    z = arb.pi()**2/6
    ell = (dd/2).log()+arb.const_euler()
    independent = 288*z/dd**5+72*(8*(ell**2-ell-1)-38*z)/dd**6
    assert approximation.overlaps(independent)
assert ctx.prec == before
"""
    result = subprocess.run([flint_python, "-c", source, str(degree)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
