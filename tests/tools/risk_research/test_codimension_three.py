"""Independent polynomial traces, norm moments and exact bound regressions."""

import itertools
import json
import math
import subprocess
import sys
from fractions import Fraction
from functools import cache
from pathlib import Path

import pytest
import sympy as sp

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import codimension_three as bounds
from tools.risk_research import codimension_three_certificate as certificate
from tools.risk_research.edge_closed import tetrahedron_edge_coefficient

ROOT = Path(__file__).resolve().parents[3]
VARIABLES = sp.symbols("x0:5")


@cache
def _mode(degree, angular, subset):
    # Build the propagated polynomial directly, rather than calling any
    # kernel or transfer implementation.  Jacobi(2,2) is the source-edge
    # angular basis for the squared bubble weight x0^2*x1^2.
    x, y = VARIABLES[:2]
    radius = x + y
    common = x * y * sp.cancel(radius**angular * sp.jacobi(angular, 2, 2, (x - y) / radius))
    length = degree - 2 - angular
    radial = 0
    selected = [VARIABLES[index + 2] for index in range(3) if subset & (1 << index)]
    for h in range(length + 1):
        moment = 0
        for powers in itertools.product(range(h + 1), repeat=len(selected)):
            if sum(powers) == h:
                coefficient = Fraction(math.factorial(h), math.prod(math.factorial(a) * (a + 1) for a in powers))
                moment += sp.Rational(coefficient.numerator, coefficient.denominator) * math.prod(
                    variable**a for variable, a in zip(selected, powers, strict=True)
                )
        radial += (-1) ** h * math.comb(length, h) * math.comb(degree + angular + 3, h) * radius ** (length - h) * moment
    return sp.Poly(sp.expand(common * radial), *VARIABLES)


def _inner(left, right, vertices):
    result = Fraction(0)
    for powers, coefficient in (left * right).terms():
        assert all(powers[index] == 0 for index in range(5) if index not in vertices)
        moment = Fraction(
            math.factorial(len(vertices) - 1) * math.prod(math.factorial(powers[index]) for index in vertices),
            math.factorial(len(vertices) - 1 + sum(powers)),
        )
        result += Fraction(int(coefficient.p), int(coefficient.q)) * moment
    return result


@pytest.mark.parametrize(("degree", "angular"), [(2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0)])
def test_correct_lift_matches_all_facets_and_is_orthogonal_to_its_bubble(degree, angular):
    full = _mode(degree, angular, 7)
    lift = sum(
        ((-1) ** (4 - subset.bit_count()) * _mode(degree, angular, subset) for subset in range(7)), sp.Poly(0, *VARIABLES)
    )
    difference = full - lift
    if degree >= 5 and angular == 0:
        assert difference != 0  # Exercise a genuine full bubble, not only degree<M.
    for coordinate in VARIABLES:
        assert difference.as_expr().subs(coordinate, 0) == 0
    assert _inner(full, difference, tuple(range(5))) == 0
    assert _inner(full, full, tuple(range(5))) <= _inner(lift, lift, tuple(range(5)))


@pytest.mark.parametrize(("degree", "angular"), [(2, 0), (3, 1), (4, 0), (4, 2)])
@pytest.mark.parametrize("subset", [0, 1, 2, 3, 5, 6])
def test_embedding_factor_matches_independent_dirichlet_integration(degree, angular, subset):
    mode = _mode(degree, angular, subset)
    face = (0, 1, *(index + 2 for index in range(3) if subset & (1 << index)))
    smaller = len(face)
    expected_ratio = Fraction(
        math.factorial(smaller + 2 * degree - 1) * math.factorial(4),
        math.factorial(smaller - 1) * math.factorial(4 + 2 * degree),
    )
    assert _inner(mode, mode, tuple(range(5))) == expected_ratio * _inner(mode, mode, face)


@pytest.mark.parametrize("degree", [2, 3, 7, 19])
def test_five_vertex_transfer_weights_have_the_proved_allocation_factors(degree):
    d = degree
    assert bounds.transfer_squared_weights(2, 3, d) == (
        Fraction(2400, (2 * d + 2) * (2 * d + 3) * (2 * d + 4)),
        Fraction(1200, (2 * d + 3) * (2 * d + 4)),
        Fraction(100, 2 * d + 4),
    )


def _root_interval(value):
    scale = 1 << 80
    lower = math.isqrt(value.numerator * scale * scale // value.denominator)
    return Fraction(lower, scale), Fraction(lower + 1, scale)


@pytest.mark.parametrize(("size", "outside", "degree"), [(2, 3, 2), (2, 3, 4), (2, 3, 8), (3, 3, 3), (3, 3, 6), (2, 4, 5)])
def test_general_transfer_matches_independent_scalar_coefficients(size, outside, degree):
    actual = _scalar_risk_sum(size + outside, degree, size)
    terms = [
        weight * _scalar_risk_sum(size + selected, degree, size)
        for selected, weight in enumerate(bounds.transfer_squared_weights(size, outside, degree))
    ]
    assert _root_interval(actual)[1] < sum((_root_interval(value)[0] for value in terms), Fraction(0))


def test_five_vertex_bound_covers_scalar_prefix_and_independent_gram_start():
    assert bounds.five_vertex_edge_upper_bound(1) == 0
    for degree in range(2, 13):
        assert 0 < _scalar_risk_sum(5, degree, 2) < bounds.five_vertex_edge_upper_bound(degree)
        assert bounds.five_vertex_edge_upper_bound(degree) <= Fraction(30000, degree**2)
    assert _exact_coefficients(5, 2)[0][1] < bounds.five_vertex_edge_upper_bound(2)


@pytest.mark.parametrize("degree", [2, 3, 5, 13, 31, 67])
def test_new_tetrahedron_inequality_agrees_with_exact_fixed_state_values(degree):
    assert 0 < tetrahedron_edge_coefficient(degree) < Fraction(864, (degree + 4) * (degree + 5))


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 2.0, "2", None])
def test_degree_validation_does_not_coerce_inputs(invalid):
    with pytest.raises(ValueError, match="degree"):
        bounds.five_vertex_edge_upper_bound(invalid)
    with pytest.raises(ValueError, match="degree"):
        bounds.transfer_squared_weights(2, 3, invalid)


@pytest.mark.parametrize(("size", "outside"), [(1, 3), (True, 3), (2, 0), (2, 1.0)])
def test_transfer_domain_requires_a_nonvertex_source_and_outside_coordinates(size, outside):
    with pytest.raises(ValueError):
        bounds.transfer_squared_weights(size, outside, 2)


def test_algebraic_certificate_checks_boundary_telescope_and_domain():
    report = certificate.verify_lemmas()
    assert report["status"] == "certified_algebraic_lemmas"
    assert report["outside_facet_coefficients"] == [{"6": 1}, {"5": 1}, {"3": 1}]
    assert report["harmonic_telescope_residual"] == "0"
    assert report["edge_power_bound_numerator"] == 30000


def test_altered_bound_constant_and_boundary_cancellation_are_rejected(monkeypatch):
    monkeypatch.setattr(bounds, "EDGE_BOUND_NUMERATOR", 29999)
    with pytest.raises(ArithmeticError, match="constant"):
        certificate.verify_lemmas()
    monkeypatch.setattr(bounds, "EDGE_BOUND_NUMERATOR", 30000)
    monkeypatch.setattr(certificate, "_trace_coefficients", lambda outside, facet: {0: 1})
    with pytest.raises(ArithmeticError, match="cancellation"):
        certificate.verify_lemmas()


def test_certificate_cli_roundtrip_rejects_changed_domain_or_source(tmp_path):
    original = tmp_path / "certificate.json"
    output = tmp_path / "rechecked.json"
    assert certificate.main(["--output", str(original)]) == 0
    assert certificate.main(["--verify", str(original), "--output", str(output)]) == 0
    assert output.read_bytes() == original.read_bytes()
    report = json.loads(original.read_bytes())
    for field, value in [("domain", "all integers"), ("bound_source_sha256", "0" * 64)]:
        changed = dict(report)
        changed[field] = value
        original.write_text(json.dumps(changed))
        with pytest.raises(SystemExit) as error:
            certificate.main(["--verify", str(original), "--output", str(output)])
        assert error.value.code == 2
        assert json.loads(output.read_bytes()) == report


def test_bound_cli_works_without_symbolic_dependencies_and_preserves_existing_output(tmp_path):
    output = tmp_path / "bounds.json"
    script = """
import sys
sys.modules['sage'] = None
sys.modules['sympy'] = None
from tools.risk_research.codimension_three import main
raise SystemExit(main(sys.argv[1:]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, "--max-degree", "5", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_bytes())
    assert json.loads(completed.stdout) == report
    assert report["proof_status"] == "proved_analytic_upper_bound"
    for degree, value in report["bounds"].items():
        assert Fraction(int(value["numerator"]), int(value["denominator"])) == bounds.five_vertex_edge_upper_bound(int(degree))
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        bounds.main(["--max-degree", "0", "--output", str(output)])
    assert output.read_bytes() == before
