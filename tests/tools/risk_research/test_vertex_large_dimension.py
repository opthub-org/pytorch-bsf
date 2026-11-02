"""Exact elementary enclosures and independent positive-integral checks."""

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

from tools.risk_research import vertex_large_dimension as large

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("bits", [32, 64, 128])
def test_elementary_series_enclose_independently_known_decimal_brackets(bits):
    logarithm, pi = large.elementary_enclosures(bits)
    brackets = [
        (
            Fraction(6931471805599453094172321214581765680755, 10**40),
            Fraction(6931471805599453094172321214581765680756, 10**40),
        ),
        (
            Fraction(31415926535897932384626433832795028841971, 10**40),
            Fraction(31415926535897932384626433832795028841972, 10**40),
        ),
    ]
    for computed, known in zip((logarithm, pi), brackets, strict=True):
        assert max(computed[0], known[0]) <= min(computed[1], known[1])
        assert 0 < computed[1] - computed[0] <= Fraction(1, 1 << bits)


@pytest.mark.parametrize("value", [Fraction(0), Fraction(2), Fraction(4), Fraction(7, 3)])
@pytest.mark.parametrize("bits", [32, 128])
def test_integer_square_root_rounding_is_outward_even_for_exact_squares(value, bits):
    lower, upper = large._sqrt_enclosure(value, bits)
    assert lower**2 <= value <= upper**2
    assert 0 <= upper - lower <= Fraction(1, 1 << bits)
    if value in (0, 4):
        assert lower == upper


def test_fourth_derivative_identity_and_nonzero_quadratic_maximum():
    x = sp.Symbol("x", real=True)
    h = 4 * sp.cos(x / 2) ** 2 * sp.log(2 * sp.cos(x / 2)) + x * sp.sin(x)
    expected = x * sp.sin(x) + 2 * sp.cos(x) * sp.log(2 * sp.cos(x / 2)) + sp.cos(x) / (1 + sp.cos(x)) - 2
    assert sp.trigsimp(sp.diff(h, x, 4) - expected, method="fu") == 0
    assert h.subs(x, 0) == 4 * sp.log(2)
    assert sp.diff(h, x).subs(x, 0) == 0
    assert sp.diff(h, x, 2).subs(x, 0) == 1 - 2 * sp.log(2)
    assert sp.diff(h, x, 3).subs(x, 0) == 0


def test_positive_kernel_derivative_proves_unique_maximum_without_sampling():
    u, c = sp.symbols("u c", real=True)
    integrand = (1 + u) * (1 - c) / (1 - 2 * u * c + u**2)
    expected = -((1 - u) ** 2) * (1 + u) / (1 - 2 * u * c + u**2) ** 2
    assert sp.factor(sp.diff(integrand, c) - expected) == 0
    assert Fraction(7, 8) + Fraction(7, 15) - 2 > -1
    assert Fraction(1, 4) + 2 * Fraction(7, 10) + Fraction(1, 2) - 2 < 1
    assert Fraction(1, 64) + Fraction(33, 256) ** 2 / (2 * (1 - Fraction(33, 1024))) < Fraction(1, 32)
    assert Fraction(1, 16) - Fraction(1, 256) > Fraction(1, 20)
    assert Fraction(5373, 256) < 21
    assert 18 * Fraction(7, 10) < Fraction(1022, 80)
    assert 2**18 > 4 * 1024 * 32


@pytest.mark.parametrize("vertices", [4, 5, 32, 1025, 1026, 2502, 100000])
def test_normalized_intervals_have_ordered_exact_endpoints(vertices):
    lower, upper = large.normalized_constant_enclosure(vertices)
    leading_lower, leading_upper = large.leading_normalized_enclosure(vertices)
    assert isinstance(lower, Fraction) and isinstance(upper, Fraction)
    assert 0 <= lower < upper
    assert lower <= leading_lower < leading_upper <= upper
    if vertices >= large.MINIMUM_VERTICES:
        assert lower > 0
        assert large.relative_remainder_bound(vertices) == Fraction(25, vertices - 2)


@pytest.mark.parametrize("target", [Fraction(1, 100), Fraction(1, 1000), Fraction(2), Fraction(7, 300)])
def test_relative_threshold_applies_to_every_larger_vertex_count(target):
    threshold = large.relative_error_threshold(target)
    assert threshold >= 1026
    for vertices in (threshold, threshold + 1, 2 * threshold, threshold**2):
        assert large.relative_remainder_bound(vertices) <= target
    if threshold > 1026:
        assert large.relative_remainder_bound(threshold - 1) > target


@pytest.mark.parametrize("vertices", [True, 3, 0, 6.0])
@pytest.mark.parametrize(
    "function", [large.leading_normalized_enclosure, large.relative_remainder_bound, large.normalized_constant_enclosure]
)
def test_invalid_vertex_domains_are_rejected(function, vertices):
    with pytest.raises(ValueError, match="vertices"):
        function(vertices)


@pytest.mark.parametrize("target", [0, True, Fraction(0), Fraction(-1), 0.1])
def test_invalid_relative_targets_are_rejected(target):
    with pytest.raises(ValueError, match="target"):
        large.relative_error_threshold(target)


@pytest.mark.parametrize("bits", [True, 31, 0, 64.0])
@pytest.mark.parametrize("function", [large.constant_ball, large.integral_normalized_ball])
def test_invalid_precision_is_rejected_before_importing_optional_backend(function, bits, monkeypatch):
    def forbidden(name):
        raise AssertionError("optional import occurred before precision validation")

    monkeypatch.setattr(large, "import_module", forbidden)
    with pytest.raises(ValueError, match="bits"):
        function(1026, bits=bits)


def test_cli_is_standard_library_only_and_records_source_bound_exact_intervals(tmp_path):
    output = tmp_path / "constant.json"
    script = """
import sys
sys.modules['flint']=None
from tools.risk_research.vertex_large_dimension import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "--vertices", "2502", "--relative-target", "1/100", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_bytes())
    assert record == json.loads(result.stdout)
    assert record["relative_error_threshold"] == "2502"
    assert record["bits"] == 128
    assert record["simple_remainder_start_vertices"] == 1026
    assert "constant_ball" not in record and "normalized_integral_ball" not in record
    for name, digest in record["source_sha256"].items():
        assert digest == hashlib.sha256((Path(large.__file__).parent / name).read_bytes()).hexdigest()
    for name in (
        "normalized_lower",
        "normalized_upper",
        "leading_normalized_lower",
        "leading_normalized_upper",
        "relative_error_bound",
    ):
        value = record[name]
        exact = Fraction(int(value["numerator"]), int(value["denominator"]))
        assert str(exact.numerator) == value["numerator"] and str(exact.denominator) == value["denominator"]


@pytest.mark.parametrize(
    "option", [["--vertices", "3"], ["--bits", "31"], ["--relative-target", "0"], ["--relative-target", "1/0"]]
)
def test_invalid_cli_preserves_existing_output(tmp_path, option):
    output = tmp_path / "constant.json"
    output.write_bytes(b"previous\n")
    with pytest.raises(SystemExit) as error:
        large.main(["--vertices", "1026", "--output", str(output), *option])
    assert error.value.code == 2
    assert output.read_bytes() == b"previous\n"


def test_concurrent_source_change_preserves_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "constant.json"
    output.write_bytes(b"previous\n")
    snapshots = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr(large, "_sources", lambda: next(snapshots))
    with pytest.raises(SystemExit) as error:
        large.main(["--vertices", "1026", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


@pytest.mark.parametrize("failure", [ModuleNotFoundError("flint"), RuntimeError("integration failed")])
def test_optional_backend_failures_preserve_previous_report(tmp_path, monkeypatch, failure):
    output = tmp_path / "constant.json"
    output.write_bytes(b"previous\n")

    def fail(name):
        raise failure

    monkeypatch.setattr(large, "import_module", fail)
    with pytest.raises(SystemExit) as error:
        large.main(["--vertices", "1026", "--with-arb", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


def test_output_failure_preserves_previous_report(tmp_path, monkeypatch):
    output = tmp_path / "constant.json"
    output.write_bytes(b"previous\n")

    def fail(path, record):
        raise OSError("cannot replace output")

    monkeypatch.setattr(large, "atomic_json", fail)
    with pytest.raises(SystemExit) as error:
        large.main(["--vertices", "1026", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == b"previous\n"


@pytest.fixture
def flint_python():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("optional python-flint is unavailable")
        executable = sys.executable
    return executable


def test_native_integral_matches_rational_series_and_known_constants(flint_python):
    script = """
from fractions import Fraction
from flint import arb,ctx,fmpq
from tools.risk_research.vertex_general import leading_constant_enclosure
from tools.risk_research.vertex_large_dimension import integral_normalized_ball,_rational_ball,constant_ball
with ctx.workprec(128):
    for m in (4,5,6,8,12,16):
        direct=integral_normalized_ball(m,bits=128)
        lower,upper=leading_constant_enclosure(m,128)
        scale=m**2*arb(m).gamma()*(2*arb(2).log())**(m-1)
        series=_rational_ball(lower,upper,__import__('flint'))/scale
        assert direct.overlaps(series),(m,direct,series)
    assert constant_ball(4).contains(48*arb.pi()**2)
    assert constant_ball(5).contains(7200*(arb(2).zeta()-arb(3).zeta()))
print('independent series and known constants agree')
"""
    result = subprocess.run([flint_python, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [32, 128])
def test_native_integral_is_contained_in_large_vertex_theorem_and_precision_restores(flint_python, bits):
    script = f"""
from flint import arb,ctx
from tools.risk_research.vertex_large_dimension import *
from tools.risk_research.vertex_large_dimension import _rational_ball
import flint
ctx.prec=73
for m in (1026,2502,4097):
    direct=integral_normalized_ball(m,bits={bits})
    assert ctx.prec==73
    lower,upper=normalized_constant_enclosure(m,bits={bits})
    with ctx.workprec(128):
        enclosure=_rational_ball(lower,upper,flint)
        assert enclosure.contains(direct),(m,enclosure,direct)
        saved=arb(direct.str(50,more=True))
        assert saved.contains(direct),(saved,direct)
    restored=constant_ball(m,bits={bits})
    assert ctx.prec==73 and restored.is_finite()
print('large vertex theorem independently enclosed; precision restored')
"""
    result = subprocess.run([flint_python, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [32, 128])
def test_saved_optional_balls_retain_independent_integral_after_reload(flint_python, bits, tmp_path):
    output = tmp_path / "constant.json"
    script = f"""
import contextlib,io,json
from flint import arb,ctx
from tools.risk_research.vertex_large_dimension import main,integral_normalized_ball
ctx.prec=73
with contextlib.redirect_stdout(io.StringIO()):
    assert main(['--vertices','1026','--bits','{bits}','--with-arb','--integral-check','--output',{str(output)!r}])==0
assert ctx.prec==73
record=json.loads(open({str(output)!r}).read())
with ctx.workprec(160):
    direct=integral_normalized_ball(1026,bits=128)
    saved=arb(record['normalized_integral_ball'])
    original=integral_normalized_ball(1026,bits={bits})
    assert saved.contains(original)
    assert saved.overlaps(direct)
    scale=1026**2*arb(1026).gamma()*(2*arb(2).log())**1025
    assert arb(record['constant_ball']).contains(scale*direct)
print('saved intervals retain independent constant')
"""
    result = subprocess.run([flint_python, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
