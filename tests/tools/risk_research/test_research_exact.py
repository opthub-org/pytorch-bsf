"""Check independent integer kernel multiplication and scalar evaluation."""

import hashlib
import json
from argparse import Namespace
from fractions import Fraction
from math import comb
from types import SimpleNamespace

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import research_exact
from tools.risk_research.research_exact import integer_kernel, scalar_coefficient


def _fraction_kernel(power, h, ell):
    if power == 0:
        return Fraction(int(h == ell == 0))
    return sum(
        (
            Fraction(comb(i + j, i), (i + 1) * (j + 1)) * _fraction_kernel(power - 1, h - i, ell - j)
            for i in range(h + 1)
            for j in range(ell + 1)
        ),
        Fraction(0),
    )


@pytest.mark.parametrize("power", [0, 1, 2, 3])
@pytest.mark.parametrize("backend", ["python", "flint"])
def test_kernel_matches_independent_fraction_convolution(power, backend):
    if backend == "flint":
        pytest.importorskip("flint")
    kernel = integer_kernel(power, 3, backend=backend)
    for h in range(4):
        for ell in range(4):
            assert kernel.value(h, ell) == _fraction_kernel(power, h, ell)


@pytest.mark.parametrize(("vertices", "degree", "skeleton"), [(2, 5, 0), (3, 5, 0), (3, 5, 1), (4, 5, 0), (4, 5, 1)])
def test_integer_sum_matches_existing_rational_sum(vertices, degree, skeleton):
    kernel = integer_kernel(vertices - skeleton - 1, degree - skeleton - 1, backend="python")
    assert scalar_coefficient(vertices, degree, skeleton, kernel) == _scalar_risk_sum(vertices, degree, skeleton + 1)


@pytest.mark.parametrize("skeleton", [0, 1, 2])
def test_integer_sum_matches_independent_gram_oracle(skeleton):
    kernel = integer_kernel(3 - skeleton, 3 - skeleton, backend="python")
    expected, _, _ = _exact_coefficients(4, 4)
    assert scalar_coefficient(4, 4, skeleton, kernel) == expected[skeleton]


@pytest.mark.parametrize(("power", "maximum"), [(-1, 0), (0, -1), (True, 2), (2, 1.5)])
def test_kernel_rejects_invalid_dimensions(power, maximum):
    with pytest.raises(ValueError):
        integer_kernel(power, maximum, backend="python")


def test_scalar_rejects_incompatible_kernel():
    with pytest.raises(ValueError, match="does not cover"):
        scalar_coefficient(4, 5, 0, integer_kernel(2, 5, backend="python"))
    with pytest.raises(ValueError, match="does not cover"):
        scalar_coefficient(4, 5, 0, integer_kernel(3, 2, backend="python"))


def test_absent_coefficient_is_zero_without_kernel_requirements():
    assert scalar_coefficient(4, 1, 1, integer_kernel(0, 0, backend="python")) == 0


def test_source_change_during_kernel_is_rejected_and_initial_source_preserved(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"original")

    def changing_kernel(power, maximum):
        source.write_bytes(b"changed")
        return integer_kernel(power, maximum, backend="python")

    monkeypatch.setattr(research_exact, "__file__", str(source))
    monkeypatch.setattr(research_exact, "integer_kernel", changing_kernel)
    monkeypatch.setattr(research_exact, "import_module", lambda name: SimpleNamespace(__version__="test"))
    output = tmp_path / "result.json"
    arguments = Namespace(vertices=2, skeleton=0, degrees=[2], output=output)
    with pytest.raises(RuntimeError, match="source changed"):
        research_exact._worker(arguments)
    assert output.with_suffix(".source.py").read_bytes() == b"original"
    assert json.loads(output.read_text())["source_sha256"] == hashlib.sha256(b"original").hexdigest()
