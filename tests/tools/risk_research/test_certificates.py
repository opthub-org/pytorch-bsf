"""Independent symbolic checks and mutations of research certificates."""

import copy
import json
from pathlib import Path

import pytest
import sympy as sp

from tools.risk_research.certificates import CertificateError, D, H, _parse_expression, main, verify_certificate

EXAMPLE = Path(__file__).resolve().parents[3] / "tools/risk_research/examples/triangle_vertex_certificate.json"


@pytest.fixture
def triangle():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_triangle_certificate_is_symbolic_and_includes_both_boundaries(triangle):
    result = verify_certificate(triangle)
    assert result["status"] == "certified_identity"
    assert result["boundaries"]["upper"] == "0"
    contribution, _ = _parse_expression(result["boundary_contribution"])
    assert sp.cancel(contribution - 1 / ((D + 2) * (D + 3))) == 0
    assert {"lower_boundary", "upper_boundary", "symbolic_residual"} <= set(result["checks"])


def test_degree_one_endpoint_is_included(triangle):
    weight, _ = _parse_expression(triangle["weight"], allow_binomial=True)
    factor, _ = _parse_expression(triangle["summand_factor"])
    certificate, _ = _parse_expression(triangle["certificate_factor"])
    actual = (weight * factor).subs({D: 1, H: 0})
    difference = (weight * certificate).subs({D: 1, H: 1}) - (weight * certificate).subs({D: 1, H: 0})
    assert actual == difference == sp.Rational(1, 12)


@pytest.mark.parametrize("field", ["weight_ratio", "summand_factor", "certificate_factor"])
def test_changed_identity_is_rejected(triangle, field):
    triangle[field] = f"({triangle[field]}) + 1"
    with pytest.raises(CertificateError, match="identity|residual|boundary"):
        verify_certificate(triangle)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_changed_boundary_is_rejected(triangle, side):
    triangle["boundaries"][side] = "1"
    with pytest.raises(CertificateError, match="boundary"):
        verify_certificate(triangle)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_missing_boundary_is_rejected(triangle, side):
    del triangle["boundaries"][side]
    with pytest.raises(CertificateError, match="Both"):
        verify_certificate(triangle)


def test_wrong_weight_is_not_certified_from_supplied_ratio(triangle):
    triangle["weight"] = "binomial(d + 1, h + 1) * binomial(d + 1, h)"
    with pytest.raises(CertificateError, match="weight ratio"):
        verify_certificate(triangle)


def test_cancelled_pole_is_rejected(triangle):
    triangle["summand_factor"] = f"({triangle['summand_factor']}) * ((d - 1) / (d - 1))"
    with pytest.raises(CertificateError, match="denominator"):
        verify_certificate(triangle)


def test_pole_only_at_upper_boundary_is_rejected(triangle):
    triangle["certificate_factor"] = f"({triangle['certificate_factor']}) * ((d - h) / (d - h))"
    with pytest.raises(CertificateError, match="upper boundary"):
        verify_certificate(triangle)


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "d.__class__", "[d][0]", "1.0", "other", "d**h"])
def test_expression_parser_rejects_executable_or_unsupported_syntax(triangle, expression):
    triangle["summand_factor"] = expression
    with pytest.raises(CertificateError):
        verify_certificate(triangle)


def test_constant_weight_certificate():
    result = verify_certificate(
        {
            "schema_version": 1,
            "kind": "rational_telescoping",
            "domain": {"degree_min": 1, "lower": 0, "upper_offset": 1},
            "weight": "1",
            "weight_ratio": "1",
            "summand_factor": "1",
            "certificate_factor": "h",
            "boundaries": {"lower": "0", "upper": "d"},
        }
    )
    assert result["boundaries"] == {"lower": "0", "upper": "d"}


def test_cli_rejects_certificate_mutation(triangle, tmp_path, capsys):
    altered = copy.deepcopy(triangle)
    altered["boundaries"]["upper"] = "1"
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert main([str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"


def test_cli_example_passes(capsys):
    assert main([str(EXAMPLE)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "certified_identity"
