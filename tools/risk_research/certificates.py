"""Check rational telescoping certificates independently of their discovery tool.

The supported domain is integer ``d >= degree_min`` and
``lower <= h <= d - upper_offset``.  Rational expressions use only ``d``,
``h``, integers and arithmetic; the hypergeometric weight may also use
``binomial``.  The checker proves a symbolic identity and its two boundary
values.  It does not turn a guessed recurrence or a finite list of matching
values into a proof of a risk formula.
"""

import argparse
import ast
import json
import operator
from pathlib import Path
from typing import Any

import sympy as sp

D = sp.Symbol("d", integer=True, positive=True)
H = sp.Symbol("h", integer=True, nonnegative=True)


class CertificateError(ValueError):
    """Identify an invalid or unsupported certificate."""


def _binary_expression(node: ast.BinOp, left: Any, right: Any, denominators: list[Any]) -> Any:
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    if type(node.op) in operations:
        return operations[type(node.op)](left, right)
    if isinstance(node.op, ast.Div):
        denominators.append(right)
        return left / right
    if isinstance(node.op, ast.Pow) and right.is_Integer and abs(right) <= 20:
        if right < 0:
            denominators.append(left)
        return left**right
    raise CertificateError("Only arithmetic and bounded integer exponents are supported.")


def _binomial_expression(node: ast.Call, denominators: list[Any], allowed: bool) -> Any:
    if not allowed or not isinstance(node.func, ast.Name) or node.func.id != "binomial":
        raise CertificateError("Only permitted binomial calls are supported.")
    if len(node.args) != 2 or node.keywords:
        raise CertificateError("Binomial requires two positional arguments.")
    return sp.binomial(*(_visit_expression(argument, denominators, allowed) for argument in node.args))


def _visit_expression(node: ast.AST, denominators: list[Any], allow_binomial: bool) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if node.value.bit_length() > 1024:
            raise CertificateError("Integer literal is too large.")
        return sp.Integer(node.value)
    if isinstance(node, ast.Name):
        if node.id in {"d", "h"}:
            return D if node.id == "d" else H
        raise CertificateError("Only d and h are supported as variable names.")
    if isinstance(node, ast.UnaryOp):
        value = _visit_expression(node.operand, denominators, allow_binomial)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise CertificateError("Only unary plus and minus are supported.")
    if isinstance(node, ast.BinOp):
        left = _visit_expression(node.left, denominators, allow_binomial)
        right = _visit_expression(node.right, denominators, allow_binomial)
        return _binary_expression(node, left, right, denominators)
    if isinstance(node, ast.Call):
        return _binomial_expression(node, denominators, allow_binomial)
    raise CertificateError("Only integer arithmetic, d, h and permitted binomial calls are supported.")


def _parse_expression(source: object, *, allow_binomial: bool = False) -> tuple[Any, list[Any]]:
    """Parse a small arithmetic grammar without executing source text."""
    if not isinstance(source, str) or len(source) > 4000:
        raise CertificateError("Expressions must be strings of at most 4000 characters.")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise CertificateError("Invalid expression syntax.") from error
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise CertificateError("Expression is too large.")
    denominators: list[Any] = []
    try:
        result = _visit_expression(tree.body, denominators, allow_binomial)
    except (TypeError, ZeroDivisionError, RecursionError) as error:
        raise CertificateError("Invalid arithmetic expression.") from error
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise CertificateError("Expression is not finite.")
    return result, denominators


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise CertificateError(f"{name} must be an integer >= {minimum}.")
    return value


def _constant_sign(polynomial: Any, variables: tuple[Any, ...], *, strict: bool = True) -> bool:
    """Prove a sign from polynomial coefficients on a nonnegative orthant.

    This sufficient criterion deliberately rejects expressions whose sign
    requires a stronger inequality argument.
    """
    try:
        poly = sp.Poly(sp.expand(polynomial), *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.CoercionFailed):
        return False
    coefficients = poly.coeffs()
    constant = poly.coeff_monomial(1)
    if all(coefficient >= 0 for coefficient in coefficients):
        return bool(constant > 0 if strict else True)
    if strict and all(coefficient <= 0 for coefficient in coefficients):
        return bool(constant < 0)
    return False


def _nonzero_on_domain(expression: Any, lower: int, offset: int) -> bool:
    # h = lower + x, d = lower + offset + x + y parametrizes the
    # triangular domain. A larger degree_min only restricts this domain.
    x, y = sp.symbols("x y", nonnegative=True)
    numerator = sp.fraction(sp.cancel(expression))[0]
    transformed = numerator.subs({D: lower + offset + x + y, H: lower + x}, simultaneous=True)
    return _constant_sign(transformed, (x, y))


def _nonzero_at_endpoint(expression: Any, endpoint: Any, degree_min: int) -> bool:
    x = sp.Symbol("x", nonnegative=True)
    numerator = sp.fraction(sp.cancel(expression.subs(H, endpoint)))[0]
    return _constant_sign(numerator.subs(D, degree_min + x), (x,))


def _check_denominators(denominators: list[Any], lower: int, offset: int, degree_min: int) -> None:
    for denominator in denominators:
        if not _nonzero_on_domain(denominator, lower, offset):
            raise CertificateError("A denominator may vanish on the summation domain, or its sign is unsupported.")
        if not _nonzero_at_endpoint(denominator, D - offset + 1, degree_min):
            raise CertificateError("A denominator may vanish at the upper boundary, or its sign is unsupported.")


def _check_weight_arguments(weight: Any, lower: int, offset: int, degree_min: int) -> None:
    x, y = sp.symbols("x y", nonnegative=True)
    for call in weight.atoms(sp.binomial):
        for argument in call.args:
            try:
                poly = sp.Poly(argument, D, H, domain=sp.ZZ)
            except (sp.PolynomialError, sp.CoercionFailed) as error:
                raise CertificateError("Binomial arguments must be integer affine polynomials.") from error
            if poly.total_degree() > 1:
                raise CertificateError("Binomial arguments must be integer affine polynomials.")
            transformed = argument.subs({D: lower + offset + x + y, H: lower + x}, simultaneous=True)
            boundary = argument.subs(H, D - offset + 1).subs(D, degree_min + x)
            if not _constant_sign(transformed, (x, y), strict=False) or not _constant_sign(boundary, (x,), strict=False):
                raise CertificateError("Binomial arguments must be nonnegative on the supported domain and boundary.")


def _certificate_domain(document: dict[str, Any]) -> tuple[int, int, int]:
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
    ):
        raise CertificateError("Expected a schema_version 1 certificate object.")
    if document.get("kind") != "rational_telescoping":
        raise CertificateError("Only rational_telescoping certificates are supported.")
    domain = document.get("domain")
    if not isinstance(domain, dict):
        raise CertificateError("A domain object is required.")
    degree_min = _integer(domain.get("degree_min"), "degree_min", 1)
    lower = _integer(domain.get("lower"), "lower", 0)
    offset = _integer(domain.get("upper_offset"), "upper_offset", 0)
    if degree_min < lower + offset:
        raise CertificateError("The summation interval must be nonempty at degree_min.")
    boundaries = document.get("boundaries")
    if not isinstance(boundaries, dict) or set(boundaries) != {"lower", "upper"}:
        raise CertificateError("Both lower and upper boundary expressions are required.")
    return degree_min, lower, offset


def _certificate_expressions(document: dict[str, Any], degree_min: int, lower: int, offset: int) -> dict[str, Any]:
    expressions: dict[str, Any] = {}
    for field in ("weight", "weight_ratio", "summand_factor", "certificate_factor"):
        expression, denominators = _parse_expression(document.get(field), allow_binomial=field == "weight")
        _check_denominators(denominators, lower, offset, degree_min)
        if field != "weight" and not expression.is_rational_function(D, H):
            raise CertificateError("Term ratio and normalized factors must be rational functions.")
        expressions[field] = expression
    return expressions


def _certificate_endpoints(
    weight: Any, certificate: Any, boundaries: dict[str, str], degree_min: int, lower: int, offset: int
) -> dict[str, Any]:
    endpoint_expressions = {}
    for side, endpoint in (("lower", sp.Integer(lower)), ("upper", D - offset + 1)):
        expected, denominators = _parse_expression(boundaries[side])
        if expected.has(H):
            raise CertificateError("Boundary expressions may depend only on d.")
        for denominator in denominators:
            if not _nonzero_at_endpoint(denominator, endpoint, degree_min):
                raise CertificateError("A boundary expression may have a pole.")
        actual = sp.combsimp(weight.subs(H, endpoint) * certificate.subs(H, endpoint))
        if actual.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or sp.combsimp(actual - expected) != 0:
            raise CertificateError(f"The {side} boundary value does not match the symbolic product.")
        endpoint_expressions[side] = actual
    return endpoint_expressions


def verify_certificate(document: dict[str, Any]) -> dict[str, Any]:
    """Verify a supported symbolic telescoping identity and both endpoints.

    A successful result certifies ``sum(w(h)*factor(h)) = upper - lower``
    under the recorded integer-domain assumptions.  The expressions for the
    weight and its ratio are checked against one another, rather than taking
    a supplied term ratio on trust.
    """
    degree_min, lower, offset = _certificate_domain(document)
    expressions = _certificate_expressions(document, degree_min, lower, offset)
    weight, ratio = expressions["weight"], expressions["weight_ratio"]
    factor, certificate = expressions["summand_factor"], expressions["certificate_factor"]
    _check_weight_arguments(weight, lower, offset, degree_min)
    if weight == 0:
        raise CertificateError("The weight must not be identically zero.")
    if sp.combsimp(weight.subs(H, H + 1) - ratio * weight) != 0:
        raise CertificateError("The supplied weight ratio is not a symbolic shift identity.")
    # The shifted certificate must also be finite on every summation index.
    _, shifted_denominator = sp.fraction(sp.cancel(certificate.subs(H, H + 1)))
    if not _nonzero_on_domain(shifted_denominator, lower, offset):
        raise CertificateError("The shifted certificate may have a pole on the summation domain.")
    if sp.cancel(ratio * certificate.subs(H, H + 1) - certificate - factor) != 0:
        raise CertificateError("The normalized telescoping residual is not identically zero.")
    endpoints = _certificate_endpoints(weight, certificate, document["boundaries"], degree_min, lower, offset)

    return {
        "schema_version": 1,
        "status": "certified_identity",
        "name": document.get("name", "unnamed"),
        "domain": {"degree_min": degree_min, "lower": lower, "upper_offset": offset},
        "checks": ["weight_shift", "nonvanishing_denominators", "symbolic_residual", "lower_boundary", "upper_boundary"],
        "boundaries": {side: str(value) for side, value in endpoints.items()},
        "boundary_contribution": str(sp.cancel(endpoints["upper"] - endpoints["lower"])),
        "scope": "Telescoping identity and both boundary values; a complete risk formula is not certified.",
    }


def main(argv: list[str] | None = None) -> int:
    """Print a machine-readable certificate verification report."""
    parser = argparse.ArgumentParser(description="Verify a symbolic rational telescoping certificate.")
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.certificate.read_text(encoding="utf-8"))
        report = verify_certificate(document)
    except (OSError, json.JSONDecodeError, CertificateError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
