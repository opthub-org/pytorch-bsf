"""Derive and certify a degree recurrence for tetrahedron vertex risk.

Four triangular polynomial difference equations replace a generic multivariate
creative-telescoping calculation. SageMath is needed only for derivation and
symbolic verification; evaluating the recorded recurrence uses Python integers
and fractions.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any

from .edge_recurrence import _integer_rows
from .exact import atomic_json
from .vertex_single import tetrahedron_vertex_coefficient

SCHEMA = "skeleton-risk-vertex-degree-recurrence-v1"
MAX_ORDER = 10
STATE = ["H_j^2", "H_j^(2)", "H_j", "1"]
DEFAULT_OPERATOR = Path(__file__).with_name("examples") / "tetrahedron_vertex_recurrence_order10.json"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _falling(value: Any, count: int) -> Any:
    return math.prod(value - offset for offset in range(count))


def _normalization_ratio(degree: Any, shift: int) -> Any:
    """Return d_(D+shift)/d_D, d_D=(2D+3)!/((D-1)!(D+1)!)."""
    return (
        degree**0
        * math.prod(2 * degree + 4 + offset for offset in range(2 * shift))
        / (math.prod(degree + offset for offset in range(shift)) * math.prod(degree + 2 + offset for offset in range(shift)))
    )


def _symbolic_context() -> tuple[Any, Any, Any]:
    sage = import_module("sage.all")
    parameter_ring = sage.PolynomialRing(sage.QQ, "N")
    field = parameter_ring.fraction_field()
    polynomials = sage.PolynomialRing(field, "j")
    return field(parameter_ring.gen()), polynomials, polynomials.gen()


def _reduce_polynomial(polynomial: Any, operator: Callable[[Any], Any], drop: int) -> tuple[Any, Any]:
    """Invert the triangular leading terms, retaining the low-degree remainder."""
    ring = polynomial.parent()
    variable = ring.gen()
    quotient = ring.zero()
    remainder = polynomial
    while remainder.degree() >= drop:
        monomial = variable ** (remainder.degree() - drop)
        image = operator(monomial)
        coefficient = remainder.leading_coefficient() / image.leading_coefficient()
        quotient += coefficient * monomial
        remainder -= coefficient * image
    return quotient, remainder


def _certificate(polynomial: Any, parameter: Any, variable: Any) -> tuple[list[Any], list[Any]]:
    """Solve the four polynomial equations, exposing every residual."""
    quadratic = (parameter - variable) * (parameter + 2 - variable)

    def second(poly: Any) -> Any:
        return quadratic * (variable + 1) * poly(variable + 1) - variable**2 * (variable + 3) * poly

    def third(poly: Any) -> Any:
        return quadratic * (variable + 1) ** 2 * poly(variable + 1) - variable**2 * (variable + 2) * (variable + 3) * poly

    first, first_remainder = _reduce_polynomial(polynomial, second, 2)
    second_poly, second_remainder = _reduce_polynomial(polynomial * (variable + 2), second, 2)
    third_poly, third_remainder = _reduce_polynomial(
        -polynomial * variable * (variable + 1) - 2 * quadratic * (variable + 1) * (variable + 2) * first(variable + 1),
        third,
        3,
    )
    fourth, fourth_remainder = _reduce_polynomial(
        2 * polynomial
        - quadratic * (variable + 2) * (first(variable + 1) + second_poly(variable + 1))
        - quadratic * (variable + 1) * third_poly(variable + 1),
        third,
        3,
    )
    return [first, second_poly, third_poly, fourth], [first_remainder, second_remainder, third_remainder, fourth_remainder]


def _rational_residuals(polynomial: Any, certificate: list[Any], parameter: Any, variable: Any) -> list[Any]:
    """Check the telescoping row directly, separately from polynomial reduction."""
    field = polynomial.parent().fraction_field()
    variable = field(variable)
    first_denominator = (variable + 1) * (variable + 2)
    second_denominator = (variable + 1) ** 2 * (variable + 2)
    row = [
        variable**2 * field(poly) / (first_denominator if index < 2 else second_denominator)
        for index, poly in enumerate(certificate)
    ]
    shifted = [entry(variable + 1) for entry in row]
    increment = 1 / (variable + 1)
    ratio = (parameter - variable) * (parameter + 2 - variable) * increment**2
    advanced = [
        shifted[0],
        shifted[1],
        2 * increment * shifted[0] + shifted[2],
        increment**2 * (shifted[0] + shifted[1]) + increment * shifted[2] + shifted[3],
    ]
    output = [
        1 / ((variable + 1) * (variable + 2) * (variable + 3)),
        1 / ((variable + 1) * (variable + 3)),
        -variable / ((variable + 1) * (variable + 2) ** 2 * (variable + 3)),
        2 / ((variable + 1) ** 2 * (variable + 2) ** 2 * (variable + 3)),
    ]
    return [ratio * advanced[index] - row[index] - field(polynomial) * output[index] for index in range(4)]


def _check_parameter_denominators(certificate: list[Any], order: int) -> int:
    """Independently prove positivity by shifting N=D+order-1."""
    count = 0
    for polynomial in certificate:
        for coefficient in polynomial.list():
            denominator = coefficient.denominator()
            shifted = denominator(denominator.parent().gen() + order - 1)
            if not shifted or any(value < 0 for value in shifted.list()):
                raise ValueError("certificate parameter denominator positivity needs an additional proof")
            count += 1
    return count


def _primitive_polynomials(entries: list[Any], ring: Any, sage: Any) -> list[Any]:
    denominator = sage.lcm([entry.denominator() for entry in entries])
    polynomials = [ring(entry * denominator) for entry in entries]
    divisor = sage.gcd(polynomials)
    polynomials = [polynomial // divisor for polynomial in polynomials]
    multiplier = sage.lcm([coefficient.denominator() for polynomial in polynomials for coefficient in polynomial.list()])
    polynomials = [ring(polynomial * multiplier) for polynomial in polynomials]
    content = math.gcd(*(int(coefficient) for polynomial in polynomials for coefficient in polynomial.list()))
    polynomials = [ring(polynomial / content) for polynomial in polynomials]
    if polynomials[-1].leading_coefficient() < 0:
        polynomials = [-polynomial for polynomial in polynomials]
    return polynomials


def _evaluate_polynomial(coefficients: list[int], value: int) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _leading_domain_certificate(coefficients: list[int]) -> dict[str, object]:
    """Exclude integer zeros by Descartes' rule and an exact root bracket."""
    signs = [1 if coefficient > 0 else -1 for coefficient in coefficients if coefficient]
    changes = sum(first != second for first, second in zip(signs, signs[1:], strict=False))
    # The recorded leading coefficient has exactly one positive real root,
    # strictly between 4 and 5. Descartes plus these two integer evaluations
    # excludes every positive integer, despite the root inside the domain.
    if changes != 1 or not _evaluate_polynomial(coefficients, 4) < 0 < _evaluate_polynomial(coefficients, 5):
        raise ValueError("leading coefficient needs a different domain proof")
    return {"positive_sign_variations": 1, "positive_real_root_bracket": [4, 5], "exceptional_indices": []}


def _read_operator(source: bytes) -> tuple[dict[str, Any], list[list[int]]]:
    document = json.loads(source)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("expected a tetrahedron vertex degree recurrence")
    order, degree = document.get("order"), document.get("polynomial_degree")
    if type(order) is not int or not 1 <= order <= MAX_ORDER or type(degree) is not int or degree < 0:
        raise ValueError("expected integer order 1..10 and nonnegative polynomial degree")
    if type(document.get("domain_start")) is not int or document["domain_start"] != 1:
        raise ValueError("the vertex degree recurrence domain must start at D=1")
    if document.get("harmonic_basis") != STATE:
        raise ValueError("inconsistent harmonic basis")
    if document.get("exceptional_indices") != []:
        raise ValueError("this domain certificate requires no exceptional indices")
    if document.get("required_initial_indices") != list(range(1, order + 1)):
        raise ValueError("inconsistent required initial indices")
    coefficients = document.get("coefficients")
    if not isinstance(coefficients, list):
        raise ValueError("invalid coefficient array")
    integers = _integer_rows(coefficients, order)
    if integers[-1][-1] <= 0 or max(len(row) - 1 for row in integers) != degree or any(row[-1] == 0 for row in integers):
        raise ValueError("inconsistent polynomial degree or leading sign")
    return document, integers


def _provenance() -> dict[str, str]:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("verifier source changed during execution; rerun with stable sources")
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "python": platform.python_version(),
        "sage": str(import_module("sage.env").SAGE_VERSION),
    }


def derive_operator() -> dict[str, object]:
    """Derive the operator from ten remainder conditions without sequence fitting."""
    started = time.monotonic()
    sage = import_module("sage.all")
    parameter, ring, variable = _symbolic_context()
    field = ring.base_ring()
    columns = []
    polynomial = ring.one()
    for count in range(MAX_ORDER + 1):
        if count:
            polynomial *= (parameter - variable - count + 1) * (parameter + 3 - variable - count)
        _, remainders = _certificate(polynomial, parameter, variable)
        columns.append(
            [remainder[index] for remainder, size in zip(remainders, (2, 2, 3, 3), strict=True) for index in range(size)]
        )
    kernel = sage.matrix(field, columns).transpose().right_kernel()
    if kernel.dimension() != 1:
        raise RuntimeError("the structured order-ten certificate unexpectedly changed rank")
    multipliers = kernel.basis()[0]
    combined = sum(
        multipliers[count] * _falling(parameter - variable, count) * _falling(parameter + 2 - variable, count)
        for count in range(MAX_ORDER + 1)
    )
    certificate, remainders = _certificate(combined, parameter, variable)
    if any(remainders) or any(_rational_residuals(combined, certificate, parameter, variable)):
        raise RuntimeError("derived certificate failed the direct rational telescoping identity")
    degree_ring = sage.PolynomialRing(sage.QQ, "D")
    degree_field = degree_ring.fraction_field()
    degree = degree_field(degree_ring.gen())
    entries = []
    for shift in range(MAX_ORDER + 1):
        count = MAX_ORDER - shift
        value = multipliers[count] * _falling(parameter, count) * _falling(parameter + 2, count)
        substituted = degree_field(value.numerator()(degree + MAX_ORDER - 1)) / value.denominator()(degree + MAX_ORDER - 1)
        entries.append(substituted * _normalization_ratio(degree, shift))
    polynomials = _primitive_polynomials(entries, degree_ring, sage)
    integers = [[int(coefficient) for coefficient in polynomial.list()] for polynomial in polynomials]
    return {
        "schema": SCHEMA,
        "proof_status": "exact_rational_telescoping_identity",
        "equation": "sum_i p_i(D)*a_0(4,D+i)=0; coefficients in ascending powers of D",
        "order": MAX_ORDER,
        "polynomial_degree": int(max(polynomial.degree() for polynomial in polynomials)),
        "coefficients": [[str(coefficient) for coefficient in row] for row in integers],
        "domain_start": 1,
        "exceptional_indices": [],
        "required_initial_indices": list(range(1, MAX_ORDER + 1)),
        "initial_values": {str(index): str(tetrahedron_vertex_coefficient(index)) for index in range(1, MAX_ORDER + 1)},
        "harmonic_basis": STATE.copy(),
        "leading_domain_certificate": _leading_domain_certificate(integers[-1]),
        "proof_document": "docs/advanced/skeleton_risk_vertex_recurrence.rst",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def verify_operator(path: Path) -> dict[str, object]:
    """Reconstruct and verify four rational telescoping identities exactly.

    Polynomial reduction denominators are m-2N-4 and m-2N-5, with
    N=D+r-1 and m<=2r-1. All are strictly negative on D>=1. The remaining
    normalizing factors are positive there, and certificate j-denominators
    have factors j+1 and j+2 only. Boundary terms vanish at j=0 and N+1.
    """
    started = time.monotonic()
    original = path.read_bytes()
    document, integers = _read_operator(original)
    order = document["order"]
    parameter, ring, variable = _symbolic_context()
    degree = parameter - order + 1
    polynomial = ring.zero()
    for shift, row in enumerate(integers):
        count = order - shift
        coefficient = sum(value * degree**index for index, value in enumerate(row))
        coefficient /= _normalization_ratio(degree, shift) * _falling(parameter, count) * _falling(parameter + 2, count)
        polynomial += coefficient * _falling(parameter - variable, count) * _falling(parameter + 2 - variable, count)
    certificate, remainders = _certificate(polynomial, parameter, variable)
    if any(remainders) or any(_rational_residuals(polynomial, certificate, parameter, variable)):
        raise ValueError("operator fails the exact rational telescoping identity")
    certificate_degrees = [int(poly.degree()) for poly in certificate]
    if max(certificate_degrees) > 2 * order - 1:
        raise ValueError("certificate denominator domain requires an additional proof")
    denominator_checks = _check_parameter_denominators(certificate, order)
    leading_proof = _leading_domain_certificate(integers[-1])
    if leading_proof != document.get("leading_domain_certificate"):
        raise ValueError("inconsistent leading-domain certificate")
    initial_values = {str(index): str(tetrahedron_vertex_coefficient(index)) for index in range(1, order + 1)}
    if initial_values != document.get("initial_values"):
        raise ValueError("initial values do not match the proved positive sum")
    if path.read_bytes() != original:
        raise RuntimeError("operator changed during verification")
    return {
        "status": "certified_by_rational_telescoping",
        "schema": SCHEMA,
        "operator_sha256": hashlib.sha256(original).hexdigest(),
        "order": order,
        "polynomial_degree": document["polynomial_degree"],
        "domain_start": 1,
        "exceptional_indices": [],
        "symbolic_harmonic_residuals": ["0"] * 4,
        "certificate_polynomial_degrees": certificate_degrees,
        "certificate_denominators_nonzero_on_domain": True,
        "positive_parameter_denominator_checks": denominator_checks,
        "lower_boundary": "j^2 at j=0",
        "upper_boundary": "binom(N,j)=0 at j=N+1; N=D+order-1",
        "harmonic_basis": STATE.copy(),
        "leading_domain_certificate": leading_proof,
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def recurrence_coefficients(maximum_degree: int, operator: Path = DEFAULT_OPERATOR) -> dict[int, Fraction]:
    """Evaluate the recorded recurrence exactly, using its ten initial values.

    This is an evaluator for a previously certified artifact, not a substitute
    for ``verify_operator``. It does not import SageMath.
    """
    if type(maximum_degree) is not int or maximum_degree < 1:
        raise ValueError("maximum_degree must be a positive integer")
    document, coefficients = _read_operator(operator.read_bytes())
    _leading_domain_certificate(coefficients[-1])
    order = document["order"]
    values = {index: Fraction(document["initial_values"][str(index)]) for index in range(1, order + 1)}
    for degree in range(1, maximum_degree - order + 1):
        numerator = sum(
            (_evaluate_polynomial(row, degree) * values[degree + shift] for shift, row in enumerate(coefficients[:-1])),
            Fraction(0),
        )
        values[degree + order] = -numerator / _evaluate_polynomial(coefficients[-1], degree)
    return {index: values[index] for index in range(1, maximum_degree + 1)}


def main(argv: list[str] | None = None) -> None:
    """Derive or verify an operator and optionally save the complete proof report."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", type=Path)
    action.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output:
        atomic_json(arguments.output, derive_operator())
        report = verify_operator(arguments.output)
    else:
        report = verify_operator(arguments.verify)
    if arguments.report:
        atomic_json(arguments.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
