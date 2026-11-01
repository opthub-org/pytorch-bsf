"""Derive and certify the five-vertex vertex degree recurrence.

Eight triangular harmonic equations leave at most21 polynomial residuals.
Exact state reduction produces the supplied order17 operator, with degree188
and no positive integer exceptions. Sage is lazy and only needed for the
symbolic proof; the recorded recurrence evaluates with standard Python.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any

from .edge_recurrence import _integer_rows
from .exact import atomic_json, peak_rss_bytes
from .vertex_degree_recurrence import (
    _check_parameter_denominators,
    _evaluate_polynomial,
    _falling,
    _primitive_polynomials,
    _reduce_polynomial,
    _symbolic_context,
)
from .vertex_five import five_vertex_coefficient

SCHEMA = "skeleton-risk-five-vertex-degree-recurrence-v1"
MAX_ORDER = 17
STATE = ["H_j^3", "H_j^2", "H_j*H_j^(2)", "H_j^(3)", "A_j", "H_j", "H_j^(2)", "1"]
DROPS = (2, 3, 2, 2, 2, 3, 3, 4)
DEFAULT_OPERATOR = Path(__file__).with_name("examples") / "five_vertex_vertex_recurrence_order17.json"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalization_ratio(degree: Any, shift: int) -> Any:
    """Return d_(D+shift)/d_D, d_D=(2D+4)!/((D-1)!(D+1)!)."""
    return (
        degree**0
        * math.prod(2 * degree + 5 + offset for offset in range(2 * shift))
        / (math.prod(degree + offset for offset in range(shift)) * math.prod(degree + 2 + offset for offset in range(shift)))
    )


def _structure(parameter: Any, variable: Any) -> tuple[list[Any], list[Any], dict[int, list[tuple[int, Any]]]]:
    n = variable + 1
    low = n * (n + 1) * (n + 2)
    middle = n**2 * (n + 1) ** 2 * (n + 2)
    high = n**3 * (n + 1) ** 2 * (n + 2)
    denominators = [low, middle, low, low, low, middle, middle, high]
    output = [
        24 / (n * (n + 1) * (n + 2) * (n + 3)),
        -12 * (5 * n**3 + 7 * n**2 - 6 * n - 12) / (n**2 * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3)),
        24 * (2 * n + 3) / (n * (n + 1) * (n + 2) * (n + 3)),
        -48 / (n * (n + 1) * (n + 3)),
        -48 / (n * (n + 2)),
        -24 * (2 * n**4 + 15 * n**3 + 44 * n**2 + 47 * n + 24) / (n**2 * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3)),
        12 * (4 * n**4 + 22 * n**3 + 41 * n**2 + 26 * n + 12) / (n**2 * (n + 1) * (n + 2) ** 2 * (n + 3)),
        144 / (n**2 * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3)),
    ]
    transfers = {
        1: [(0, 3 / n)],
        5: [(0, 3 / n**2), (1, 2 / n), (2, 1 / n**2), (4, 1 / n**2)],
        6: [(2, 1 / n)],
        7: [(0, 1 / n**3), (1, 1 / n**2), (2, 1 / n**3), (3, 1 / n**3), (5, 1 / n), (6, 1 / n**2)],
    }
    return denominators, output, transfers


def _certificate(polynomial: Any, parameter: Any, variable: Any) -> tuple[list[Any], list[Any]]:
    """Solve eight triangular polynomial equations, retaining every remainder."""
    ring = polynomial.parent()
    field = ring.fraction_field()
    quadratic = (parameter - variable) * (parameter + 2 - variable)
    denominators, output, transfers = _structure(parameter, variable)
    quotients: list[Any] = []
    remainders: list[Any] = []
    for index, denominator in enumerate(denominators):
        omega = denominator.lcm(denominator(variable + 1))
        rhs = field(polynomial) * output[index]
        for upstream, increment in transfers.get(index, []):
            rhs -= quadratic * field(quotients[upstream](variable + 1)) * increment / denominators[upstream](variable + 1)
        scaled = rhs * omega
        if scaled.denominator() != 1:
            raise RuntimeError("harmonic reduction has an unexpected j-pole")

        def operator(poly: Any, omega: Any = omega, denominator: Any = denominator) -> Any:
            return (
                quadratic * ring(omega / denominator(variable + 1)) * poly(variable + 1)
                - variable**2 * ring(omega / denominator) * poly
            )

        quotient, remainder = _reduce_polynomial(ring(scaled.numerator()), operator, DROPS[index])
        quotients.append(quotient)
        remainders.append(remainder)
    return quotients, remainders


def _rational_residuals(polynomial: Any, certificate: list[Any], parameter: Any, variable: Any) -> list[Any]:
    """Check the rational telescoping row directly, outside polynomial reduction."""
    ring = polynomial.parent()
    field = ring.fraction_field()
    denominators, output, transfers = _structure(parameter, variable)
    row = [variable**2 * field(poly) / denominator for poly, denominator in zip(certificate, denominators, strict=True)]
    shifted = [entry(variable + 1) for entry in row]
    ratio = (parameter - variable) * (parameter + 2 - variable) / (variable + 1) ** 2
    return [
        ratio
        * (
            shifted[index]
            + sum((increment * shifted[upstream] for upstream, increment in transfers.get(index, [])), field.zero())
        )
        - row[index]
        - field(polynomial) * output[index]
        for index in range(len(STATE))
    ]


def _leading_domain_certificate(coefficients: list[int]) -> dict[str, object]:
    if not coefficients or any(coefficient <= 0 for coefficient in coefficients):
        raise ValueError("leading coefficient needs strictly positive polynomial coefficients")
    return {
        "criterion": "strictly positive polynomial coefficients",
        "coefficient_count": len(coefficients),
        "exceptional_indices": [],
    }


def _read_operator(source: bytes) -> tuple[dict[str, Any], list[list[int]]]:
    document = json.loads(source)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("expected a five-vertex vertex degree recurrence")
    order, degree = document.get("order"), document.get("polynomial_degree")
    if type(order) is not int or not 1 <= order <= MAX_ORDER or type(degree) is not int or degree < 0:
        raise ValueError("expected integer order 1..17 and nonnegative polynomial degree")
    if type(document.get("domain_start")) is not int or document["domain_start"] != 1:
        raise ValueError("the five-vertex recurrence domain must start at D=1")
    if document.get("harmonic_basis") != STATE or document.get("remainder_dimensions") != list(DROPS):
        raise ValueError("inconsistent harmonic basis or residual dimensions")
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


def _provenance() -> dict[str, object]:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("verifier source changed during execution")
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "python": platform.python_version(),
        "sage": str(import_module("sage.env").SAGE_VERSION),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def derive_operator() -> dict[str, object]:
    """Construct the recorded order17 operator without fitting sequence values."""
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
        columns.append([remainder[index] for remainder, size in zip(remainders, DROPS, strict=True) for index in range(size)])
    kernel = sage.matrix(field, columns).transpose().right_kernel()
    if kernel.dimension() != 1:
        raise RuntimeError("the structured order17 certificate changed rank")
    multipliers = kernel.basis()[0]
    combined = sum(
        (
            multipliers[count] * _falling(parameter - variable, count) * _falling(parameter + 2 - variable, count)
            for count in range(MAX_ORDER + 1)
        ),
        ring.zero(),
    )
    certificate, remainders = _certificate(combined, parameter, variable)
    if any(remainders) or any(_rational_residuals(combined, certificate, parameter, variable)):
        raise RuntimeError("derived certificate fails its exact rational identities")
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
        "equation": "sum_i p_i(D)*a_0(5,D+i)=0; coefficients in ascending powers of D",
        "order": MAX_ORDER,
        "polynomial_degree": int(max(polynomial.degree() for polynomial in polynomials)),
        "coefficients": [[str(coefficient) for coefficient in row] for row in integers],
        "domain_start": 1,
        "exceptional_indices": [],
        "required_initial_indices": list(range(1, MAX_ORDER + 1)),
        "initial_values": {str(index): str(five_vertex_coefficient(index)) for index in range(1, MAX_ORDER + 1)},
        "harmonic_basis": STATE.copy(),
        "remainder_dimensions": list(DROPS),
        "leading_domain_certificate": _leading_domain_certificate(integers[-1]),
        "proof_document": "docs/advanced/skeleton_risk_five_vertex_recurrence.rst",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def verify_operator(path: Path) -> dict[str, object]:
    """Reconstruct eight identities and prove their positive-integer domain."""
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
        raise ValueError("operator fails the exact rational telescoping identities")
    certificate_degrees = [int(poly.degree()) for poly in certificate]
    if max(certificate_degrees) > 2 * order + 2:
        raise ValueError("certificate domain requires an additional degree proof")
    denominator_checks = _check_parameter_denominators(certificate, order)
    leading = _leading_domain_certificate(integers[-1])
    if leading != document.get("leading_domain_certificate"):
        raise ValueError("inconsistent leading-domain certificate")
    initials = {str(index): str(five_vertex_coefficient(index)) for index in range(1, order + 1)}
    if initials != document.get("initial_values"):
        raise ValueError("initial values do not match the proved harmonic sum")
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
        "symbolic_harmonic_residuals": ["0"] * len(STATE),
        "certificate_polynomial_degrees": certificate_degrees,
        "certificate_denominators_nonzero_on_domain": True,
        "positive_parameter_denominator_checks": denominator_checks,
        "lower_boundary": "j^2 at j=0",
        "upper_boundary": "binom(N,j)=0 at j=N+1; N=D+order-1",
        "harmonic_basis": STATE.copy(),
        "remainder_dimensions": list(DROPS),
        "leading_domain_certificate": leading,
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def recurrence_coefficients(maximum_degree: int, operator: Path = DEFAULT_OPERATOR) -> dict[int, Fraction]:
    """Evaluate a previously certified artifact exactly without SageMath."""
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
    """Derive or verify an operator and optionally save its complete proof report."""
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
