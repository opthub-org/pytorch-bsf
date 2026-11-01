"""Derive and certify five-vertex triangular-face recurrences over QQ(D).

The eight-dimensional state is rebuilt explicitly. SageMath is imported only
when deriving or proving an operator; ordinary exact arithmetic can evaluate
the output row and check examples without the optional symbolic environment.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from .edge_recurrence import _integer_rows
from .exact import atomic_json
from .triangle_faces import triangle_face_coefficients

SCHEMA = "skeleton-risk-triangle-recurrence-v1"
STATE = ["H_(2n)^2-H_(2n)^(2)", "H_(2n)", "H_n", "F_(2n)-H_(2n)^(2)", "G_(2n)", "J_n", "1/binom(2n,n)", "1"]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _output_coefficients(n: Any) -> list[Any]:
    """Return the rational output row for a_2(5,D), n=D+2.

    Pass an exact Fraction or an element of QQ(D), rather than a float.
    The coefficients result from collecting the proved finite-state formula.
    """
    denominator = (n - 2) * (n - 1) * (n + 1) ** 2 * (n + 2) * (n + 3)
    a = -3 / (4 * n * (n + 1) * (2 * n + 1))
    c = (2 * n**7 + 9 * n**6 - 22 * n**5 - 90 * n**4 + 56 * n**3 + 177 * n**2 - 48 * n - 36) / (4 * n**3 * denominator)
    d = -(n**6 + 3 * n**5 - 14 * n**4 - 33 * n**3 + 43 * n**2 + 60 * n - 36) / (n**2 * denominator)
    e = 1 / (2 * n)
    f = (2 * n**3 + 9 * n**2 + 4 * n + 3) / (4 * n**2 * (n + 1) * (2 * n + 1))
    g = -3 * (n - 1) / (4 * n**3 * (n + 1))
    w = -(3 * n**9 + n**8 - 36 * n**7 + 24 * n**6 + 129 * n**5 - 177 * n**4 + 24 * n**3 + 272 * n**2 - 288 * n + 144) / (
        4 * n**4 * (n - 2) ** 2 * (n - 1) ** 2 * (n + 1) ** 2 * (n + 2) * (n + 3)
    )
    z = (
        n**10 + n**9 - 14 * n**8 - 12 * n**7 + 87 * n**6 - 21 * n**5 - 226 * n**4 + 344 * n**3 + 224 * n**2 - 528 * n + 288
    ) / (4 * n**4 * (n - 2) ** 2 * (n - 1) ** 2 * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3))
    return [2400 * coefficient for coefficient in (a, c, d, f - e / 2, g, e, w, z)]


def _state_system() -> tuple[Any, Any, Callable[[Any], Any], Callable[[Any], Any]]:
    sage = import_module("sage.all")
    ring = sage.PolynomialRing(sage.QQ, "D")
    field = ring.fraction_field()

    def transition(degree: Any) -> Any:
        n = degree + 2
        denominator = (2 * n + 1) * (2 * n + 2)
        harmonic_increment = 1 / (2 * n + 1) + 1 / (2 * n + 2)
        square_increment = 1 / (2 * n + 1) ** 2 + 1 / (2 * n + 2) ** 2
        result = sage.identity_matrix(field, 8)
        result[0, 1], result[0, 7] = 2 * harmonic_increment, harmonic_increment**2 - square_increment
        result[1, 7], result[2, 7] = harmonic_increment, 1 / (n + 1)
        result[3, 4], result[3, 7] = -(6 * n + 5) / (2 * denominator), -1 / denominator
        result[4, 4], result[4, 7] = field(1) / 4, 1 / (2 * (2 * n + 1)) + 1 / (2 * n + 2)
        result[5, 6], result[5, 7] = -((3 * n + 2) ** 2) / (2 * n**2 * denominator), 1 / (n + 1) ** 2
        result[6, 6] = (n + 1) / (2 * (2 * n + 1))
        return result

    def output_row(degree: Any) -> Any:
        return sage.vector(field, _output_coefficients(degree + 2))

    return ring, field, transition, output_row


def _rows(order: int) -> tuple[Any, Any, Any]:
    sage = import_module("sage.all")
    ring, field, transition, output_row = _state_system()
    degree = field(ring.gen())
    accumulated = sage.identity_matrix(field, 8)
    rows = []
    for shift in range(order + 1):
        rows.append(output_row(degree + shift) * accumulated)
        accumulated = transition(degree + shift) * accumulated
    return ring, field, sage.matrix(field, rows)


def _provenance() -> dict[str, str]:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("state verifier source changed during execution; rerun with stable sources")
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": SOURCE_SHA256,
        "python": platform.python_version(),
        "sage": str(import_module("sage.env").SAGE_VERSION),
    }


def _read_operator(source: bytes) -> tuple[dict[str, Any], list[list[int]]]:
    document = json.loads(source)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("expected a triangular-face recurrence")
    order, degree = document.get("order"), document.get("polynomial_degree")
    if type(order) is not int or not 1 <= order <= 8 or type(degree) is not int or degree < 0:
        raise ValueError("expected integer order 1..8 and nonnegative polynomial degree")
    if type(document.get("domain_start")) is not int or document["domain_start"] != 1:
        raise ValueError("the triangular-face recurrence domain must start at D=1")
    exceptions = document.get("exceptional_indices")
    if not isinstance(exceptions, list) or any(type(index) is not int or index < 1 for index in exceptions):
        raise ValueError("exceptional indices must be positive integers")
    if document.get("state_basis") != STATE:
        raise ValueError("inconsistent state basis")
    coefficients = document.get("coefficients")
    if not isinstance(coefficients, list):
        raise ValueError("invalid coefficient array")
    return document, _integer_rows(coefficients, order)


def _exceptional_indices(polynomial: Any, sage: Any) -> list[int]:
    return sorted(int(root) for root, _ in polynomial.roots(sage.QQ) if root in sage.ZZ and root >= 1)


def verify_operator(path: Path) -> dict[str, object]:
    """Certify the supplied operator by eight rational identities over QQ(D).

    Denominator positivity and every positive-integer leading-coefficient
    zero are checked separately. Initial values come from the independently
    evaluated finite-state formula, including inactive degrees one and two.
    """
    started = time.monotonic()
    original = path.read_bytes()
    document, integers = _read_operator(original)
    order = document["order"]
    sage = import_module("sage.all")
    ring, field, system = _rows(order)
    polynomials = [ring(row) for row in integers]
    if not polynomials[-1] or max(polynomial.degree() for polynomial in polynomials) != document["polynomial_degree"]:
        raise ValueError("inconsistent order or degree metadata")
    if any(sage.vector(field, polynomials) * system):
        raise ValueError("operator fails the exact rational state identity")
    if any(any(coefficient < 0 for coefficient in entry.denominator().list()) for entry in system.list()):
        raise ValueError("state denominator positivity needs an additional proof")
    exceptional = _exceptional_indices(polynomials[-1], sage)
    if exceptional != document["exceptional_indices"]:
        raise ValueError("inconsistent exceptional-index metadata")
    initial_values = {str(degree): str(value) for degree, value in triangle_face_coefficients(order).items()}
    if document.get("initial_values") != initial_values:
        raise ValueError("initial values do not match the proved representation")
    if path.read_bytes() != original:
        raise RuntimeError("operator changed during verification")
    return {
        "status": "certified_by_state_identity",
        "schema": SCHEMA,
        "operator_sha256": hashlib.sha256(original).hexdigest(),
        "order": order,
        "polynomial_degree": document["polynomial_degree"],
        "domain_start": 1,
        "exceptional_indices": exceptional,
        "symbolic_state_residuals": ["0"] * 8,
        "state_denominators_positive_on_domain": True,
        "state_basis": STATE,
        "state_parameter": "n=D+2",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def derive_operator() -> dict[str, object]:
    """Eliminate the eight-state system without fitting sequence values."""
    started = time.monotonic()
    sage = import_module("sage.all")
    ring, field, system = _rows(8)
    basis = system.left_kernel().basis()
    if not basis:
        raise RuntimeError("eight-dimensional state unexpectedly has no recurrence")
    coefficients = basis[0]
    denominator = sage.lcm([entry.denominator() for entry in coefficients])
    polynomials = [ring(entry * denominator) for entry in coefficients]
    divisor = sage.gcd(polynomials)
    polynomials = [polynomial // divisor for polynomial in polynomials]
    multiplier = sage.lcm([coefficient.denominator() for polynomial in polynomials for coefficient in polynomial.list()])
    polynomials = [ring(polynomial * multiplier) for polynomial in polynomials]
    content = math.gcd(*(int(coefficient) for polynomial in polynomials for coefficient in polynomial.list()))
    polynomials = [ring(polynomial / content) for polynomial in polynomials]
    while not polynomials[-1]:
        polynomials.pop()
    if polynomials[-1].leading_coefficient() < 0:
        polynomials = [-polynomial for polynomial in polynomials]
    order = len(polynomials) - 1
    if any(sage.vector(field, polynomials) * system[: order + 1, :]):
        raise RuntimeError("derived operator failed its rational state identity")
    return {
        "schema": SCHEMA,
        "proof_status": "exact_rational_state_identity",
        "equation": "sum_j p_j(D)*a_2(5,D+j)=0; coefficients in ascending powers of D",
        "order": order,
        "polynomial_degree": int(max(polynomial.degree() for polynomial in polynomials)),
        "coefficients": [[str(coefficient) for coefficient in polynomial.list()] for polynomial in polynomials],
        "domain_start": 1,
        "exceptional_indices": _exceptional_indices(polynomials[-1], sage),
        "required_initial_indices": list(range(1, order + 1)),
        "initial_values": {str(degree): str(value) for degree, value in triangle_face_coefficients(order).items()},
        "state_basis": STATE,
        "state_parameter": "n=D+2",
        "proof_document": "docs/advanced/skeleton_risk_extensions.rst",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def main(argv: list[str] | None = None) -> None:
    """Derive an operator or verify one, optionally saving a proof report."""
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
