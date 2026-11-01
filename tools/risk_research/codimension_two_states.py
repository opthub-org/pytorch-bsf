"""Proved eight-state evaluation of a_3(6,D), with optional Sage certificates.

The quadratic angular multiplicity reduces to three half-range moments of
the universal codimension-two kernel. Ordinary Python evaluates the state
using Fraction; Sage is loaded only to derive or verify a scalar operator.
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
from .exact import atomic_json

SCHEMA = "skeleton-risk-six-vertex-tetrahedral-face-recurrence-v1"
STATE = ["H_N^2-H_N^(2)", "H_N", "H_m", "F_N-H_N^(2)", "G_N", "J_m", "1/binom(2m+1,m)", "1"]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _transition_rows(m: Any) -> list[list[Any]]:
    """Rational matrix taking the state at m to its value at m+1."""
    zero = m - m
    one = zero + 1
    n = 2 * m + 1
    denominator = (n + 1) * (n + 2)
    increment = 1 / (n + 1) + 1 / (n + 2)
    square_increment = 1 / (n + 1) ** 2 + 1 / (n + 2) ** 2
    rows = [[one if i == j else zero for j in range(8)] for i in range(8)]
    rows[0][1], rows[0][7] = 2 * increment, increment**2 - square_increment
    rows[1][7], rows[2][7] = increment, 1 / (m + 1)
    rows[3][4], rows[3][7] = -(3 * n + 5) / (2 * denominator), -1 / denominator
    rows[4][4], rows[4][7] = one / 4, 1 / (2 * (n + 1)) + 1 / (n + 2)
    rows[5][6], rows[5][7] = -3 / (4 * (m + 1) ** 2), 1 / (m + 1) ** 2
    rows[6][6] = (m + 2) / (2 * (2 * m + 3))
    return rows


def _boundary_row(m: Any, offset: int) -> list[Any]:
    """Coefficients of g_(2m+1)(m-offset), for offset=0,1,2."""
    zero = m - m
    n, j = 2 * m + 1, m - offset
    delta = j * (j + 1) * (n - j) * (n - j + 1)
    c = (n + 1) * (j * (n - j) - 1) / delta
    shift = sum((1 / (m + k) for k in range(1, offset + 3)), zero)
    if offset == 0:
        shift += 1 / (m + 1)
    elif offset == 2:
        shift -= 1 / m
    ratio = zero + 1
    for k in range(1, offset + 1):
        ratio *= (m + k + 1) / (m - k + 1)
    reciprocal = ratio / (j**2 * (j + 1) * (n - j) ** 2 * (n - j + 1))
    return [zero, -1 / delta, 2 / delta, zero, zero, zero, -reciprocal, (shift - c) / delta]


def _output_coefficients(m: Any) -> list[Any]:
    """Return the rational output row for a_3(6,D), m=D+2 >= 3."""
    zero = m - m
    n = 2 * m + 1
    # S=sum(g), X=sum(j(N-j)g), E=sum((N/2-j)g), all j=1..m.
    s = [
        1 / (n * (n + 1) * (n + 2)),
        -2 * (n - 2) / (n**3 * (n + 2)),
        zero,
        -(n * n + n + 2) / (n**2 * (n + 1) * (n + 2)),
        2 * (n - 2) / (n**3 * (n + 2)),
        zero,
        zero,
        zero,
    ]
    x = [1 / (2 * (n + 2)), 2 / (n * (n + 2)), zero, 1 / (n + 2), -2 / (n * (n + 2)), zero, zero, zero]
    e = [zero, 1 / (2 * (m + 1) ** 2), -1 / (m + 1) ** 2, zero, zero, 1 / n, -1 / (2 * (m + 1) ** 3), zero]
    result = [(n * n + 63) * a / 8 - b / 2 - 4 * c for a, b, c in zip(s, x, e, strict=True)]
    for offset, weight in [(2, 1), (1, 3), (0, 6)]:
        result = [a - weight * b for a, b in zip(result, _boundary_row(m, offset), strict=True)]
    return [9000 * entry for entry in result]


def tetrahedral_face_coefficients(max_degree: int) -> dict[int, Fraction]:
    """Evaluate a_3(6,D), 1<=D<=max_degree, with eight rational states.

    Degrees one, two and three are inactive and the same nonsingular output
    formula returns zero. Work is linear in max_degree in rational operations.
    """
    _positive_integer(max_degree, "max_degree")
    state = [
        Fraction(value)
        for value in (2, Fraction(11, 6), 1, Fraction(-5, 6), Fraction(5, 6), Fraction(1, 4), Fraction(1, 3), 1)
    ]
    result = {}
    for m in range(1, max_degree + 3):
        if m >= 3:
            result[m - 2] = sum((a * b for a, b in zip(_output_coefficients(Fraction(m)), state, strict=True)), Fraction(0))
        if m < max_degree + 2:
            state = [
                sum((a * b for a, b in zip(row, state, strict=True)), Fraction(0)) for row in _transition_rows(Fraction(m))
            ]
    return result


def tetrahedral_face_coefficient(degree: int) -> Fraction:
    """Return the exact six-vertex tetrahedral-face coefficient at a degree."""
    _positive_integer(degree, "degree")
    return tetrahedral_face_coefficients(degree)[degree]


def _rows(order: int) -> tuple[Any, Any, Any]:
    sage = import_module("sage.all")
    ring = sage.PolynomialRing(sage.QQ, "D")
    field = ring.fraction_field()
    degree = field(ring.gen())
    accumulated = sage.identity_matrix(field, 8)
    rows = []
    for shift in range(order + 1):
        m = degree + shift + 2
        rows.append(sage.vector(field, _output_coefficients(m)) * accumulated)
        accumulated = sage.matrix(field, _transition_rows(m)) * accumulated
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
        raise ValueError("expected a six-vertex tetrahedral-face recurrence")
    order, degree = document.get("order"), document.get("polynomial_degree")
    if type(order) is not int or not 1 <= order <= 8 or type(degree) is not int or degree < 0:
        raise ValueError("expected integer order 1..8 and nonnegative polynomial degree")
    if type(document.get("domain_start")) is not int or document["domain_start"] != 1:
        raise ValueError("the tetrahedral-face recurrence domain must start at D=1")
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
    """Prove a supplied operator by eight rational identities over QQ(D)."""
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
    initial_values = {str(degree): str(value) for degree, value in tetrahedral_face_coefficients(order).items()}
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
        "state_basis": STATE.copy(),
        "state_parameter": "m=D+2; N=2m+1",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def derive_operator() -> dict[str, object]:
    """Eliminate the eight-state system, without fitting sequence values."""
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
        "equation": "sum_j p_j(D)*a_3(6,D+j)=0; coefficients in ascending powers of D",
        "order": order,
        "polynomial_degree": int(max(polynomial.degree() for polynomial in polynomials)),
        "coefficients": [[str(coefficient) for coefficient in polynomial.list()] for polynomial in polynomials],
        "domain_start": 1,
        "exceptional_indices": _exceptional_indices(polynomials[-1], sage),
        "required_initial_indices": list(range(1, order + 1)),
        "initial_values": {str(degree): str(value) for degree, value in tetrahedral_face_coefficients(order).items()},
        "state_basis": STATE.copy(),
        "state_parameter": "m=D+2; N=2m+1",
        "proof_document": "docs/advanced/codimension_two_states.rst",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def main(argv: list[str] | None = None) -> None:
    """Derive or verify an operator, optionally saving a proof report."""
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
