"""Derive and certify tetrahedron edge recurrences by exact state elimination.

Run with the optional Sage Python. Imports of Sage are deferred, so ordinary
research tooling can inspect this module without installing SageMath.
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

from .edge_closed import tetrahedron_edge_coefficients

SCHEMA = "skeleton-risk-edge-recurrence-v1"
STATE = ["H_N^2", "H_N^(2)", "H_N", "H_D", "F_N", "G_N", "1", "1/binom(N,D)"]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _state_system() -> tuple[Any, Any, Callable[[Any], Any], Callable[[Any], Any]]:
    sage = import_module("sage.all")
    ring = sage.PolynomialRing(sage.QQ, "D")
    field = ring.fraction_field()

    def transition(d: Any) -> Any:
        n = 2 * d + 3
        a = 1 / (n + 1) + 1 / (n + 2)
        b = 1 / (n + 1) ** 2 + 1 / (n + 2) ** 2
        result = sage.identity_matrix(field, 8)
        result[0, 2], result[0, 6] = 2 * a, a * a
        result[1, 6], result[2, 6], result[3, 6] = b, a, 1 / (d + 1)
        result[4, 5] = -1 / (n + 1) - 1 / (2 * (n + 2))
        result[4, 6] = b - 1 / ((n + 1) * (n + 2))
        result[5, 5] = field(1) / 4
        result[5, 6] = 1 / (2 * (n + 1)) + 1 / (n + 2)
        result[7, 7] = (d + 1) * (d + 4) / ((2 * d + 4) * (2 * d + 5))
        return result

    def output_row(d: Any) -> Any:
        n = 2 * d + 3
        den1 = d * (d + 1) * (d + 3) * (d + 4)
        den2 = (d + 1) * (d + 2) ** 2 * (d + 3)
        shift1 = 2 / (d + 1) + 1 / (d + 2) + 1 / (d + 3) + 1 / (d + 4)
        shift2 = 2 / (d + 1) + 2 / (d + 2) + 1 / (d + 3)
        c1 = (n + 1) * (d * (d + 3) - 1) / den1
        c2 = (n + 1) * ((d + 1) * (d + 2) - 1) / den2
        return sage.vector(
            field,
            [
                1 / (n * (n + 1) * (n + 2)),
                (n * n + 2) / (n * n * (n + 1) * (n + 2)),
                -2 * (n - 2) / (n**3 * (n + 2)) + 1 / den1 + 1 / den2,
                -2 / den1 - 2 / den2,
                -(n * n + n + 2) / (n * n * (n + 1) * (n + 2)),
                2 * (n - 2) / (n**3 * (n + 2)),
                -(n * n - 3 * n + 6) / (n**4 * (n + 2))
                + (n**3 - 2 * n * n + 3 * n + 6) / (n**4 * (n + 1) * (n + 2))
                + (c1 - shift1) / den1
                + (c2 - shift2) / den2,
                1 / (d * d * (d + 1) * (d + 3) ** 2 * (d + 4)) + 1 / ((d + 1) * (d + 2) ** 3 * (d + 3) ** 2),
            ],
        )

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


def _integer_rows(rows: list[Any], order: int) -> list[list[int]]:
    if len(rows) != order + 1 or any(not isinstance(row, list) or not row for row in rows):
        raise ValueError("invalid coefficient array")
    integers = []
    for row in rows:
        parsed = []
        for coefficient in row:
            if not isinstance(coefficient, str):
                raise ValueError("coefficients must be canonical decimal integer strings")
            integer = int(coefficient)
            if str(integer) != coefficient:
                raise ValueError("coefficients must be canonical decimal integer strings")
            parsed.append(integer)
        integers.append(parsed)
    return integers


def verify_operator(path: Path) -> dict[str, object]:
    """Prove a supplied operator using eight exact rational state identities.

    Metadata alone is not accepted as evidence: the coefficient vector is
    multiplied by the independently rebuilt state rows over QQ(D).
    """
    started = time.monotonic()
    document = json.loads(path.read_text(encoding="utf-8"))
    order = document["order"]
    if document.get("schema") != SCHEMA or type(order) is not int or not 1 <= order <= 8:
        raise ValueError("expected an edge recurrence of order 1..8")
    integers = _integer_rows(document["coefficients"], order)
    sage = import_module("sage.all")
    ring, field, system = _rows(order)
    polynomials = [ring(row) for row in integers]
    if not polynomials[-1] or max(p.degree() for p in polynomials) != document["polynomial_degree"]:
        raise ValueError("inconsistent order or degree metadata")
    residual = sage.vector(field, polynomials) * system
    if any(residual):
        raise ValueError("operator fails the exact rational state identity")
    if any(any(c < 0 for c in entry.denominator().list()) for entry in system.list()):
        raise ValueError("state denominator positivity needs an additional proof")
    exceptional = sorted(int(root) for root, _ in polynomials[-1].roots(sage.QQ) if root in sage.ZZ and root >= 1)
    if exceptional != document["exceptional_indices"] or document["domain_start"] != 1:
        raise ValueError("inconsistent domain or exceptional-index metadata")
    initial_values = tetrahedron_edge_coefficients(order)
    if document["initial_values"] != {str(d): str(value) for d, value in initial_values.items()}:
        raise ValueError("initial values do not match the proved representation")
    return {
        "status": "certified_by_state_identity",
        "schema": SCHEMA,
        "operator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "order": order,
        "polynomial_degree": document["polynomial_degree"],
        "domain_start": 1,
        "exceptional_indices": exceptional,
        "symbolic_state_residuals": ["0"] * 8,
        "state_denominators_positive_on_domain": True,
        "state_basis": STATE,
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def derive_operator() -> dict[str, object]:
    """Derive an order-seven operator by exact elimination, without data fits."""
    started = time.monotonic()
    sage = import_module("sage.all")
    ring, field, full_system = _rows(8)
    for order in range(1, 9):
        system = full_system[: order + 1, :]
        basis = system.left_kernel().basis()
        if basis:
            break
    else:
        raise RuntimeError("eight-dimensional state unexpectedly has no recurrence")
    coefficients = basis[0]
    denominator = sage.lcm([item.denominator() for item in coefficients])
    polynomials = [ring(item * denominator) for item in coefficients]
    divisor = ring.zero()
    for polynomial in polynomials:
        divisor = divisor.gcd(polynomial)
    polynomials = [item // divisor for item in polynomials]
    multiplier = sage.lcm([item.denominator() for item in polynomials])
    polynomials = [item * multiplier for item in polynomials]
    content = math.gcd(*(int(c) for item in polynomials for c in item.list()))
    polynomials = [item / content for item in polynomials]
    if polynomials[-1].leading_coefficient() < 0:
        polynomials = [-item for item in polynomials]
    if any(sage.vector(field, polynomials) * system):
        raise RuntimeError("derived operator failed its rational state identity")
    exceptional = sorted(int(root) for root, _ in polynomials[-1].roots(sage.QQ) if root in sage.ZZ and root >= 1)
    return {
        "schema": SCHEMA,
        "status": "certified_by_state_identity",
        "proof_status": "exact_rational_state_identity",
        "equation": "sum_j p_j(D)*a_1(4,D+j)=0; coefficients in ascending powers of D",
        "order": order,
        "polynomial_degree": int(max(item.degree() for item in polynomials)),
        "coefficients": [[str(c) for c in item.list()] for item in polynomials],
        "domain_start": 1,
        "exceptional_indices": exceptional,
        "required_initial_indices": list(range(1, order + 1)),
        "initial_values": {str(d): str(value) for d, value in tetrahedron_edge_coefficients(order).items()},
        "leading_coefficients_positive": all(c > 0 for c in polynomials[-1].list()),
        "state_basis": STATE,
        "symbolic_state_residuals": ["0"] * 8,
        "proof_document": "docs/advanced/skeleton_risk_extensions.rst",
        "elapsed_seconds": time.monotonic() - started,
        **_provenance(),
    }


def main(argv: list[str] | None = None) -> None:
    """Derive an operator or verify one, optionally saving its certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", type=Path)
    action.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.output:
        document = derive_operator()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        report = verify_operator(args.output)
    else:
        report = verify_operator(args.verify)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
