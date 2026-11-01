"""Check exact algebra and denominator certificates for the uniform family.

SageMath is optional and loaded only after validating the source-face size.
The finite-dimensional construction theorem is proved in the companion text;
this verifier checks the emitted rational reductions over QQ(h)(j).
"""

import argparse
import hashlib
import json
import math
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from . import codimension_two_family as family
from .exact import atomic_json, peak_rss_bytes


def _positive_denominator(value: Any, lower: int, sage: Any) -> bool:
    denominator = value.denominator()
    variable = denominator.parent().gen()
    shifted = denominator(variable + lower)
    coefficients = shifted.list()
    return bool(coefficients and coefficients[0] > 0 and all(c >= 0 for c in coefficients))


def _universal_identities(sage: Any) -> list[str]:
    ring = sage.PolynomialRing(sage.QQ, names=("j", "C", "A", "K", "H", "A2", "K2", "Z", "Z2"))
    field = ring.fraction_field()
    j, c, a, k, h, a2, k2, z, z2 = field.gens()

    def prefix(aa: Any, kk: Any, aa2: Any, kk2: Any) -> Any:
        return (aa**2 + aa2 - z**2 - z2 + kk**2 + kk2) / 2 + aa * kk - h * (aa - z + kk)

    difference = prefix(a, k, a2, k2) - prefix(a - 1 / j, k + 1 / (c - j), a2 - 1 / j**2, k2 + 1 / (c - j) ** 2)
    expected = (1 / j - 1 / (c - j)) * (a + k + 1 / (c - j) - h)
    residuals = [difference - expected]
    n = c
    q = (n + 1) * (n + 2)
    ratio = (n - j + 1) * (n - j + 2) / q
    residuals.extend(
        [
            ratio / j - (1 / j - (2 * n + 3) / q + j / q),
            ratio / j**2 - (1 / j**2 - (2 * n + 3) / (q * j) + 1 / q),
            ratio / (j + 1) - ((n + 3) / ((n + 1) * (j + 1)) - 2 / (n + 1) + j / q),
            (n - j + 1) / ((n + 1) * (j + 1)) - ((n + 2) / ((n + 1) * (j + 1)) - 1 / (n + 1)),
        ]
    )
    if any(residuals):
        raise ArithmeticError("a universal algebra identity failed")
    return ["0"] * len(residuals)


def _check_inverse_polynomial(quotient: list[Any], half: Any, epsilon: int, jring: Any, sage: Any) -> None:
    """Use a separate dense solve to check polynomial inverse-binomial reduction."""
    n = 2 * half + epsilon
    j, field = jring.gen(), jring.base_ring()
    qpoly = jring(quotient)
    size = max(0, qpoly.degree())
    if not size:
        if family._combine(n, (qpoly[0], family._half_binomial_sum(half, epsilon))) != family._inverse_polynomial(
            quotient, half, epsilon
        ):
            raise ArithmeticError("constant inverse-binomial reduction differs")
        return
    columns = [jring((j + 1) ** (k + 1) - (n - j + 1) * j**k) for k in range(size)] + [jring(1)]
    matrix = sage.matrix(field, size + 1, size + 1, lambda row, col: columns[col][row])
    solution = matrix.solve_right(sage.vector(field, [qpoly[k] for k in range(size + 1)]))
    primitive = sum((solution[k] * j**k for k in range(size)), jring(0))
    independent = family._combine(
        n,
        (solution[-1], family._half_binomial_sum(half, epsilon)),
        ((half + 1) * primitive(half + 1), family._basis(n, 13)),
        (-primitive(1), family._basis(n, 14)),
    )
    if independent != family._inverse_polynomial(quotient, half, epsilon):
        raise ArithmeticError("independent inverse-binomial reduction differs")


def _check_boundaries(polynomial: Any, degree: Any, face_size: int) -> None:
    for t in range(face_size // 2 + 1):
        target = (-1) ** (face_size - 2) * math.comb(t + face_size - 2, face_size - 2)
        if polynomial(degree + t) != target:
            raise ArithmeticError("boundary polynomial identity failed")
    for t in range(1, face_size - 1):
        if polynomial(degree - t):
            raise ArithmeticError("zero-weight boundary identity failed")


def _check_output_assembly(row: list[Any], polynomial: Any, half: Any, epsilon: int, face_size: int, sage: Any) -> None:
    """Reassemble the row from an independently formed weight and boundary values."""
    n, degree = 2 * half + epsilon, half - face_size // 2
    expected = family._weighted_half_row(list(polynomial.list()), half, epsilon)
    for t in range(face_size // 2 + 1):
        offset = face_size // 2 - t
        j = degree + t
        x = j * (n - j)
        delta = x * (x + n + 1)
        lower_shift = 1 - offset
        first = sum((1 / (half + k) for k in range(1, lower_shift + 1)), n - n)
        if lower_shift < 0:
            first = -sum((1 / (half - k) for k in range(-lower_shift)), n - n)
        second = sum((1 / (half + k) for k in range(1, epsilon + offset + 2)), n - n)
        reciprocal = sage.prod((n - half + k) / (half - k + 1) for k in range(1, offset + 1))
        boundary = [n - n] * 15
        boundary[3], boundary[4] = -1 / delta, 2 / delta
        boundary[13] = -reciprocal / (x * delta)
        boundary[14] = (first + second) / delta - (n + 1) * (x - 1) / delta**2
        weight = (-1) ** (face_size - 2) * math.comb(t + face_size - 2, face_size - 2)
        expected = [a - weight * b for a, b in zip(expected, boundary, strict=True)]
    multiplier = math.comb(face_size + 2, face_size) ** 2 * 2 * face_size * (face_size + 1)
    if row != [multiplier * entry for entry in expected]:
        raise ArithmeticError("output assembly identity failed")


def verify_construction(face_size: int) -> dict[str, object]:
    """Certify emitted reductions, both-domain conventions and denominator signs."""
    family._integer(face_size, 2, "face_size")
    source = Path(__file__).read_bytes()
    family_source = Path(family.__file__).read_bytes()
    started = time.monotonic()
    sage = import_module("sage.all")
    ring = sage.PolynomialRing(sage.QQ, "h")
    field = ring.fraction_field()
    half = field(ring.gen())
    epsilon = 1 if face_size % 2 == 0 else 2
    n, degree = 2 * half + epsilon, half - face_size // 2
    jring = sage.PolynomialRing(field, "j")
    j = jring.gen()
    rational_field = jring.fraction_field()
    polynomial = jring(sage.prod(degree - j - 1 - k for k in range(face_size - 2)) / sage.factorial(face_size - 2))
    raw = [field(c) for c in polynomial.list()]
    kernels = [
        (raw, [(field(0), 1), (field(-1), 1), (n, 1), (n + 1, 1)]),
        (
            family._scale(family._mul(raw, [field(-1), n, field(-1)]), n + 1),
            [(field(0), 2), (field(-1), 2), (n, 2), (n + 1, 2)],
        ),
        (family._scale(raw, field(-1)), [(field(0), 2), (field(-1), 1), (n, 2), (n + 1, 1)]),
    ]
    partial_residuals = []
    polynomial_residuals = []
    for coefficients, poles in kernels:
        denominator = sage.prod((j - root) ** order for root, order in poles)
        quotient, fractions = family._partial(coefficients, poles)
        reconstructed = rational_field(jring(quotient))
        for index, order, coefficient in fractions:
            reconstructed += coefficient / rational_field((j - poles[index][0]) ** order)
        residual = rational_field(jring(coefficients)) / denominator - reconstructed
        if residual:
            raise ArithmeticError("partial-fraction identity failed")
        partial_residuals.append("0")
        primitive = jring(family._antidifference(quotient))
        if primitive - primitive(j - 1) != jring(quotient) or primitive(0):
            raise ArithmeticError("polynomial antidifference failed")
        polynomial_residuals.append("0")

    inverse_quotient = family._partial(kernels[-1][0], kernels[-1][1])[0]
    _check_inverse_polynomial(inverse_quotient, half, epsilon, jring, sage)
    _check_boundaries(polynomial, degree, face_size)
    row = family._output_row(face_size, half)
    _check_output_assembly(row, polynomial, half, epsilon, face_size, sage)
    minimum = face_size + face_size // 2
    if not all(_positive_denominator(entry, minimum, sage) for entry in row):
        raise ArithmeticError("output denominator sign certificate failed")
    for parity in (1, 2):
        if not all(_positive_denominator(entry, 1, sage) for r in family._transition_rows(half, parity) for entry in r):
            raise ArithmeticError("transition denominator sign certificate failed")
    residuals = _universal_identities(sage)
    if Path(__file__).read_bytes() != source or Path(family.__file__).read_bytes() != family_source:
        raise RuntimeError("construction source changed during verification")
    return {
        "schema": "skeleton-risk-codimension-two-construction-certificate-v1",
        "status": "certified_algebraic_construction",
        "face_size": face_size,
        "vertices": face_size + 2,
        "state_dimension": 15,
        "state_basis": list(family.STATE),
        "state_parameter": f"h=D+{face_size // 2}; N=2h+{epsilon}",
        "active_domain_start": face_size,
        "inactive_indices": list(range(1, face_size)),
        "partial_fraction_residuals": partial_residuals,
        "polynomial_antidifference_residuals": polynomial_residuals,
        "universal_identity_residuals": residuals,
        "independent_polynomial_gosper_check": True,
        "boundary_polynomial_identities": True,
        "output_assembly_residuals": ["0"] * 15,
        "output_denominators_positive_on_domain": True,
        "both_transition_denominators_positive_from_h": 1,
        "output_row": [str(entry) for entry in row],
        "family_source_sha256": hashlib.sha256(family_source).hexdigest(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "sage": str(import_module("sage.env").SAGE_VERSION),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "proof": "docs/advanced/codimension_two_family.rst",
    }


def main(argv: list[str] | None = None) -> int:
    """Write an exact construction certificate for a chosen source-face size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = verify_construction(arguments.face_size)
        atomic_json(arguments.output, report)
    except (ValueError, RuntimeError, ArithmeticError, ModuleNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
