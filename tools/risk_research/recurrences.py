"""Guess exact recurrences while keeping finite evidence distinct from proof."""

import json
import math
from collections.abc import Mapping
from fractions import Fraction
from functools import reduce
from pathlib import Path

import sympy as sp

from .exact import SCHEMA, Case, record_fraction


def sequence_from_manifest(path: Path, skeleton: int, vertices: int | None = None) -> tuple[int, dict[int, Fraction]]:
    """Select one coefficient sequence, rejecting conflicting reference values."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA or not isinstance(document.get("records"), list):
        raise ValueError("input must be a skeleton-risk-exact-v1 manifest with a records list")
    for record in document["records"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("status"), str)
            or record["status"] not in {"exact", "timeout", "error"}
        ):
            raise ValueError("malformed exact-data record")
        if record["status"] == "exact":
            try:
                Case(**record["case"])
                record_fraction(record["value"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed exact-data record: {error}") from error
    records = [
        record
        for record in document["records"]
        if record["status"] == "exact"
        and record["case"]["skeleton"] == skeleton
        and (vertices is None or record["case"]["vertices"] == vertices)
    ]
    sizes = {record["case"]["vertices"] for record in records}
    if len(sizes) != 1:
        raise ValueError("select exactly one vertex count with nonempty exact data")
    sequence: dict[int, Fraction] = {}
    for record in records:
        degree = record["case"]["degree"]
        value = record_fraction(record["value"])
        if degree in sequence and sequence[degree] != value:
            raise ValueError(f"conflicting exact values at degree {degree}")
        sequence[degree] = value
    return sizes.pop(), sequence


def _primitive_coefficients(vector: sp.Matrix) -> list[int]:
    denominator = sp.ilcm(*(entry.q for entry in vector))
    integers = [int(entry * denominator) for entry in vector]
    divisor = reduce(math.gcd, integers)
    integers = [value // divisor for value in integers]
    if next(value for value in reversed(integers) if value) < 0:
        integers = [-value for value in integers]
    return integers


def guess_recurrence(
    sequence: Mapping[int, Fraction], *, order: int, polynomial_degree: int, train_through: int
) -> dict[str, object]:
    """Fit a unique polynomial-coefficient recurrence and test unseen windows.

    Every returned candidate remains a conjecture. Passing finitely many exact
    residual checks does not establish a recurrence at all future degrees.
    """
    if type(order) is not int or order < 1 or type(polynomial_degree) is not int or polynomial_degree < 0:
        raise ValueError("order must be positive and polynomial_degree nonnegative integers")
    if type(train_through) is not int:
        raise ValueError("train_through must be an integer")
    if not sequence or any(type(index) is not int or index < 1 for index in sequence):
        raise ValueError("the sequence must have positive integer degree indices")
    values = {index: sp.Rational(value.numerator, value.denominator) for index, value in sequence.items()}
    windows = [index for index in sorted(values) if all(index + shift in values for shift in range(order + 1))]
    training = [index for index in windows if index + order <= train_through]
    held_out = [index for index in windows if index > train_through]
    unknowns = (order + 1) * (polynomial_degree + 1)
    if len(training) < unknowns - 1:
        raise ValueError(f"need at least {unknowns - 1} complete training windows")
    if not held_out:
        raise ValueError("need at least one complete held-out window using only unseen degrees")
    matrix = sp.Matrix(
        [
            [values[index + shift] * index**power for shift in range(order + 1) for power in range(polynomial_degree + 1)]
            for index in training
        ]
    )
    basis = matrix.nullspace()
    if len(basis) != 1:
        raise ValueError(f"expected a unique candidate up to scaling; nullspace dimension is {len(basis)}")
    integers = _primitive_coefficients(basis[0])
    width = polynomial_degree + 1
    coefficients = [integers[shift * width : (shift + 1) * width] for shift in range(order + 1)]
    if not any(coefficients[-1]):
        raise ValueError("candidate has lower order; retry with a smaller order")
    residuals = {
        index: sum(
            sum(coefficient * index**power for power, coefficient in enumerate(polynomial)) * values[index + shift]
            for shift, polynomial in enumerate(coefficients)
        )
        for index in held_out
    }
    symbol = sp.Symbol("D")
    leading = sp.Poly(sum(coefficient * symbol**power for power, coefficient in enumerate(coefficients[-1])), symbol)
    exceptions = sorted(int(root) for root in leading.ground_roots() if root.is_Integer and root >= min(values))
    required_initial_indices = sorted(
        {index for start in [min(values), *(root + 1 for root in exceptions)] for index in range(start, start + order)}
    )
    return {
        "schema": "skeleton-risk-recurrence-v1",
        "status": "conjecture" if all(value == 0 for value in residuals.values()) else "rejected",
        "proof_status": "finite_exact_checks_only",
        "equation": "sum_j p_j(D) a(D+j) = 0; coefficients in ascending powers of D",
        "order": order,
        "polynomial_degree": polynomial_degree,
        "coefficients": [[str(value) for value in polynomial] for polynomial in coefficients],
        "training_window_starts": training,
        "training_degrees": sorted({index + shift for index in training for shift in range(order + 1)}),
        "held_out_window_starts": held_out,
        "held_out_degrees": sorted({index + shift for index in held_out for shift in range(order + 1)}),
        "held_out_residuals": {str(index): str(value) for index, value in residuals.items()},
        "exceptional_indices": exceptions,
        "required_initial_indices": required_initial_indices,
        "missing_initial_indices": [index for index in required_initial_indices if index not in values],
        "initial_values": {str(index): str(values[index]) for index in required_initial_indices if index in values},
        "domain_start": min(values),
    }
