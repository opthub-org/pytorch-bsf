"""Certify an all-degree linear sufficient region for inductive risk.

For every integer D>=1 and M>=9720*D, the optimal inductive asymptotic
risk is strictly smaller than four fifths of the all-at-once risk.
The proof uses coefficientwise kernel bounds, without constructing any
degree-specific dimension polynomial or evaluating individual risks.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from .exact import atomic_json

THRESHOLD_MULTIPLIER = 9720
RATIO_BOUND = Fraction(4, 5)
SCHEMA = "skeleton-risk-uniform-dimension-tail-v1"


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def dimension_threshold(degree: int) -> int:
    """Return the proved sufficient vertex threshold 9720*degree."""
    _positive_integer(degree, "degree")
    return THRESHOLD_MULTIPLIER * degree


def coefficient_upper_bounds(degree: int) -> tuple[Fraction, ...]:
    """Bound a_(s-1)(M,D) for s=1,...,D and every M>=4D.

    These conservative positive bounds come directly from the scalar
    formula and apply simultaneously to all integer vertex counts in
    their domain. They are not computed dimension-polynomial norms.
    """
    _positive_integer(degree, "degree")
    common = degree**2 * 36**degree
    return tuple(
        Fraction(
            common * 4 ** (size - 1) * math.comb(degree - 1, size - 1) * (3 * degree) ** (2 * size),
            math.factorial(size) ** 2,
        )
        for size in range(1, degree + 1)
    )


def risk_upper_bound(degree: int) -> Fraction:
    """Bound optimal inductive risk for every integer M>=4D.

    The closed bound is U_D=D^3*36^D*5^(D-1)*((3D)^D/D!)^2.
    Its evaluation does not construct coefficients or mode sums.
    """
    _positive_integer(degree, "degree")
    return Fraction(
        degree**3 * 36**degree * 5 ** (degree - 1) * (3 * degree) ** (2 * degree),
        math.factorial(degree) ** 2,
    )


def _sources() -> dict[str, bytes]:
    return {name: Path(__file__).with_name(name).read_bytes() for name in ["uniform_dimension_tail.py", "exact.py"]}


def _decimal(value: int) -> str:
    """Serialize exact integers locally without changing Python's digit limit."""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    chunks = []
    while value:
        value, remainder = divmod(value, 10**500)
        chunks.append(f"{remainder:0500d}" if value else str(remainder))
    return sign + "".join(reversed(chunks))


def _fraction_record(value: Fraction) -> dict[str, str]:
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def certify_degree(degree: int) -> dict[str, object]:
    """Record the all-M tail theorem and its exact arithmetic at one D.

    The general proof is in skeleton_risk_uniform_dimension_tail.rst.
    The certificate checks its specialized arithmetic; a finite set of
    such records is not the proof of the all-degree theorem.
    """
    threshold = dimension_threshold(degree)
    factorial = math.factorial(degree)
    risk_bound = risk_upper_bound(degree)
    target_lower = Fraction(threshold**degree, factorial)
    ratio_margin = RATIO_BOUND * target_lower - risk_bound
    if degree**3 > 4 * 2**degree or degree**degree >= 3**degree * factorial or ratio_margin <= 0:
        raise ArithmeticError("the specialized proof inequalities did not hold")
    return {
        "schema": SCHEMA,
        "degree": degree,
        "from_vertices": str(threshold),
        "coefficient_bound_from_vertices": str(4 * degree),
        "decision": "inductive_wins",
        "proof_status": "specialization_of_proved_all_degree_all_vertex_tail",
        "proof": "docs/advanced/skeleton_risk_uniform_dimension_tail.rst",
        "risk_upper_bound": _fraction_record(risk_bound),
        "all_at_once_lower_bound": _fraction_record(target_lower),
        "strict_ratio_upper_bound": _fraction_record(RATIO_BOUND),
        "strict_margin_below_ratio_bound": _fraction_record(ratio_margin),
        "source_sha256": {name: hashlib.sha256(source).hexdigest() for name, source in _sources().items()},
    }


def verify_certificate(record: object) -> None:
    """Reject altered values, domains, fields and source fingerprints."""
    if not isinstance(record, dict) or type(record.get("degree")) is not int:
        raise ValueError("certificate must be an object with an integer degree")
    if record != certify_degree(record["degree"]):
        raise ValueError("certificate does not match the exact proof specialization and sources")


def main(argv: list[str] | None = None) -> int:
    """Emit an atomic exact certificate or verify a saved certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--degree", type=int)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.degree is not None:
            if arguments.output is None:
                parser.error("--degree requires --output")
            sources = _sources()
            report = certify_degree(arguments.degree)
            if _sources() != sources:
                raise RuntimeError("proof sources changed during certification")
            atomic_json(arguments.output, report)
        else:
            if arguments.output is not None:
                parser.error("--output is only supported with --degree")
            verify_certificate(json.loads(arguments.verify.read_text(encoding="utf-8")))
            print("verified uniform dimension-tail certificate")
    except (OSError, ValueError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
