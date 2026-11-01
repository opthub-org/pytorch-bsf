"""Enclose every fixed-M vertex coefficient with an explicit degree error.

The proof in ``skeleton_risk_vertex_general_asymptotics.rst`` supplies a
conservative error of order log(D)**(M-2)/D**(M+2), for every fixed M>=4.
All interval endpoints and analytic errors use exact fractions. Optional
python-flint adds a real-ball representation with arithmetic rounding.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from importlib.metadata import version
from math import comb, factorial
from pathlib import Path
from typing import Any

from .exact import atomic_json, fraction_record
from .vertex_general import leading_constant_enclosure, vertex_coefficient

MINIMUM_DEGREE = 256


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _majorants(power: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Construct the proved phi, b, e and first-difference majorants."""
    phi = [0] + [factorial(r) * r ** (r - 1) for r in range(1, power + 1)]
    coefficient = [0] + [2 ** (r + 1) * r * factorial(r) * (r + 1) ** (r - 1) for r in range(1, power + 1)]
    cross, difference = [0, 0], [0, 0]
    for r in range(2, power + 1):
        cross.append(sum(comb(r, a) * phi[a] * phi[r - a] for a in range(1, r)))
        difference.append(
            2 ** (r - 2) * sum(comb(r, a) * (coefficient[a] * phi[r - a] + phi[a] * coefficient[r - a]) for a in range(1, r))
        )
    return phi, coefficient, cross, difference


@dataclass(frozen=True)
class ErrorConstants:
    """Record explicit constants; no asymptotic or fitted values are stored."""

    vertices: int
    diagonal: int
    limiting: int
    remainder: Fraction
    logarithm_power: int


def error_constants(vertices: int) -> ErrorConstants:
    """Return K_q, an upper bound for c_q, and the finite-degree constant."""
    _integer(vertices, 4, "vertices")
    q = vertices - 1
    phi, coefficient, cross, difference = _majorants(q)
    limiting = 2 * q * cross[q - 1]
    r = q - 1
    diagonal = phi[q] + cross[q] + q * (2 ** (r + 1) * difference[r] + (20 + 3 ** (r - 1)) * cross[r])
    for k in range(2, q - 1):
        r = q - k
        diagonal += comb(q, k) * coefficient[k] * (2 ** (r + 2) * difference[r] + 5 * 2**k * cross[r])
    diagonal += 8 * (1 + q**2 + limiting)
    scale = Fraction(vertices**2 * factorial(q), 2**q)
    remainder = scale * (96 * diagonal + (273 + q * (q + 1)) * limiting)
    return ErrorConstants(vertices, diagonal, limiting, remainder, q - 1)


def remainder_bound(vertices: int, degree: int) -> Fraction:
    """Majorize |a_0-C_M/D**(M+1)| exactly, for M>=4 and D>=256."""
    _integer(degree, MINIMUM_DEGREE, "degree")
    constants = error_constants(vertices)
    logarithm = 1 + (degree - 1).bit_length()
    logarithmic_factor: int = logarithm**constants.logarithm_power
    denominator: int = degree ** (vertices + 2)
    return constants.remainder * logarithmic_factor / denominator


def relative_error_threshold(vertices: int, target: Fraction = Fraction(1, 100), *, terms: int = 128) -> int:
    """Certify the reported analytic bound is at most target*C_M/D**(M+1).

    The returned power-of-two threshold applies to every larger degree,
    including jumps of the integer logarithm majorant. A positive certified
    lower endpoint for C_M is required; increase ``terms`` if it is zero.
    """
    if not isinstance(target, Fraction) or target <= 0:
        raise ValueError("target must be a strictly positive Fraction")
    constants = error_constants(vertices)
    lower, _ = leading_constant_enclosure(vertices, terms)
    if lower <= 0:
        raise ValueError("constant lower bound is zero; increase terms")
    exponent = max(8, 2 * constants.logarithm_power)
    while constants.remainder * (exponent + 2) ** constants.logarithm_power > target * lower * 2**exponent:
        exponent += 1
    return 1 << exponent


def rational_enclosure(vertices: int, degree: int, *, terms: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose the coefficient; use the exact construction below degree256."""
    _integer(vertices, 4, "vertices")
    _integer(degree, 1, "degree")
    _integer(terms, 1, "terms")
    if degree < MINIMUM_DEGREE:
        exact = vertex_coefficient(vertices, degree)
        return exact, exact
    lower, upper = leading_constant_enclosure(vertices, terms)
    radius = remainder_bound(vertices, degree)
    normalization = degree ** (vertices + 1)
    return max(Fraction(0), lower / normalization - radius), upper / normalization + radius


def _rational_ball(lower: Fraction, upper: Fraction, flint: Any) -> Any:
    center, radius = (lower + upper) / 2, (upper - lower) / 2
    midpoint = flint.arb(flint.fmpq(center.numerator, center.denominator))
    error = flint.arb(flint.fmpq(radius.numerator, radius.denominator))
    return midpoint + error * flint.arb(0, 1)


def coefficient_ball(vertices: int, degree: int, *, terms: int = 128, bits: int = 128) -> Any:
    """Represent the exact rational enclosure as a ball, restoring precision."""
    _integer(bits, 32, "bits")
    lower, upper = rational_enclosure(vertices, degree, terms=terms)
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        result = _rational_ball(lower, upper, flint)
    return result


def _sources() -> dict[str, str]:
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (Path(__file__).name, "vertex_general.py", "exact.py")
    }


def _target(text: str) -> Fraction:
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("relative target must be a positive rational number") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("relative target must be a positive rational number")
    return result


def main(argv: list[str] | None = None) -> int:
    """Write source-bound exact intervals, with an optional rounded real ball."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--constant-terms", type=int, default=128)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--with-arb", action="store_true")
    parser.add_argument("--relative-target", type=_target)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    sources = _sources()
    try:
        _integer(arguments.bits, 32, "bits")
        constants = error_constants(arguments.vertices)
        _integer(arguments.degree, 1, "degree")
        lower, upper = leading_constant_enclosure(arguments.vertices, arguments.constant_terms)
        coefficient_lower, coefficient_upper = rational_enclosure(
            arguments.vertices, arguments.degree, terms=arguments.constant_terms
        )
        method = "asymptotic" if arguments.degree >= MINIMUM_DEGREE else "exact"
        record: dict[str, object] = {
            "schema": "skeleton-risk-general-vertex-error-v1",
            "vertices": arguments.vertices,
            "degree": arguments.degree,
            "method": method,
            "domain_start": MINIMUM_DEGREE,
            "constant_terms": arguments.constant_terms,
            "constant_lower": fraction_record(lower),
            "constant_upper": fraction_record(upper),
            "logarithm_power": constants.logarithm_power,
            "remainder_constant": fraction_record(constants.remainder),
            "coefficient_lower": fraction_record(coefficient_lower),
            "coefficient_upper": fraction_record(coefficient_upper),
            "proof": "docs/advanced/skeleton_risk_vertex_general_asymptotics.rst",
            "source_sha256": sources,
            "scope": "each fixed M>=4; no simultaneous M,D limit",
        }
        if method == "asymptotic":
            record["analytic_error_bound"] = fraction_record(remainder_bound(arguments.vertices, arguments.degree))
        if arguments.relative_target is not None:
            threshold = relative_error_threshold(arguments.vertices, arguments.relative_target, terms=arguments.constant_terms)
            record["relative_error_target"] = fraction_record(arguments.relative_target)
            record["relative_error_threshold"] = str(threshold)
        if arguments.with_arb:
            flint = import_module("flint")
            with flint.ctx.workprec(arguments.bits):
                ball = _rational_ball(coefficient_lower, coefficient_upper, flint)
                serialized = ball.str(arguments.bits * 30103 // 100000 + 5, more=True)
            record.update(bits=arguments.bits, python_flint=version("python-flint"), coefficient_ball=serialized)
        if _sources() != sources:
            raise RuntimeError("an enclosure source changed during calculation")
        atomic_json(arguments.output, record)
    except (ValueError, ModuleNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
