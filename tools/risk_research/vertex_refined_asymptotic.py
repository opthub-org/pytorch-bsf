"""Enclose fixed-M vertex risks with a single logarithmic degree error.

The proof in ``skeleton_risk_vertex_refined_asymptotics.rst`` applies for
every M>=5 and D>=256. All constants and interval endpoints are rational.
The existing squared-log M4 theorem and sharper M5 evaluator are unchanged.
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
from .vertex_general_asymptotic import MINIMUM_DEGREE, _integer, _majorants, _rational_ball, _target
from .vertex_general_asymptotic import remainder_bound as previous_remainder_bound

SOURCE_HASH_POLICY = "sha256-crlf-to-lf-v1"


def logarithmic_integral(power: int, decay: int) -> Fraction:
    """Integrate (1+log(x))**power/x**decay over x>=1 exactly."""
    _integer(power, 0, "power")
    _integer(decay, 2, "decay")
    return sum(
        (Fraction(factorial(power), factorial(power - j) * (decay - 1) ** (j + 1)) for j in range(power + 1)),
        Fraction(0),
    )


@dataclass(frozen=True)
class ErrorConstants:
    """Explicit diagonal, limiting and finite-degree error constants."""

    vertices: int
    diagonal: Fraction
    limiting: int
    remainder: Fraction
    logarithm_power: int = 1


def error_constants(vertices: int) -> ErrorConstants:
    """Return the proved single-log constants for every integer M>=5."""
    _integer(vertices, 5, "vertices")
    q = vertices - 1
    phi, coefficient, cross, difference = _majorants(q)
    limiting = 2 * q * cross[q - 1]
    r = q - 1
    cross_moment = cross[r] * (1 + logarithmic_integral(r - 2, r - 1))
    convolution = 2 ** (r + 1) * difference[r] + (2 + 3 ** (r - 1)) * cross[r] + 18 * cross_moment
    diagonal = 2 * phi[q] + cross[q] + q * convolution
    for k in range(2, q - 1):
        r = q - k
        coefficient_moment = coefficient[k] * (1 + logarithmic_integral(k - 1, k))
        convolution = 2 ** (r + 1) * difference[r] * coefficient_moment + 5 * 2**k * coefficient[k] * cross[r]
        diagonal += comb(q, k) * convolution
    diagonal += 8 * (1 + q**2 + limiting)
    scale = Fraction(vertices**2 * factorial(q), 2**q)
    remainder = scale * (96 * diagonal + (273 + q * (q + 1)) * limiting)
    return ErrorConstants(vertices, diagonal, limiting, remainder)


def remainder_bound(vertices: int, degree: int) -> Fraction:
    """Bound |a_0-C_M/D**(M+1)| by a rational single-log majorant."""
    _integer(degree, MINIMUM_DEGREE, "degree")
    constants = error_constants(vertices)
    logarithm = 1 + (degree - 1).bit_length()
    denominator: int = degree ** (vertices + 2)
    return constants.remainder * logarithm / denominator


def relative_error_threshold(vertices: int, target: Fraction = Fraction(1, 100), *, terms: int = 128) -> int:
    """Return a threshold valid at every larger degree, including log jumps."""
    if not isinstance(target, Fraction) or target <= 0:
        raise ValueError("target must be a strictly positive Fraction")
    constants = error_constants(vertices)
    lower, _ = leading_constant_enclosure(vertices, terms)
    if lower <= 0:
        raise ValueError("constant lower bound is zero; increase terms")
    exponent = 8
    while constants.remainder * (exponent + 2) > target * lower * 2**exponent:
        exponent += 1
    return 1 << exponent


def rational_enclosure(vertices: int, degree: int, *, terms: int = 128) -> tuple[Fraction, Fraction]:
    """Intersect old/new error bounds, with exact values below degree256."""
    _integer(vertices, 5, "vertices")
    _integer(degree, 1, "degree")
    _integer(terms, 1, "terms")
    if degree < MINIMUM_DEGREE:
        exact = vertex_coefficient(vertices, degree)
        return exact, exact
    lower, upper = leading_constant_enclosure(vertices, terms)
    radius = min(remainder_bound(vertices, degree), previous_remainder_bound(vertices, degree))
    normalization = degree ** (vertices + 1)
    return max(Fraction(0), lower / normalization - radius), upper / normalization + radius


def coefficient_ball(vertices: int, degree: int, *, terms: int = 128, bits: int = 128) -> Any:
    """Represent the exact enclosure as a real ball, restoring precision."""
    _integer(bits, 32, "bits")
    lower, upper = rational_enclosure(vertices, degree, terms=terms)
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        result = _rational_ball(lower, upper, flint)
    return result


def _sources() -> dict[str, bytes]:
    directory = Path(__file__).parent
    return {
        name: (directory / name).read_bytes()
        for name in (Path(__file__).name, "vertex_general_asymptotic.py", "vertex_general.py", "exact.py")
    }


def _source_hashes(sources: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() for name, content in sources.items()}


def main(argv: list[str] | None = None) -> int:
    """Write exact intervals and separated analytic and rounding budgets."""
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
        method = "asymptotic-intersection" if arguments.degree >= MINIMUM_DEGREE else "exact"
        record: dict[str, object] = {
            "schema": "skeleton-risk-refined-vertex-error-v1",
            "vertices": arguments.vertices,
            "degree": arguments.degree,
            "method": method,
            "domain_start": MINIMUM_DEGREE,
            "constant_terms": arguments.constant_terms,
            "constant_lower": fraction_record(lower),
            "constant_upper": fraction_record(upper),
            "logarithm_power": constants.logarithm_power,
            "diagonal_constant": fraction_record(constants.diagonal),
            "limiting_upper": str(constants.limiting),
            "remainder_constant": fraction_record(constants.remainder),
            "coefficient_lower": fraction_record(coefficient_lower),
            "coefficient_upper": fraction_record(coefficient_upper),
            "proof": "docs/advanced/skeleton_risk_vertex_refined_asymptotics.rst",
            "source_hash_policy": SOURCE_HASH_POLICY,
            "source_sha256": _source_hashes(sources),
            "scope": "each fixed M>=5; no simultaneous M,D limit",
        }
        if arguments.degree >= MINIMUM_DEGREE:
            refined = remainder_bound(arguments.vertices, arguments.degree)
            previous = previous_remainder_bound(arguments.vertices, arguments.degree)
            record.update(
                refined_analytic_error_bound=fraction_record(refined),
                previous_analytic_error_bound=fraction_record(previous),
                analytic_error_bound=fraction_record(min(refined, previous)),
            )
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
