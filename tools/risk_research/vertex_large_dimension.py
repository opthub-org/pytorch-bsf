"""Enclose the vertex leading constant as the vertex count tends to infinity.

The proof in ``skeleton_risk_vertex_large_dimension.rst`` gives a relative
error at most 25/(M-2), for every integer M>=1026. Exact rational elementary
constant enclosures need no optional dependency. Optional Arb integration
uses a separate positive integral and a certified omitted tail.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from importlib import import_module
from importlib.metadata import version
from math import isqrt
from pathlib import Path
from typing import Any

from .exact import atomic_json, fraction_record

MINIMUM_VERTICES = 1026


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _atan_enclosure(denominator: int, bits: int) -> tuple[Fraction, Fraction]:
    partial, k = Fraction(0), 0
    while True:
        term = Fraction(1, (2 * k + 1) * denominator ** (2 * k + 1))
        if term <= Fraction(1, 1 << bits):
            other = partial + (-1) ** k * term
            return min(partial, other), max(partial, other)
        partial += (-1) ** k * term
        k += 1


def elementary_enclosures(bits: int = 128) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """Return exact enclosures of log(2) and pi from convergent series."""
    _integer(bits, 32, "bits")
    partial, count = Fraction(0), 0
    while True:
        tail = Fraction(9, 4 * (2 * count + 1) * 3 ** (2 * count + 1))
        if tail <= Fraction(1, 1 << bits):
            break
        partial += Fraction(2, (2 * count + 1) * 3 ** (2 * count + 1))
        count += 1
    five_lower, five_upper = _atan_enclosure(5, bits + 5)
    other_lower, other_upper = _atan_enclosure(239, bits + 5)
    return (partial, partial + tail), (16 * five_lower - 4 * other_upper, 16 * five_upper - 4 * other_lower)


def _sqrt_enclosure(value: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    scale = 1 << bits
    numerator = value.numerator * scale**2
    floor = isqrt(numerator // value.denominator)
    lower = Fraction(floor, scale)
    upper = lower if floor**2 * value.denominator == numerator else Fraction(floor + 1, scale)
    return lower, upper


def leading_normalized_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose sqrt(M-1)/(4*sqrt(pi*log(2)*(log(2)-1/2))).

    The normalization removes M**2*(M-1)!*(2*log(2))**(M-1), so reports
    remain small even when the unnormalized constant has many digits.
    """
    _integer(vertices, 4, "vertices")
    logarithm, pi = elementary_enclosures(bits)
    lower_denominator = 16 * pi[0] * logarithm[0] * (logarithm[0] - Fraction(1, 2))
    upper_denominator = 16 * pi[1] * logarithm[1] * (logarithm[1] - Fraction(1, 2))
    return (
        _sqrt_enclosure(Fraction(vertices - 1, 1) / upper_denominator, bits)[0],
        _sqrt_enclosure(Fraction(vertices - 1, 1) / lower_denominator, bits)[1],
    )


def relative_remainder_bound(vertices: int) -> Fraction:
    """Bound |C_M/B_M-1|, with B_M the displayed large-M leading term.

    The simple 25/(M-2) bound applies for M>=1026. Below that start, the
    same proof supplies a conservative elementary bound, possibly >1.
    """
    _integer(vertices, 4, "vertices")
    n = vertices - 2
    if vertices >= MINIMUM_VERTICES:
        return Fraction(25, n)
    root_lower, root_upper = _sqrt_enclosure(Fraction(n), 32)
    natural_error = (
        Fraction(21, n) + 4 * root_upper * Fraction(79, 80) ** (n - 2) + Fraction(16, 3) / root_lower * Fraction(64, 65) ** n
    )
    return (1 + Fraction(1, 2 * n)) * natural_error + Fraction(1, 2 * n)


def normalized_constant_enclosure(vertices: int, *, bits: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose C_M/(M**2*(M-1)!*(2*log(2))**(M-1)) exactly."""
    lower, upper = leading_normalized_enclosure(vertices, bits=bits)
    error = relative_remainder_bound(vertices)
    return max(Fraction(0), 1 - error) * lower, (1 + error) * upper


def relative_error_threshold(target: Fraction = Fraction(1, 100)) -> int:
    """Return a threshold valid for every larger integer vertex count."""
    if not isinstance(target, Fraction) or target <= 0:
        raise ValueError("target must be a strictly positive Fraction")
    required = (25 * target.denominator + target.numerator - 1) // target.numerator
    return max(MINIMUM_VERTICES, required + 2)


def _rational_ball(lower: Fraction, upper: Fraction, flint: Any) -> Any:
    center, radius = (lower + upper) / 2, (upper - lower) / 2
    return flint.arb(flint.fmpq(center.numerator, center.denominator)) + flint.arb(
        flint.fmpq(radius.numerator, radius.denominator)
    ) * flint.arb(0, 1)


def constant_ball(vertices: int, *, bits: int = 128) -> Any:
    """Return an Arb enclosure of C_M without serializing giant integers."""
    _integer(bits, 32, "bits")
    lower, upper = normalized_constant_enclosure(vertices, bits=bits)
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        scale = vertices**2 * flint.arb(vertices).gamma() * (2 * flint.arb(2).log()) ** (vertices - 1)
        result = scale * _rational_ball(lower, upper, flint)
        if not result.is_finite():
            raise RuntimeError("constant enclosure is not finite")
    return result


def integral_normalized_ball(vertices: int, *, bits: int = 128) -> Any:
    """Independently enclose the normalized constant by positive integration.

    Integrate an analytic part of the original one-dimensional integral;
    bound its omitted endpoint interval by the exactly known second moment.
    This path does not use the large-M remainder inequality.
    """
    _integer(vertices, 4, "vertices")
    _integer(bits, 32, "bits")
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        logarithm, pi = flint.arb(2).log(), flint.arb.pi()
        n, q = vertices - 2, vertices - 1
        cutoff = flint.arb(1) / 2 if vertices >= MINIMUM_VERTICES else pi - flint.arb(1) / 65536

        def integrand(x: Any, analytic: bool) -> Any:
            cosine = (x / 2).cos()
            h = 4 * cosine**2 * (2 * cosine).log(analytic=analytic) + x * x.sin()
            return (h / (4 * logarithm)) ** n / cosine**2

        integral = flint.acb.integral(integrand, 0, cutoff, eval_limit=100000)
        cosine = (cutoff / 2).cos()
        rho = (4 * cosine**2 * (2 * cosine).log() + cutoff * cutoff.sin()) / (4 * logarithm)
        if not 0 < rho < 1:
            raise RuntimeError("integral cutoff is outside the positive decreasing kernel")
        tail = 12 * rho ** (n - 2)
        result = q / (4 * pi * logarithm) * (integral.real + tail / 2 * (1 + flint.arb(0, 1)))
        if not integral.is_finite() or not integral.imag.contains(0) or not result.is_finite():
            raise RuntimeError("integral enclosure did not converge")
    return result


def _sources() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in (Path(__file__).name, "exact.py")}


def _target(text: str) -> Fraction:
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("relative target must be a positive rational number") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("relative target must be a positive rational number")
    return result


def main(argv: list[str] | None = None) -> int:
    """Write source-bound rational intervals and optional independent balls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--relative-target", type=_target)
    parser.add_argument("--with-arb", action="store_true")
    parser.add_argument("--integral-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    sources = _sources()
    try:
        lower, upper = normalized_constant_enclosure(arguments.vertices, bits=arguments.bits)
        leading_lower, leading_upper = leading_normalized_enclosure(arguments.vertices, bits=arguments.bits)
        record: dict[str, object] = {
            "schema": "skeleton-risk-large-vertex-constant-v1",
            "vertices": arguments.vertices,
            "bits": arguments.bits,
            "normalization": "C_M/(M^2*(M-1)!*(2*log(2))^(M-1))",
            "normalized_lower": fraction_record(lower),
            "normalized_upper": fraction_record(upper),
            "leading_normalized_lower": fraction_record(leading_lower),
            "leading_normalized_upper": fraction_record(leading_upper),
            "relative_error_bound": fraction_record(relative_remainder_bound(arguments.vertices)),
            "simple_remainder_start_vertices": MINIMUM_VERTICES,
            "proof": "docs/advanced/skeleton_risk_vertex_large_dimension.rst",
            "source_sha256": sources,
            "scope": "leading constant as M tends to infinity; no joint M,D limit",
        }
        if arguments.relative_target is not None:
            record["relative_error_target"] = fraction_record(arguments.relative_target)
            record["relative_error_threshold"] = str(relative_error_threshold(arguments.relative_target))
        if arguments.with_arb or arguments.integral_check:
            flint = import_module("flint")
            with flint.ctx.workprec(arguments.bits):
                digits = arguments.bits * 30103 // 100000 + 5
                if arguments.with_arb:
                    record["constant_ball"] = constant_ball(arguments.vertices, bits=arguments.bits).str(digits, more=True)
                if arguments.integral_check:
                    ball = integral_normalized_ball(arguments.vertices, bits=arguments.bits)
                    if not ball.overlaps(_rational_ball(lower, upper, flint)):
                        raise RuntimeError("independent integral contradicts the rational enclosure")
                    record["normalized_integral_ball"] = ball.str(digits, more=True)
            record["python_flint"] = version("python-flint")
        if _sources() != sources:
            raise RuntimeError("a constant enclosure source changed during calculation")
        atomic_json(arguments.output, record)
    except (ValueError, ModuleNotFoundError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
