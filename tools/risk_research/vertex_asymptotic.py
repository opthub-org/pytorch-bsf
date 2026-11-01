"""Enclose the two-term tetrahedron vertex expansion with an explicit error.

The proof is in ``skeleton_risk_vertex_asymptotics.rst``. The rational
remainder majorant needs only the standard library. Evaluating the expansion
as a rigorous real ball additionally needs the optional python-flint package.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .exact import atomic_json, fraction_record

MINIMUM_DEGREE = 256
REMAINDER_CONSTANT = 10_000_000


def _degree(value: int) -> None:
    if type(value) is not int or value < MINIMUM_DEGREE:
        raise ValueError(f"degree must be an integer >= {MINIMUM_DEGREE}")


def tetrahedron_vertex_remainder_bound(degree: int) -> Fraction:
    """Return an exact upper bound on the two-term expansion error.

    The analytic bound is 10^7*(1+log(D))^2/D^7 for D>=256.
    Since log(D)<=ceil(log2(D)), an integer bit count gives a rational
    upper bound without rounded transcendental evaluation.
    """
    _degree(degree)
    logarithm_upper = (degree - 1).bit_length()
    return Fraction(REMAINDER_CONSTANT * (1 + logarithm_upper) ** 2, degree**7)


def tetrahedron_vertex_expansion_ball(degree: int, *, bits: int = 128) -> tuple[Any, Any]:
    """Return real balls for the expansion and the enclosed exact coefficient.

    python-flint is imported lazily. The second ball includes the proved
    analytic error as well as all arithmetic and transcendental rounding.
    """
    radius = tetrahedron_vertex_remainder_bound(degree)
    if type(bits) is not int or bits < 32:
        raise ValueError("bits must be an integer >= 32")
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        d = flint.arb(degree)
        logarithm = (d / 2).log() + flint.arb.const_euler()
        pi_squared = flint.arb.pi() ** 2
        expansion = 48 * pi_squared / d**5 + (576 * (logarithm**2 - logarithm - 1) - 456 * pi_squared) / d**6
        error = flint.arb(flint.fmpq(radius.numerator, radius.denominator))
        enclosure = expansion + error * flint.arb(0, 1)
    return expansion, enclosure


def main(argv: list[str] | None = None) -> int:
    """Write a source-bound real-ball enclosure without changing global precision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--bound-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    source = Path(__file__).read_bytes()
    try:
        bound = tetrahedron_vertex_remainder_bound(arguments.degree)
        record: dict[str, object] = {
            "schema": "skeleton-risk-vertex-asymptotic-v1",
            "degree": arguments.degree,
            "domain_start": MINIMUM_DEGREE,
            "error_upper_bound": fraction_record(bound),
            "proof": "docs/advanced/skeleton_risk_vertex_asymptotics.rst",
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
        if not arguments.bound_only:
            expansion, enclosure = tetrahedron_vertex_expansion_ball(arguments.degree, bits=arguments.bits)
            digits = arguments.bits * 30103 // 100000 + 5
            record.update(
                {
                    "bits": arguments.bits,
                    "python_flint": version("python-flint"),
                    "expansion": expansion.str(digits, more=True),
                    "coefficient_enclosure": enclosure.str(digits, more=True),
                }
            )
        if Path(__file__).read_bytes() != source:
            raise RuntimeError("asymptotic evaluator source changed during calculation")
        atomic_json(arguments.output, record)
    except (ValueError, ModuleNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
