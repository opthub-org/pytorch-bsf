"""Enclose the five-vertex vertex leading term with an explicit error.

The proof is in ``skeleton_risk_five_vertex_asymptotics.rst``. Its rational
error majorant needs only Python's standard library. Evaluating the leading
term as a rigorous real ball additionally needs optional python-flint.
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
REMAINDER_CONSTANT = 3_000_000


def five_vertex_remainder_bound(degree: int) -> Fraction:
    """Return an exact majorant of the proved leading-term error.

    The analytic error is at most 3*10^6*(1+log(D))/D^7 for D>=256.
    An integer bit count bounds the logarithm without rounded arithmetic.
    """
    if type(degree) is not int or degree < MINIMUM_DEGREE:
        raise ValueError(f"degree must be an integer >= {MINIMUM_DEGREE}")
    logarithm_upper = (degree - 1).bit_length()
    return Fraction(REMAINDER_CONSTANT * (1 + logarithm_upper), degree**7)


def five_vertex_leading_ball(degree: int, *, bits: int = 128) -> tuple[Any, Any]:
    """Return balls for the leading term and the enclosed exact coefficient.

    The second ball includes both arithmetic rounding and the proved
    analytic error. python-flint is imported lazily; its ambient precision
    is restored on leaving the local precision context.
    """
    radius = five_vertex_remainder_bound(degree)
    if type(bits) is not int or bits < 32:
        raise ValueError("bits must be an integer >= 32")
    flint = import_module("flint")
    with flint.ctx.workprec(bits):
        d = flint.arb(degree)
        constant = flint.arb.pi() ** 2 / 6 - flint.arb(3).zeta()
        leading = 7200 * constant / d**6
        error = flint.arb(flint.fmpq(radius.numerator, radius.denominator))
        enclosure = leading + error * flint.arb(0, 1)
    return leading, enclosure


def main(argv: list[str] | None = None) -> int:
    """Write a source-bound enclosure while preserving ambient precision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--bound-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    source = Path(__file__).read_bytes()
    try:
        bound = five_vertex_remainder_bound(arguments.degree)
        record: dict[str, object] = {
            "schema": "skeleton-risk-five-vertex-asymptotic-v1",
            "degree": arguments.degree,
            "domain_start": MINIMUM_DEGREE,
            "error_upper_bound": fraction_record(bound),
            "proof": "docs/advanced/skeleton_risk_five_vertex_asymptotics.rst",
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
        if not arguments.bound_only:
            leading, enclosure = five_vertex_leading_ball(arguments.degree, bits=arguments.bits)
            digits = arguments.bits * 30103 // 100000 + 5
            record.update(
                {
                    "bits": arguments.bits,
                    "python_flint": version("python-flint"),
                    "leading_term": leading.str(digits, more=True),
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
