"""Evaluate nonvertex risks by positive univariate transformed diagonals.

The all-degree Laguerre diagonalization is proved in
``skeleton_risk_codimension_three_exact.rst``.  At codimension three the
diagonal is an explicit positive harmonic expression.  The remaining
angular/binomial sum has quadratically many terms in the degree; this is
not a bounded-length classical-special-function formula.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

from .exact import atomic_json, fraction_record
from .vertex_general import leading_constant_enclosure as vertex_leading_constant_enclosure
from .vertex_general import transformed_diagonals

HASH_POLICY = "sha256-crlf-to-lf-v1"


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def harmonic_diagonals(count: int) -> list[Fraction]:
    """Return delta_3(j)=6G_j, 0<=j<count, with strictly positive G_j."""
    _integer(count, 0, "count")
    harmonic = Fraction(0)
    harmonic2 = Fraction(0)
    result = []
    for j in range(count):
        if j:
            harmonic += Fraction(1, j)
            harmonic2 += Fraction(1, j * j)
        value = (
            (harmonic * harmonic - Fraction(j, j + 2) * harmonic) / ((j + 1) * (j + 2) * (j + 3))
            + harmonic2 / ((j + 1) * (j + 3))
            + Fraction(2, (j + 1) ** 2 * (j + 2) ** 2 * (j + 3))
        )
        result.append(6 * value)
    return result


def _diagonals(outside: int, count: int) -> list[Fraction]:
    if outside == 0:
        return [Fraction(1)] * count
    if outside == 3:
        return harmonic_diagonals(count)
    return transformed_diagonals(outside, count)


def _average(diagonals: list[Fraction], length: int, beta: int) -> Fraction:
    weight = 1
    total = Fraction(0)
    for j in range(length + 1):
        total += weight * diagonals[j]
        weight = weight * (length - j) * (length + beta - j) // (j + 1) ** 2
    return total / comb(2 * length + beta, length)


def angular_mode_ratio(face_size: int, outside: int, degree: int, angular: int) -> Fraction:
    """Return the propagated/source squared-norm ratio for one angular mode.

    The active domain is face_size>=2, outside>=0, degree>=face_size and
    0<=angular<=degree-face_size.  All modes of one angular degree have
    this same ratio.  The source normalization cancels.
    """
    _integer(face_size, 2, "face_size")
    _integer(outside, 0, "outside")
    _integer(degree, face_size, "degree")
    _integer(angular, 0, "angular")
    if angular > degree - face_size:
        raise ValueError("angular must be <= degree-face_size")
    length = degree - face_size - angular
    beta = 3 * face_size + 2 * angular - 1
    scale = Fraction(
        prod(range(face_size, face_size + outside)),
        prod(range(2 * degree + face_size, 2 * degree + face_size + outside)),
    )
    return scale * _average(_diagonals(outside, length + 1), length, beta)


def nonvertex_coefficient(vertices: int, degree: int, face_size: int) -> Fraction:
    """Return a_(face_size-1)(vertices,degree) by positive diagonal averages.

    Integers vertices>=face_size>=2 and degree>=1 are supported.  The
    inactive prefix degree<face_size returns zero.  No optional algebra
    backend or bivariate outside kernel is used.
    """
    _integer(face_size, 2, "face_size")
    _integer(vertices, face_size, "vertices")
    _integer(degree, 1, "degree")
    if degree < face_size:
        return Fraction(0)
    outside = vertices - face_size
    if outside == 0:
        return Fraction(comb(degree - 1, face_size - 1))
    count = degree - face_size + 1
    diagonals = _diagonals(outside, count)
    total = Fraction(0)
    for angular in range(count):
        length = degree - face_size - angular
        beta = 3 * face_size + 2 * angular - 1
        total += comb(angular + face_size - 2, face_size - 2) * _average(diagonals, length, beta)
    scale = Fraction(
        comb(vertices, face_size) ** 2 * prod(range(face_size, vertices)),
        prod(range(2 * degree + face_size, 2 * degree + vertices)),
    )
    return scale * total


def codimension_three_coefficient(face_size: int, degree: int) -> Fraction:
    """Return a_(face_size-1)(face_size+3,degree) by its harmonic double sum."""
    _integer(face_size, 2, "face_size")
    return nonvertex_coefficient(face_size + 3, degree, face_size)


def nonvertex_leading_constant_enclosure(face_size: int, outside: int, terms: int = 128) -> tuple[Fraction, Fraction]:
    """Enclose A_(s,q) in a_(s-1)(s+q,D)~A_(s,q)*D**(s-q-2).

    The theorem fixes integers s>=2 and q>=2 as D tends to infinity.
    This interval encloses the leading constant, not the finite-D risk.
    """
    _integer(face_size, 2, "face_size")
    _integer(outside, 2, "outside")
    lower, upper = vertex_leading_constant_enclosure(outside + 2, terms)
    scale = Fraction(
        comb(face_size + outside, face_size) ** 2 * prod(range(face_size, face_size + outside)),
        2 * factorial(face_size - 2) * (outside + 1) * (outside + 2) ** 2 * factorial(outside + 1),
    )
    return scale * lower, scale * upper


def _sources() -> dict[str, bytes]:
    return {
        name: Path(__file__).with_name(name).read_bytes() for name in [Path(__file__).name, "vertex_general.py", "exact.py"]
    }


def _report(face_size: int, outside: int, degree: int, sources: dict[str, bytes]) -> dict[str, object]:
    _integer(outside, 0, "outside")
    value = nonvertex_coefficient(face_size + outside, degree, face_size)
    return {
        "schema": "skeleton-risk-nonvertex-diagonals-v1",
        "proof_status": "proved_all_degree_identity",
        "proof": "docs/advanced/skeleton_risk_codimension_three_exact.rst",
        "algorithm": "positive_harmonic_double_sum" if outside == 3 else "positive_univariate_diagonal_averages",
        "face_size": face_size,
        "outside": outside,
        "vertices": face_size + outside,
        "degree": degree,
        "coefficient": fraction_record(value),
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in sources.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Save or independently recompute a source-bound exact coefficient record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=int, required=True)
    parser.add_argument("--outside", type=int, default=3)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    sources = _sources()
    try:
        report = _report(arguments.face_size, arguments.outside, arguments.degree, sources)
        if arguments.verify is not None:
            previous = json.loads(arguments.verify.read_bytes())
            if previous != report:
                raise ValueError("saved coefficient record differs from exact recomputation")
        if _sources() != sources:
            raise RuntimeError("nonvertex diagonal sources changed during calculation")
        atomic_json(arguments.output, report)
    except (ValueError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
