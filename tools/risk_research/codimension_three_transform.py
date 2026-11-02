"""Evaluate the codimension-three family by one homogeneous polynomial transform.

The all-degree identity and the subquadratic integer-operation bound are
proved in ``skeleton_risk_codimension_three_transform.rst``.  All arithmetic
is exact; the existing positive double sum remains an independent algorithm.
This is not a fixed-state degree recurrence or a bounded special-function
formula.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, lcm
from pathlib import Path

from .codimension_three_exact import harmonic_diagonals
from .exact import atomic_json

HASH_POLICY = "sha256-crlf-to-lf-v1"


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _add(left: list[int], right: list[int]) -> list[int]:
    result = [0] * max(len(left), len(right))
    for index, value in enumerate(left):
        result[index] += value
    for index, value in enumerate(right):
        result[index] += value
    return result


def _multiply(left: list[int], right: list[int]) -> list[int]:
    """Multiply dense integer polynomials using balanced Karatsuba recursion."""
    if not left or not right:
        return []
    if min(len(left), len(right)) <= 16:
        result = [0] * (len(left) + len(right) - 1)
        for first, value in enumerate(left):
            for second, other in enumerate(right):
                result[first + second] += value * other
        return result
    middle = max(len(left), len(right)) // 2
    low = _multiply(left[:middle], right[:middle])
    high = _multiply(left[middle:], right[middle:])
    cross = _multiply(_add(left[:middle], left[middle:]), _add(right[:middle], right[middle:]))
    for index, value in enumerate(low):
        cross[index] -= value
    for index, value in enumerate(high):
        cross[index] -= value
    result = [0] * (len(left) + len(right) - 1)
    for offset, part in ((0, low), (middle, cross), (2 * middle, high)):
        for index, value in enumerate(part):
            if offset + index < len(result):
                result[offset + index] += value
    return result


def _binomial_power(exponent: int) -> list[int]:
    result = [1]
    for index in range(exponent):
        result.append(result[-1] * (exponent - index) // (index + 1))
    return result


def _homogeneous(coefficients: list[int], powers: dict[int, list[int]]) -> list[int]:
    """Return (1+z)^(2n) F(z/(1+z)^2), n=len(coefficients)-1."""
    if len(coefficients) == 1:
        return coefficients.copy()
    middle = len(coefficients) // 2
    upper_count = len(coefficients) - middle
    if upper_count not in powers:
        powers[upper_count] = _binomial_power(2 * upper_count)
    low = _multiply(_homogeneous(coefficients[:middle], powers), powers[upper_count])
    high = _homogeneous(coefficients[middle:], powers)
    for index, value in enumerate(high):
        low[middle + index] += value
    return low


def _averages(diagonals: list[Fraction], total: int) -> list[Fraction]:
    """Apply the coefficient identity to one nonempty diagonal prefix."""
    last = len(diagonals) - 1
    denominator = lcm(*(value.denominator for value in diagonals))
    coefficients = []
    weight = 1
    for index, value in enumerate(diagonals):
        coefficients.append(value.numerator * (denominator // value.denominator) * weight)
        if index != last:
            weight = weight * (total - 2 * index) * (total - 2 * index - 1) // (index + 1) ** 2
    transformed = _multiply(_homogeneous(coefficients, {}), _binomial_power(total - 2 * last))
    normalizer = 1
    result = []
    for length in range(last + 1):
        result.append(Fraction(transformed[length], denominator * normalizer**2))
        normalizer = normalizer * (total - length) // (length + 1)
    return result


def mode_averages(face_size: int, degree: int) -> list[Fraction]:
    """Return q3 diagonal averages in radial-length order L=0,...,D-s.

    Integers face_size>=2 and degree>=1 are required.  On the inactive
    prefix degree<face_size the result is empty.  Angular degree k uses
    entry L=degree-face_size-k.  The common radial scale is omitted.
    """
    _integer(face_size, 2, "face_size")
    _integer(degree, 1, "degree")
    if degree < face_size:
        return []
    return _averages(harmonic_diagonals(degree - face_size + 1), 2 * degree + face_size - 1)


def coefficient(face_size: int, degree: int) -> Fraction:
    """Return a_(s-1)(s+3,D) by the exact homogeneous transform.

    This optional checkout tool supports every integer s>=2 and D>=1,
    including the zero inactive prefix.  It uses only Python integers
    and fractions, with no optional symbolic or numerical backend.
    """
    averages = mode_averages(face_size, degree)
    if not averages:
        return Fraction(0)
    last = degree - face_size
    total = sum(
        (comb(last - length + face_size - 2, face_size - 2) * value for length, value in enumerate(averages)),
        Fraction(0),
    )
    return total * Fraction(
        comb(face_size + 3, face_size) ** 2 * face_size * (face_size + 1) * (face_size + 2),
        (2 * degree + face_size) * (2 * degree + face_size + 1) * (2 * degree + face_size + 2),
    )


def _decimal(value: int) -> str:
    if not value:
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    parts = []
    while value:
        value, part = divmod(value, 1_000_000_000)
        parts.append(part)
    return sign + str(parts[-1]) + "".join(f"{part:09d}" for part in reversed(parts[:-1]))


def _raw_sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "tools/risk_research/codimension_three_transform.py",
        "tools/risk_research/codimension_three_exact.py",
        "tools/risk_research/vertex_general.py",
        "tools/risk_research/exact.py",
        "docs/advanced/skeleton_risk_codimension_three_transform.rst",
    ]
    return {name: (root / name).read_bytes() for name in names}


def _record(face_size: int, degree: int, sources: dict[str, bytes]) -> dict[str, object]:
    value = coefficient(face_size, degree)
    count = max(0, degree - face_size + 1)
    return {
        "schema": "skeleton-risk-codimension-three-transform-v1",
        "proof_status": "proved_all_degree_identity",
        "proof": "docs/advanced/skeleton_risk_codimension_three_transform.rst",
        "algorithm": "homogeneous_karatsuba_binomial_transform",
        "face_size": face_size,
        "vertices": face_size + 3,
        "outside": 3,
        "degree": degree,
        "diagonal_terms": count,
        "reference_double_sum_terms": count * (count + 1) // 2,
        "coefficient": {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)},
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in sources.items()
        },
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _canonical(record: object) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    """Save or strictly recheck a source-bound exact coefficient record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=int, required=True)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        sources = _raw_sources()
        root = Path(__file__).resolve().parents[2]
        if arguments.output.resolve() in {(root / name).resolve() for name in sources}:
            raise ValueError("output must not overwrite evaluator or proof sources")
        if arguments.verify is not None and arguments.output.resolve() == arguments.verify.resolve():
            raise ValueError("verification output must differ from its saved input")
        report = _record(arguments.face_size, arguments.degree, sources)
        if arguments.verify is not None:
            previous = json.loads(arguments.verify.read_bytes(), object_pairs_hook=_unique_object)
            if _canonical(previous) != _canonical(report):
                raise ValueError("saved transform record differs from exact recomputation")
        if _raw_sources() != sources:
            raise RuntimeError("transform sources changed during calculation")
        if arguments.output.resolve() in {(root / name).resolve() for name in sources}:
            raise ValueError("output must not overwrite evaluator or proof sources")
        if arguments.verify is not None and arguments.output.resolve() == arguments.verify.resolve():
            raise ValueError("verification output must differ from its saved input")
        atomic_json(arguments.output, report)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(arguments.output), "face_size": arguments.face_size, "degree": arguments.degree}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
