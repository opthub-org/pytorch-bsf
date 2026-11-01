"""Construct every nonvertex codimension-two coefficient with fifteen states.

For a fixed source-face size, all polynomial and rational reductions have
degree depending only on that size. Evaluation never sums a degree-dependent
angular range. The proof is in ``codimension_two_family.rst``.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from .exact import atomic_json, fraction_record

STATE = (
    "H_N^2",
    "H_N*H_h",
    "H_h^2",
    "H_N",
    "H_h",
    "H_N^(2)",
    "H_h^(2)",
    "F_N",
    "G_N",
    "P_N",
    "T_1",
    "T_2",
    "U",
    "1/binom(N,h)",
    "1",
)


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _trim(p: list[Any]) -> list[Any]:
    while len(p) > 1 and not p[-1]:
        p.pop()
    return p


def _add(a: list[Any], b: list[Any]) -> list[Any]:
    z = a[0] - a[0]
    return _trim([(a[k] if k < len(a) else z) + (b[k] if k < len(b) else z) for k in range(max(len(a), len(b)))])


def _scale(a: list[Any], c: Any) -> list[Any]:
    return _trim([c * x for x in a])


def _mul(a: list[Any], b: list[Any]) -> list[Any]:
    z = a[0] - a[0]
    out = [z] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(out)


def _at(p: list[Any], x: Any) -> Any:
    out = p[0] - p[0]
    for c in reversed(p):
        out = out * x + c
    return out


def _shift(p: list[Any], a: int) -> list[Any]:
    z = p[0] - p[0]
    return _trim([sum((p[k] * math.comb(k, j) * a ** (k - j) for k in range(j, len(p))), z) for j in range(len(p))])


def _divide(a: list[Any], b: list[Any]) -> tuple[list[Any], list[Any]]:
    a = _trim(a.copy())
    z = a[0] - a[0]
    out = [z] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and any(a):
        k, c = len(a) - len(b), a[-1] / b[-1]
        out[k] += c
        a = _add(a, [z] * k + _scale(b, -c))
    return _trim(out), a


def _antidifference(p: list[Any]) -> list[Any]:
    """Return A with A(0)=0 and A(j)-A(j-1)=p(j)."""
    z = p[0] - p[0]
    remaining, out = p.copy(), [z]
    while any(remaining):
        k = len(remaining)
        monomial = [z] * k + [remaining[-1] / k]
        remaining = _add(remaining, _scale(_add(monomial, _scale(_shift(monomial, -1), -1)), -1))
        out = _add(out, monomial)
    return out


def _partial(p: list[Any], poles: list[tuple[Any, int]]) -> tuple[list[Any], list[tuple[int, int, Any]]]:
    """Partial fractions with distinct poles of order at most two."""
    one = p[0] - p[0] + 1
    denominator = [one]
    for root, order in poles:
        for _ in range(order):
            denominator = _mul(denominator, [-root, one])
    quotient, remainder = _divide(p, denominator)
    fractions = []
    for i, (root, order) in enumerate(poles):
        factor = [one]
        for _ in range(order):
            factor = _mul(factor, [-root, one])
        other, residual = _divide(denominator, factor)
        if any(residual):
            raise ArithmeticError("partial-fraction denominator did not divide")
        r, d = _at(remainder, root), _at(other, root)
        fractions.append((i, order, r / d))
        if order == 2:
            dr = _at([k * remainder[k] for k in range(1, len(remainder))] or [one - one], root)
            dd = _at([k * other[k] for k in range(1, len(other))] or [one - one], root)
            fractions.append((i, 1, (dr * d - r * dd) / d**2))
    return quotient, fractions


def _basis(n: Any, index: int, coefficient: Any = 1) -> list[Any]:
    out = [n - n] * 15
    out[index] = coefficient + n - n
    return out


def _combine(n: Any, *terms: tuple[Any, list[Any]]) -> list[Any]:
    return [sum((c * row[i] for c, row in terms), n - n) for i in range(15)]


def _product(a: list[Any], b: list[Any]) -> list[Any]:
    """Multiply affine functions of H_N,H_h into the quadratic basis."""
    n = a[14]
    if any(a[i] or b[i] for i in range(15) if i not in (3, 4, 14)):
        raise ArithmeticError("harmonic product is not affine")
    out = [n - n] * 15
    out[0], out[1], out[2] = a[3] * b[3], a[3] * b[4] + a[4] * b[3], a[4] * b[4]
    out[3], out[4] = a[3] * b[14] + a[14] * b[3], a[4] * b[14] + a[14] * b[4]
    out[14] = a[14] * b[14]
    return out


def _harmonic(n: Any, h: Any, offset: int, order: int = 1, *, full: bool = False) -> list[Any]:
    base, index = (n, 3 if order == 1 else 5) if full else (h, 4 if order == 1 else 6)
    one = n - n + 1
    increment = (
        sum((one / (base + k) ** order for k in range(1, offset + 1)), n - n)
        if offset >= 0
        else -sum((one / (base - k) ** order for k in range(-offset)), n - n)
    )
    return _combine(n, (one, _basis(n, index)), (increment, _basis(n, 14)))


def _rational_sum(
    p: list[Any], poles: list[tuple[Any, int]], h: Any, epsilon: int, labels: list[tuple[bool, int]], shift: int = 0
) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    polynomial, fractions = _partial(p, poles)
    out = _basis(n, 14, _at(_antidifference(polynomial), h + shift))
    for i, order, coefficient in fractions:
        full, offset = labels[i]
        if full:
            row = _combine(
                n,
                (one, _harmonic(n, h, offset - 1, order, full=True)),
                (-one, _harmonic(n, h, epsilon + offset - shift - 1, order)),
            )
            coefficient *= (-1) ** order
        else:
            # Pole at -offset, with a fixed nonnegative integer offset.
            constant = sum((one / k**order for k in range(1, offset + 1)), n - n)
            row = _combine(n, (one, _harmonic(n, h, shift + offset, order)), (-constant, _basis(n, 14)))
        out = _combine(n, (one, out), (coefficient, row))
    return out


def _anti(h: Any, epsilon: int, cshift: int, lshift: int) -> list[Any]:
    """Harmonic antisymmetric prefix, with C=N+cshift and L=h+lshift."""
    n, one = 2 * h + epsilon, h - h + 1
    a = _harmonic(n, h, lshift)
    c = _harmonic(n, h, cshift - 1, full=True)
    k = _harmonic(n, h, epsilon + cshift - lshift - 1)
    difference = _combine(n, (one, a), (-one, c), (one, k))
    return _combine(
        n,
        (one / 2, _product(a, a)),
        (one / 2, _harmonic(n, h, lshift, 2)),
        (-one / 2, _product(c, c)),
        (-one / 2, _harmonic(n, h, cshift - 1, 2, full=True)),
        (one / 2, _product(k, k)),
        (one / 2, _harmonic(n, h, epsilon + cshift - lshift - 1, 2)),
        (one, _product(a, k)),
        (-one, _product(_basis(n, 3), difference)),
    )


def _harmonic_poles(h: Any, epsilon: int) -> list[list[Any]]:
    n, one = 2 * h + epsilon, h - h + 1
    poles = [(n - n, 1), (-one, 1), (n, 1), (n + 1, 1)]
    labels = [(False, 0), (False, 1), (True, 0), (True, 1)]
    symmetric0 = _combine(
        n,
        (one / 2, _basis(n, 0)),
        (-one / 2, _basis(n, 5)),
        (2 / (n + 1), _basis(n, 3)),
        (one - 1 / n - (one + 1 / n) / (n + 1), _basis(n, 14)),
    )
    u, v = one / (n + 1), one / (n + 2)
    symmetric1 = _combine(
        n,
        (one / 2, _basis(n, 0)),
        (-one / 2, _basis(n, 5)),
        (2 * (u + v), _basis(n, 3)),
        (2 * u * v - 2 * u - 1, _basis(n, 14)),
    )
    if epsilon == 2:
        center = _combine(n, (2, _harmonic(n, h, 2)), (-one, _basis(n, 3)))
        symmetric0 = _combine(n, (one, symmetric0), (-one / (h + 1), center))
        symmetric1 = _combine(n, (one, symmetric1), (-one / (h + 2), center))
    correction = _rational_sum([n * (n + 2), -2 * (n + 2)], poles, h, epsilon, labels)
    anti0 = _combine(n, (one, _anti(h, epsilon, 0, 0)), (one, correction))
    anti1 = _combine(n, (one, _anti(h, epsilon, 2, 1)), (-one + 1 / (n + 1) ** 2, _basis(n, 14)))
    return [symmetric0, symmetric1, anti0, anti1]


def _harmonic_sum(p: list[Any], h: Any, epsilon: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    poles = [(n - n, 1), (-one, 1), (n, 1), (n + 1, 1)]
    polynomial, fractions = _partial(p, poles)
    primitive = _antidifference(polynomial)
    boundary = _combine(n, (one, _harmonic(n, h, 1)), (one, _harmonic(n, h, epsilon + 1)), (-one, _basis(n, 3)))
    out = _combine(n, (_at(primitive, h), boundary))
    # Summation by parts leaves A(j)/(j+2)-A(j)/(N-j+1).
    left = _rational_sum(primitive, [(-2 * one, 1)], h, epsilon, [(False, 2)], -1)
    right = _rational_sum(_scale(primitive, -one), [(n + 1, 1)], h, epsilon, [(True, 1)], -1)
    out = _combine(n, (one, out), (-one, left), (one, right))
    coefficients = {i: coefficient for i, _, coefficient in fractions}
    s0, s1, a0, a1 = _harmonic_poles(h, epsilon)
    return _combine(
        n,
        (one, out),
        ((coefficients[0] - coefficients[2]) / 2, s0),
        ((coefficients[0] + coefficients[2]) / 2, a0),
        ((coefficients[1] - coefficients[3]) / 2, s1),
        ((coefficients[1] + coefficients[3]) / 2, a1),
    )


def _half_binomial_sum(h: Any, epsilon: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    center = (h + 1) / (h + 2) if epsilon == 2 else n - n
    return _combine(n, ((n + 1) / 4, _basis(n, 8)), (-one / 2, _basis(n, 14)), (-center / 2, _basis(n, 13)))


def _inverse_polynomial(p: list[Any], h: Any, epsilon: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    remaining, primitive = p.copy(), [n - n]
    while len(remaining) > 1:
        k = len(remaining) - 2
        monomial = [n - n] * k + [remaining[-1] / 2]
        image = _add(_mul([one, one], _shift(monomial, 1)), _scale(_mul([n + 1, -one], monomial), -one))
        remaining = _add(remaining, _scale(image, -one))
        primitive = _add(primitive, monomial)
    return _combine(
        n,
        (remaining[0], _half_binomial_sum(h, epsilon)),
        ((h + 1) * _at(primitive, h + 1), _basis(n, 13)),
        (-_at(primitive, one), _basis(n, 14)),
    )


def _inverse_sum(p: list[Any], h: Any, epsilon: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    polynomial, fractions = _partial(p, [(n - n, 2), (-one, 1), (n, 2), (n + 1, 1)])
    out = _inverse_polynomial(polynomial, h, epsilon)
    center = (h + 1) / (h + 2) if epsilon == 2 else n - n
    for i, order, coefficient in fractions:
        if i == 0:
            row = _basis(n, 10 if order == 1 else 11)
        elif i == 1:
            row = _basis(n, 12)
        elif i == 2:
            row = _combine(
                n,
                (one, _basis(n, 8 if order == 1 else 7)),
                (-one, _basis(n, 10 if order == 1 else 11)),
                (-one / n**order, _basis(n, 14)),
                (-center / (h + 1) ** order, _basis(n, 13)),
            )
            coefficient *= (-1) ** order
        else:
            row = _combine(
                n,
                (one, _basis(n, 9)),
                (-one, _basis(n, 12)),
                (-one - 1 / (n + 1), _basis(n, 14)),
                (-center / (h + 2), _basis(n, 13)),
            )
            coefficient = -coefficient
        out = _combine(n, (one, out), (coefficient, row))
    return out


def _weighted_half_row(p: list[Any], h: Any, epsilon: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    poles = [(n - n, 2), (-one, 2), (n, 2), (n + 1, 2)]
    rational = _rational_sum(
        _scale(_mul(p, [-one, n, -one]), n + 1), poles, h, epsilon, [(False, 0), (False, 1), (True, 0), (True, 1)]
    )
    inverse = _inverse_sum(_scale(p, -one), h, epsilon)
    return _combine(n, (one, _harmonic_sum(p, h, epsilon)), (-one, rational), (-one, inverse))


def _boundary_row(h: Any, epsilon: int, offset: int) -> list[Any]:
    n, one = 2 * h + epsilon, h - h + 1
    j = h - offset
    x = j * (n - j)
    delta = x * (x + n + 1)
    harmonic = _combine(
        n, (one, _harmonic(n, h, 1 - offset)), (one, _harmonic(n, h, epsilon + offset + 1)), (-one, _basis(n, 3))
    )
    ratio = one
    for k in range(1, offset + 1):
        ratio *= (n - h + k) / (h - k + 1)
    return _combine(
        n, (one / delta, harmonic), (-(n + 1) * (x - 1) / delta**2, _basis(n, 14)), (-ratio / (x * delta), _basis(n, 13))
    )


def _output_row(face_size: int, h: Any) -> list[Any]:
    epsilon = 1 if face_size % 2 == 0 else 2
    n, one = 2 * h + epsilon, h - h + 1
    degree = h - face_size // 2
    polynomial = [one]
    for k in range(face_size - 2):
        polynomial = _scale(_mul(polynomial, [degree - 1 - k, -one]), one / (k + 1))
    out = _weighted_half_row(polynomial, h, epsilon)
    for offset in range(face_size // 2 + 1):
        t = face_size // 2 - offset
        weight = (-1) ** (face_size - 2) * math.comb(t + face_size - 2, face_size - 2)
        out = _combine(n, (one, out), (-weight, _boundary_row(h, epsilon, offset)))
    return [math.comb(face_size + 2, face_size) ** 2 * 2 * face_size * (face_size + 1) * c for c in out]


def _transition_rows(h: Any, epsilon: int) -> list[list[Any]]:
    n, one = 2 * h + epsilon, h - h + 1
    rows = [_basis(n, i) for i in range(15)]
    a, b = one / (n + 1) + one / (n + 2), one / (h + 1)
    a2 = one / (n + 1) ** 2 + one / (n + 2) ** 2
    rows[0] = _combine(n, (one, rows[0]), (2 * a, rows[3]), (a * a, rows[14]))
    rows[1] = _combine(n, (one, rows[1]), (b, rows[3]), (a, rows[4]), (a * b, rows[14]))
    rows[2] = _combine(n, (one, rows[2]), (2 * b, rows[4]), (b * b, rows[14]))
    rows[3], rows[4] = _combine(n, (one, rows[3]), (a, rows[14])), _combine(n, (one, rows[4]), (b, rows[14]))
    rows[5], rows[6] = _combine(n, (one, rows[5]), (a2, rows[14])), _combine(n, (one, rows[6]), (b * b, rows[14]))
    q = (n + 1) * (n + 2)
    rows[7] = _combine(n, (one, rows[7]), (-(3 * n + 5) / (2 * q), _basis(n, 8)), (a2 - one / q, rows[14]))
    g1 = _combine(n, (one / 2, _basis(n, 8)), (one / (n + 1), rows[14]))
    p1 = _combine(n, ((n + 2) / (n + 1), _basis(n, 9)), (-one / 2, _basis(n, 8)), (-one / (n + 1) + one / (n + 2), rows[14]))
    rows[8] = _combine(n, (one / 4, _basis(n, 8)), (one / (2 * (n + 1)) + one / (n + 2), rows[14]))
    rows[9] = _combine(n, ((n + 3) / (n + 2), p1), (-one / 2, g1), (-one / (n + 2) + one / (n + 3), rows[14]))
    total = _half_binomial_sum(h, epsilon)
    moment = _combine(n, (n / 2, total), ((h + 1) / 2, _basis(n, 13)), (-one / 2, rows[14]))
    w = (h + 1) * (n - h + 1) / q
    rows[10] = _combine(n, (one, _basis(n, 10)), (-(2 * n + 3) / q, total), (one / q, moment), (w / (h + 1), _basis(n, 13)))
    rows[11] = _combine(
        n, (one, _basis(n, 11)), (-(2 * n + 3) / q, _basis(n, 10)), (one / q, total), (w / (h + 1) ** 2, _basis(n, 13))
    )
    rows[12] = _combine(
        n, ((n + 3) / (n + 1), _basis(n, 12)), (-2 * one / (n + 1), total), (one / q, moment), (w / (h + 2), _basis(n, 13))
    )
    rows[13] = _basis(n, 13, w)
    return rows


def _initial_state(epsilon: int) -> list[Fraction]:
    n = 2 + epsilon
    hn = sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))
    hn2 = sum((Fraction(1, j**2) for j in range(1, n + 1)), Fraction(0))
    f = sum((Fraction(1, j**2 * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    g = sum((Fraction(1, j * math.comb(n, j)) for j in range(1, n + 1)), Fraction(0))
    p = sum((Fraction(1, (j + 1) * math.comb(n, j)) for j in range(n + 1)), Fraction(0))
    return [
        hn**2,
        hn,
        Fraction(1),
        hn,
        Fraction(1),
        hn2,
        Fraction(1),
        f,
        g,
        p,
        Fraction(1, n),
        Fraction(1, n),
        Fraction(1, 2 * n),
        Fraction(1, n),
        Fraction(1),
    ]


def family_coefficients(face_size: int, maximum: int) -> dict[int, Fraction]:
    """Evaluate a_(s-1)(s+2,D), for fixed s>=2 and 1<=D<=maximum.

    The fifteen states are independent of s. Polynomial construction cost
    depends on s, and rational bit lengths still grow with degree.
    """
    _integer(face_size, 2, "face_size")
    _integer(maximum, 1, "maximum")
    epsilon = 1 if face_size % 2 == 0 else 2
    state = _initial_state(epsilon)
    result = {degree: Fraction(0) for degree in range(1, min(face_size, maximum + 1))}
    if maximum < face_size:
        return result
    for h in range(1, maximum + face_size // 2 + 1):
        degree = h - face_size // 2
        if degree >= face_size:
            result[degree] = sum((a * b for a, b in zip(_output_row(face_size, Fraction(h)), state, strict=True)), Fraction(0))
        if h < maximum + face_size // 2:
            state = [
                sum((a * b for a, b in zip(row, state, strict=True)), Fraction(0))
                for row in _transition_rows(Fraction(h), epsilon)
            ]
    return result


def family_coefficient(face_size: int, degree: int) -> Fraction:
    """Return one exact nonvertex codimension-two coefficient."""
    _integer(face_size, 2, "face_size")
    _integer(degree, 1, "degree")
    if degree < face_size:
        return Fraction(0)
    epsilon = 1 if face_size % 2 == 0 else 2
    state = _initial_state(epsilon)
    target = degree + face_size // 2
    for h in range(1, target):
        state = [
            sum((a * b for a, b in zip(row, state, strict=True)), Fraction(0))
            for row in _transition_rows(Fraction(h), epsilon)
        ]
    return sum((a * b for a, b in zip(_output_row(face_size, Fraction(target)), state, strict=True)), Fraction(0))


def main(argv: list[str] | None = None) -> int:
    """Generate a source-bound exact family evaluation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=int, required=True)
    parser.add_argument("--max-degree", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    source = Path(__file__).read_bytes()
    try:
        values = family_coefficients(arguments.face_size, arguments.max_degree)
        record = {
            "schema": "skeleton-risk-codimension-two-family-v1",
            "face_size": arguments.face_size,
            "vertices": arguments.face_size + 2,
            "state_dimension": len(STATE),
            "state_basis": list(STATE),
            "values": {str(d): fraction_record(v) for d, v in values.items()},
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "proof": "docs/advanced/codimension_two_family.rst",
        }
        if Path(__file__).read_bytes() != source:
            raise RuntimeError("family source changed during evaluation")
        atomic_json(arguments.output, record)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
