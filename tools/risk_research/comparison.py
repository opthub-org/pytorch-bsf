"""Certify fixed-degree skeleton-risk comparisons for every simplex dimension.

Exact binomial expansion constructs the dimension polynomials. Integer square
roots settle a finite prefix, and a rational bound covers the infinite tail.
These optional research tools use no numerical or computer-algebra dependency.
"""

import argparse
import hashlib
import math
from fractions import Fraction
from pathlib import Path

from .__main__ import integer_list
from .exact import atomic_json, fraction_record

Polynomial = tuple[Fraction, ...]
PolynomialFamily = tuple[Polynomial, ...]
Kernel = list[list[Fraction]]


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _falling_binomial(offset: int, order: int) -> Polynomial:
    """Return binom(M-offset, order) in ascending powers of M."""
    row = [Fraction(1)]
    for j in range(order):
        shifted = [Fraction(0)] * (len(row) + 1)
        for i, coefficient in enumerate(row):
            shifted[i] -= (offset + j) * coefficient
            shifted[i + 1] += coefficient
        row = shifted
    return tuple(value / math.factorial(order) for value in row)


def _kernel_powers(maximum: int) -> list[Kernel]:
    """Construct the relevant rectangle of (F-1)^r, 0<=r<=2*maximum."""
    base = [
        [Fraction(math.comb(h + ell, h), (h + 1) * (ell + 1)) if h + ell else Fraction(0) for ell in range(maximum + 1)]
        for h in range(maximum + 1)
    ]
    powers = [[[Fraction(int(h == ell == 0)) for ell in range(maximum + 1)] for h in range(maximum + 1)]]
    for order in range(1, 2 * maximum + 1):
        previous = powers[-1]
        current = [[Fraction(0)] * (maximum + 1) for _ in range(maximum + 1)]
        for h in range(maximum + 1):
            for ell in range(maximum + 1):
                if h + ell >= order:
                    current[h][ell] = sum(
                        (previous[h - i][ell - j] * base[i][j] for i in range(h + 1) for j in range(ell + 1) if i + j),
                        Fraction(0),
                    )
        powers.append(current)
    return powers


def _angular_weight(degree: int, size: int, kernel: Kernel, factorials: list[int]) -> Fraction:
    n = degree - size
    total = Fraction(0)
    for k in range(n + 1 if size >= 2 else 1):
        multiplicity = math.comb(k + size - 2, size - 2) if size >= 2 else 1
        radial = [
            (-1) ** h * math.comb(n - k, h) * math.comb(n + k + 3 * size - 1, h) * factorials[h] for h in range(n - k + 1)
        ]
        total += multiplicity * sum(
            (
                radial[h] * radial[ell] * factorials[2 * degree + size - h - ell - 1] * kernel[h][ell]
                for h in range(n - k + 1)
                for ell in range(n - k + 1)
            ),
            Fraction(0),
        )
    return total / factorials[size - 1]


def dimension_polynomials(degree: int) -> PolynomialFamily:
    """Construct every P_(D,s-1)(M) directly, in ascending powers of M.

    The identity a_(s-1)=binom(M,s)^2*P_(D,s-1)(M)/(M)_(2D) holds
    for every integer M>=s. There is no interpolation or propagation solve.
    """
    _integer(degree, 1, "degree")
    kernels = _kernel_powers(degree - 1)
    factorials = [math.factorial(j) for j in range(3 * degree)]
    result = []
    for size in range(1, degree + 1):
        row = [Fraction(0)] * (2 * (degree - size) + 1)
        for order in range(len(row)):
            weight = _angular_weight(degree, size, kernels[order], factorials)
            for i, coefficient in enumerate(_falling_binomial(size, order)):
                row[i] += weight * coefficient
        result.append(tuple(row))
    return tuple(result)


def _evaluate(polynomial: Polynomial, at: int) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(polynomial):
        value = value * at + coefficient
    return value


def _validate_polynomials(polynomials: PolynomialFamily) -> None:
    degree = len(polynomials)
    if not degree or any(len(row) != 2 * (degree - size) + 1 for size, row in enumerate(polynomials, 1)):
        raise ValueError("dimension polynomial rows must have lengths 2*(D-s)+1 for s=1,...,D")
    if any(not isinstance(value, Fraction) for row in polynomials for value in row):
        raise ValueError("dimension polynomial coefficients must be exact Fractions")


def coefficient_values(polynomials: PolynomialFamily, vertices: int) -> tuple[Fraction, ...]:
    """Evaluate the rational dimension polynomials, including absent-face zeros."""
    _integer(vertices, 2, "vertices")
    _validate_polynomials(polynomials)
    denominator = math.prod(range(vertices, vertices + 2 * len(polynomials)))
    values = tuple(
        math.comb(vertices, size) ** 2 * _evaluate(row, vertices) / denominator if size <= vertices else Fraction(0)
        for size, row in enumerate(polynomials, 1)
    )
    if any(value < 0 for value in values):
        raise ValueError("dimension polynomials gave a negative risk coefficient")
    return values


def _root_bounds(value: Fraction, bits: int) -> tuple[int, int]:
    """Return integer bounds for 2^bits*sqrt(value), with outward rounding."""
    numerator = value.numerator << (2 * bits)
    lower = math.isqrt(numerator // value.denominator)
    upper = lower if lower * lower * value.denominator == numerator else lower + 1
    return lower, upper


def radical_comparison(values: tuple[Fraction, ...], target: int, *, max_bits: int = 4096) -> dict[str, object]:
    """Certify the strict sign of (sum sqrt(values))^2-target using integers.

    A possible equality or insufficient precision raises ArithmeticError;
    an overlapping enclosure is never reported as a proved comparison.
    """
    _integer(target, 1, "target")
    _integer(max_bits, 1, "max_bits")
    if not values or any(not isinstance(value, Fraction) or value < 0 for value in values):
        raise ValueError("values must be a nonempty tuple of nonnegative Fractions")
    bits = min(32, max_bits)
    while True:
        bounds = [_root_bounds(value, bits) for value in values]
        lower, upper = sum(pair[0] for pair in bounds), sum(pair[1] for pair in bounds)
        scaled_target = target << (2 * bits)
        if upper * upper < scaled_target:
            decision, margin = "inductive_wins", scaled_target - upper * upper
        elif lower * lower > scaled_target:
            decision, margin = "all_at_once_wins", lower * lower - scaled_target
        elif bits == max_bits:
            raise ArithmeticError(f"unresolved strict comparison at {max_bits} bits; equality is possible")
        else:
            bits = min(2 * bits, max_bits)
            continue
        return {
            "decision": decision,
            "bits": bits,
            "sqrt_sum_lower_numerator": str(lower),
            "sqrt_sum_upper_numerator": str(upper),
            "target": str(target),
            "strict_margin_numerator": str(margin),
        }


def _tail_bounds(polynomials: PolynomialFamily, start: int) -> tuple[tuple[Fraction, ...], Fraction]:
    degree = len(polynomials)
    bounds = tuple(
        sum((abs(value) / start ** (2 * (degree - size) - i) for i, value in enumerate(row)), Fraction(0))
        / math.factorial(size) ** 2
        for size, row in enumerate(polynomials, 1)
    )
    return bounds, degree * sum(bounds, Fraction(0))


def _certify_tail(polynomials: PolynomialFamily, max_tail: int) -> dict[str, object]:
    degree = len(polynomials)
    for start in range(max(2, degree), max_tail + 1):
        bounds, risk_bound = _tail_bounds(polynomials, start)
        target = math.comb(start + degree - 1, degree)
        if risk_bound < target:
            return {
                "from_vertices": start,
                "coefficient_upper_bounds": [fraction_record(value) for value in bounds],
                "risk_upper_bound": fraction_record(risk_bound),
                "all_at_once_lower_bound": str(target),
                "strict_margin": fraction_record(target - risk_bound),
                "argument": "a_s(M)<=B_s(L); Cauchy-Schwarz; increasing binom(M+D-1,D), for all M>=L>=D",
            }
    raise ArithmeticError(f"no sufficient infinite-tail bound found through M={max_tail}")


def certify_degree(degree: int, *, max_tail: int = 10000, max_bits: int = 4096) -> dict[str, object]:
    """Prove all strict comparisons at a fixed degree or fail without a result.

    The returned JSON-compatible certificate contains the exact polynomial
    coefficients, every finite radical comparison, and the infinite-tail
    inequality. Loss sets need not be intervals. No formula for their
    endpoints, or for arbitrary degrees, is inferred from a computed row.
    """
    _integer(degree, 1, "degree")
    _integer(max_tail, max(2, degree), "max_tail")
    _integer(max_bits, 1, "max_bits")
    polynomials = dimension_polynomials(degree)
    tail = _certify_tail(polynomials, max_tail)
    start = tail["from_vertices"]
    assert isinstance(start, int)
    points = []
    losses = []
    for vertices in range(2, start):
        point = radical_comparison(
            coefficient_values(polynomials, vertices), math.comb(vertices + degree - 1, degree), max_bits=max_bits
        )
        point["vertices"] = vertices
        points.append(point)
        if point["decision"] == "all_at_once_wins":
            losses.append(vertices)
    return {
        "degree": degree,
        "proof_status": "strict_comparisons_certified_for_all_integer_vertices_ge_2",
        "construction": "direct finite binomial expansion of F**(M-s); no interpolation",
        "polynomial_order": "face size s=1,...,D; ascending powers of M within each row",
        "polynomials": [[fraction_record(value) for value in row] for row in polynomials],
        "finite_comparisons": points,
        "all_at_once_wins_vertices": losses,
        "tail": tail,
    }


def main(argv: list[str] | None = None) -> int:
    """Write complete certificates atomically, preserving old output on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=integer_list, required=True)
    parser.add_argument("--max-tail", type=int, default=10000)
    parser.add_argument("--max-bits", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    original = Path(__file__).read_bytes()
    try:
        records = [
            certify_degree(degree, max_tail=arguments.max_tail, max_bits=arguments.max_bits) for degree in arguments.degrees
        ]
        if Path(__file__).read_bytes() != original:
            raise RuntimeError("comparison source changed during certification")
        atomic_json(
            arguments.output,
            {
                "schema": "skeleton-risk-comparison-v1",
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "records": records,
            },
        )
    except (ValueError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
