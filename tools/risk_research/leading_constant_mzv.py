"""Construct finite convergent multiple-zeta expressions for leading constants.

The constant index q is M-1 at a vertex. For every fixed integer q>=3,
the construction terminates by decreasing harmonic-word depth. A caller
work limit bounds selected symbolic steps, not CPU time or integer bits.
Direct MZV intervals can be broad: this is a symbolic construction, not
an improved numerical method or a finite-degree error certificate.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb, factorial
from pathlib import Path
from typing import TypeVar

from .exact import atomic_json
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-leading-constant-mzv-v1"
PROOF = "docs/advanced/skeleton_risk_leading_constant_mzv.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
DEFAULT_WORK_LIMIT = 200_000
MAX_WORK_LIMIT = 1_000_000
Word = tuple[int, ...]
Pole = tuple[int, int]
PartialFractions = dict[Pole, Fraction]
Key = TypeVar("Key")


def _integer(value: object, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass
class _Budget:
    limit: int
    used: int = 0

    def spend(self, amount: int = 1) -> None:
        self.used += amount
        if self.used > self.limit:
            raise ValueError("construction or evaluation exceeds work limit")


def _budget(limit: int) -> _Budget:
    _integer(limit, 1, "work limit")
    if limit > MAX_WORK_LIMIT:
        raise ValueError("work limit exceeds the supported operational maximum")
    return _Budget(limit)


def _add(target: dict[Key, Fraction], key: Key, value: Fraction, budget: _Budget) -> None:
    budget.spend()
    total = target.get(key, Fraction(0)) + value
    if total:
        target[key] = total
    else:
        target.pop(key, None)


def _partial_fractions(factors: dict[int, int], budget: _Budget) -> PartialFractions:
    result: PartialFractions = {}
    for shift, power in factors.items():
        coefficient, derivative = Fraction(1), Fraction(0)
        for other, multiplicity in factors.items():
            budget.spend()
            if shift != other:
                coefficient /= (other - shift) ** multiplicity
                derivative -= Fraction(multiplicity, other - shift)
        result[shift, power] = coefficient
        if power == 2:
            _add(result, (shift, 1), coefficient * derivative, budget)
    return result


def _telescope(poles: PartialFractions, budget: _Budget) -> tuple[Fraction, PartialFractions]:
    simple_sum = sum((c for (_, power), c in poles.items() if power == 1), Fraction(0))
    if simple_sum:
        raise ArithmeticError("a rational kernel is not of order n**(-2)")
    residual = sum((c for (_, power), c in poles.items() if power == 2), Fraction(0))
    antiderivative: PartialFractions = {}
    for (shift, power), coefficient in poles.items():
        for inner in range(shift):
            _add(antiderivative, (inner, power), coefficient, budget)
    return residual, antiderivative


def _shift_divide(poles: PartialFractions, leading: int, budget: _Budget) -> PartialFractions:
    result: PartialFractions = {}
    for (shift, power), coefficient in poles.items():
        for key, value in _partial_fractions({0: leading, shift + 1: power}, budget).items():
            _add(result, key, -coefficient * value, budget)
    return result


def _ones_product(left: int, right: int, budget: _Budget) -> dict[Word, Fraction]:
    result: dict[Word, Fraction] = {}
    for merged in range(min(left, right) + 1):
        ones = left + right - 2 * merged
        length = ones + merged
        coefficient = Fraction(comb(ones, left - merged))
        for positions in combinations(range(length), merged):
            budget.spend(length + 1)
            twos = set(positions)
            result[tuple(2 if i in twos else 1 for i in range(length))] = coefficient
    return result


def _phi_polynomial(order: int, budget: _Budget) -> tuple[Fraction, ...]:
    elementary = [Fraction(1)] + [Fraction(0)] * (order - 1)
    for denominator in range(1, order):
        for index in range(order - 1, 0, -1):
            budget.spend()
            elementary[index] += elementary[index - 1] / denominator
    scale = factorial(order) * factorial(order - 1)
    return tuple((-1) ** index * scale * elementary[order - 1 - index] for index in range(order))


def _initial(index: int, budget: _Budget) -> dict[Word, PartialFractions]:
    result: dict[Word, PartialFractions] = {}
    order = index - 1
    for left in range(1, order):
        right = order - left
        poles = _partial_fractions({i: int(i < left) + int(i < right) for i in range(max(left, right))}, budget)
        for a, coefficient_a in enumerate(_phi_polynomial(left, budget)):
            for b, coefficient_b in enumerate(_phi_polynomial(right, budget)):
                scale = index * comb(order, left) * coefficient_a * coefficient_b
                for word, coefficient in _ones_product(a, b, budget).items():
                    kernel = result.setdefault(word, {})
                    for key, value in poles.items():
                        _add(kernel, key, scale * coefficient * value, budget)
    return {word: poles for word, poles in result.items() if poles}


def _transform(poles: PartialFractions, word: Word, budget: _Budget) -> dict[Word, Fraction]:
    residual, antiderivative = _telescope(poles, budget)
    result: dict[Word, Fraction] = {}
    _add(result, (2,) + word, residual, budget)
    if word:
        child = _shift_divide(antiderivative, word[0], budget)
        for key, value in _transform(child, word[1:], budget).items():
            _add(result, key, value, budget)
    else:
        boundary = -sum((c / (1 + shift) ** power for (shift, power), c in antiderivative.items()), Fraction(0))
        _add(result, (), boundary, budget)
    return result


def _construct(index: int, budget: _Budget) -> dict[Word, Fraction]:
    _integer(index, 3, "constant index")
    budget.spend(index * index)
    result: dict[Word, Fraction] = {}
    for word, poles in _initial(index, budget).items():
        for key, coefficient in _transform(poles, word, budget).items():
            _add(result, key, coefficient, budget)
    return dict(sorted(result.items()))


def leading_constant_expression(index: int, work_limit: int = DEFAULT_WORK_LIMIT) -> dict[Word, Fraction]:
    """Construct c_index as a rational combination of convergent MZVs.

    Indices are in descending-summation order. The empty word denotes 1.
    Every other word starts with 2, has remaining entries 1 or 2 and has
    weight at most index-1. A work-limit failure produces no partial result.
    The theorem has no upper index bound; the implementation has an
    explicit operational work limit and can reject costly finite inputs.
    """
    return _construct(index, _budget(work_limit))


def _harmonic_values(words: set[Word], terms: int, budget: _Budget) -> dict[Word, Fraction]:
    budget.spend(sum(len(word) ** 2 + sum(word) for word in words))
    suffixes = {word[i:] for word in words for i in range(len(word) + 1)}
    ordered = sorted((word for word in suffixes if word), key=len, reverse=True)
    budget.spend(terms * sum(sum(word) + len(word) for word in ordered))
    values = {word: Fraction(0) for word in ordered}
    values[()] = Fraction(1)
    for n in range(1, terms + 1):
        for word in ordered:
            values[word] += values[word[1:]] / n ** word[0]
    return values


def _tail(word: Word, terms: int) -> Fraction:
    depth, decay = len(word) - 1, word[0] - 1
    logarithm = 2 + terms.bit_length()
    return sum(
        (Fraction(logarithm**k, factorial(k) * decay ** (depth - k + 1)) for k in range(depth + 1)),
        Fraction(0),
    ) * Fraction(1, terms**decay)


def mzv_enclosure(word: Word, terms: int = 64, work_limit: int = DEFAULT_WORK_LIMIT) -> tuple[Fraction, Fraction]:
    """Enclose one convergent strict descending-index multiple zeta value.

    The elementary tail uses a harmonic majorant and can be conservative.
    No rounded logarithm or numerical special-function value is used.
    """
    if type(word) is not tuple or not word or any(type(k) is not int or k < 1 for k in word) or word[0] < 2:
        raise ValueError("MZV indices must be a nonempty tuple of positive integers starting at least 2")
    _integer(terms, 1, "terms")
    lower = _harmonic_values({word}, terms, _budget(work_limit))[word]
    return lower, lower + _tail(word, terms)


def _enclose(expression: dict[Word, Fraction], terms: int, budget: _Budget) -> tuple[Fraction, Fraction]:
    values = _harmonic_values(set(expression), terms, budget)
    lower = upper = expression.get((), Fraction(0))
    for word, coefficient in expression.items():
        if word:
            partial, tail = values[word], _tail(word, terms)
            lower += coefficient * (partial if coefficient > 0 else partial + tail)
            upper += coefficient * (partial + tail if coefficient > 0 else partial)
    if upper < 0:
        raise ArithmeticError("a positive leading constant has a negative enclosure")
    return max(Fraction(0), lower), upper


def constant_enclosure(index: int, terms: int = 64, work_limit: int = DEFAULT_WORK_LIMIT) -> tuple[Fraction, Fraction]:
    """Enclose c_index using only its constructed finite MZV expression.

    This deliberately keeps the symbolic expression and its conservative
    numerical truncation separate from original-definition evaluation.
    """
    _integer(terms, 1, "terms")
    budget = _budget(work_limit)
    return _enclose(_construct(index, budget), terms, budget)


def _decimal(value: int) -> str:
    if value == 0:
        return "0"
    sign, value = ("-", -value) if value < 0 else ("", value)
    chunks = []
    while value:
        value, chunk = divmod(value, 10**500)
        chunks.append(f"{chunk:0500d}" if value else str(chunk))
    return sign + "".join(reversed(chunks))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "tools/risk_research/leading_constant_mzv.py",
        "tools/risk_research/exact.py",
        "tools/risk_research/saved_records.py",
        PROOF,
        "docs/advanced/skeleton_risk_vertex_general.rst",
    )
    return {path: (root / path).read_bytes() for path in paths}


def build_record(index: int = 5, terms: int = 64, work_limit: int = DEFAULT_WORK_LIMIT) -> dict[str, object]:
    """Record a source-bound finite expression and signed rational enclosure."""
    sources = _sources()
    _integer(terms, 1, "terms")
    budget = _budget(work_limit)
    expression = _construct(index, budget)
    lower, upper = _enclose(expression, terms, budget)
    vertices = index + 1
    scale = Fraction(vertices**2 * factorial(index), 2 ** (index - 2))
    result: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_finite_convergent_mzv_construction",
        "proof": PROOF,
        "constant_index": index,
        "vertices": vertices,
        "terms": terms,
        "work_limit": work_limit,
        "selected_work_units": budget.used,
        "mzv_convention": "strict-descending-indices; empty word is rational unit",
        "mzv_terms": [{"indices": list(word), "coefficient": _fraction(c)} for word, c in expression.items()],
        "c_index": {"lower": _fraction(lower), "upper": _fraction(upper)},
        "vertex_scale": _fraction(scale),
        "vertex_leading_constant": {"lower": _fraction(scale * lower), "upper": _fraction(scale * upper)},
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
        "scope": (
            "fixed-index leading constants; no general classical reduction, "
            "minimal basis, finite-degree error or numeric speedup"
        ),
    }
    if _sources() != sources:
        raise RuntimeError("a construction or proof source changed during calculation")
    return result


def verify_record(record: object) -> None:
    """Restore every generation parameter and verify the complete typed record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported leading-constant MZV record")
    index = _integer(record.get("constant_index"), 3, "constant index")
    terms = _integer(record.get("terms"), 1, "terms")
    limit = _integer(record.get("work_limit"), 1, "work limit")
    if canonical_json(record) != canonical_json(build_record(index, terms, limit)):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace construction or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a finite convergent-MZV construction record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant-index", type=int)
    parser.add_argument("--terms", type=int)
    parser.add_argument("--work-limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        if args.verify is not None:
            if any(x is not None for x in (args.constant_index, args.terms, args.work_limit, args.output)):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(args.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            print("verified leading-constant MZV record")
            return 0
        if args.output is None:
            raise ValueError("generation requires --output")
        _protect_output(args.output, sources)
        record = build_record(
            5 if args.constant_index is None else args.constant_index,
            64 if args.terms is None else args.terms,
            DEFAULT_WORK_LIMIT if args.work_limit is None else args.work_limit,
        )
        if _sources() != sources:
            raise RuntimeError("a construction or proof source changed during calculation")
        output = args.output.parent.resolve() / args.output.name
        _protect_output(output, sources)
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
