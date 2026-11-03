"""Enclose convergent multiple zeta values by a dyadic integral split.

All arithmetic is rational. The proof gives a polynomial times 2**(-T)
truncation bound for every admissible index, and absolute target widths
for signed finite expressions. Operational work limits can reject costly
inputs; they do not bound CPU time, memory, or integer bit lengths.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

from .exact import atomic_json
from .leading_constant_mzv import DEFAULT_WORK_LIMIT, _Budget, _budget, _construct, _fraction, _integer
from .saved_records import _unique_object, canonical_json

SCHEMA = "skeleton-risk-mzv-dyadic-v1"
PROOF = "docs/advanced/skeleton_risk_mzv_dyadic.rst"
HASH_POLICY = "sha256-crlf-to-lf-v1"
Word = tuple[int, ...]
Interval = tuple[Fraction, Fraction]


def _word(indices: object, *, admissible: bool = True) -> Word:
    if (
        type(indices) is not tuple
        or not indices
        or any(type(k) is not int or k < 1 for k in indices)
        or (admissible and indices[0] < 2)
    ):
        raise ValueError("indices must be a nonempty tuple of positive integers, starting at least 2 for an MZV")
    return indices


def _decode(binary: str) -> Word:
    indices, count = [], 1
    for letter in binary:
        if letter == "0":
            count += 1
        else:
            indices.append(count)
            count = 1
    if count != 1:
        raise ArithmeticError("an internal dyadic word does not end in 1")
    return tuple(indices)


def _split(indices: Word, budget: _Budget) -> tuple[tuple[Word, Word], ...]:
    weight = sum(indices)
    budget.spend((weight + 1) ** 2)
    binary = "".join("0" * (k - 1) + "1" for k in indices)
    return tuple(
        (_decode("".join("1" if bit == "0" else "0" for bit in reversed(binary[:j]))), _decode(binary[j:]))
        for j in range(weight + 1)
    )


def _tail(indices: Word, terms: int) -> Fraction:
    if not indices:
        return Fraction(0)
    depth = len(indices)
    return Fraction(sum(comb(terms, k) for k in range(min(depth - 1, terms) + 1)), 2**terms * (terms + 1) ** indices[0])


def _prepare(expression: dict[Word, Fraction], budget: _Budget) -> dict[Word, tuple[tuple[Word, Word], ...]]:
    return {word: _split(word, budget) for word in expression if word}


def _width_bound(
    expression: dict[Word, Fraction], cuts: dict[Word, tuple[tuple[Word, Word], ...]], terms: int, budget: _Budget
) -> Fraction:
    words = {part for pairs in cuts.values() for pair in pairs for part in pair}
    budget.spend(terms + sum(len(word) + 1 for word in words))
    tails = {word: _tail(word, terms) for word in words}
    return sum(
        (
            abs(expression[word]) * sum((tails[left] + tails[right] for left, right in pairs), Fraction(0))
            for word, pairs in cuts.items()
        ),
        Fraction(0),
    )


def _cutoff(
    expression: dict[Word, Fraction], cuts: dict[Word, tuple[tuple[Word, Word], ...]], bits: int, budget: _Budget
) -> int:
    _integer(bits, 1, "bits")
    budget.spend(bits)
    target, terms = Fraction(1, 2**bits), 1
    while _width_bound(expression, cuts, terms, budget) > target:
        terms *= 2
    return terms


def _polylogs(words: set[Word], terms: int, budget: _Budget) -> dict[Word, Interval]:
    budget.spend(sum(len(word) ** 2 + sum(word) for word in words))
    suffixes = {word[i:] for word in words for i in range(len(word) + 1)}
    ordered = sorted((word for word in suffixes if word), key=len, reverse=True)
    budget.spend(terms * sum(sum(word) + len(word) for word in ordered))
    harmonic = {word: Fraction(0) for word in ordered}
    harmonic[()] = Fraction(1)
    partial = {word: Fraction(0) for word in words}
    for n in range(1, terms + 1):
        for word in words:
            if word:
                partial[word] += harmonic[word[1:]] / (2**n * n ** word[0])
        # Decreasing depth preserves H_(n-1) in each strict suffix update.
        for word in ordered:
            harmonic[word] += harmonic[word[1:]] / n ** word[0]
    return {
        word: (partial[word], min(Fraction(1), partial[word] + _tail(word, terms))) if word else (Fraction(1), Fraction(1))
        for word in words
    }


def polylog_enclosure(indices: Word, terms: int = 64, work_limit: int = DEFAULT_WORK_LIMIT) -> Interval:
    """Enclose Li_indices(1/2), allowing a first index of one."""
    _word(indices, admissible=False)
    _integer(terms, 1, "terms")
    return _polylogs({indices}, terms, _budget(work_limit))[indices]


def _enclose(
    expression: dict[Word, Fraction], cuts: dict[Word, tuple[tuple[Word, Word], ...]], terms: int, budget: _Budget
) -> Interval:
    words = {part for pairs in cuts.values() for pair in pairs for part in pair}
    values = _polylogs(words, terms, budget)
    lower = upper = expression.get((), Fraction(0))
    for word, pairs in cuts.items():
        left = sum((values[a][0] * values[b][0] for a, b in pairs), Fraction(0))
        right = sum((values[a][1] * values[b][1] for a, b in pairs), Fraction(0))
        coefficient = expression[word]
        lower += coefficient * (left if coefficient >= 0 else right)
        upper += coefficient * (right if coefficient >= 0 else left)
    return lower, upper


def mzv_enclosure(indices: Word, terms: int = 64, work_limit: int = DEFAULT_WORK_LIMIT) -> Interval:
    """Enclose a strict descending-index MZV using a fixed dyadic cutoff."""
    _word(indices)
    _integer(terms, 1, "terms")
    budget = _budget(work_limit)
    expression = {indices: Fraction(1)}
    cuts = _prepare(expression, budget)
    return _enclose(expression, cuts, terms, budget)


def _precision(expression: dict[Word, Fraction], bits: int, budget: _Budget) -> tuple[Interval, int]:
    cuts = _prepare(expression, budget)
    terms = _cutoff(expression, cuts, bits, budget)
    interval = _enclose(expression, cuts, terms, budget)
    if interval[1] - interval[0] > Fraction(1, 2**bits):
        raise ArithmeticError("the dyadic interval exceeds its proved target width")
    return interval, terms


def mzv_precision_enclosure(indices: Word, bits: int = 32, work_limit: int = DEFAULT_WORK_LIMIT) -> Interval:
    """Enclose an admissible MZV with absolute width at most 2**(-bits)."""
    _word(indices)
    return _precision({indices: Fraction(1)}, bits, _budget(work_limit))[0]


def constant_enclosure(index: int, bits: int = 32, work_limit: int = DEFAULT_WORK_LIMIT) -> Interval:
    """Enclose c_index with target absolute width, including signed cancellation."""
    budget = _budget(work_limit)
    return _precision(_construct(index, budget), bits, budget)[0]


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: (root / name).read_bytes()
        for name in (
            "tools/risk_research/mzv_dyadic.py",
            "tools/risk_research/leading_constant_mzv.py",
            "tools/risk_research/exact.py",
            "tools/risk_research/saved_records.py",
            PROOF,
            "docs/advanced/skeleton_risk_leading_constant_mzv.rst",
            "docs/advanced/skeleton_risk_vertex_general.rst",
        )
    }


def _record(expression: dict[Word, Fraction], bits: int, budget: _Budget) -> dict[str, object]:
    (lower, upper), terms = _precision(expression, bits, budget)
    return {
        "schema": SCHEMA,
        "proof_status": "proved_dyadic_mzv_enclosure",
        "proof": PROOF,
        "bits": bits,
        "terms": terms,
        "work_limit": budget.limit,
        "selected_work_units": budget.used,
        "mzv_convention": "strict-descending-indices; empty word is rational unit",
        "mzv_terms": [{"indices": list(word), "coefficient": _fraction(c)} for word, c in expression.items()],
        "value": {"lower": _fraction(lower), "upper": _fraction(upper)},
        "width": _fraction(upper - lower),
        "target_absolute_width": _fraction(Fraction(1, 2**bits)),
        "scope": "exact MZV or fixed-index leading constant; no finite-degree risk or relative-error claim",
    }


def _finish(record: dict[str, object], sources: dict[str, bytes]) -> dict[str, object]:
    record.update(
        source_hash_policy=HASH_POLICY,
        source_sha256={name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()},
    )
    if _sources() != sources:
        raise RuntimeError("a dyadic evaluator or proof source changed during calculation")
    return record


def build_record(indices: Word = (2,), bits: int = 32, work_limit: int = DEFAULT_WORK_LIMIT) -> dict[str, object]:
    """Build a complete, source-bound record for one admissible MZV."""
    sources = _sources()
    _word(indices)
    record = _record({indices: Fraction(1)}, bits, _budget(work_limit))
    record.update(kind="multiple_zeta", indices=list(indices))
    return _finish(record, sources)


def build_constant_record(index: int = 7, bits: int = 32, work_limit: int = DEFAULT_WORK_LIMIT) -> dict[str, object]:
    """Build a complete record for a constructed, signed c_index expression."""
    sources, budget = _sources(), _budget(work_limit)
    expression = _construct(index, budget)
    record = _record(expression, bits, budget)
    scale = Fraction((index + 1) ** 2 * factorial(index), 2 ** (index - 2))
    interval = record["value"]
    assert isinstance(interval, dict)
    lower = Fraction(int(interval["lower"]["numerator"]), int(interval["lower"]["denominator"]))
    upper = Fraction(int(interval["upper"]["numerator"]), int(interval["upper"]["denominator"]))
    record.update(
        kind="leading_constant",
        constant_index=index,
        vertices=index + 1,
        vertex_scale=_fraction(scale),
        vertex_leading_constant={"lower": _fraction(scale * lower), "upper": _fraction(scale * upper)},
    )
    return _finish(record, sources)


def verify_record(record: object) -> None:
    """Restore every generation option and compare the entire strictly typed record."""
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported dyadic MZV record")
    bits = _integer(record.get("bits"), 1, "bits")
    work_limit = _integer(record.get("work_limit"), 1, "work limit")
    if record.get("kind") == "multiple_zeta":
        raw = record.get("indices")
        if type(raw) is not list:
            raise ValueError("saved MZV indices must be a list")
        expected = build_record(tuple(raw), bits, work_limit)
    elif record.get("kind") == "leading_constant":
        index = _integer(record.get("constant_index"), 3, "constant index")
        expected = build_constant_record(index, bits, work_limit)
    else:
        raise ValueError("unsupported dyadic MZV record kind")
    if canonical_json(record) != canonical_json(expected):
        raise ValueError("saved record differs from exact recomputation or sources")


def _protect_output(path: Path, sources: dict[str, bytes]) -> None:
    root = Path(__file__).resolve().parents[2]
    if path.resolve() in {(root / name).resolve() for name in sources}:
        raise ValueError("output must not replace evaluator or proof sources")


def _indices(text: str) -> Word:
    try:
        return _word(tuple(int(token) for token in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    """Generate or verify a dyadic source-bound MZV or leading-constant record."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--indices", type=_indices)
    group.add_argument("--constant-index", type=int)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--work-limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    sources = _sources()
    try:
        if args.verify is not None:
            if any(
                value is not None for value in (args.indices, args.constant_index, args.bits, args.work_limit, args.output)
            ):
                raise ValueError("--verify cannot be combined with generation options")
            verify_record(json.loads(args.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
            print("verified dyadic MZV record")
            return 0
        if args.output is None:
            raise ValueError("generation requires --output")
        _protect_output(args.output, sources)
        bits = 32 if args.bits is None else args.bits
        limit = DEFAULT_WORK_LIMIT if args.work_limit is None else args.work_limit
        record = (
            build_record((2,) if args.indices is None else args.indices, bits, limit)
            if args.constant_index is None
            else build_constant_record(args.constant_index, bits, limit)
        )
        if _sources() != sources:
            raise RuntimeError("a dyadic evaluator or proof source changed during calculation")
        output = args.output.parent.resolve() / args.output.name
        _protect_output(output, sources)
        atomic_json(output, record)
    except (OSError, ValueError, TypeError, ArithmeticError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
