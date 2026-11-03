"""Check finite MZV algebra against separate finite sums and saved contracts."""

import json
import sys
from fractions import Fraction as F
from itertools import combinations
from math import comb, prod
from pathlib import Path
from typing import Any

import pytest

from tools.risk_research import leading_constant_mzv as mzv
from tools.risk_research.vertex_general import _phi_rows


def _harmonic(n: int, word: tuple[int, ...]) -> F:
    return sum(
        (
            prod((F(1, k**s) for k, s in zip(reversed(indices), word, strict=True)), start=F(1))
            for indices in combinations(range(1, n + 1), len(word))
        ),
        F(0),
    )


def _rational(poles: mzv.PartialFractions, n: int) -> F:
    return sum((c / (n + shift) ** power for (shift, power), c in poles.items()), F(0))


@pytest.mark.parametrize(
    "index, expected",
    [
        (3, {(2,): F(6)}),
        (4, {(2,): F(48), (2, 1): F(-48)}),
        (5, {(2,): F(-120), (2, 1): F(-600), (2, 1, 1): F(720), (2, 2): F(240)}),
        (
            6,
            {
                (2,): F(600),
                (2, 1): F(2520),
                (2, 1, 1): F(11280),
                (2, 1, 1, 1): F(-14400),
                (2, 1, 2): F(-4320),
                (2, 2): F(3600),
                (2, 2, 1): F(-4320),
            },
        ),
    ],
)
def test_known_low_weight_coefficients(index: int, expected: dict[tuple[int, ...], F]) -> None:
    assert mzv.leading_constant_expression(index) == expected


@pytest.mark.parametrize("index", [3, 4, 5, 6])
@pytest.mark.parametrize("n", [1, 2, 7, 13])
def test_initial_kernel_matches_original_elementary_rows(index: int, n: int) -> None:
    rows = _phi_rows(index - 1, n)
    expected = index * sum((comb(index - 1, a) * rows[a][-1] * rows[index - 1 - a][-1] for a in range(1, index - 1)), F(0))
    kernels = mzv._initial(index, mzv._budget(200000))
    actual = sum((_rational(poles, n) * _harmonic(n - 1, word) for word, poles in kernels.items()), F(0))
    assert actual == expected


@pytest.mark.parametrize("left,right", [(0, 0), (0, 3), (1, 1), (1, 3), (2, 2), (3, 2)])
def test_finite_product_counts_descending_chains(left: int, right: int) -> None:
    expansion = mzv._ones_product(left, right, mzv._budget(10000))
    for n in (0, 1, 4, 7):
        assert _harmonic(n, (1,) * left) * _harmonic(n, (1,) * right) == sum(
            (c * _harmonic(n, word) for word, c in expansion.items()), F(0)
        )


@pytest.mark.parametrize("factors", [{0: 2}, {0: 1, 1: 1}, {0: 2, 1: 2, 3: 1}, {0: 1, 2: 2, 4: 1}])
def test_partial_fractions_equal_original_product(factors: dict[int, int]) -> None:
    poles = mzv._partial_fractions(factors, mzv._budget(10000))
    for n in (1, 2, 19):
        assert _rational(poles, n) == F(1, prod((n + shift) ** power for shift, power in factors.items()))


@pytest.mark.parametrize("word", [(), (1,), (2,), (1, 2), (2, 1, 1)])
def test_exact_finite_summation_by_parts_boundary(word: tuple[int, ...]) -> None:
    poles = mzv._partial_fractions({0: 2, 1: 2, 3: 1}, mzv._budget(10000))
    a, g = mzv._telescope(poles, mzv._budget(10000))
    for n in (1, 2, 9):
        assert _rational(poles, n) == _rational(g, n + 1) - _rational(g, n) + a / n**2
        left = sum((_rational(poles, k) * _harmonic(k - 1, word) for k in range(1, n + 1)), F(0))
        right = a * _harmonic(n, (2,) + word) + _rational(g, n + 1) * _harmonic(n, word)
        right -= _rational(g, 1) * _harmonic(0, word)
        if word:
            right -= sum((_rational(g, k + 1) / k ** word[0] * _harmonic(k - 1, word[1:]) for k in range(1, n + 1)), F(0))
        assert left == right


def test_nondecaying_kernel_is_rejected() -> None:
    with pytest.raises(ArithmeticError, match="order"):
        mzv._telescope({(0, 1): F(1)}, mzv._budget(100))


@pytest.mark.parametrize("word", [(2,), (2, 1), (2, 2), (3, 1), (2, 1, 1)])
def test_enclosures_contain_independent_larger_finite_prefix(word: tuple[int, ...]) -> None:
    lower, upper = mzv.mzv_enclosure(word, 7)
    assert lower == _harmonic(7, word)
    assert lower <= _harmonic(41, word) <= upper


def test_zeta_two_has_sharp_elementary_integral_bracket() -> None:
    lower, upper = mzv.mzv_enclosure((2,), 64)
    # A separate larger prefix and both elementary integral tails bracket zeta(2).
    assert upper - lower == F(1, 64)
    prefix = sum((F(1, n * n) for n in range(1, 129)), F(0))
    assert lower < prefix + F(1, 129) < prefix + F(1, 128) < upper
    c_lower, c_upper = mzv.constant_enclosure(3, 64)
    assert (c_lower, c_upper) == (6 * lower, 6 * upper)


@pytest.mark.parametrize("bad", [None, True, False, 2, -1, 3.0, "3"])
def test_constant_index_validation(bad: object) -> None:
    value = json.loads(json.dumps(bad))
    with pytest.raises(ValueError):
        mzv.leading_constant_expression(value)


@pytest.mark.parametrize("bad", [None, True, 0, -1, 1.0, "1"])
def test_truncation_validation(bad: object) -> None:
    value = json.loads(json.dumps(bad))
    with pytest.raises(ValueError):
        mzv.constant_enclosure(3, value)


@pytest.mark.parametrize("word", [(), (1,), (2, 0), (2, True), (2.0,), [2], None])
def test_word_validation(word: Any) -> None:
    # The deliberately heterogeneous JSON-like boundary is tested at runtime.
    with pytest.raises(ValueError):
        mzv.mzv_enclosure(word)


def test_resource_refusals_are_local_and_preserve_integer_conversion_limit() -> None:
    before = sys.get_int_max_str_digits()
    for action in (
        lambda: mzv.leading_constant_expression(10**5000),
        lambda: mzv.leading_constant_expression(6, 1),
        lambda: mzv.leading_constant_expression(3, mzv.MAX_WORK_LIMIT + 1),
        lambda: mzv.constant_enclosure(3, 10**5000),
        lambda: mzv.mzv_enclosure((10**100,), 2),
        lambda: mzv.mzv_enclosure((2,) + (1,) * 10000, 1),
    ):
        with pytest.raises(ValueError, match="work limit"):
            action()
    encoded = mzv._fraction(F(10**5000 + 1, 3))
    assert len(encoded["numerator"]) == 5001
    assert sys.get_int_max_str_digits() == before


def test_saved_record_restores_nondefault_options_and_preserves_input() -> None:
    record = json.loads(json.dumps(mzv.build_record(6, 23, 190000)))
    before = json.dumps(record, sort_keys=True)
    mzv.verify_record(record)
    assert json.dumps(record, sort_keys=True) == before
    assert record["constant_index"] == 6 and record["terms"] == 23 and record["work_limit"] == 190000
    assert record["vertex_scale"] == {"numerator": "2205", "denominator": "1"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("constant_index", True),
        ("terms", 0),
        ("work_limit", False),
        ("vertices", 7.0),
        ("selected_work_units", False),
        ("extra", "field"),
    ],
)
def test_record_types_and_extra_fields_are_checked(field: str, value: object) -> None:
    record = json.loads(json.dumps(mzv.build_record(5, 11)))
    record[field] = value
    with pytest.raises(ValueError):
        mzv.verify_record(record)


def test_changed_coefficient_and_source_are_rejected() -> None:
    for field in ("mzv_terms", "source_sha256"):
        record = json.loads(json.dumps(mzv.build_record(5, 11)))
        if field == "mzv_terms":
            record[field][0]["coefficient"]["numerator"] = "0"
        else:
            key = next(iter(record[field]))
            record[field][key] = "0" * 64
        with pytest.raises(ValueError):
            mzv.verify_record(record)


def test_cli_duplicate_keys_and_option_collisions_preserve_files(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"schema":"a","schema":"b"}')
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        mzv.main(["--verify", str(path)])
    assert path.read_bytes() == before
    with pytest.raises(SystemExit):
        mzv.main(["--verify", str(path), "--terms", "11"])
    assert path.read_bytes() == before


def test_cli_roundtrip_and_failed_generation_preserve_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    assert mzv.main(["--constant-index", "6", "--terms", "17", "--work-limit", "180000", "--output", str(path)]) == 0
    before = path.read_bytes()
    assert mzv.main(["--verify", str(path)]) == 0
    with pytest.raises(SystemExit):
        mzv.main(["--constant-index", "6", "--work-limit", "1", "--output", str(path)])
    assert path.read_bytes() == before


def test_output_cannot_replace_a_proof_source_or_symlink_alias(tmp_path: Path) -> None:
    source = Path(mzv.__file__).resolve()
    alias = tmp_path / "source-alias.py"
    alias.symlink_to(source)
    before = source.read_bytes()
    for output in (source, alias):
        with pytest.raises(SystemExit):
            mzv.main(["--output", str(output)])
        assert source.read_bytes() == before


def test_writer_uses_fixed_canonical_parent_after_alias_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, other = tmp_path / "original", tmp_path / "other"
    original.mkdir()
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    protect = mzv._protect_output
    calls: list[Path] = []

    def change_alias(path: Path, sources: dict[str, bytes]) -> None:
        protect(path, sources)
        calls.append(path)
        if len(calls) == 2:
            alias.unlink()
            alias.symlink_to(other, target_is_directory=True)

    monkeypatch.setattr(mzv, "_protect_output", change_alias)
    assert mzv.main(["--constant-index", "3", "--terms", "7", "--output", str(alias / "record.json")]) == 0
    assert (original / "record.json").is_file()
    assert not (other / "record.json").exists()
