"""Contracts for growing-domain and fully computable center certificates."""

import importlib.util
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tools.risk_research import vertex_composite_asymptotic as composite
from tools.risk_research.checkout_harness import checkout_state, run_python
from tools.risk_research.vertex_joint_growth_refined import maximum_vertices as old_ceiling
from tools.risk_research.vertex_refined_asymptotic import error_constants


def _fraction(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _copy(record: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = json.loads(composite._canonical(record), parse_int=composite._parse_integer)
    return copied


@pytest.mark.parametrize(
    ("exponent", "ceiling"),
    [(256, 4), (323, 4), (324, 5), (384, 5), (510, 7), (511, 6), (1024, 12), (40000, 313)],
)
def test_exact_domain_boundaries(exponent: int, ceiling: int) -> None:
    assert composite.maximum_vertices(exponent) == ceiling
    assert composite.degree_error_exponent(ceiling, exponent) >= 3 * exponent // 4
    with pytest.raises(ValueError, match="ceiling"):
        composite.degree_error_exponent(ceiling + 1, exponent)


@pytest.mark.parametrize("exponent", [True, 255, 256.0, "256"])
def test_degree_domain_rejects_nonintegers_and_outside(exponent: Any) -> None:
    with pytest.raises(ValueError, match="integer"):
        composite.maximum_vertices(exponent)


@pytest.mark.parametrize("vertices", [True, 3, 4.0, "4"])
def test_vertex_validation(vertices: Any) -> None:
    with pytest.raises(ValueError, match="integer"):
        composite.build_record(vertices, 1024)


@pytest.mark.parametrize("bits", [True, 31, 47.0, "47"])
def test_precision_validation(bits: Any) -> None:
    with pytest.raises(ValueError, match="integer"):
        composite.build_record(5, 384, bits=bits)


@pytest.mark.parametrize("target", [True, 1, 0.01, Fraction(0), Fraction(-1)])
def test_target_validation(target: Any) -> None:
    with pytest.raises(ValueError, match="positive Fraction"):
        composite.build_record(258, 40000, target=target)


def test_old_domain_inclusion_including_logarithm_jumps() -> None:
    exponents = set(range(384, 4097))
    exponents.update((1 << k) + offset for k in range(12, 100) for offset in (-2, -1, 0, 1))
    for exponent in exponents:
        assert composite.maximum_vertices(exponent) >= old_ceiling(exponent)
    assert composite.maximum_vertices(1024) == 12 > old_ceiling(1024) == 8


@pytest.mark.parametrize(("vertices", "exponent"), [(4, 256), (5, 324), (7, 510), (12, 1024), (37, 4501)])
def test_original_constant_budget_fits_new_domain(vertices: int, exponent: int) -> None:
    q = vertices - 1
    logarithm_power = 2 if vertices == 4 else 1
    original_budget = (512 * q**2) ** q * (exponent + 2) ** logarithm_power
    assert original_budget <= 1 << (exponent // 4)
    assert composite.degree_error_exponent(vertices, exponent) >= 3 * exponent // 4


@pytest.mark.parametrize("vertices", [5, 8, 13, 20])
def test_single_log_constants_independently_obey_arc_majorant(vertices: int) -> None:
    from math import factorial

    original = error_constants(vertices)
    q = vertices - 1
    arc = Fraction(vertices**2 * factorial(q) * q, 4)
    assert original.remainder / arc <= (512 * q**2) ** q


@pytest.mark.parametrize(
    ("vertices", "method"),
    [
        (4, "local_leading"),
        (161, "local_leading"),
        (162, "refined_leading"),
        (257, "refined_leading"),
        (258, "second_order_corrected"),
        (4099, "second_order_corrected"),
    ],
)
def test_center_branches_cover_every_transition(vertices: int, method: str) -> None:
    record = composite.build_record(vertices, 1_000_000, bits=47)
    assert record["center"]["method"] == method
    lower = _fraction(record["center"]["normalized_lower"])
    upper = _fraction(record["center"]["normalized_upper"])
    midpoint = _fraction(record["center"]["normalized_rational_midpoint"])
    assert 0 < lower <= midpoint <= upper
    assert midpoint == (lower + upper) / 2
    assert _fraction(record["elementary_error_relative_to_rational_midpoint"]) == (upper - lower) / (upper + lower)


def test_constant_radius_uses_corrected_center_only_after_its_domain() -> None:
    assert composite.constant_relative_error(162) == Fraction(6, 160)
    assert composite.constant_relative_error(257) == Fraction(6, 255)
    assert composite.constant_relative_error(258) == Fraction(84, 256**2)
    assert composite.constant_relative_error(4) > 1


@pytest.mark.parametrize("vertices", [162, 258, 4099])
def test_more_elementary_precision_narrows_the_same_center(vertices: int) -> None:
    low = composite.center_enclosure(vertices, bits=32)
    high = composite.center_enclosure(vertices, bits=79)
    assert low[0] <= high[0] <= high[1] <= low[1]
    assert high[1] - high[0] < low[1] - low[0]


def test_composite_bound_includes_all_three_product_cross_terms() -> None:
    record = composite.build_record(258, 40000, bits=47, target=Fraction(1, 100))
    constant = _fraction(record["constant_error_relative_to_exact_center"])
    numerical = _fraction(record["elementary_error_relative_to_rational_midpoint"])
    bound = record["composite_relative_to_computable_center"]
    base, coefficient = _fraction(bound["base"]), _fraction(bound["tail_coefficient"])
    assert base == constant + numerical + constant * numerical
    assert coefficient == 1 + base
    degree = Fraction(1, 1 << bound["tail_denominator_power_of_two"])
    assert base + coefficient * degree == (1 + constant) * (1 + numerical) * (1 + degree) - 1
    assert record["target_met_on_entire_block"] is True
    assert composite.build_record(257, 40000, bits=47, target=Fraction(1, 100))["target_met_on_entire_block"] is False


def test_compressed_target_comparison_matches_exact_fractions() -> None:
    for base in (Fraction(0), Fraction(1, 7), Fraction(2)):
        for exponent in range(16):
            for target in (Fraction(1, 100), Fraction(1), Fraction(20)):
                expected = base + (1 + base) / 2**exponent <= target
                assert composite._affine_target(base, 1 + base, exponent, target) is expected
    assert composite._affine_target(Fraction(1, 3), Fraction(4, 3), 10**6000, Fraction(1, 2))
    assert not composite._affine_target(Fraction(1), Fraction(2), 10**6000, Fraction(1))


def test_roundtrip_restores_nondefault_precision_and_target() -> None:
    record = composite.build_record(258, 38017, bits=47, target=Fraction(1, 777))
    before = composite._canonical(record)
    composite.verify_record(record)
    assert composite._canonical(record) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vertices", True),
        ("bits", 47.0),
        ("degree_floor_exponent", "38017"),
        ("schema", "other"),
        ("degree_budget_margin", -1),
        ("extra", 0),
    ],
)
def test_replay_rejects_changed_top_fields(field: str, value: Any) -> None:
    record = _copy(composite.build_record(258, 38017, bits=47))
    record[field] = value
    with pytest.raises(ValueError):
        composite.verify_record(record)


@pytest.mark.parametrize("change", ["bound", "source", "center", "target", "nested_type"])
def test_replay_rejects_changed_nested_fields(change: str) -> None:
    record = _copy(composite.build_record(258, 38017, bits=47, target=Fraction(1, 100)))
    if change == "bound":
        record["composite_relative_to_computable_center"]["base"]["numerator"] = "0"
    elif change == "source":
        record["source_sha256"][composite.PROOF] = "0" * 64
    elif change == "center":
        record["center"]["normalized_rational_midpoint"]["denominator"] = "1"
    elif change == "target":
        record["relative_target"]["denominator"] = "+100"
    else:
        record["degree_error_relative_to_C_M"]["numerator"] = True
    with pytest.raises(ValueError):
        composite.verify_record(record)


def test_empty_and_nonobject_records_are_rejected() -> None:
    values: tuple[object, ...] = (None, [], {}, "record")
    for value in values:
        with pytest.raises(ValueError):
            composite.verify_record(value)


def test_very_large_degree_and_vertex_inputs_keep_local_digit_policy() -> None:
    before = sys.get_int_max_str_digits()
    vertices, exponent = 10**4500, 10**4550
    record = composite.build_record(vertices, exponent, bits=32, target=Fraction(1, 100))
    serialized = composite._canonical(record)
    parsed = json.loads(serialized, object_pairs_hook=composite._unique, parse_int=composite._parse_integer)
    composite.verify_record(parsed)
    assert parsed["vertices"] == vertices and parsed["degree_floor_exponent"] == exponent
    assert parsed["target_met_on_entire_block"] is True
    assert len(serialized) < 500_000
    assert sys.get_int_max_str_digits() == before


def test_raw_source_guard_rejects_even_a_line_ending_change(monkeypatch: pytest.MonkeyPatch) -> None:
    original = composite._sources()
    changed = {name: data.replace(b"\n", b"\r\n") for name, data in original.items()}
    calls = iter([original, changed])
    monkeypatch.setattr(composite, "_sources", lambda: next(calls))
    with pytest.raises(RuntimeError, match="sources changed"):
        composite.build_record(5, 384, bits=47)


def test_source_fingerprints_are_portable_between_lf_and_crlf(monkeypatch: pytest.MonkeyPatch) -> None:
    original = composite._sources()
    first = composite.build_record(5, 384, bits=47)
    monkeypatch.setattr(composite, "_sources", lambda: {name: data.replace(b"\n", b"\r\n") for name, data in original.items()})
    second = composite.build_record(5, 384, bits=47)
    assert first == second


def test_cli_generation_replay_and_input_preservation(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    with redirect_stdout(io.StringIO()):
        assert (
            composite.main(
                [
                    "--vertices",
                    "258",
                    "--degree-floor-exponent",
                    "40000",
                    "--bits",
                    "47",
                    "--relative-target",
                    "1/777",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    original = output.read_bytes()
    with redirect_stdout(io.StringIO()):
        assert composite.main(["--verify", str(output)]) == 0
    assert output.read_bytes() == original
    record = json.loads(original)
    assert record["bits"] == 47 and record["relative_target"] == {"numerator": "1", "denominator": "777"}
    with pytest.raises(SystemExit) as error:
        composite.main(["--verify", str(output), "--bits", "79"])
    assert error.value.code == 2 and output.read_bytes() == original


def test_cli_duplicate_key_and_invalid_generation_preserve_output(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    output.write_text("original output")
    with pytest.raises(SystemExit) as error:
        composite.main(["--vertices", "5", "--degree-floor-exponent", "323", "--output", str(output)])
    assert error.value.code == 2 and output.read_text() == "original output"
    invalid = tmp_path / "duplicate.json"
    record = composite._canonical(composite.build_record(5, 384, bits=47))
    invalid.write_text('{"vertices":5,' + record[1:])
    original = invalid.read_bytes()
    with pytest.raises(SystemExit):
        composite.main(["--verify", str(invalid)])
    assert invalid.read_bytes() == original


def test_output_cannot_alias_source_before_calculation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(composite.__file__)
    alias = tmp_path / "source-link.json"
    alias.symlink_to(source)
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise RuntimeError("calculation must not start")

    monkeypatch.setattr(composite, "build_record", forbidden)
    with pytest.raises(SystemExit):
        composite.main(["--vertices", "5", "--degree-floor-exponent", "384", "--output", str(alias)])
    assert not called and alias.is_symlink()


def test_unrelated_leaf_symlink_is_replaced_without_changing_target(tmp_path: Path) -> None:
    target, output = tmp_path / "target.txt", tmp_path / "record.json"
    target.write_text("target stays")
    output.symlink_to(target)
    with redirect_stdout(io.StringIO()):
        composite.main(["--vertices", "5", "--degree-floor-exponent", "384", "--bits", "47", "--output", str(output)])
    assert target.read_text() == "target stays" and not output.is_symlink()
    composite.verify_record(json.loads(output.read_text()))


def test_parent_alias_is_pinned_before_calculation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first, second, alias = tmp_path / "first", tmp_path / "second", tmp_path / "alias"
    first.mkdir()
    second.mkdir()
    alias.symlink_to(first, target_is_directory=True)
    original = composite.build_record

    def calculate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        return result

    monkeypatch.setattr(composite, "build_record", calculate)
    with redirect_stdout(io.StringIO()):
        composite.main(
            ["--vertices", "5", "--degree-floor-exponent", "384", "--bits", "47", "--output", str(alias / "saved.json")]
        )
    assert (first / "saved.json").exists() and not (second / "saved.json").exists()


@pytest.fixture
def native_python() -> str:
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("optional python-flint is unavailable")
        executable = sys.executable
    return executable


@pytest.fixture
def native_checkout(tmp_path: Path) -> Path:
    """Copy only the source closure into a clean independent Git fixture."""
    checkout = tmp_path / "native-checkout"
    checkout.mkdir()
    for name, data in composite._sources().items():
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    return checkout


@pytest.mark.parametrize("vertices", [163, 271, 619])
def test_original_native_integral_is_inside_center_and_constant_budget(
    vertices: int, native_python: str, native_checkout: Path
) -> None:
    script = f"""
from fractions import Fraction
from pathlib import Path
import flint
from tools.risk_research import vertex_composite_asymptotic as composite
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
assert Path(composite.__file__).resolve().is_relative_to(Path.cwd())
def fraction(value):
    return Fraction(int(value['numerator']),int(value['denominator']))
record=composite.build_record({vertices},100000,bits=47)
lower=fraction(record['center']['normalized_lower'])
upper=fraction(record['center']['normalized_upper'])
radius=fraction(record['constant_error_relative_to_exact_center'])
flint.ctx.prec=73
original=integral_normalized_ball({vertices},bits=83)
assert flint.ctx.prec==73
with flint.ctx.workprec(100):
    lo=flint.arb(flint.fmpq((lower*(1-radius)).numerator,(lower*(1-radius)).denominator))
    hi=flint.arb(flint.fmpq((upper*(1+radius)).numerator,(upper*(1+radius)).denominator))
    assert original.lower()>lo and original.upper()<hi
assert flint.ctx.prec==73
"""
    before = checkout_state(native_checkout)
    result = run_python(
        native_checkout,
        ["-c", script],
        modules=("tools.risk_research.vertex_composite_asymptotic",),
        executable=native_python,
        expected_head=before.head,
        expected_tree=before.tree,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert checkout_state(native_checkout) == before
