"""Check broader domains, original budgets and independent constant bounds."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from math import factorial
from pathlib import Path

import pytest

from tests.sampling import _scalar_risk_sum
from tools.risk_research import vertex_joint_growth as old
from tools.risk_research import vertex_joint_growth_refined as joint
from tools.risk_research.vertex_general import leading_constant_enclosure, vertex_coefficient
from tools.risk_research.vertex_general_asymptotic import error_constants

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("outside", [3, 5, 9, 17, 31, 67, 127, 161])
def test_refined_budget_covers_original_exact_error_constants(outside):
    vertices = outside + 1
    constants = error_constants(vertices)
    arc = Fraction(vertices**2 * factorial(outside) * outside, 4)
    assert constants.remainder / arc <= (512 * outside**2) ** outside
    if vertices >= 162:
        normalized = 1 << (outside // 3)
        gain = joint.large_vertex_gain(vertices)
        improved = vertices**2 * factorial(outside) * normalized
        assert constants.remainder / improved <= Fraction((512 * outside**2) ** outside, 1 << gain)


@pytest.mark.parametrize("vertices", [4, 5, 7, 13])
def test_arc_bound_covers_small_vertices_using_original_exact_series(vertices):
    outside = vertices - 1
    lower, upper = leading_constant_enclosure(vertices, 192)
    assert Fraction(vertices**2 * factorial(outside) * outside, 4) < lower < upper


@pytest.mark.parametrize(("vertices", "degree"), [(4, 5), (5, 7), (6, 9)])
def test_old_scalar_oracle_checks_finite_normalization_outside_huge_domain(vertices, degree):
    coefficient = vertex_coefficient(vertices, degree)
    assert coefficient == _scalar_risk_sum(vertices, degree, 1)
    lower, upper = leading_constant_enclosure(vertices, 96)
    assert coefficient > 0 and 0 < lower < upper
    # This feasible prefix is deliberately outside the simultaneous domain.
    with pytest.raises(ValueError, match="degree_floor_exponent"):
        joint.build_certificate(vertices, degree.bit_length() - 1)


@pytest.mark.parametrize(("exponent", "ceiling"), [(384, 4), (1024, 8), (2046, 16), (2047, 15), (10**9, 2777778)])
def test_new_ceiling_has_correct_start_and_binary_log_jumps(exponent, ceiling):
    assert joint.maximum_vertices(exponent) == ceiling
    record = joint.build_certificate(ceiling, exponent)
    joint.verify_certificate(record)
    logarithm = record["logarithm_ceiling"]
    assert 1 << (logarithm - 1) < exponent + 2 <= 1 << logarithm
    assert all(value >= 0 for value in record["margins"].values())
    with pytest.raises(ValueError, match="ceiling"):
        joint.build_certificate(ceiling + 1, exponent)


def test_entire_old_domain_is_contained_even_at_binary_log_jumps():
    exponents = set(range(1024, 12000))
    exponents.update(2**power + delta for power in range(11, 65) for delta in (-2, -1, 0, 1))
    for exponent in exponents:
        assert joint.maximum_vertices(exponent) >= old.maximum_vertices(exponent)
    assert joint.maximum_vertices(1024) > old.maximum_vertices(1024)
    assert joint.build_certificate(4, 384)["old_domain_included"] is False
    with pytest.raises(ValueError):
        old.build_certificate(8, 1024)


@pytest.mark.parametrize("exponent", [384, 385, 510, 511, 512, 1024, 2046, 2047, 40000])
def test_exact_compressed_block_bound_covers_both_endpoints(exponent):
    vertices = joint.maximum_vertices(exponent)
    outside = vertices - 1
    record = joint.build_certificate(vertices, exponent)
    logarithm = record["logarithm_ceiling"]
    gain = record["large_vertex_gain"]
    assert 512 * outside**2 <= (exponent + 2) ** 2
    assert 12 * logarithm * outside <= exponent
    numerator = (512 * outside**2) ** outside * (exponent + 2) ** (outside - 1)
    assert numerator**4 <= 1 << exponent
    for degree in (1 << exponent, (1 << (exponent + 1)) - 1):
        bound = Fraction(numerator, degree * (1 << gain))
        power = record["relative_error_block_upper"]["denominator_power_of_two"]
        assert bound <= Fraction(1, 1 << power)
        sharper = record["budget_block_upper"]["denominator_power_of_two"]
        assert bound <= Fraction(1, 1 << sharper)


@pytest.mark.parametrize("outside", [160, 161, 162, 163, 164, 255, 256, 257, 8191, 8192, 8193])
def test_large_vertex_branch_and_extra_gain_have_no_missing_boundary(outside):
    gain = joint.large_vertex_gain(outside + 1)
    if outside == 160:
        assert gain == 0
    else:
        assert gain >= 1
        assert Fraction(outside, 4 * (1 << (outside // 3))) <= Fraction(1, 1 << gain)
    record = joint.build_certificate(outside + 1, max(40000, 1000 * outside))
    assert record["constant_lower_method"] == ("positive_arc" if outside == 160 else "refined_large_vertex")


@pytest.mark.parametrize("digits", [100, 5000])
def test_enormous_exponents_remain_compressed_without_global_conversion_changes(digits):
    limit = sys.get_int_max_str_digits()
    exponent = 10**digits
    record = joint.build_certificate(joint.maximum_vertices(exponent), exponent)
    serialized = joint._canonical(record)
    assert len(serialized) < 30 * (digits + 1) + 5000
    decoded = json.loads(serialized, parse_int=joint._parse_integer)
    joint.verify_certificate(decoded)
    assert sys.get_int_max_str_digits() == limit


@pytest.mark.parametrize("vertices", [True, False, 3, 4.0, "4", None])
def test_invalid_vertex_types_are_rejected(vertices):
    with pytest.raises(ValueError, match="vertices"):
        joint.build_certificate(vertices, 1024)
    with pytest.raises(ValueError, match="vertices"):
        joint.large_vertex_gain(vertices)


@pytest.mark.parametrize("exponent", [True, False, 383, 384.0, "384", None])
def test_invalid_exponent_types_and_one_below_start_are_rejected(exponent):
    with pytest.raises(ValueError, match="degree_floor_exponent"):
        joint.maximum_vertices(exponent)


def test_every_saved_field_and_type_is_recomputed_strictly():
    record = joint.build_certificate(162, 40000)

    def paths(value, prefix=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from paths(item, (*prefix, key))
        else:
            yield prefix

    for path in paths(record):
        changed = copy.deepcopy(record)
        parent = changed
        for key in path[:-1]:
            parent = parent[key]
        original = parent[path[-1]]
        parent[path[-1]] = original + 1 if type(original) is int else "altered"
        with pytest.raises(ValueError):
            joint.verify_certificate(changed)
    for value in (True, 1.0, "1"):
        changed = copy.deepcopy(record)
        changed["relative_error_block_upper"]["numerator"] = value
        with pytest.raises(ValueError):
            joint.verify_certificate(changed)
    changed = copy.deepcopy(record)
    changed["unknown"] = 1
    with pytest.raises(ValueError):
        joint.verify_certificate(changed)
    del changed["source_sha256"]
    with pytest.raises(ValueError):
        joint.verify_certificate(changed)


@pytest.mark.parametrize("record", [None, [], 1, "record", {}, {"vertices": 4, "degree_floor_exponent": 383}])
def test_invalid_saved_record_shapes_are_rejected(record):
    with pytest.raises(ValueError):
        joint.verify_certificate(record)


def test_cli_roundtrip_rejects_duplicate_keys_and_generation_overrides(tmp_path):
    output = tmp_path / "joint.json"
    arguments = ["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(output)]
    assert joint.main(arguments) == 0
    previous = output.read_bytes()
    assert joint.main(["--verify", str(output)]) == 0
    assert output.read_bytes() == previous
    for extra in (["--vertices", "8"], ["--output", str(output)], ["--degree-floor-exponent", "1024"]):
        with pytest.raises(SystemExit):
            joint.main(["--verify", str(output), *extra])
        assert output.read_bytes() == previous
    output.write_bytes(previous.rstrip()[:-1] + b',"vertices":8}')
    with pytest.raises(SystemExit):
        joint.main(["--verify", str(output)])


@pytest.mark.parametrize("extra", [["--vertices", "9"], ["--degree-floor-exponent", "383"]])
def test_invalid_generation_preserves_previous_output(tmp_path, extra):
    output = tmp_path / "previous.json"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        joint.main(["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(output), *extra])
    assert output.read_bytes() == b"keep"


def test_source_collisions_are_rejected_before_calculation(monkeypatch):
    def forbidden(*args):
        pytest.fail("calculation started before protected-source rejection")

    monkeypatch.setattr(joint, "build_certificate", forbidden)
    for name in joint._sources():
        with pytest.raises(SystemExit):
            joint.main(["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(ROOT / name)])


def test_output_leaf_symlink_is_replaced_without_mutating_its_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b"preserve target")
    output = tmp_path / "output.json"
    output.symlink_to(target)
    assert joint.main(["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(output)]) == 0
    assert not output.is_symlink() and target.read_bytes() == b"preserve target"
    joint.verify_certificate(json.loads(output.read_bytes()))


def test_original_parent_alias_retarget_at_writer_entry_does_not_redirect_output(tmp_path, monkeypatch):
    safe, other = tmp_path / "safe", tmp_path / "other"
    safe.mkdir()
    other.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(safe, target_is_directory=True)
    writer = joint.atomic_json

    def retarget(path, record):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        writer(path, record)

    monkeypatch.setattr(joint, "atomic_json", retarget)
    assert joint.main(["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(alias / "record.json")]) == 0
    assert (safe / "record.json").exists() and not (other / "record.json").exists()


def test_failed_write_and_raw_source_change_preserve_previous_output(tmp_path, monkeypatch):
    output = tmp_path / "previous.json"
    output.write_bytes(b"keep")
    arguments = ["--vertices", "8", "--degree-floor-exponent", "1024", "--output", str(output)]

    def fail(*args):
        raise OSError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(joint, "atomic_json", fail)
        with pytest.raises(SystemExit):
            joint.main(arguments)
    sources = joint._sources()
    changed = {name: value.replace(b"\n", b"\r\n") for name, value in sources.items()}
    iterator = iter([sources, sources, changed])
    monkeypatch.setattr(joint, "_sources", lambda: next(iterator))
    with pytest.raises(SystemExit):
        joint.main(arguments)
    assert output.read_bytes() == b"keep"


def test_real_lf_and_crlf_checkouts_reproduce_without_optional_dependencies(tmp_path):
    records = []
    for ending in (b"\n", b"\r\n"):
        checkout = tmp_path / ("lf" if ending == b"\n" else "crlf")
        for name, data in joint._sources().items():
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", ending))
        output = tmp_path / (checkout.name + ".json")
        command = [sys.executable, "-S", "-m", "tools.risk_research.vertex_joint_growth_refined"]
        result = subprocess.run(
            [*command, "--vertices", "162", "--degree-floor-exponent", "40000", "--output", str(output)],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        records.append(json.loads(output.read_bytes()))
        result = subprocess.run(
            [*command, "--verify", str(output)],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
    assert records[0] == records[1]


def test_native_independent_integral_confirms_large_vertex_normalized_lower_bounds():
    executable = os.environ.get("BSF_FLINT_PYTHON") or os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        if importlib.util.find_spec("flint") is None:
            pytest.skip("set BSF_FLINT_PYTHON for independent positive-integral checks")
        executable = sys.executable
    script = """
from flint import arb,ctx
from tools.risk_research.vertex_large_dimension import integral_normalized_ball
ctx.prec=73
for m in (162,164,258,603):
    direct=integral_normalized_ball(m,bits=96)
    assert ctx.prec==73
    with ctx.workprec(128):
        q=m-1
        normalized=direct*(2*arb(2).log())**q
        assert normalized>arb(2)**(q//3),(m,normalized)
assert ctx.prec==73
print('independent positive integrals and exact normalization passed')
"""
    result = subprocess.run(
        [executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
