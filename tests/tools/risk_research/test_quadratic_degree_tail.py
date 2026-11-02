"""Independent coefficient and certificate checks for the quadratic region."""

import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research import fixed_vertex_tail as old
from tools.risk_research import quadratic_degree_tail as tail
from tools.risk_research.comparison import _root_bounds


def test_frozen_original_kernel_fixtures_include_the_new_region_boundaries():
    fixture = json.loads((Path(__file__).parent / "fixtures/quadratic_degree_tail_holdouts.json").read_text())
    root = Path(__file__).resolve().parents[3]
    for source, digest in fixture["oracle_sources_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    certified_cases = []
    for row in fixture["cases"]:
        vertices, degree = row["vertices"], row["degree"]
        values = tuple(Fraction(int(value["numerator"]), int(value["denominator"])) for value in row["values"])
        assert len(values) == vertices
        bounds = tail.lower_skeleton_upper_bounds(vertices, degree)
        assert all(value <= bound for value, bound in zip(values[:-1], bounds, strict=True))
        if degree >= tail.degree_threshold(vertices):
            upper = sum(_root_bounds(value, 128)[1] for value in values)
            risk_upper = Fraction(upper * upper, 1 << 256)
            b0 = Fraction(degree ** (vertices - 1), math.factorial(vertices - 1))
            assert math.comb(degree + vertices - 1, vertices - 1) - risk_upper > b0 * tail.normalized_risk_margin(
                vertices, degree
            )
            certified_cases.append((vertices, degree))
    assert {(4, 128), (4, 129), (5, 200), (6, 288)} <= set(certified_cases)


@pytest.mark.parametrize("vertices,degree", [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 2), (5, 3), (6, 3)])
def test_direct_majorants_bound_independent_gram_coefficients(vertices, degree):
    actual = _exact_coefficients(vertices, degree)[0]
    actual += [Fraction(0)] * (vertices - len(actual))
    bounds = tail.lower_skeleton_upper_bounds(vertices, degree)
    assert len(bounds) == vertices - 1
    assert all(Fraction(0) <= value <= bound for value, bound in zip(actual, bounds, strict=False))
    assert all(bound == 0 for size, bound in enumerate(bounds, 1) if size > degree)


@pytest.mark.parametrize("vertices,degree", [(4, 11), (5, 9), (6, 7), (7, 5), (8, 6)])
def test_frozen_scalar_holdouts_bound_every_lower_skeleton(vertices, degree):
    for size, bound in enumerate(tail.lower_skeleton_upper_bounds(vertices, degree), 1):
        assert _scalar_risk_sum(vertices, degree, size) <= bound


def test_exact_constant_inequalities_used_in_the_uniform_proof():
    assert sum(Fraction(5**k, math.factorial(k)) for k in range(8)) > 128
    assert Fraction(6, 64) < Fraction(5, 16) ** 2
    assert Fraction(5, 16) + Fraction(2, 15) < Fraction(9, 20)
    assert 2 * Fraction(9, 20) + Fraction(9, 20) ** 2 / 8 == Fraction(2961, 3200)
    assert 1 - Fraction(2961, 3200) == tail.NORMALIZED_MARGIN > 0


@pytest.mark.parametrize("vertices", [4, 5, 6, 7, 12, 20])
def test_specializations_improve_old_tail_and_preserve_an_exact_margin(vertices):
    record = tail.certify_vertices(vertices)
    degree_text = record["from_degree"]
    assert isinstance(degree_text, str)
    threshold = int(degree_text)
    assert threshold == 8 * vertices**2 < old.degree_threshold(vertices)
    p = Fraction(vertices * (vertices - 1), threshold)
    assert 0 < p < Fraction(1, 8)
    for degree in (threshold, threshold + 1, 2 * threshold):
        expected = tail.NORMALIZED_MARGIN * Fraction(vertices * (vertices - 1), degree)
        assert tail.normalized_risk_margin(vertices, degree) == expected > 0
    tail.verify_certificate(json.loads(json.dumps(record)))


@pytest.mark.parametrize("invalid", [True, False, 3, 0, -1, 4.0, "4", None])
def test_invalid_vertex_domains_fail_without_widening_the_theorem(invalid):
    for function in (tail.degree_threshold, tail.coefficient_majorants, tail.certify_vertices):
        with pytest.raises(ValueError, match="vertices"):
            function(invalid)


@pytest.mark.parametrize("degree", [True, False, 0, -1, 1.0, "1", None])
def test_coefficient_bound_degree_validation(degree):
    with pytest.raises(ValueError, match="degree"):
        tail.lower_skeleton_upper_bounds(4, degree)


@pytest.mark.parametrize("degree", [True, 127, 0, 128.0, "128", None])
def test_margin_rejects_degrees_outside_the_certified_domain(degree):
    with pytest.raises(ValueError, match="degree"):
        tail.normalized_risk_margin(4, degree)


@pytest.mark.parametrize("record", [None, [], {}, {"vertices": True}, {"vertices": 4.0}, {"vertices": 3}])
def test_malformed_records_are_not_certificates(record):
    with pytest.raises(ValueError):
        tail.verify_certificate(record)


@pytest.mark.parametrize(
    "field",
    ["from_degree", "all_vertex_domain", "all_degree_domain", "strict_margin_over_p_b0_lower_bound", "source_sha256"],
)
def test_changed_theorem_fields_and_extra_fields_are_rejected(field):
    record = copy.deepcopy(tail.certify_vertices(6))
    record[field] = "altered"
    with pytest.raises(ValueError, match="does not match"):
        tail.verify_certificate(record)
    record = tail.certify_vertices(6)
    record["unexpected"] = 1
    with pytest.raises(ValueError, match="does not match"):
        tail.verify_certificate(record)


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_nested_json_integer_types_are_exact(replacement):
    record = tail.certify_vertices(6)
    majorants = record["lower_skeleton_majorants"]
    assert isinstance(majorants, list)
    majorants[0]["size"] = replacement
    with pytest.raises(ValueError, match="does not match"):
        tail.verify_certificate(record)


def test_cli_round_trip_preservation_and_duplicate_keys(tmp_path):
    output = tmp_path / "quadratic.json"
    assert tail.main(["--vertices", "6", "--output", str(output)]) == 0
    record = json.loads(output.read_text())
    assert tail.main(["--verify", str(output)]) == 0
    assert record["source_sha256"] == {
        name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in tail._sources().items()
    }
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        tail.main(["--vertices", "3", "--output", str(output)])
    assert output.read_bytes() == original
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"vertices": 5, "vertices": 6}')
    with pytest.raises(SystemExit):
        tail.main(["--verify", str(malformed)])
    assert output.read_bytes() == original


@pytest.mark.parametrize("name", list(tail._sources()))
def test_cli_rejects_each_fingerprinted_source_before_calling_writer(monkeypatch, name):
    root = Path(__file__).resolve().parents[3]
    before = tail._sources()
    calls = []
    monkeypatch.setattr(tail, "atomic_json", lambda path, record: calls.append(path))
    with pytest.raises(SystemExit) as error:
        tail.main(["--vertices", "4", "--output", str(root / name)])
    assert error.value.code == 2
    assert calls == []
    assert tail._sources() == before


def test_cli_rejects_a_resolved_source_alias_and_preserves_the_link(tmp_path, monkeypatch):
    protected = Path(tail.__file__).resolve()
    alias = tmp_path / "source-alias.json"
    try:
        alias.symlink_to(protected)
    except (OSError, NotImplementedError) as link_error:
        pytest.skip(f"symlink creation is unavailable: {link_error}")
    before = tail._sources()
    calls = []
    monkeypatch.setattr(tail, "atomic_json", lambda path, record: calls.append(path))
    with pytest.raises(SystemExit) as error:
        tail.main(["--vertices", "4", "--output", str(alias)])
    assert error.value.code == 2
    assert not calls and alias.is_symlink() and alias.resolve() == protected
    assert tail._sources() == before


@pytest.mark.parametrize("crlf", [False, True])
def test_hashes_are_portable_and_raw_concurrent_edits_preserve_output(tmp_path, monkeypatch, crlf):
    lf = {name: value.replace(b"\r\n", b"\n") for name, value in tail._sources().items()}
    windows = {name: value.replace(b"\n", b"\r\n") for name, value in lf.items()}
    monkeypatch.setattr(tail, "_sources", lambda: lf)
    expected = tail.certify_vertices(6)
    monkeypatch.setattr(tail, "_sources", lambda: windows)
    assert tail.certify_vertices(6) == expected
    count = 0

    def changing_sources():
        nonlocal count
        count += 1
        return lf if count == 1 else windows

    monkeypatch.setattr(tail, "_sources", changing_sources)
    output = tmp_path / "existing.json"
    output.write_bytes(b"previous complete certificate\n")
    with pytest.raises(SystemExit):
        tail.main(["--vertices", "6", "--output", str(output)])
    assert output.read_bytes() == b"previous complete certificate\n"
    monkeypatch.setattr(tail, "_sources", lambda: windows if crlf else lf)
    tail.verify_certificate(expected)


@pytest.mark.parametrize("arguments", [[], ["--vertices", "4"], ["--verify", "unused", "--output", "unused"]])
def test_cli_invalid_argument_combinations_fail(arguments):
    with pytest.raises(SystemExit) as error:
        tail.main(arguments)
    assert error.value.code == 2
