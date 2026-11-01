"""Validate recurrence screening's data and field assumptions."""

import json
from argparse import Namespace
from fractions import Fraction

import pytest

from tools.risk_research import research_scan
from tools.risk_research.research_scan import modular_attempt, read_experiment


def _manifest(path, records):
    path.write_text(json.dumps({"schema": "skeleton-risk-integer-experiment-v1", "records": records}), encoding="utf-8")
    return path


def test_exact_normalizations_preserve_degree_indices(tmp_path):
    path = _manifest(tmp_path / "sequence.json", [{"degree": 3, "numerator": "2", "denominator": "7"}])
    assert read_experiment(path) == {3: Fraction(2, 7)}
    assert read_experiment(path, rising=2, central_binomial=True) == {3: Fraction(480, 7)}


def test_conflicting_records_are_rejected(tmp_path):
    path = _manifest(
        tmp_path / "sequence.json",
        [{"degree": 3, "numerator": "1", "denominator": "2"}, {"degree": 3, "numerator": "1", "denominator": "3"}],
    )
    with pytest.raises(ValueError, match="conflicting"):
        read_experiment(path)


@pytest.mark.parametrize("degree", [True, 0, -1, "3"])
def test_invalid_indices_are_rejected(tmp_path, degree):
    path = _manifest(tmp_path / "sequence.json", [{"degree": degree, "numerator": "1", "denominator": "2"}])
    with pytest.raises(ValueError, match="positive integers"):
        read_experiment(path)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(1.9, 2.9), (True, "2"), (1, 2), ("01", "2"), ("1", "+2"), ("1", "0"), ("1", "-2"), (None, "2")],
)
def test_malformed_exact_values_are_rejected(tmp_path, numerator, denominator):
    path = _manifest(tmp_path / "sequence.json", [{"degree": 1, "numerator": numerator, "denominator": denominator}])
    with pytest.raises(ValueError, match="malformed exact-data record"):
        read_experiment(path)


@pytest.mark.parametrize(
    "document",
    [None, [], {}, {"schema": "skeleton-risk-integer-experiment-v1", "records": {}}],
)
def test_malformed_manifest_is_rejected(tmp_path, document):
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        read_experiment(path)


@pytest.mark.parametrize("record", [None, [], {}, {"degree": 1}, {"degree": 1, "numerator": "1"}])
def test_malformed_record_is_rejected(tmp_path, record):
    path = _manifest(tmp_path / "sequence.json", [record])
    with pytest.raises(ValueError, match="record"):
        read_experiment(path)


def test_modular_screen_detects_recurrence_and_excludes_smaller_space():
    pytest.importorskip("flint")
    sequence = {index: Fraction(index) for index in range(1, 15)}
    assert modular_attempt(sequence, order=1, degree=0, train_through=10, prime=1009)["status"] == "excluded_by_full_rank"
    assert modular_attempt(sequence, order=1, degree=1, train_through=10, prime=1009)["status"] == "possible_nullspace"


def test_missing_data_and_bad_modulus_are_not_exclusions():
    pytest.importorskip("flint")
    sequence = {index: Fraction(1, 7) for index in range(1, 15)}
    assert modular_attempt(sequence, order=2, degree=4, train_through=4, prime=1009)["status"] == "insufficient_data"
    assert modular_attempt(sequence, order=1, degree=1, train_through=10, prime=7)["status"] == "noninvertible_denominator"
    with pytest.raises(ValueError, match="must be prime"):
        modular_attempt(sequence, order=1, degree=1, train_through=10, prime=1000)


@pytest.mark.parametrize(
    ("degrees", "status", "skipped"), [([1, 2, 3, 4, 10], "incomplete_search", 1), (list(range(1, 8)), "conjecture_found", 0)]
)
def test_exact_fit_without_held_out_windows_is_incomplete(tmp_path, monkeypatch, degrees, status, skipped):
    source = _manifest(
        tmp_path / "input.json", [{"degree": degree, "numerator": "1", "denominator": "1"} for degree in degrees]
    )
    monkeypatch.setattr(research_scan, "modular_attempt", lambda *args, **kwargs: {"status": "possible_nullspace"})
    output = tmp_path / "output.json"
    research_scan._worker(
        Namespace(
            input=source,
            output=output,
            rising=0,
            central_binomial=False,
            train_through=4,
            prime=1009,
            max_order=1,
            max_degree=0,
            max_unknowns=2,
        )
    )
    document = json.loads(output.read_text())
    assert document["status"] == status
    assert document["skipped_spaces"] == skipped
    assert document["excluded_spaces"] == 0
    if skipped:
        assert "held-out window" in document["attempts"][0]["exact_fit_error"]
        assert document["candidates"] == []
    else:
        assert document["candidates"][0]["proof_status"] == "finite_exact_checks_only"


def test_input_replaced_during_screen_is_rejected_before_output(tmp_path, monkeypatch):
    path = _manifest(
        tmp_path / "sequence.json", [{"degree": index, "numerator": str(index), "denominator": "1"} for index in range(1, 8)]
    )

    def changing_attempt(*args, **kwargs):
        path.write_text("changed", encoding="utf-8")
        return {"status": "excluded_by_full_rank"}

    monkeypatch.setattr(research_scan, "modular_attempt", changing_attempt)
    output = tmp_path / "output.json"
    arguments = Namespace(
        input=path,
        output=output,
        rising=0,
        central_binomial=False,
        train_through=4,
        prime=1009,
        max_order=1,
        max_degree=0,
        max_unknowns=2,
    )
    with pytest.raises(RuntimeError, match="input data changed"):
        research_scan._worker(arguments)
    assert not output.exists()
