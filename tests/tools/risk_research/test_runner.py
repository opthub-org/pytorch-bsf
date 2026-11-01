"""Check independent oracles, resumability and honest recurrence evidence."""

import json
import subprocess
from fractions import Fraction
from unittest.mock import patch

import pytest

from tools.risk_research.__main__ import integer_list, main
from tools.risk_research.exact import SCHEMA, Case, exact_coefficient, run_case, source_fingerprint, worker
from tools.risk_research.recurrences import guess_recurrence, sequence_from_manifest


@pytest.mark.parametrize(("vertices", "degree", "skeleton"), [(2, 1, 0), (3, 3, 1), (4, 2, 0), (4, 1, 3)])
def test_scalar_and_gram_oracles_agree(vertices, degree, skeleton):
    assert exact_coefficient(Case(vertices, degree, skeleton, "scalar")) == exact_coefficient(
        Case(vertices, degree, skeleton, "gram")
    )


@pytest.mark.parametrize("arguments", [(1, 2, 0), (3, 0, 0), (3, 2, 3), (3, 2, -1), (True, 2, 0), (3, 2, 0, "unknown")])
def test_invalid_cases(arguments):
    with pytest.raises(ValueError):
        Case(*arguments)


def test_resume_only_successful_results_and_invalidate_source(tmp_path):
    metadata = {"source_fingerprint": "abc", "git_sha": "head", "git_dirty": True}
    response = subprocess.CompletedProcess(
        [],
        0,
        json.dumps({"value": {"numerator": "2", "denominator": "3"}, "runtime_seconds": 0.01, "source_fingerprint": "abc"}),
        "",
    )
    changed = subprocess.CompletedProcess([], 0, response.stdout.replace('"abc"', '"changed"'), "")
    with patch("tools.risk_research.exact.subprocess.run", side_effect=[response, changed]) as launch:
        first = run_case(Case(2, 1, 0), tmp_path, metadata)
        second = run_case(Case(2, 1, 0), tmp_path, metadata, role="validation")
        third = run_case(Case(2, 1, 0), tmp_path, {**metadata, "source_fingerprint": "changed"})
    assert first["status"] == "exact" and not first["cache_hit"]
    assert second["cache_hit"] and second["role"] == "validation"
    assert not third["cache_hit"] and launch.call_count == 2


def test_timeout_has_no_value_or_success_cache_and_can_be_retried(tmp_path):
    metadata = {"source_fingerprint": "abc", "git_sha": "head", "git_dirty": True}
    with patch("tools.risk_research.exact.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 0.01)) as launch:
        for _ in range(2):
            result = run_case(Case(2, 1, 0), tmp_path, metadata, timeout=0.01)
            assert result["status"] == "timeout" and "value" not in result
    assert launch.call_count == 2
    assert not list(tmp_path.rglob("*.json"))


def test_source_fingerprint_captures_uncommitted_oracle_changes(tmp_path):
    path = tmp_path / "tests" / "sampling.py"
    path.parent.mkdir()
    path.write_text("original", encoding="utf-8")
    first = source_fingerprint(tmp_path)
    path.write_text("modified", encoding="utf-8")
    assert source_fingerprint(tmp_path) != first


def test_successfully_checked_recurrence_remains_a_conjecture():
    result = guess_recurrence(
        {index: Fraction(index) for index in range(1, 13)}, order=1, polynomial_degree=1, train_through=7
    )
    assert result["status"] == "conjecture"
    assert result["proof_status"] == "finite_exact_checks_only"
    assert result["coefficients"] == [["-1", "-1"], ["0", "1"]]
    training, held_out, residuals = result["training_degrees"], result["held_out_degrees"], result["held_out_residuals"]
    assert isinstance(training, list) and isinstance(held_out, list) and isinstance(residuals, dict)
    assert set(training).isdisjoint(held_out)
    assert all(value == "0" for value in residuals.values())


def test_false_recurrence_is_rejected_on_unseen_data():
    sequence = {index: Fraction(index) for index in range(1, 13)}
    sequence[10] = Fraction(1000)
    result = guess_recurrence(sequence, order=1, polynomial_degree=1, train_through=7)
    assert result["status"] == "rejected"
    residuals = result["held_out_residuals"]
    assert isinstance(residuals, dict)
    assert any(value != "0" for value in residuals.values())


def test_recurrence_retains_singular_indices_and_restart_values():
    sequence = {index: Fraction((index - 3) ** 2) for index in range(1, 16)}
    result = guess_recurrence(sequence, order=1, polynomial_degree=2, train_through=10)
    assert result["status"] == "conjecture"
    assert result["exceptional_indices"] == [3]
    assert result["initial_values"] == {"1": "4", "4": "1"}


def test_recurrence_requires_a_disjoint_complete_validation_window():
    with pytest.raises(ValueError, match="held-out"):
        guess_recurrence({index: Fraction(index) for index in range(1, 8)}, order=1, polynomial_degree=1, train_through=7)


def test_recurrence_reports_missing_startup_values_in_sparse_input():
    previous, current = 0, 1
    sequence = {}
    for index in range(1, 21):
        sequence[index] = Fraction(current)
        previous, current = current, previous + current
    del sequence[2]
    result = guess_recurrence(sequence, order=2, polynomial_degree=0, train_through=12)
    assert result["status"] == "conjecture"
    assert result["domain_start"] == 1
    assert result["required_initial_indices"] == [1, 2]
    assert result["missing_initial_indices"] == [2]
    assert result["initial_values"] == {"1": "1"}


def test_recurrence_reports_restart_values_beyond_observed_data():
    sequence = {index: Fraction(1, index - 50) for index in range(1, 13)}
    result = guess_recurrence(sequence, order=1, polynomial_degree=1, train_through=7)
    assert result["status"] == "conjecture"
    assert result["exceptional_indices"] == [49]
    assert result["required_initial_indices"] == [1, 50]
    assert result["missing_initial_indices"] == [50]


def test_manifest_rejects_conflicting_reference_values(tmp_path):
    records = [
        {
            "status": "exact",
            "case": {"vertices": 4, "degree": 2, "skeleton": 0},
            "value": {"numerator": value, "denominator": "1"},
        }
        for value in ("1", "2")
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"schema": SCHEMA, "records": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        sequence_from_manifest(path, 0)


def test_cli_rejects_invalid_case_without_starting_process(tmp_path):
    assert main(["generate", "--vertices", "1", "--degrees", "1:2", "--output", str(tmp_path / "data.json")]) == 2
    assert not (tmp_path / "data.json").exists()


def test_interval_parser():
    assert integer_list("3:5") == [3, 4, 5]
    assert integer_list("2,1,2") == [1, 2]


@pytest.mark.parametrize("corruption", [[], {"status": []}, {"schema": SCHEMA, "records": [{}]}])
def test_bad_manifest_is_not_used(tmp_path, corruption):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(corruption), encoding="utf-8")
    with pytest.raises(ValueError):
        sequence_from_manifest(path, 0)


def test_corrupted_cache_is_recomputed(tmp_path):
    directory = tmp_path / "abc"
    directory.mkdir()
    (directory / "M2-D1-m0-scalar.json").write_text("[]", encoding="utf-8")
    metadata = {"source_fingerprint": "abc", "git_sha": "head", "git_dirty": True}
    with patch("tools.risk_research.exact.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 0.01)) as launch:
        result = run_case(Case(2, 1, 0), tmp_path, metadata, timeout=0.01)
    assert result["status"] == "timeout" and launch.call_count == 1


def test_worker_rejects_changed_source_before_computing():
    with (
        patch("tools.risk_research.exact.source_fingerprint", return_value="changed"),
        patch("tools.risk_research.exact.exact_coefficient") as compute,
        pytest.raises(ValueError, match="source changed"),
    ):
        worker(Case(2, 1, 0), expected_fingerprint="original")
    compute.assert_not_called()


def test_worker_rejects_changed_source_during_computation():
    with (
        patch("tools.risk_research.exact.source_fingerprint", side_effect=["original", "changed"]),
        patch("tools.risk_research.exact.exact_coefficient", return_value=Fraction(4, 3)),
        pytest.raises(ValueError, match="source changed"),
    ):
        worker(Case(2, 1, 0), expected_fingerprint="original")


def test_real_worker_subprocess_and_cache(tmp_path):
    output = tmp_path / "data.json"
    arguments = [
        "generate",
        "--vertices",
        "2",
        "--degrees",
        "1",
        "--skeletons",
        "0",
        "--cache",
        str(tmp_path / "cache"),
        "--output",
        str(output),
        "--timeout",
        "30",
    ]
    assert main(arguments) == 0
    result = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert result["status"] == "exact" and result["value"] == {"numerator": "4", "denominator": "3"}
    assert not result["cache_hit"]
    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["records"][0]["cache_hit"]
