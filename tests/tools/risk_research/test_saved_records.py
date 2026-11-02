"""Replay only supported saved schemas with their actual generation options."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from tools.risk_research import saved_records
from tools.risk_research import vertex_general_asymptotic as general
from tools.risk_research import vertex_large_dimension as large
from tools.risk_research import vertex_refined_asymptotic as refined

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(params=["general", "refined", "large"])
def generated(request, tmp_path):
    module, arguments = {
        "general": (general, ["--vertices", "5", "--degree", "257", "--constant-terms", "17", "--relative-target", "1/7"]),
        "refined": (refined, ["--vertices", "6", "--degree", "257", "--constant-terms", "17", "--relative-target", "1/7"]),
        "large": (large, ["--vertices", "37", "--bits", "47", "--relative-target", "1/7"]),
    }[request.param]
    output = tmp_path / (request.param + ".json")
    with redirect_stdout(StringIO()):
        assert module.main([*arguments, "--output", str(output)]) == 0
    return output, json.loads(output.read_text())


def test_nondefault_generation_round_trips_through_plain_verify_without_writes(generated):
    output, record = generated
    before = output.read_bytes()
    assert saved_records.main(["--verify", str(output)]) == 0
    assert output.read_bytes() == before
    saved_records.verify_record(record)


@pytest.mark.parametrize("mutation", ["extra", "source", "options", "float", "bool", "nan"])
def test_altered_record_or_options_are_rejected_and_preserved(generated, mutation):
    output, record = generated
    sources = {path: path.read_bytes() for path in Path(saved_records.__file__).parent.glob("*.py")}
    if mutation == "extra":
        record["unexpected"] = 1
    elif mutation == "source":
        record["source_sha256"] = {}
    elif mutation == "options":
        key = "bits" if "normalized_lower" in record else "constant_terms"
        record[key] += 1
    else:
        record["vertices"] = {"float": float(record["vertices"]), "bool": True, "nan": float("nan")}[mutation]
    output.write_text(json.dumps(record), encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        saved_records.main(["--verify", str(output)])
    assert error.value.code == 2 and output.read_bytes() == before
    assert all(path.read_bytes() == content for path, content in sources.items())


@pytest.mark.parametrize("record", [None, [], {}, {"schema": []}, {"schema": "unknown"}])
def test_unsupported_schema_is_rejected(record):
    with pytest.raises(ValueError, match="unsupported"):
        saved_records.verify_record(record)


@pytest.mark.parametrize(
    "target",
    [None, [], {"numerator": "1"}, {"numerator": 1, "denominator": "7"}, {"numerator": "1", "denominator": "0"}],
)
def test_malformed_relative_target_is_rejected(generated, target):
    _, record = generated
    record["relative_error_target"] = target
    with pytest.raises(ValueError, match="target"):
        saved_records.verify_record(record)


def test_generation_overrides_and_output_are_not_supported(generated, tmp_path):
    output, _ = generated
    for extra in [["--bits", "47"], ["--output", str(tmp_path / "existing.json")]]:
        with pytest.raises(SystemExit) as error:
            saved_records.main(["--verify", str(output), *extra])
        assert error.value.code == 2


def test_record_line_endings_do_not_change_replay_or_saved_bytes(generated):
    output, record = generated
    # JSON whitespace is independent of each schema's source-hash policy.
    output.write_bytes(json.dumps(record, indent=2).replace("\n", "\r\n").encode())
    before = output.read_bytes()
    assert saved_records.main(["--verify", str(output)]) == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_json_keys_are_rejected_without_changing_record(generated, line_ending, nested):
    output, record = generated
    text = json.dumps(record, indent=2)
    if nested:
        first_source = next(iter(record["source_sha256"]))
        key = json.dumps(first_source) + ": " + json.dumps(record["source_sha256"][first_source])
    else:
        key = '"schema": ' + json.dumps(record["schema"])
    text = text.replace(key, key + "," + key, 1)
    output.write_bytes(text.replace("\n", line_ending).encode())
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        saved_records.main(["--verify", str(output)])
    assert error.value.code == 2 and output.read_bytes() == before


@pytest.mark.parametrize(
    "module,arguments",
    [
        (
            "vertex_general_asymptotic",
            ["--vertices", "5", "--degree", "257", "--constant-terms", "17", "--bits", "47", "--with-arb"],
        ),
        (
            "vertex_refined_asymptotic",
            ["--vertices", "5", "--degree", "257", "--constant-terms", "17", "--bits", "47", "--with-arb"],
        ),
        ("vertex_large_dimension", ["--vertices", "1026", "--bits", "47", "--with-arb", "--integral-check"]),
    ],
)
def test_native_flint_precision_and_backend_record_round_trip(tmp_path, module, arguments):
    executable = os.environ.get("BSF_FLINT_PYTHON")
    if not executable:
        pytest.skip("set BSF_FLINT_PYTHON for native Arb replay regressions")
    source = """
import hashlib,json,sys
from importlib import import_module
from pathlib import Path
from tools.risk_research import saved_records
module=import_module('tools.risk_research.'+sys.argv[1])
output=Path(sys.argv[2])
sources=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(module.__file__).parent.glob('*.py')}
source_before=sources()
module.main([*json.loads(sys.argv[3]),'--output',str(output)])
before=output.read_bytes()
saved_records.main(['--verify',str(output)])
assert output.read_bytes()==before and sources()==source_before
record=json.loads(before)
assert record['bits']==47 and 'python_flint' in record
"""
    result = subprocess.run(
        [executable, "-c", source, module, str(tmp_path / "arb.json"), json.dumps(arguments)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_from_another_directory_does_not_change_input_or_sources(generated, tmp_path, monkeypatch):
    output, record = generated
    before = output.read_bytes()
    sources = {path: path.read_bytes() for path in Path(saved_records.__file__).parent.glob("*.py")}
    monkeypatch.chdir(tmp_path)
    assert saved_records.main(["--verify", output.name]) == 0
    assert output.read_bytes() == before
    assert all(path.read_bytes() == content for path, content in sources.items())


def test_concurrent_replays_leave_process_stdout_and_input_unchanged(generated, monkeypatch):
    output, record = generated
    before = output.read_bytes()
    original_stdout, original_run = sys.stdout, subprocess.run
    overlap = threading.Barrier(2)

    def overlapping_run(*arguments, **keywords):
        overlap.wait(timeout=15)
        assert sys.stdout is original_stdout
        result = original_run(*arguments, **keywords)
        assert sys.stdout is original_stdout
        return result

    monkeypatch.setattr(subprocess, "run", overlapping_run)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(saved_records.verify_record, record) for _ in range(2)]
        for future in futures:
            future.result(timeout=30)
    assert sys.stdout is original_stdout and output.read_bytes() == before


def test_replay_timeout_is_bounded_and_preserves_input(generated, monkeypatch):
    output, record = generated
    before = output.read_bytes()

    def timeout(command, **keywords):
        assert keywords["timeout"] == 60
        raise subprocess.TimeoutExpired(command, keywords["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        saved_records.verify_record(record)
    assert output.read_bytes() == before
