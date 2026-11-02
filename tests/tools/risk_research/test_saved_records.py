"""Replay only supported saved schemas with their actual generation options."""

import json
import os
import subprocess
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


def test_native_flint_precision_and_backend_record_round_trip(tmp_path):
    executable = os.environ.get("BSF_FLINT_PYTHON")
    if not executable:
        pytest.skip("set BSF_FLINT_PYTHON for native Arb replay regressions")
    source = """
import sys
from pathlib import Path
from tools.risk_research import vertex_refined_asymptotic as module, saved_records
output=Path(sys.argv[1])
module.main(['--vertices','5','--degree','257','--constant-terms','17','--bits','47','--with-arb','--output',str(output)])
before=output.read_bytes()
saved_records.main(['--verify',str(output)])
assert output.read_bytes()==before
"""
    result = subprocess.run(
        [executable, "-c", source, str(tmp_path / "arb.json")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
