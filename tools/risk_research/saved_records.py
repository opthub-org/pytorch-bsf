"""Replay selected saved vertex enclosures using their recorded options.

Only the three listed schemas have this regeneration contract. This checks
the complete source-bound record; it does not establish a new mathematical
proof or provide a universal verifier for every research output.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

SCHEMAS = {
    "skeleton-risk-general-vertex-error-v1": "vertex_general_asymptotic",
    "skeleton-risk-refined-vertex-error-v1": "vertex_refined_asymptotic",
    "skeleton-risk-large-vertex-constant-v1": "vertex_large_dimension",
}


def canonical_json(record: object) -> str:
    """Compare JSON scalar types exactly and reject nonfinite JSON numbers."""
    return json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _integer(record: dict[str, object], name: str) -> str:
    value = record.get(name)
    if type(value) is not int or value < 1:
        raise ValueError(f"recorded {name} must be a positive integer")
    return str(value)


def _target_arguments(record: dict[str, object]) -> list[str]:
    if "relative_error_target" not in record:
        return []
    target = record["relative_error_target"]
    if type(target) is not dict or set(target) != {"numerator", "denominator"}:
        raise ValueError("invalid recorded relative error target")
    if any(type(value) is not str for value in target.values()):
        raise ValueError("relative error target must contain decimal strings")
    try:
        value = Fraction(int(target["numerator"]), int(target["denominator"]))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid recorded relative error target") from error
    return ["--relative-target", str(value)]


def _generation_arguments(record: dict[str, object], module_name: str) -> list[str]:
    arguments = ["--vertices", _integer(record, "vertices")]
    if module_name != "vertex_large_dimension":
        arguments.extend(["--degree", _integer(record, "degree"), "--constant-terms", _integer(record, "constant_terms")])
        if "coefficient_ball" in record:
            arguments.extend(["--with-arb", "--bits", _integer(record, "bits")])
    else:
        arguments.extend(["--bits", _integer(record, "bits")])
        if "constant_ball" in record:
            arguments.append("--with-arb")
        if "normalized_integral_ball" in record:
            arguments.append("--integral-check")
    return arguments + _target_arguments(record)


def _regenerate(module_name: str, arguments: list[str], output: Path) -> None:
    # Execute the origin check and the generator in the same process. Capturing
    # child output avoids changing this process's global streams during replay.
    root = Path(__file__).resolve().parents[2]
    source = """
import importlib
import sys
from pathlib import Path
module = importlib.import_module(sys.argv[1])
origin = Path(module.__file__).resolve()
if not origin.is_relative_to(Path.cwd().resolve()) or origin != Path(sys.argv[2]).resolve():
    raise RuntimeError('saved-record generator import escaped its source checkout')
raise SystemExit(module.main(sys.argv[3:]))
"""
    command = [
        sys.executable,
        "-c",
        source,
        "tools.risk_research." + module_name,
        str(root / "tools" / "risk_research" / (module_name + ".py")),
        *arguments,
        "--output",
        str(output),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("saved enclosure regeneration timed out") from error
    if result.returncode:
        raise ValueError("saved enclosure could not be regenerated: " + result.stderr.strip())


def verify_record(record: object) -> None:
    """Regenerate a supported enclosure without changing the saved record."""
    if type(record) is not dict or type(record.get("schema")) is not str or record["schema"] not in SCHEMAS:
        raise ValueError("unsupported saved enclosure schema")
    supplied = canonical_json(record)
    module_name = SCHEMAS[record["schema"]]
    arguments = _generation_arguments(record, module_name)
    with tempfile.TemporaryDirectory(prefix="bsf-record-replay-") as directory:
        output = Path(directory) / "regenerated.json"
        _regenerate(module_name, arguments, output)
        expected = json.loads(output.read_text(encoding="utf-8"))
    if supplied != canonical_json(expected):
        raise ValueError("saved enclosure differs from its recorded options, source or recomputation")


def main(argv: list[str] | None = None) -> int:
    """Verify a saved enclosure with no generation overrides or output writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        verify_record(json.loads(arguments.verify.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
    except (OSError, ValueError, TypeError, ModuleNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print("verified saved vertex enclosure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
