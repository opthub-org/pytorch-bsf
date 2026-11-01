"""Generate exact risk data in bounded, restartable subprocesses."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "skeleton-risk-exact-v1"


@dataclass(frozen=True)
class Case:
    """Specify one simplex coefficient and its independent reference method."""

    vertices: int
    degree: int
    skeleton: int
    oracle: str = "scalar"

    def __post_init__(self) -> None:
        """Reject invalid cases before starting a process."""
        for name in ("vertices", "degree", "skeleton"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.vertices < 2 or self.degree < 1 or not 0 <= self.skeleton < self.vertices:
            raise ValueError("require vertices >= 2, degree >= 1 and 0 <= skeleton < vertices")
        if not isinstance(self.oracle, str) or self.oracle not in {"scalar", "gram"}:
            raise ValueError("oracle must be scalar or gram")


def fraction_record(value: Fraction) -> dict[str, str]:
    """Encode an exact value without JSON integer-precision loss."""
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def record_fraction(value: dict[str, str]) -> Fraction:
    """Decode a canonical exact value, rejecting invalid denominators."""
    numerator_text, denominator_text = value["numerator"], value["denominator"]
    if not isinstance(numerator_text, str) or not isinstance(denominator_text, str):
        raise ValueError("exact numerators and denominators must be decimal strings")
    numerator, denominator = int(numerator_text), int(denominator_text)
    if str(numerator) != numerator_text or str(denominator) != denominator_text:
        raise ValueError("exact values must use canonical decimal integers")
    if denominator <= 0:
        raise ValueError("an exact denominator must be positive")
    return Fraction(numerator, denominator)


def exact_coefficient(case: Case) -> Fraction:
    """Evaluate a coefficient with a reference separate from production code."""
    if case.skeleton >= case.degree:
        return Fraction(0)
    # Lazy imports keep the coordinator lightweight and bound oracle imports too.
    from tests.sampling import _exact_coefficients, _scalar_risk_sum

    if case.oracle == "scalar":
        return _scalar_risk_sum(case.vertices, case.degree, case.skeleton + 1)
    return _exact_coefficients(case.vertices, case.degree)[0][case.skeleton]


def source_fingerprint(root: Path = ROOT) -> str:
    """Hash oracle/tool sources and environment versions, including dirty edits."""
    digest = hashlib.sha256()
    paths = set(root.glob("torch_bsf/**/*.py"))
    paths.update(root.glob("tests/*.py"))
    paths.update(root.glob("tools/risk_research/*.py"))
    paths.update(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file())
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ("torch", "numpy", "sympy", "pytest", "hypothesis"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return digest.hexdigest()


def provenance(root: Path = ROOT) -> dict[str, object]:
    """Record the checkout revision alongside a content-based fingerprint."""

    def git(*arguments: str) -> str:
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    dirty = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "tools/risk_research",
        "tests",
        "torch_bsf",
        "pyproject.toml",
        "uv.lock",
    )
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "source_fingerprint": source_fingerprint(root),
        "python": sys.version.split()[0],
    }


def atomic_json(path: Path, value: object) -> None:
    """Replace a JSON artifact only after its complete contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def worker(case: Case, memory_mib: int | None = None, expected_fingerprint: str | None = None) -> None:
    """Compute one case, optionally enforcing the OS address-space limit."""
    fingerprint = source_fingerprint()
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("source changed before the oracle started; retry the batch")
    if memory_mib is not None:
        import resource

        memory_bytes = memory_mib * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    started = time.monotonic()
    value = exact_coefficient(case)
    elapsed = time.monotonic() - started
    if source_fingerprint() != fingerprint:
        raise ValueError("source changed during the oracle computation; retry the batch")
    print(json.dumps({"value": fraction_record(value), "runtime_seconds": elapsed, "source_fingerprint": fingerprint}))


def _cached_result(path: Path, case: Case, metadata: dict[str, object]) -> dict[str, object] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        valid = (
            isinstance(result, dict)
            and result["schema"] == SCHEMA
            and result["status"] == "exact"
            and Case(**result["case"]) == case
            and result["source_fingerprint"] == metadata["source_fingerprint"]
        )
        if valid:
            record_fraction(result["value"])
            for field in ("runtime_seconds", "elapsed_seconds"):
                number = result[field]
                if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                    return None
            if not isinstance(result["git_sha"], str) or type(result["git_dirty"]) is not bool:
                return None
            return cast(dict[str, object], result)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _worker_output(stdout: str, fingerprint: object) -> tuple[Fraction, float]:
    payload = json.loads(stdout)
    value = record_fraction(payload["value"])
    if payload["source_fingerprint"] != fingerprint:
        raise ValueError("source fingerprint differs from the requested oracle source")
    runtime = payload["runtime_seconds"]
    if type(runtime) not in (int, float) or not math.isfinite(runtime) or runtime < 0:
        raise ValueError("invalid worker runtime")
    return value, float(runtime)


def run_case(
    case: Case,
    cache_dir: Path,
    metadata: dict[str, object],
    *,
    timeout: float = 60,
    memory_mib: int | None = None,
    role: str = "exploration",
) -> dict[str, object]:
    """Reuse exact results, or launch one bounded and independently retryable case."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if memory_mib is not None and (type(memory_mib) is not int or memory_mib <= 0):
        raise ValueError("memory_mib must be a positive integer")
    if role not in {"exploration", "validation"}:
        raise ValueError("role must be exploration or validation")
    name = f"M{case.vertices}-D{case.degree}-m{case.skeleton}-{case.oracle}.json"
    path = cache_dir / str(metadata["source_fingerprint"]) / name
    cached = _cached_result(path, case, metadata)
    if cached is not None:
        return {**cached, "role": role, "cache_hit": True}
    result = {
        **metadata,
        "schema": SCHEMA,
        "case": asdict(case),
        "role": role,
        "cache_hit": False,
        "timeout_seconds": timeout,
        "memory_mib": memory_mib,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    command = [
        sys.executable,
        "-m",
        "tools.risk_research",
        "_worker",
        json.dumps(asdict(case)),
        "--source-fingerprint",
        str(metadata["source_fingerprint"]),
    ]
    if memory_mib is not None:
        command.extend(["--memory-mib", str(memory_mib)])
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {**result, "status": "timeout", "elapsed_seconds": time.monotonic() - started, "timeout_seconds": timeout}
    if completed.returncode:
        return {
            **result,
            "status": "error",
            "elapsed_seconds": time.monotonic() - started,
            "returncode": completed.returncode,
            "error": completed.stderr[-4000:],
        }
    try:
        value, runtime = _worker_output(completed.stdout, metadata["source_fingerprint"])
    except (ValueError, KeyError, TypeError) as error:
        return {**result, "status": "error", "error": f"invalid oracle output: {error}"}
    result["value"] = fraction_record(value)
    result["runtime_seconds"] = runtime
    result["status"] = "exact"
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(path, result)
    return result
