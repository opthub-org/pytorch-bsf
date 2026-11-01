"""Protect toolchain verification against silently disabled assertions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/risk_research/sage_smoke.py"


@pytest.mark.parametrize("optimized_environment", [False, True])
def test_optimized_python_cannot_report_smoke_checks_as_passed(tmp_path, optimized_environment):
    report = tmp_path / "report.json"
    environment = os.environ.copy()
    if optimized_environment:
        environment["PYTHONOPTIMIZE"] = "1"
    command = [sys.executable, *([] if optimized_environment else ["-O"]), str(SCRIPT), "--output", str(report)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Run without -O or PYTHONOPTIMIZE" in result.stderr
    assert not report.exists()
