"""Keep optional resident-memory accounting portable across research workers."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.risk_research import exact, research_exact, research_scan


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 17), ("linux", 17 * 1024)])
def test_peak_rss_units(monkeypatch, platform, expected):
    resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=17))
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(sys, "platform", platform)
    assert exact.peak_rss_bytes() == expected


def test_peak_rss_is_unavailable_without_resource(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    assert exact.peak_rss_bytes() is None


def test_research_modules_import_without_resource():
    code = """
import sys
sys.modules['resource'] = None
from tools.risk_research import research_exact, research_scan
assert research_exact.peak_rss_bytes() is None
assert research_scan.peak_rss_bytes() is None
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[3], check=True)


@pytest.mark.parametrize("worker", ["exact", "scan"])
def test_tool_cli_without_resource(tmp_path, worker):
    # sitecustomize simulates Windows in both the coordinator and its real child.
    (tmp_path / "sitecustomize.py").write_text("import sys\nsys.modules['resource'] = None\n", encoding="utf-8")
    output = tmp_path / "output.json"
    arguments = ["--output", str(output)]
    if worker == "exact":
        # The tiny power-one experiment only constructs a polynomial; no native
        # arithmetic is needed to check the subprocess/import/report boundary.
        (tmp_path / "flint.py").write_text("__version__ = 'test'\nfmpz_poly = list\n", encoding="utf-8")
        arguments += ["--vertices", "2", "--skeleton", "0", "--degrees", "1"]
    else:
        source = tmp_path / "input.json"
        source.write_text(
            json.dumps(
                {
                    "schema": "skeleton-risk-integer-experiment-v1",
                    "records": [{"degree": index, "numerator": "1", "denominator": "1"} for index in range(1, 8)],
                }
            ),
            encoding="utf-8",
        )
        arguments += ["--input", str(source), "--train-through", "4", "--max-unknowns", "0"]
    root = Path(__file__).resolve().parents[3]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root)))}
    result = subprocess.run(
        [sys.executable, "-m", f"tools.risk_research.research_{worker}", *arguments],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["peak_rss_bytes"] is None


@pytest.mark.parametrize("worker", ["exact", "scan"])
def test_workers_record_unavailable_peak_rss(tmp_path, monkeypatch, worker):
    monkeypatch.setitem(sys.modules, "resource", None)
    output = tmp_path / "output.json"
    if worker == "exact":
        monkeypatch.setattr(research_exact, "import_module", lambda name: SimpleNamespace(__version__="test"))
        kernel = research_exact.integer_kernel(1, 2, backend="python")
        monkeypatch.setattr(research_exact, "integer_kernel", lambda power, maximum: kernel)
        research_exact._worker(Namespace(vertices=2, skeleton=0, degrees=[2], output=output))
    else:
        source = tmp_path / "input.json"
        source.write_text(
            json.dumps(
                {
                    "schema": "skeleton-risk-integer-experiment-v1",
                    "records": [{"degree": index, "numerator": "1", "denominator": "1"} for index in range(1, 8)],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(research_scan, "modular_attempt", lambda *args, **kwargs: {"status": "excluded_by_full_rank"})
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
    assert json.loads(output.read_text())["peak_rss_bytes"] is None
