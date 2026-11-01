"""Round-trip the generated recurrence artifacts and their proof reports."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("module", "artifact", "status", "residual_field", "dimension"),
    [
        (
            "codimension_two_states",
            "six_vertex_tetrahedral_face_recurrence_order8.json",
            "certified_by_state_identity",
            "symbolic_state_residuals",
            8,
        ),
        (
            "vertex_degree_recurrence",
            "tetrahedron_vertex_recurrence_order10.json",
            "certified_by_rational_telescoping",
            "symbolic_harmonic_residuals",
            4,
        ),
    ],
)
def test_native_cli_generates_and_verifies_source_bound_reports(tmp_path, module, artifact, status, residual_field, dimension):
    executable = os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        pytest.skip("set BSF_SAGE_PYTHON for native Sage CLI regressions")
    source = """
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
module = import_module('tools.risk_research.' + sys.argv[1])
reference = json.loads(Path(sys.argv[2]).read_bytes())
directory = Path(sys.argv[3])
operator = directory / 'operator.json'
report = directory / 'proof.json'
for action in ('--output', '--verify'):
    stream = io.StringIO()
    with redirect_stdout(stream):
        module.main([action, str(operator), '--report', str(report)])
    proof = json.loads(stream.getvalue())
    assert proof == json.loads(report.read_bytes())
    assert proof['status'] == sys.argv[4]
    assert proof[sys.argv[5]] == ['0'] * int(sys.argv[6])
    assert proof['operator_sha256'] == hashlib.sha256(operator.read_bytes()).hexdigest()
    assert proof['source_sha256'] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    generated = json.loads(operator.read_bytes())
    assert generated['coefficients'] == reference['coefficients']
    assert generated['initial_values'] == reference['initial_values']
"""
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).resolve().parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [
            executable,
            "-c",
            source,
            module,
            str(ROOT / "tools/risk_research/examples" / artifact),
            str(tmp_path),
            status,
            residual_field,
            str(dimension),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
