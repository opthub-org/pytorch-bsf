"""Execute contributor entry-point examples from their actual document sources."""

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from torch_bsf.io import load_model

ROOT = Path(__file__).resolve().parents[1]


def test_contributing_fit_predict(tmp_path):
    text = (ROOT / "CONTRIBUTING.md").read_text()
    section = text.split("### 2. Try the CLI", 1)[1].split("### 3.", 1)[0]
    commands = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
    assert len(commands) == 1
    for name in ("params.csv", "values.csv"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for line in commands[0].splitlines():
        arguments = shlex.split(line)
        if not arguments:
            continue
        assert arguments[0] == "python"
        # Use the pytest interpreter on every OS; a Windows bash can select
        # WSL or a different Python, which does not verify this installation.
        subprocess.run([sys.executable, *arguments[1:]], cwd=tmp_path, env=env, check=True)
    params = np.loadtxt(tmp_path / "params.csv", delimiter=",", ndmin=2)
    values = np.loadtxt(tmp_path / "values.csv", delimiter=",", ndmin=2)
    predicted = np.loadtxt(tmp_path / "predictions.csv", delimiter=",", ndmin=2)
    assert predicted.shape == values.shape
    assert np.isfinite(predicted).all()
    model = load_model(tmp_path / "model.bsf")
    np.testing.assert_allclose(predicted, model(torch.from_numpy(params)).detach().numpy())


@pytest.mark.parametrize("form,field", [("bug_report", "reproduce"), ("feature_request", "proposal")])
def test_issue_form_python_examples(form, field):
    document = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / f"{form}.yml").read_text())
    example = next(item["attributes"]["placeholder"] for item in document["body"] if item.get("id") == field)
    blocks = re.findall(r"```python\n(.*?)```", example, flags=re.DOTALL)
    assert blocks
    for code in blocks:
        # The examples include behavioral assertions, so execution verifies their results.
        exec(compile(code, f"{form}.yml", "exec"), {})
