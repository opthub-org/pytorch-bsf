"""Run the release documentation dispatch script offline with its API mocked."""

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml


def test_retired_pages_workflow_is_manual_inert_and_unprivileged():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/sphinx-pages.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow) == {"name", "on", "permissions", "jobs"}
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert not workflow["on"]["workflow_dispatch"]
    assert workflow["permissions"] == {}
    assert workflow["jobs"]
    for job in workflow["jobs"].values():
        assert set(job) <= {"if", "runs-on", "steps", "permissions"}
        assert job["if"] == "false"
        assert job.get("permissions", {}) == {}
        assert job["steps"]
        for step in job["steps"]:
            assert set(step) <= {"name", "run"}
            assert step["run"] == ":"


def _job() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/release-please-action.yml").read_text())
    result: dict[str, Any] = workflow["jobs"]["docs"]
    return result


def _script() -> str:
    lines = _job()["steps"][0]["run"].splitlines()
    assert lines[0] == "python3 -I - <<'PY'"
    assert lines[-1] == "PY"
    return "\n".join(lines[1:-1])


@pytest.mark.parametrize("tag", ["v0.24.1", "v1.2.3", "v10.20.30"])
def test_stable_release_dispatches_exact_tag_as_structured_json(tag, monkeypatch):
    monkeypatch.setenv("DOCS_TAG", tag)
    with patch("subprocess.run") as run:
        exec(compile(_script(), "release-docs-inline", "exec"), {})
    run.assert_called_once()
    assert run.call_args.args[0] == [
        "gh",
        "api",
        "--method",
        "POST",
        "repos/opthub-org/pytorch-bsf/actions/workflows/sphinx-pages-build.yml/dispatches",
        "--input",
        "-",
    ]
    assert json.loads(run.call_args.kwargs["input"]) == {"ref": tag}
    assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "master",
        "v1.2.3-rc.1",
        "v1.2.3+build",
        "v1.2",
        "v1x2x3",
        "v1.2.3\n",
        "v１.2.3",
        "v1.2.3; echo bad",
        "$(touch bad)",
        "refs/tags/v1.2.3",
    ],
)
def test_invalid_tag_fails_before_any_dispatch(tag, monkeypatch):
    monkeypatch.setenv("DOCS_TAG", tag)
    with patch("subprocess.run") as run, pytest.raises(SystemExit, match="stable"):
        exec(compile(_script(), "release-docs-inline", "exec"), {})
    run.assert_not_called()


def test_dispatch_api_failure_is_not_reported_as_success(monkeypatch):
    monkeypatch.setenv("DOCS_TAG", "v1.2.3")
    with (
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "gh")),
        pytest.raises(subprocess.CalledProcessError),
    ):
        exec(compile(_script(), "release-docs-inline", "exec"), {})


def test_job_is_dispatch_only_and_skips_uncreated_or_prerelease_tags():
    job = _job()
    assert job["needs"] == "github-release"
    assert job["permissions"] == {"actions": "write"}
    assert len(job["steps"]) == 1
    assert "uses" not in job["steps"][0]
    assert job["steps"][0]["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "DOCS_TAG": "${{ needs.github-release.outputs.tag_name }}",
    }
    assert " ".join(job["if"].split()) == (
        "${{ needs.github-release.outputs.release_created == 'true' && "
        "!contains(needs.github-release.outputs.tag_name, '-') && "
        "!contains(needs.github-release.outputs.tag_name, '+') }}"
    )
