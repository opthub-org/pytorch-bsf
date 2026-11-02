"""Exercise the documentation producer's source and artifact boundary."""

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.pages_build import freeze, package, source_name


@pytest.mark.parametrize("ref", ["refs/heads/master", "refs/heads/fix/topic", "refs/tags/v1.2.3"])
def test_version_urls_are_preserved(ref):
    assert source_name(ref) == ref.split("/", 2)[2]


@pytest.mark.parametrize(
    "ref",
    [
        "master",
        "refs/heads/gh-pages",
        "refs/heads/release-please-topic",
        "refs/tags/v1x2x3",
        "refs/heads/.git",
        "refs/heads/a/../b",
        "refs/heads/index.html",
        "refs/heads/a\\b",
        "refs/heads/a//b",
    ],
)
def test_reserved_and_unsafe_version_paths_are_rejected(ref):
    with pytest.raises(ValueError):
        source_name(ref)


def test_freeze_uses_event_sha_and_does_not_update_a_different_checkout():
    sha = "a" * 40
    with (
        patch("tools.pages_build.subprocess.check_output", return_value=sha + "\n"),
        patch("tools.pages_build.subprocess.run") as run,
    ):
        freeze("refs/heads/fix/topic", sha)
        run.assert_called_once_with(["git", "update-ref", "refs/heads/fix/topic", sha], check=True)
    with (
        patch("tools.pages_build.subprocess.check_output", return_value="b" * 40),
        patch("tools.pages_build.subprocess.run") as run,
    ):
        with pytest.raises(ValueError, match="Checkout"):
            freeze("refs/heads/master", sha)
        run.assert_not_called()


def _build(tmp_path):
    build = tmp_path / "build"
    target = build / "fix/topic"
    target.mkdir(parents=True)
    (target / "index.html").write_bytes(b"new preview")
    (build / "master").mkdir()
    (build / "master/index.html").write_bytes(b"unrelated master")
    metadata = dict(repository="owner/repo", run_id=12, run_attempt=2, source_ref="refs/heads/fix/topic", source_sha="a" * 40)
    return build, target, metadata


def test_artifact_contains_only_triggering_ref_and_byte_digests(tmp_path):
    build, target, metadata = _build(tmp_path)
    (target / "_static").mkdir()
    (target / "_static/style.css").write_bytes(b"body{}\r\n")
    output = tmp_path / "artifact"
    package(build, output, metadata)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest == dict(
        metadata,
        schema=1,
        files={
            "index.html": {"size": 11, "sha256": hashlib.sha256(b"new preview").hexdigest()},
            "_static/style.css": {"size": 8, "sha256": hashlib.sha256(b"body{}\r\n").hexdigest()},
        },
    )
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) == [
        "manifest.json",
        "site/_static/style.css",
        "site/index.html",
    ]


def test_missing_version_and_existing_output_fail_without_overwriting(tmp_path):
    build, target, metadata = _build(tmp_path)
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "sentinel").write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        package(build, output, metadata)
    assert (output / "sentinel").read_bytes() == b"keep"
    (target / "index.html").unlink()
    with pytest.raises(ValueError, match="not built"):
        package(build, tmp_path / "missing", metadata)


@pytest.mark.parametrize("special", [".git/config", ".gitattributes", ".gitmodules"])
def test_git_control_files_are_not_packaged(tmp_path, special):
    build, target, metadata = _build(tmp_path)
    path = target / special
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"control")
    with pytest.raises(ValueError, match="Git control"):
        package(build, tmp_path / "output", metadata)


def test_symlink_is_not_packaged(tmp_path):
    build, target, metadata = _build(tmp_path)
    try:
        (target / "outside").symlink_to(tmp_path)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError, match="nonregular"):
        package(build, tmp_path / "output", metadata)


def test_copy_failure_removes_partial_artifact(tmp_path):
    build, _, metadata = _build(tmp_path)
    output = tmp_path / "output"
    original = Path.write_bytes

    def fail(path, data):
        if output in path.parents:
            raise OSError("disk full")
        return original(path, data)

    with patch.object(Path, "write_bytes", fail), pytest.raises(OSError, match="disk full"):
        package(build, output, metadata)
    assert not output.exists()


def test_workflow_keeps_complete_build_readonly_and_outside_publish_lock():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/sphinx-pages-build.yml").read_text()
    assert workflow.encode() == (root / ".github/pages-build-template.yml").read_bytes()
    assert "sphinx-multiversion docs/ docs/_build" in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "concurrency:" not in workflow
    assert "secrets." not in workflow
    assert "include-hidden-files: true" in workflow


def test_security_postmerge_dispatch_uses_the_new_readonly_producer():
    from tools.security_automation import POST_MERGE

    assert "sphinx-pages-build.yml" in POST_MERGE
    assert "sphinx-pages.yml" not in POST_MERGE


def test_invalid_sha_is_not_sent_to_git():
    with patch("tools.pages_build.subprocess.check_output") as read, pytest.raises(ValueError, match="SHA"):
        freeze("refs/heads/master", "-bad")
    read.assert_not_called()


@pytest.mark.parametrize("limit", ["MAX_FILES", "MAX_SITE_BYTES", "MAX_FILE_BYTES", "MAX_MANIFEST_BYTES"])
def test_oversized_artifact_does_not_leave_partial_output(tmp_path, monkeypatch, limit):
    build, _, metadata = _build(tmp_path)
    monkeypatch.setattr("tools.pages_build." + limit, 0)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="too large"):
        package(build, output, metadata)
    assert not output.exists()


def test_real_git_freeze_preserves_detached_commit(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "source",
        ],
        check=True,
    )
    sha = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.chdir(tmp_path)
    freeze("refs/heads/fix/topic", sha)
    assert subprocess.check_output(["git", "rev-parse", "fix/topic"], text=True).strip() == sha
