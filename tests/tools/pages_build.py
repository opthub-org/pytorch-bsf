"""Exercise the documentation producer's source and artifact boundary."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tools.pages_build import freeze, package, source_name

# Stricter than tools.security_automation.ACTION_PIN: at the trust boundary, the comment
# must be a single ASCII version token, so that it cannot hide a YAML line break.
# The optional CR keeps CRLF checkouts on windows-latest matching.
ACTION_PIN = re.compile(rb"^( *(?:- )?uses: ([\w./-]+)@)([0-9a-f]{40}) # v?[0-9][\w.+-]*(\r?)$", re.MULTILINE)
CHECKOUT_PIN = re.compile(rb"(?<=actions/checkout@)[0-9a-f]{40} # v[\d.]+")
RUN_BLOCK_WITH_PIN_TEXT = b"      - run: |\n          uses: evil/action@%s # v1\n"


def _without_action_pins(workflow: bytes) -> bytes:
    """Mask only the SHA and version comment of immutable ``uses:`` pins."""
    return ACTION_PIN.sub(rb"\1<pin>\4", workflow)


def _assert_fully_pinned_and_only_pins_differ(workflow: bytes, template: bytes) -> None:
    """Require SHA pins on every step and allow only those pins to differ.

    The two files differ while a two-step pin update is in flight. The publisher
    still requires an exact match.
    """
    assert _without_action_pins(workflow) == _without_action_pins(template)
    for definition in (workflow, template):
        # The mask skips lines it cannot match, and it can match text outside a step,
        # so the masked pins must be exactly the steps that the YAML parser sees.
        jobs = yaml.safe_load(definition)["jobs"].values()
        uses = [step["uses"] for job in jobs for step in job["steps"] if "uses" in step]
        pins = [f"{pin[2].decode()}@{pin[3].decode()}" for pin in ACTION_PIN.finditer(definition)]
        assert uses
        assert pins == uses


@pytest.fixture
def template() -> bytes:
    return (Path(__file__).resolve().parents[2] / ".github/pages-build-template.yml").read_bytes()


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


def test_competing_packager_does_not_remove_the_completed_artifact(tmp_path, monkeypatch):
    from tools import pages_build

    build, _, metadata = _build(tmp_path)
    output = tmp_path / "output"
    original = pages_build.inventory
    competing_writer_finished = False

    def finish_competitor(source):
        nonlocal competing_writer_finished
        if not competing_writer_finished:
            competing_writer_finished = True
            package(build, output, metadata)
        return original(source)

    monkeypatch.setattr(pages_build, "inventory", finish_competitor)
    with pytest.raises(FileExistsError):
        package(build, output, metadata)
    assert (output / "site/index.html").read_bytes() == b"new preview"
    assert json.loads((output / "manifest.json").read_text())["run_id"] == 12


def test_changed_source_rejects_copy_and_removes_its_partial_output(tmp_path, monkeypatch):
    from tools import pages_build

    build, _, metadata = _build(tmp_path)
    output = tmp_path / "output"
    original = pages_build.inventory

    def change_after_inventory(source):
        files = original(source)
        (source / "index.html").write_bytes(b"changed during packaging")
        return files

    monkeypatch.setattr(pages_build, "inventory", change_after_inventory)
    with pytest.raises(ValueError, match="changed during"):
        package(build, output, metadata)
    assert not output.exists()


def test_cli_records_event_provenance_and_preserves_existing_output(tmp_path):
    build, _, metadata = _build(tmp_path)
    output = tmp_path / "output"
    environment = {
        **os.environ,
        "PYTHONPATH": ".",
        "GITHUB_REPOSITORY": str(metadata["repository"]),
        "GITHUB_RUN_ID": str(metadata["run_id"]),
        "GITHUB_RUN_ATTEMPT": str(metadata["run_attempt"]),
        "GITHUB_REF": str(metadata["source_ref"]),
        "GITHUB_SHA": str(metadata["source_sha"]),
    }
    command = [sys.executable, "-m", "tools.pages_build", "--build", str(build), "--output", str(output)]
    root = Path(__file__).resolve().parents[2]
    first = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    saved = (output / "manifest.json").read_bytes()
    record = json.loads(saved)
    assert {key: record[key] for key in metadata} == metadata
    second = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert (output / "manifest.json").read_bytes() == saved


def test_workflow_keeps_complete_build_readonly_and_outside_publish_lock(template):
    root = Path(__file__).resolve().parents[2]
    raw = (root / ".github/workflows/sphinx-pages-build.yml").read_bytes()
    _assert_fully_pinned_and_only_pins_differ(raw, template)
    workflow = raw.decode()
    assert "sphinx-multiversion docs/ docs/_build" in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "concurrency:" not in workflow
    assert "secrets." not in workflow
    assert "include-hidden-files: true" in workflow


@pytest.mark.parametrize("version", [b"v99.0.0", b"1.2.3", b"v1.2.3-rc.1"])
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_pin_bump_on_either_side_is_tolerated(template, line_ending, version):
    template = template.replace(b"\r\n", b"\n").replace(b"\n", line_ending)
    bumped, count = CHECKOUT_PIN.subn(b"a" * 40 + b" # " + version, template)
    assert count == 1
    assert bumped != template
    _assert_fully_pinned_and_only_pins_differ(template, bumped)
    _assert_fully_pinned_and_only_pins_differ(bumped, template)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"actions/checkout@", b"attacker/checkout@"),
        (b"contents: read", b"contents: write"),
        (b"timeout-minutes: 40", b"timeout-minutes: 41"),
        (b"      - uses: actions/checkout@", b"      - run: curl example.invalid\n      - uses: actions/checkout@"),
    ],
)
def test_non_pin_difference_is_rejected(template, old, new):
    assert old in template
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(template.replace(old, new, 1), template)


def test_pin_text_on_a_run_line_is_not_masked(template):
    line = b"      - run: echo uses: actions/checkout@%s # v1\n"
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(template + line % (b"a" * 40), template + line % (b"b" * 40))


@pytest.mark.parametrize(
    "pin",
    [
        b"v4",
        b"main",
        b"<pin>",
        b"a" * 39,
        b"a" * 41,
        b"A" * 40,
        b"a" * 40,
        b"a" * 40 + b" #",
        b"a" * 40 + b" # main",
        b"a" * 40 + b" # anything else",
        b"a" * 40 + b" # v4\r",
        b"a" * 40 + b"\r",
        b"a" * 40 + b"  # v4",
        b"a" * 40 + b"\n        if: always()",
        b"a" * 40 + b" # v4\xc2\x85        if: always()",
        b"a" * 40 + b" # v4\xe2\x80\xa8      - run: curl example.invalid",
        b"a" * 40 + b" # v4\xe2\x80\xa9      - run: curl example.invalid",
    ],
)
def test_mutable_or_malformed_pin_is_rejected(template, pin):
    changed, count = CHECKOUT_PIN.subn(pin, template)
    assert count == 1
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(changed, template)
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(template, changed)


@pytest.mark.parametrize(
    "step",
    [
        None,
        b"actions/checkout@%s" % (b"c" * 40),
        b'"actions/checkout@%s"' % (b"c" * 40),
        b"act\xd0\xbeons/checkout@%s # v1" % (b"c" * 40),
    ],
)
def test_masked_pin_in_a_run_block_must_be_a_real_step(template, step):
    if step is not None:
        template = re.sub(rb"(?<=uses: )actions/checkout@[^\r\n]*", step, template, count=1)
    left = template + RUN_BLOCK_WITH_PIN_TEXT % (b"a" * 40)
    right = template + RUN_BLOCK_WITH_PIN_TEXT % (b"b" * 40)
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(left, right)


def test_masked_pin_in_a_run_block_cannot_stand_in_for_a_mutable_pin(template):
    changed, count = CHECKOUT_PIN.subn(b"v4", template)
    assert count == 1
    changed += RUN_BLOCK_WITH_PIN_TEXT % (b"a" * 40)
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(changed, changed)


def test_definition_without_actions_is_rejected():
    definition = b"jobs:\n  build:\n    steps:\n      - run: echo\n"
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(definition, definition)


@pytest.mark.parametrize("spelling", [b'"uses":', b"uses :"])
def test_mutable_pin_in_both_files_is_rejected_for_any_uses_spelling(template, spelling):
    changed, count = re.subn(rb"uses: actions/checkout@[^\n]*", spelling + b" actions/checkout@v4", template)
    assert count == 1
    with pytest.raises(AssertionError):
        _assert_fully_pinned_and_only_pins_differ(changed, changed)


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
