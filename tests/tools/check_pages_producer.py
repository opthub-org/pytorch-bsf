"""Exercise the pull request signal for GHPages producer changes in real repositories."""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_pages_producer import PRODUCER, TEMPLATE, changes_producer_untrusted

OLD = b"uses: actions/checkout@old\n"
NEW = b"uses: actions/checkout@new\n"
OTHER = b"uses: actions/checkout@other\n"
SCRIPT = Path(__file__).parents[2] / "tools/check_pages_producer.py"


def git(root: Path, *arguments: str) -> str:
    # The repository must not inherit CI's or a developer's settings.
    command = ["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com", *arguments]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def commit(root: Path, template: bytes | None = None, producer: bytes | None = None, extra: bytes | None = None) -> str:
    for path, content in ((TEMPLATE, template), (PRODUCER, producer), ("extra.txt", extra)):
        if content is not None:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(content)
    git(root, "add", "--all")
    git(root, "commit", "--allow-empty", "-m", "change")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "core.autocrlf", "false")
    return tmp_path


@pytest.mark.parametrize(
    ("base_template", "base_producer", "head_template", "head_producer", "rejected"),
    [
        pytest.param(OLD, OLD, OLD, OLD, False, id="unrelated-pr-or-release-please"),
        pytest.param(OLD, OLD, NEW, OLD, False, id="template-pr"),
        pytest.param(NEW, OLD, NEW, NEW, False, id="workflow-pr-after-template"),
        pytest.param(NEW, OLD, NEW, OLD, False, id="unrelated-pr-between-the-two-prs"),
        pytest.param(NEW, OLD, OLD, OLD, False, id="template-revert-between-the-two-prs"),
        pytest.param(OLD, NEW, OLD, OLD, False, id="revert-of-a-premature-workflow-pr"),
        pytest.param(OLD, OLD, OLD, NEW, True, id="premature-workflow-pr"),
        pytest.param(OLD, OLD, NEW, NEW, True, id="one-pr-changes-both-files"),
        pytest.param(NEW, OLD, NEW, OTHER, True, id="workflow-pr-with-another-pin"),
        pytest.param(OLD, OLD, OLD, OLD + b"\n", True, id="trailing-byte"),
        pytest.param(OLD, OLD, OLD, OLD.replace(b"\n", b"\r\n"), True, id="line-endings"),
    ],
)
def test_only_a_producer_change_the_base_does_not_trust_is_rejected(
    root: Path, base_template: bytes, base_producer: bytes, head_template: bytes, head_producer: bytes, rejected: bool
) -> None:
    base = commit(root, base_template, base_producer)
    head = commit(root, head_template, head_producer, extra=b"change")

    assert changes_producer_untrusted(root, base, head) is rejected


@pytest.mark.parametrize("missing", [PRODUCER, TEMPLATE])
def test_a_missing_file_is_an_error_not_a_pass(root: Path, missing: str) -> None:
    head = commit(root, OLD, OLD)
    git(root, "rm", "--quiet", missing)
    without = commit(root)

    with pytest.raises(subprocess.CalledProcessError):
        changes_producer_untrusted(root, without, head)


def test_a_deleted_producer_is_an_error_not_a_pass(root: Path) -> None:
    base = commit(root, OLD, OLD)
    git(root, "rm", "--quiet", PRODUCER)
    head = commit(root)

    with pytest.raises(subprocess.CalledProcessError):
        changes_producer_untrusted(root, base, head)


def run_script(root: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), base],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_reports_a_rejected_head_and_passes_an_accepted_one(root: Path) -> None:
    base = commit(root, OLD, OLD)
    commit(root, producer=NEW)

    rejected = run_script(root, base)

    assert rejected.returncode == 1
    assert rejected.stdout.startswith(f"::error file={PRODUCER}::")
    assert "Updating GHPages action pins" in rejected.stdout
    assert run_script(root, "HEAD").returncode == 0
    assert run_script(root, "HEAD").stdout == ""


def test_script_fails_when_the_base_cannot_be_read(root: Path) -> None:
    commit(root, OLD, OLD)

    assert run_script(root, "missing-revision").returncode != 0
