"""Exercise the pull request signal for GHPages producer changes in real repositories."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_pages_producer import PRODUCER, TEMPLATE, producer_is_untrusted
from tools.pages_publish import WORKFLOW_PATH

OLD = b"uses: actions/checkout@old\n"
NEW = b"uses: actions/checkout@new\n"
OTHER = b"uses: actions/checkout@other\n"
SCRIPT = Path(__file__).parents[2] / "tools/check_pages_producer.py"


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep git from reaching a real repository or running a developer's hooks and signing."""
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(root: Path, *arguments: str) -> str:
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
    return tmp_path


def test_producer_path_is_the_one_the_publisher_authenticates() -> None:
    assert PRODUCER == WORKFLOW_PATH


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
        pytest.param(NEW, OLD, OLD, NEW, True, id="workflow-pr-reverts-the-template"),
        pytest.param(NEW, OLD, OTHER, NEW, True, id="workflow-pr-moves-the-template"),
        pytest.param(NEW, OLD, NEW, OTHER, True, id="workflow-pr-with-another-pin"),
        pytest.param(NEW, NEW, OLD, OLD, True, id="branch-behind-both-files"),
        pytest.param(OLD, OLD, OLD, OLD + b"\n", True, id="trailing-byte"),
        pytest.param(OLD, OLD, OLD, OLD.replace(b"\n", b"\r\n"), True, id="line-endings"),
    ],
)
def test_only_a_producer_the_base_does_not_trust_is_rejected(
    root: Path, base_template: bytes, base_producer: bytes, head_template: bytes, head_producer: bytes, rejected: bool
) -> None:
    base = commit(root, base_template, base_producer)
    head = commit(root, head_template, head_producer, extra=b"change")

    assert producer_is_untrusted(root, base, head) is rejected


@pytest.mark.parametrize("missing", [PRODUCER, TEMPLATE])
@pytest.mark.parametrize("revision", ["base", "head"])
def test_a_missing_file_is_an_error_not_a_pass(root: Path, missing: str, revision: str) -> None:
    complete = commit(root, OLD, OLD)
    git(root, "rm", "--quiet", missing)
    without = commit(root)
    base, head = (without, complete) if revision == "base" else (complete, without)

    with pytest.raises(subprocess.CalledProcessError) as error:
        producer_is_untrusted(root, base, head)

    assert any(argument == f"{without}:{missing}" for argument in error.value.cmd)


@pytest.mark.parametrize("empty", ["base", "head"])
def test_an_empty_revision_is_an_error_not_the_index(root: Path, empty: str) -> None:
    # git would read ":path" from the index, which may hold anything.
    revision = commit(root, OLD, OLD)
    base, head = ("", revision) if empty == "base" else (revision, "")

    with pytest.raises(ValueError, match="must not be empty"):
        producer_is_untrusted(root, base, head)


def run_script(root: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), base], cwd=root, capture_output=True, text=True, check=False)


def test_script_reports_a_rejected_head_and_passes_an_accepted_one(root: Path) -> None:
    base = commit(root, OLD, OLD)
    commit(root, producer=NEW)

    rejected = run_script(root, base)

    assert rejected.returncode == 1
    assert rejected.stdout.startswith(f"::error file={PRODUCER}::")
    assert "Updating GHPages action pins" in rejected.stdout
    assert "Changing the GHPages producer beyond action pins" in rejected.stdout
    accepted = run_script(root, "HEAD")
    assert accepted.returncode == 0
    assert accepted.stdout == ""


def test_script_reports_an_unreadable_base_apart_from_a_rejection(root: Path) -> None:
    commit(root, OLD, OLD)

    result = run_script(root, "missing-revision")

    assert result.returncode == 2
    assert result.stdout.startswith("::error::Cannot read the GHPages producer or template: fatal:")


def test_script_reports_an_empty_base_apart_from_a_rejection(root: Path) -> None:
    # The index holds a producer that HEAD's template does not trust, so reading it would reject.
    commit(root, OLD, OLD)
    (root / PRODUCER).write_bytes(NEW)
    git(root, "add", PRODUCER)

    result = run_script(root, "")

    assert result.returncode == 2
    assert result.stdout.startswith("::error::Cannot read the GHPages producer or template: ")
    assert "must not be empty" in result.stdout
