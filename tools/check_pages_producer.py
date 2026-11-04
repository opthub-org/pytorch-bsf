"""Fail a change that moves the GHPages producer away from the trusted template.

The publisher rejects a producer that differs from the default branch's template, so
landing such a change stops all publication. This is an early signal for pull requests.
The trust decision stays in ``tools/pages_publish.py``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PRODUCER = ".github/workflows/sphinx-pages-build.yml"
TEMPLATE = ".github/pages-build-template.yml"


def blob(root: Path, revision: str, path: str) -> str:
    """Return the object ID of a committed file, which identifies its exact bytes."""
    command = ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}:{path}"]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def changes_producer_untrusted(root: Path, base: str, head: str = "HEAD") -> bool:
    """Return whether ``head`` changes the producer to bytes that ``base`` does not trust.

    A head that keeps the base's producer passes even if the base's template differs. Otherwise,
    every unrelated pull request would fail between a Template PR and its Workflow PR.
    """
    producer = blob(root, head, PRODUCER)
    return producer not in {blob(root, base, PRODUCER), blob(root, base, TEMPLATE)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="revision of the default branch")
    arguments = parser.parse_args()
    if not changes_producer_untrusted(Path.cwd(), arguments.base):
        return 0
    print(
        f"::error file={PRODUCER}::This change makes the producer differ from {TEMPLATE} on the default "
        "branch, so merging it stops all documentation publication. "
        "Follow 'Updating GHPages action pins' in CONTRIBUTING.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
