"""Fail a commit whose GHPages producer the default branch does not trust.

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


def blob_id(root: Path, revision: str, path: str) -> str:
    """Return the object ID of a committed file, which identifies its exact bytes."""
    command = ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}:{path}"]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def producer_is_untrusted(root: Path, base: str, head: str = "HEAD") -> bool:
    """Return whether ``head`` moves the producer away from what both templates allow.

    A head that keeps the base's producer passes even if the base's template differs. Otherwise,
    every unrelated pull request would fail between a Template PR and its Workflow PR. A head
    that moves the producer must match the template both before and after the merge.

    A file missing from either revision raises ``subprocess.CalledProcessError``.
    """
    producer = blob_id(root, head, PRODUCER)
    head_template = blob_id(root, head, TEMPLATE)
    base_producer = blob_id(root, base, PRODUCER)
    base_template = blob_id(root, base, TEMPLATE)
    return producer != base_producer and (producer != base_template or producer != head_template)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="revision of the default branch")
    arguments = parser.parse_args()
    try:
        untrusted = producer_is_untrusted(Path.cwd(), arguments.base)
    except subprocess.CalledProcessError as error:
        print(f"::error::Cannot read the GHPages producer or template: {error.stderr.strip()}")
        return 2
    if not untrusted:
        return 0
    print(
        f"::error file={PRODUCER}::The producer differs from the producer and the template on the default "
        "branch, or the template moves with it. This head gets no GHPages publish check, and a producer that "
        "lands ahead of the template stops all documentation publication. "
        "If this branch leaves both files alone, it is behind: merge the default branch and run this again. "
        "Otherwise, follow 'Updating GHPages action pins' in CONTRIBUTING.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
