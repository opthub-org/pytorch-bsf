"""File an issue for each advisory on an action that a workflow on master pins to a commit SHA.

GitHub creates no Dependabot alert for an action pinned to a SHA, and the Dependabot merge controller looks advisories up
only before it merges. This check runs on a schedule and asks the GitHub Advisory Database about every external action in
`.github/workflows/`, with the controller's lookup. It reads the workflow files as text and runs none of the actions it
checks. A lookup that fails fails the job: it never reads as no advisories.
"""

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from tools import dependabot_merge as merge
from tools import security_automation as security

_SOURCE = merge.ADVISORY_SOURCES["github_actions"]
if _SOURCE is None:
    raise RuntimeError("The Dependabot merge controller has no advisory source for GitHub Actions")
# The lookup, and which names and versions it can use, are the pre-merge check's, so that both read a pin the same way.
ACTIONS: merge.AdvisorySource = _SOURCE

WORKFLOW_DIRECTORY = Path(".github/workflows")
# The marker names the advisory and the pins that the issue reports, as `name@sha` in lowercase.
MARKER_PREFIX = "<!-- action-advisory:v1 "
TITLE = "Advisory {} affects an action pinned in a workflow"
MARKER = re.compile(re.escape(MARKER_PREFIX) + "(" + merge.ADVISORY_ID.pattern + r")((?: [A-Za-z0-9_./-]+@[0-9a-f]{40})*) -->")
# A `uses` key anywhere on a line, quoted or not. A line that has one must be a pin or a local path, so a form that this
# check cannot read, such as a quoted key or value or a flow mapping, fails the job instead of going unchecked.
USES_KEY = re.compile(r"""(?<![\w-])["']?uses["']?[ \t]*:""")
LOCAL_USES = re.compile(r"[ \t]*(?:- )?uses:[ \t]+\./[^\s#]*(?:[ \t]+#.*)?")
SHA = re.compile(r"[0-9a-f]{40}")
# An annotated tag can point to another tag; tools/pages_publish.py:peel_commit uses the same bound.
MAX_TAG_DEPTH = 8


@dataclass(frozen=True)
class Pin:
    """One `uses:` line: the action, its commit, and the exact version that its comment names, if any."""

    path: str
    line: int
    name: str
    sha: str
    tag: str | None

    @property
    def repository(self) -> str:
        return "/".join(self.name.split("/")[:2])

    @property
    def key(self) -> str:
        """Identify the pinned commit of the action; GitHub ignores the case of the name."""
        return f"{self.name.lower()}@{self.sha}"


@dataclass(frozen=True)
class Finding:
    """A pin that an advisory affects, and the version that the lookup used; None means any version of the action."""

    pin: Pin
    version: str | None


def parse_pin(line: str) -> tuple[str, str, str | None] | None:
    """Return the action, the SHA and the comment's version of a pin line.

    Return None for a line that is not an external action: one without a `uses` key, a comment, or a local `./` path. The
    version is the first word of the comment when it is an exact version like `v7.0.1`. A moving tag like `v4`, or another
    comment, gives None. A `uses` line in any other form raises.
    """
    # YAML ignores trailing blanks and a CRLF line end, so they must not make a pin unreadable.
    line = line.rstrip(" \t\r")
    if line.lstrip().startswith("#") or USES_KEY.search(line) is None or LOCAL_USES.fullmatch(line):
        return None
    error = ValueError("A line with a `uses` key is neither a local path nor an external action pinned to a 40-character SHA")
    match = merge.PIN_LINE.fullmatch(line)
    if match is None:
        raise error
    name = match[1].split("uses:", 1)[1].strip().removesuffix("@")
    # A `.` or `..` segment would change the API path that tag_commit builds from the name.
    if ACTIONS.readable_name.fullmatch(name) is None or {".", ".."} & set(name.split("/")):
        raise error
    # PIN_LINE has matched exactly 40 hexadecimal characters after group 1.
    end = match.end(1) + 40
    words = line[end:].strip().removeprefix("#").split()
    tag = words[0] if words and ACTIONS.comparable_version.fullmatch(words[0]) else None
    return name, line[match.end(1) : end], tag


def read_pins(root: Path) -> tuple[list[Pin], list[str]]:
    """Read every external action pin in the workflow files, and the lines that no pin could be read from.

    An unreadable line does not stop the others from being checked. A directory with no
    external pin and no unreadable line raises; local `./` paths do not count.
    """
    pins = []
    errors = []
    for path in sorted(path for path in (root / WORKFLOW_DIRECTORY).iterdir() if path.suffix in {".yml", ".yaml"}):
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            try:
                parsed = parse_pin(line)
            except ValueError as error:
                errors.append(f"{relative}:{number}: {error}")
                continue
            if parsed is not None:
                pins.append(Pin(relative, number, *parsed))
    if not pins and not errors:
        raise ValueError(f"No action pins found in {WORKFLOW_DIRECTORY}")
    return pins, errors


def tag_commit(api: security.GitHub, repository: str, tag: str) -> str | None:
    """Return the commit that a tag points to, or None when the repository has no such tag."""
    try:
        target = api.request(f"/repos/{repository}/git/ref/tags/{quote(tag, safe='')}")["object"]
    except security.GitHubError as error:
        if error.code == 404:
            return None
        raise
    for _ in range(MAX_TAG_DEPTH):
        sha = target["sha"]
        if not isinstance(sha, str) or SHA.fullmatch(sha) is None:
            raise ValueError("GitHub returned a malformed tag")
        if target["type"] == "commit":
            return sha
        if target["type"] != "tag":
            raise ValueError("GitHub returned a tag that points to neither a commit nor a tag")
        target = api.request(f"/repos/{repository}/git/tags/{sha}")["object"]
    raise ValueError("A tag points to too many nested tags")


def lookup_version(api: security.GitHub, pin: Pin) -> str | None:
    """Return the version to ask about: the comment's, if its tag points to the pinned commit; otherwise any version."""
    if pin.tag is None or tag_commit(api, pin.repository, pin.tag) != pin.sha:
        return None
    # The advisories list versions without the `v` of a tag, as in merge.known_advisories.
    return pin.tag.removeprefix("v").removeprefix("V")


def scan(api: security.GitHub, pins: list[Pin]) -> tuple[dict[str, list[Finding]], list[str]]:
    """Look up each distinct pin once; return the findings by advisory, and the lookups that failed."""
    findings: dict[str, list[Finding]] = {}
    errors = []
    distinct: dict[tuple[str, str, str | None], list[Pin]] = {}
    for pin in pins:
        distinct.setdefault((pin.name, pin.sha, pin.tag), []).append(pin)
    for occurrences in distinct.values():
        first = occurrences[0]
        try:
            version = lookup_version(api, first)
            advisories = ACTIONS.lookup(api, first.name, version)
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"{first.name}@{first.sha[: merge.SHORT_SHA_CHARS]}: {type(error).__name__}: {error}")
            continue
        for ident in advisories:
            findings.setdefault(ident, []).extend(Finding(pin, version) for pin in occurrences)
    return findings, errors


def filed_pins(api: security.GitHub) -> dict[str, set[str]]:
    """Return the pins that this check has reported for each advisory, whether the issue is open or closed."""
    reported: dict[str, set[str]] = {}
    for issue in api.pages(f"issues?state=all&creator={quote(security.BOT)}"):
        if merge.author_login(issue) != security.BOT or "pull_request" in issue:
            continue
        # Other bot issues, like the security automation's status, quote outside text, so only this check's titles count.
        # Within them, a marker below text that the maintainer added still counts.
        match = MARKER.search(issue.get("body") or "")
        if match is not None and issue.get("title") == TITLE.format(match[1]):
            reported.setdefault(match[1], set()).update(match[2].split())
    return reported


def _finding_lines(findings: list[Finding]) -> Iterator[str]:
    for finding in sorted(findings, key=lambda row: (row.pin.path, row.pin.line)):
        pin = finding.pin
        where = f"- `{pin.path}:{pin.line}` `{pin.name}@{pin.sha[: merge.SHORT_SHA_CHARS]}`"
        if finding.version is not None:
            yield f"{where}: version `{finding.version}` is affected."
        elif pin.tag is None:
            yield f"{where}: its comment names no exact version, so any advisory on the action counts."
        else:
            yield f"{where}: the tag `{pin.tag}` in its comment is not this commit, so any advisory on the action counts."


def render(repository: str, ident: str, findings: list[Finding], earlier: bool) -> str:
    keys = " ".join(sorted({finding.pin.key for finding in findings}))
    return "\n".join(
        [
            f"{MARKER_PREFIX}{ident} {keys} -->",
            f"@{merge.MAINTAINER} The GitHub Advisory Database lists [{ident}](https://github.com/advisories/{ident}) "
            "for these action pins on `master`:",
            "",
            *_finding_lines(findings),
            "",
            *(["An earlier issue for this advisory reports other pins.", ""] if earlier else []),
            "Where any version counts, check the advisory's affected versions against the pinned commit. Update the pins, "
            "or close this issue if the advisory does not apply. This check never reopens an issue, and files another one "
            "for this advisory only when it affects an action and commit that no issue has reported; see "
            f"[SECURITY.md](https://github.com/{repository}/blob/master/SECURITY.md#action-advisories-after-merge).",
        ]
    )


def report(api: security.GitHub, findings: dict[str, list[Finding]]) -> tuple[list[str], list[str]]:
    """File an issue for each advisory that affects a pinned commit no issue has reported.

    Return the advisories filed and the errors. If the earlier issues cannot be read, nothing is filed.
    """
    if not findings:
        return [], []
    try:
        filed_before = filed_pins(api)
    except merge.OPERATIONAL_ERRORS as error:
        return [], [f"Reading the filed issues failed, so nothing was filed: {type(error).__name__}: {error}"]
    filed: list[str] = []
    errors: list[str] = []
    for ident in sorted(findings):
        reported = filed_before.get(ident, set())
        new = [finding for finding in findings[ident] if finding.pin.key not in reported]
        if not new:
            continue
        data = {
            "title": TITLE.format(ident),
            "body": render(api.repo, ident, new, earlier=ident in filed_before),
            "assignees": [merge.MAINTAINER],
        }
        try:
            api.write("issues", data)
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"{ident}: filing the issue failed: {type(error).__name__}: {error}")
            continue
        filed.append(ident)
    return filed, errors


def run(api: security.GitHub, root: Path) -> list[str]:
    """Check every pin and file the new advisories; return the errors, which fail the job after the rest is done."""
    pins, unreadable = read_pins(root)
    findings, errors = scan(api, pins)
    filed, filing_errors = report(api, findings)
    print(
        f"Checked {len(pins)} action pins: {len(findings)} advisories found, {len(filed)} newly filed"
        + (f" ({', '.join(filed)})" if filed else "")
        + f", {len(unreadable)} lines unreadable, {len(errors)} lookups failed, {len(filing_errors)} filing errors."
    )
    return unreadable + errors + filing_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = run(security.GitHub(), args.root)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
