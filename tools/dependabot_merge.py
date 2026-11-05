"""Merge verified Dependabot PRs after full CI, and explain every other PR in a triage comment.

The controller is level-triggered: each run re-evaluates all open Dependabot PRs from GitHub's current state, so a missed
event only delays the next decision. It reads PR metadata as data and never checks out or executes PR code.
"""

import argparse
import datetime as dt
import http.client
import itertools
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlencode

from tools import security_automation as security

Json = security.Json

AUTHOR = "dependabot[bot]"
# GitHub signs Dependabot's commits as this committer; `verified` alone does not say who signed.
COMMITTER = "web-flow"
BRANCH_PREFIX = "dependabot/"
MAINTAINER = "NaokiHamada"
TRIAGE_MARKER = "<!-- dependabot-triage:v1 -->"
OVERDUE_MARKER = "<!-- dependabot-overdue:v1 -->"
# Bounds both unfinished checks and merge states that GitHub has computed, measured from the latest commit.
CI_GRACE = dt.timedelta(hours=2)
# Dependabot rebases a conflict by itself, and a rebase restarts the clock, so a merged sibling need not notify anyone.
REBASE_GRACE = dt.timedelta(days=1)
DECISION_WINDOW = dt.timedelta(days=7)
MAX_HOLD = dt.timedelta(days=30)
MAX_REASON_CHARS = 300
MAX_LISTED_CHECKS = 8
# The branch's ecosystem segment, mapped to the Advisory Database's name. Others, like Docker and pre-commit, have no data.
ADVISORY_ECOSYSTEMS = {"pip": "pip", "github_actions": "actions"}
ADVISORY_TYPES = ("reviewed", "malware")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHORT_SHA_CHARS = 10
MERGEABLE_STATES = {"clean", "has_hooks"}
# GitHub reports these while it recomputes mergeability, which a sibling merge can trigger.
UNCOMPUTED_STATES = {"unknown", None}
WRITE_PERMISSIONS = {"admin", "write"}  # The endpoint reports the maintain role as write.
# 403: not permitted. 422: invalid merge. Other errors may be transient and fail the job instead.
MERGE_REFUSALS = {403, 422}
# 405: not mergeable, as when the base just moved. 409: the head moved. The next evaluation sees the new state.
MERGE_RETRIES = {405, 409}
# A malformed payload or a failed request for one PR must not stop the others; main() still fails the job.
OPERATIONAL_ERRORS = (OSError, RuntimeError, http.client.HTTPException, AttributeError, KeyError, TypeError, ValueError)
DEPENDENCY_FILES = {
    "pyproject.toml",
    "uv.lock",
    "docs/requirements.txt",
    ".pre-commit-config.yaml",
    "Dockerfile",
}
WORKFLOW_PATH = re.compile(r"\.github/workflows/[^/]+\.ya?ml")
# Stricter than security.ACTION_PIN, which runs on whole files: a line must not smuggle a second YAML line in its comment.
PIN_LINE = re.compile(r"([ \t]*(?:- )?uses:[ \t]+\w[\w./-]*@)[0-9a-f]{40}(?:[ \t]+#[^\x00-\x1f\x7f-\x9f\u2028\u2029]*)?")
# A near-miss hold still stops the merge, because its author meant to stop it; only the exact command releases one.
HOLD_COMMAND = re.compile(r"/dependabot[-_ ]?hold\b:?(.*)", re.IGNORECASE)
HOLD_ARGUMENTS = re.compile(r"until=(\d{4}-\d{2}-\d{2})\s+(\S.*)")
RELEASE_COMMAND = re.compile(r"/dependabot-release", re.IGNORECASE)


@dataclass(frozen=True)
class Hold:
    """A maintainer's decision not to merge, in effect through ``until``."""

    author: str
    reason: str
    until: dt.datetime


@dataclass(frozen=True)
class Evidence:
    """What the decision needs to know about one PR head."""

    created: dt.datetime
    head_committed: dt.datetime
    merge_state: str | None
    problem: str | None
    hold: Hold | None
    approved_by: list[str]
    changes_requested_by: list[str]
    failed: list[str]
    pending: list[str]
    enabled: bool
    advisories: list[str]


@dataclass(frozen=True)
class Decision:
    action: Literal["merge", "wait", "stop", "hold", "merged"]
    review: str
    outcome: str
    reason: str
    mergeable_when: str
    deadline: str


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def active_hold(comments: list[Json], has_write_access: Callable[[str], bool], now: dt.datetime) -> Hold | None:
    """Return the latest authorized hold, which its expiry or a later release command ends.

    A malformed hold still stops the merge for the decision window, because its author meant to stop it.
    """
    hold = None
    for comment in comments:
        lines = (comment["body"] or "").strip().splitlines()
        first = lines[0].strip().strip("`") if lines else ""
        released = RELEASE_COMMAND.fullmatch(first) is not None
        command = HOLD_COMMAND.fullmatch(first)
        if not released and command is None:
            continue
        author = _login(comment)
        if author is None or not has_write_access(author):
            continue
        hold = None if command is None else _parse_hold(command[1], author, parse_time(comment["created_at"]))
    return hold if hold is not None and now <= hold.until else None


def _login(row: Json) -> str | None:
    """Read the author's login; GitHub reports no user for a deleted account."""
    return (row.get("user") or {}).get("login")


def _parse_hold(arguments: str, author: str, created: dt.datetime) -> Hold:
    invalid_hold = Hold(
        author,
        f"The hold command is invalid. Use `/dependabot-hold until=YYYY-MM-DD <reason>`, at most {MAX_HOLD.days} days ahead.",
        _end_of_day((created + DECISION_WINDOW).date()),
    )
    match = HOLD_ARGUMENTS.fullmatch(arguments.strip())
    if match is None:
        return invalid_hold
    try:
        day = dt.date.fromisoformat(match[1])
    except ValueError:
        return invalid_hold
    if not created.date() <= day <= (created + MAX_HOLD).date():
        return invalid_hold
    # An unclosed HTML comment in the reason would hide the rest of the triage comment.
    return Hold(author, match[2][:MAX_REASON_CHARS].replace("<", "&lt;"), _end_of_day(day))


def _end_of_day(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, dt.time.max, dt.UTC)


def provenance_problem(pr: Json, commits: list[Json], repo: str) -> str | None:
    """Require a same-repository Dependabot PR of commits by Dependabot that GitHub itself signed."""
    if (
        pr["user"]["login"] != AUTHOR
        or pr["head"]["repo"]["full_name"] != repo
        or not pr["head"]["ref"].startswith(BRANCH_PREFIX)
        or pr["base"]["ref"] != "master"
    ):
        return "The PR is outside the Dependabot trust boundary."
    if pr["draft"]:
        return "The PR is a draft."
    if not commits:
        return "The PR has no commits to verify."
    for commit in commits:
        author = (commit.get("author") or {}).get("login")
        committer = (commit.get("committer") or {}).get("login")
        if author != AUTHOR or committer != COMMITTER or commit["commit"]["verification"]["verified"] is not True:
            return f"Commit `{commit['sha'][:SHORT_SHA_CHARS]}` is not a Dependabot commit signed by GitHub."
    return None


def change_problem(files: list[Json]) -> str | None:
    """Allow edits of dependency manifests, and only SHA and version updates of existing action pins in workflows."""
    if not files:
        return "The PR has no changed files to verify."
    for row in files:
        path = row["filename"]
        if row["status"] != "modified":
            return f"`{path}` was {row['status']}, not modified."
        if path in DEPENDENCY_FILES:
            continue
        if WORKFLOW_PATH.fullmatch(path) is None:
            return f"`{path}` is not a dependency file."
        if not _only_updates_pins(row.get("patch")):
            return f"`{path}` changes more than action pins."
    return None


def _only_updates_pins(patch: str | None) -> bool:
    if patch is None:
        return False
    # Split on newlines only: a line must stay whole for PIN_LINE to reject a control character in it.
    runs = [list(run) for changed, run in itertools.groupby(patch.split("\n"), _is_change) if changed]
    return bool(runs) and all(_run_updates_pins(run) for run in runs)


def _is_change(line: str) -> bool:
    return line.startswith(("+", "-"))


def _run_updates_pins(run: list[str]) -> bool:
    """Check one contiguous change: pairing in order forbids adding, removing, swapping or moving a step."""
    removed = [PIN_LINE.fullmatch(line[1:]) for line in run if line.startswith("-")]
    added = [PIN_LINE.fullmatch(line[1:]) for line in run if line.startswith("+")]
    if len(removed) != len(added):
        return False
    return all(old and new and old[1] == new[1] for old, new in zip(removed, added, strict=True))


def updated_dependencies(message: str) -> list[tuple[str, str | None]]:
    """Read Dependabot's metadata from a commit message, as (name, version) pairs.

    Only the last block counts: it ends the message, so quoted release notes before it cannot replace it.
    """
    _, found, block = message.rpartition("updated-dependencies:\n")
    entries: list[tuple[str, str | None]] = []
    if not found:
        return entries
    for line in block.split("\n"):
        if line == "...":
            break
        if line.startswith("- dependency-name: "):
            entries.append((line.removeprefix("- dependency-name: ").strip("'\""), None))
        elif line.startswith("  dependency-version: ") and entries:
            entries[-1] = (entries[-1][0], line.removeprefix("  dependency-version: ").strip("'\""))
    return entries


def _ecosystem(pr: Json) -> str | None:
    return ADVISORY_ECOSYSTEMS.get(pr["head"]["ref"].removeprefix(BRANCH_PREFIX).split("/")[0])


def metadata_problem(pr: Json, commits: list[Json]) -> str | None:
    """Require the metadata that the advisory lookup reads, where the Advisory Database has data at all."""
    if _ecosystem(pr) is None:
        return None
    for commit in commits:
        if not updated_dependencies(commit["commit"].get("message") or ""):
            return f"Commit `{commit['sha'][:SHORT_SHA_CHARS]}` does not say which versions it installs."
    return None


def known_advisories(api: security.GitHub, pr: Json, commits: list[Json]) -> list[str]:
    """Return the advisories that affect what the PR installs; a failed lookup raises, so nothing merges."""
    ecosystem = _ecosystem(pr)
    if ecosystem is None:
        return []
    found: set[str] = set()
    for commit in commits:
        for name, version in updated_dependencies(commit["commit"]["message"]):
            # The API accepts no SHA as a version, so a SHA pin or a missing version asks about the whole package.
            target = "" if version is None or FULL_SHA.fullmatch(version) else f"@{version}"
            # The API matches names literally: `Pillow` and `pillow` return different advisories.
            for spelling in {name, name.lower()}:
                for kind in ADVISORY_TYPES:
                    query = urlencode({"ecosystem": ecosystem, "type": kind, "affects": spelling + target, "per_page": 100})
                    rows = api.request(f"/advisories?{query}")
                    found.update(row["ghsa_id"] for row in rows if row.get("withdrawn_at") is None)
    return sorted(found)


def review_state(reviews: list[Json], has_write_access: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """Return who approved and who requested changes, from each authorized user's latest verdict.

    Anyone can review a public repository, so a review without write access must not stop a security update.
    """
    latest = {}
    for review in reviews:
        login = _login(review)
        if login is not None and review["state"] != "COMMENTED":
            latest[login] = review["state"]
    approved = sorted(user for user, state in latest.items() if state == "APPROVED" and has_write_access(user))
    changes = sorted(user for user, state in latest.items() if state == "CHANGES_REQUESTED" and has_write_access(user))
    return approved, changes


def decide(evidence: Evidence, now: dt.datetime) -> Decision:
    """Choose the next action from the current evidence; every path but the last refuses to merge."""
    review = _review_line(evidence)
    if evidence.hold is not None:
        return Decision(
            "hold",
            f"Reviewed by @{evidence.hold.author}.",
            "A maintainer decided not to merge for now.",
            evidence.hold.reason,
            "Comment `/dependabot-release`, or wait for the deadline.",
            f"{_day(evidence.hold.until)}. Automatic merging resumes after this date.",
        )
    blocker = _blocker(evidence, now)
    if blocker is not None:
        reason, mergeable_when = blocker
        outcome = "The automation will not merge this head."
        return Decision("stop", review, outcome, reason, mergeable_when, _decision_deadline(evidence.created))

    def wait(promise: str, reason: str, deadline: str, action: str | None = None) -> Decision:
        if evidence.enabled:
            return Decision("wait", review, promise, reason, action or "No action is needed.", deadline)
        resume = "Unset the variable `DEPENDABOT_MERGE_ENABLED` or set it to `true`, or merge it by hand."
        paused = "Automatic merging is paused, so this head will not merge."
        return Decision("wait", review, paused, reason, f"{action} {resume}" if action else resume, deadline)

    if evidence.merge_state == "dirty":
        return wait(
            "It merges a new head once the conflict is gone.",
            "The PR conflicts with `master`.",
            f"{_minute(evidence.head_committed + REBASE_GRACE)}. A remaining conflict then stops the merge.",
            "Dependabot normally rebases it by itself. If it does not, comment `@dependabot rebase`.",
        )
    if evidence.pending:
        return wait(
            "It merges this head once every check succeeds.",
            f"Waiting for {_names(evidence.pending)}.",
            f"{_minute(evidence.head_committed + CI_GRACE)}. Unfinished checks then stop the merge.",
        )
    if evidence.merge_state not in MERGEABLE_STATES:
        limit = _merge_state_limit(evidence.merge_state)
        return wait(
            "It merges this head once GitHub reports it as mergeable.",
            f"GitHub reports the merge state as `{evidence.merge_state}`.",
            f"{_minute(evidence.head_committed + limit)}. This state then stops the merge.",
        )
    if not evidence.enabled:
        return wait(
            "",
            "The repository variable `DEPENDABOT_MERGE_ENABLED` has paused automatic merging.",
            "None while paused. Merging resumes when the variable is cleared.",
        )
    return Decision(
        "merge",
        review,
        "Merging this head now.",
        "Every required check succeeded.",
        "No action is needed.",
        "None.",
    )


def _merge_state_limit(state: str | None) -> dt.timedelta:
    return DECISION_WINDOW if state in UNCOMPUTED_STATES else CI_GRACE


def _blocker(evidence: Evidence, now: dt.datetime) -> tuple[str, str] | None:
    """Return why a maintainer must act and what would make the PR mergeable, or None if nothing stops it."""
    age = now - evidence.head_committed
    if evidence.changes_requested_by:
        return "A reviewer requested changes.", "Dismiss the review, or have the reviewer approve this PR."
    if evidence.problem is not None:
        return evidence.problem, "Comment `@dependabot recreate`, or review and merge it by hand."
    if evidence.advisories:
        return (
            f"Known advisories affect what this PR installs: {_names(evidence.advisories)}.",
            "Wait for Dependabot to propose an unaffected version, or review the advisories and merge it by hand.",
        )
    if evidence.failed:
        return f"These checks failed: {_names(evidence.failed)}.", _failure_remedy(evidence.failed)
    if evidence.merge_state == "dirty" and age > REBASE_GRACE:
        return "The PR still conflicts with `master`.", "Comment `@dependabot rebase`, or close this PR."
    # A head that already conflicts starts no checks, so its pending checks wait for the rebase too.
    if evidence.pending and evidence.merge_state != "dirty" and age > CI_GRACE:
        return (
            f"CI is still incomplete after {_hours(CI_GRACE)} hours: {_names(evidence.pending)}.",
            "Comment `@dependabot rebase`. An old base may not run the current checks.",
        )
    if evidence.merge_state not in MERGEABLE_STATES | {"dirty"} and age > _merge_state_limit(evidence.merge_state):
        return (
            f"GitHub still reports the merge state as `{evidence.merge_state}`.",
            "Resolve what GitHub reports on the PR, or merge it by hand.",
        )
    return None


def _review_line(evidence: Evidence) -> str:
    if evidence.changes_requested_by:
        return f"Changes requested by {_mentions(evidence.changes_requested_by)}."
    if evidence.approved_by:
        return f"Approved by {_mentions(evidence.approved_by)}."
    return "Not reviewed by a person; the automation evaluated this head."


def _failure_remedy(failed: list[str]) -> str:
    if "pages-producer" in failed:
        return (
            "If this PR updates a pin in `sphinx-pages-build.yml`, land the template first, as "
            '"Updating GHPages action pins" in CONTRIBUTING.md describes. Then comment `@dependabot rebase`.'
        )
    return (
        "Fix the cause on `master` and comment `@dependabot rebase`, or rerun the failed jobs if they are flaky. "
        "If this version must wait, comment `/dependabot-hold until=YYYY-MM-DD <reason>`."
    )


def _decision_deadline(created: dt.datetime) -> str:
    return f"{_day(created + DECISION_WINDOW)}. A maintainer decides by this date."


def _mentions(users: list[str]) -> str:
    return ", ".join(f"@{user}" for user in users)


def _names(names: list[str]) -> str:
    unique = sorted(set(names))
    shown = ", ".join(f"`{name}`" for name in unique[:MAX_LISTED_CHECKS])
    hidden = len(unique) - MAX_LISTED_CHECKS
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def _hours(value: dt.timedelta) -> int:
    return int(value.total_seconds() // 3600)


def _day(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d UTC")


def _minute(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def render(decision: Decision, sha: str) -> str:
    """Render without a timestamp, so that an unchanged decision edits nothing."""
    return "\n".join(
        [
            TRIAGE_MARKER,
            f"**Dependabot triage** for `{sha[:SHORT_SHA_CHARS]}`",
            "",
            f"- **Review:** {decision.review}",
            f"- **Decision:** {decision.outcome}",
            f"- **Reason:** {decision.reason}",
            f"- **Mergeable when:** {decision.mergeable_when}",
            f"- **Deadline:** {decision.deadline}",
        ]
    )


def _own_comment(comments: list[Json], marker: str) -> Json | None:
    for comment in comments:
        if _login(comment) == security.BOT and (comment["body"] or "").startswith(marker):
            return comment
    return None


def upsert_triage(api: security.GitHub, number: int, comments: list[Json], body: str) -> None:
    comment = _own_comment(comments, TRIAGE_MARKER)
    if comment is None:
        api.write(f"issues/{number}/comments", {"body": body})
    elif comment["body"] != body:
        api.write(f"issues/comments/{comment['id']}", {"body": body}, "PATCH")


def escalate(api: security.GitHub, pr: Json, comments: list[Json], now: dt.datetime) -> None:
    """Notify the maintainer once when a PR stops, and once more when its decision is overdue.

    Editing the triage comment notifies nobody, so the overdue notice has to be a separate comment.
    """
    number = pr["number"]
    if not pr.get("assignees"):
        api.write(f"issues/{number}/assignees", {"assignees": [MAINTAINER]})
    overdue = now > parse_time(pr["created_at"]) + DECISION_WINDOW
    if overdue and _own_comment(comments, OVERDUE_MARKER) is None:
        body = f"{OVERDUE_MARKER}\n@{MAINTAINER} The decision on this Dependabot PR is overdue. See the triage comment."
        api.write(f"issues/{number}/comments", {"body": body})


def merge(api: security.GitHub, pr: Json, decision: Decision, head_committed: dt.datetime, now: dt.datetime) -> Decision:
    """Merge exactly the evaluated head, and turn a refusal into a triage decision."""
    deadline = _decision_deadline(parse_time(pr["created_at"]))
    try:
        result = api.write(f"pulls/{pr['number']}/merge", {"sha": pr["head"]["sha"], "merge_method": "squash"}, "PUT")
    except security.GitHubError as error:
        if error.code in MERGE_RETRIES and now <= head_committed + DECISION_WINDOW:
            return Decision(
                "wait",
                decision.review,
                "The next evaluation tries again.",
                f"GitHub declined this merge attempt with HTTP status {error.code}.",
                "No action is needed.",
                deadline,
            )
        if error.code not in MERGE_REFUSALS | MERGE_RETRIES:
            raise
        reason = f"GitHub refused the merge with HTTP status {error.code}."
        remedy = (
            "Comment `@dependabot rebase`, or merge it by hand."
            if error.code in MERGE_RETRIES
            else "Check the token's permissions, or merge it by hand."
        )
    else:
        if result.get("merged"):
            return Decision("merged", decision.review, "Merged this head.", decision.reason, "Already merged.", "None.")
        reason = "GitHub answered the merge request without merging."
        remedy = "Merge it by hand."
    return Decision("stop", decision.review, "GitHub refused the merge of this head.", reason, remedy, deadline)


def write_access_checker(api: security.GitHub) -> Callable[[str], bool]:
    """Check repository permission, because `author_association` also admits members without write access.

    GitHub answers 404 for a login it cannot resolve, which has no access; other errors must not read as "no access".
    """
    known: dict[str, bool] = {}

    def has_write_access(login: str) -> bool:
        if login not in known:
            try:
                permission = api.get(f"collaborators/{quote(login, safe='')}/permission")["permission"]
            except security.GitHubError as error:
                if error.code != 404:
                    raise
                permission = "none"
            known[login] = permission in WRITE_PERMISSIONS
        return known[login]

    return has_write_access


def evaluate(
    api: security.GitHub,
    number: int,
    root: Path,
    has_write_access: Callable[[str], bool],
    enabled: bool,
    now: dt.datetime,
) -> tuple[Json, list[Json], dt.datetime, Decision]:
    pr = api.get(f"pulls/{number}")
    comments = api.pages(f"issues/{number}/comments")
    commits = api.pages(f"pulls/{number}/commits")
    files = api.pages(f"pulls/{number}/files")
    approved, changes = review_state(api.pages(f"pulls/{number}/reviews"), has_write_access)
    # A Dependabot PR's read-only token cannot post checks, so unlike security.merge this does not re-authenticate them.
    failed, pending, _ = security.ci_state(api, pr["head"]["sha"], root)
    # GitHub reports no push time for a head, so the latest commit date starts the CI clock.
    problem = provenance_problem(pr, commits, api.repo) or change_problem(files) or metadata_problem(pr, commits)
    created = parse_time(pr["created_at"])
    head_committed = max((parse_time(row["commit"]["committer"]["date"]) for row in commits), default=created)
    evidence = Evidence(
        created=created,
        head_committed=head_committed,
        merge_state=pr.get("mergeable_state"),
        problem=problem,
        hold=active_hold(comments, has_write_access, now),
        approved_by=approved,
        changes_requested_by=changes,
        failed=failed,
        pending=pending,
        enabled=enabled,
        # Only a verified Dependabot commit may name what to look up.
        advisories=known_advisories(api, pr, commits) if problem is None else [],
    )
    return pr, comments, head_committed, decide(evidence, now)


def ensure_post_merge(api: security.GitHub) -> None:
    """Start the master workflows that a GITHUB_TOKEN merge does not trigger, until each has a run."""
    sha = api.get("commits/master")["sha"]
    pulls = api.get(f"commits/{sha}/pulls")
    if not any(_login(row) == AUTHOR for row in pulls):
        return
    merged_by = [(api.get(f"pulls/{row['number']}").get("merged_by") or {}).get("login") for row in pulls]
    if security.BOT not in merged_by:
        return
    runs = api.pages(f"actions/runs?head_sha={sha}", "workflow_runs")
    started = {row["path"].split("/")[-1] for row in runs}
    for workflow in security.POST_MERGE:
        if workflow not in started:
            security.dispatch_workflow(api, workflow, "master")


def reconcile(api: security.GitHub, root: Path, enabled: bool, now: dt.datetime) -> list[str]:
    """Evaluate every open Dependabot PR; one PR's error never blocks another's decision."""
    errors = []
    has_write_access = write_access_checker(api)
    for row in api.pages("pulls?state=open&base=master"):
        number = row["number"]
        try:
            if _login(row) != AUTHOR:
                continue
            pr, comments, head_committed, decision = evaluate(api, number, root, has_write_access, enabled, now)
            if decision.action == "merge":
                decision = merge(api, pr, decision, head_committed, now)
            upsert_triage(api, number, comments, render(decision, pr["head"]["sha"]))
            if decision.action == "stop":
                escalate(api, pr, comments, now)
        except OPERATIONAL_ERRORS as error:
            errors.append(f"PR #{number}: {type(error).__name__}: {error}")
    try:
        ensure_post_merge(api)
    except OPERATIONAL_ERRORS as error:
        errors.append(f"Post-merge workflows: {type(error).__name__}: {error}")
    return errors


def merging_enabled(value: str | None) -> bool:
    """Merge only when the variable is unset, empty or `true` in any case: a stop must not depend on its spelling."""
    return (value or "").strip().lower() in {"", "true"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    enabled = merging_enabled(os.environ.get("DEPENDABOT_MERGE_ENABLED"))
    errors = reconcile(security.GitHub(), args.root, enabled, dt.datetime.now(dt.UTC))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
