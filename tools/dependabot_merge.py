"""Merge verified Dependabot PRs after full CI, and explain every other PR in a triage comment.

The controller is level-triggered: each run re-evaluates all open Dependabot PRs from GitHub's current state, so a missed
event only delays the next decision. It reads PR metadata as data and never checks out or executes PR code.
"""

import argparse
import datetime as dt
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from tools import security_automation as security

Json = security.Json

AUTHOR = "dependabot[bot]"
# GitHub signs Dependabot's commits as this committer; `verified` alone does not say who signed.
COMMITTER = "web-flow"
BRANCH_PREFIX = "dependabot/"
MAINTAINER = "NaokiHamada"
TRIAGE_MARKER = "<!-- dependabot-triage:v1 -->"
OVERDUE_MARKER = "<!-- dependabot-overdue:v1 -->"
CI_GRACE = dt.timedelta(hours=2)
DECISION_WINDOW = dt.timedelta(days=7)
MAX_HOLD = dt.timedelta(days=30)
MAX_REASON_CHARS = 300
MAX_LISTED_CHECKS = 8
SHORT_SHA_CHARS = 10
MERGEABLE_STATES = {"clean", "has_hooks"}
COMMAND_PERMISSIONS = {"admin", "write"}  # The endpoint reports the maintain role as write.
# 403: not permitted. 405: not mergeable. 409: the head moved. 422: invalid merge. Other errors may be transient.
MERGE_REFUSALS = {403, 405, 409, 422}
OPERATIONAL_ERRORS = (OSError, RuntimeError, AttributeError, KeyError, TypeError, ValueError)
DEPENDENCY_FILES = {
    "pyproject.toml",
    "uv.lock",
    "environment.yml",
    "docs/requirements.txt",
    ".pre-commit-config.yaml",
    "Dockerfile",
}
WORKFLOW_PATH = re.compile(r"\.github/workflows/[^/]+\.ya?ml")
# Stricter than security.ACTION_PIN, which runs on whole files: a line must not smuggle a second YAML line in its comment.
PIN_LINE = re.compile(r"([ \t]*(?:- )?uses:[ \t]+\w[\w./-]*@)[0-9a-f]{40}(?:[ \t]+#[^\x00-\x1f\x7f-\x9f  ]*)?")
HOLD_COMMAND = re.compile(r"/dependabot-hold\b:?(.*)", re.IGNORECASE)
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
    merge_state: str | None = "clean"
    problem: str | None = None
    hold: Hold | None = None
    approved_by: list[str] = field(default_factory=list)
    changes_requested_by: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    enabled: bool = True


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


def active_hold(comments: list[Json], can_command: Callable[[str], bool], now: dt.datetime) -> Hold | None:
    """Return the latest authorized hold, which its expiry or a later release command ends.

    A malformed hold still stops the merge for the decision window, because its author meant to stop it.
    """
    hold = None
    for comment in comments:
        lines = (comment["body"] or "").strip().splitlines()
        first = lines[0].strip() if lines else ""
        released = RELEASE_COMMAND.fullmatch(first) is not None
        command = HOLD_COMMAND.fullmatch(first)
        if not released and command is None:
            continue
        author = comment["user"]["login"]
        if not can_command(author):
            continue
        hold = None if command is None else _parse_hold(command[1], author, parse_time(comment["created_at"]))
    return hold if hold is not None and now <= hold.until else None


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
    return Hold(author, match[2][:MAX_REASON_CHARS], _end_of_day(day))


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
    lines = patch.split("\n")
    removed = [PIN_LINE.fullmatch(line[1:]) for line in lines if line.startswith("-")]
    added = [PIN_LINE.fullmatch(line[1:]) for line in lines if line.startswith("+")]
    if not added or len(removed) != len(added):
        return False
    # Pairing in order keeps the indentation and the action, so a step cannot be added, removed or swapped.
    return all(old and new and old[1] == new[1] for old, new in zip(removed, added, strict=True))


def review_state(reviews: list[Json]) -> tuple[list[str], list[str]]:
    """Return who approved and who requested changes, from each user's latest review that has a verdict."""
    latest = {}
    for review in reviews:
        if review["state"] != "COMMENTED":
            latest[review["user"]["login"]] = review["state"]
    approved = sorted(user for user, state in latest.items() if state == "APPROVED")
    changes = sorted(user for user, state in latest.items() if state == "CHANGES_REQUESTED")
    return approved, changes


def decide(evidence: Evidence, now: dt.datetime) -> Decision:
    """Choose the next action from the current evidence; every path but the last refuses to merge."""
    review = _review_line(evidence)
    window = f"{_day(evidence.created + DECISION_WINDOW)}. A maintainer decides by this date."
    ci_deadline = evidence.head_committed + CI_GRACE
    if evidence.hold is not None:
        return Decision(
            "hold",
            f"Reviewed by @{evidence.hold.author}.",
            "A maintainer decided not to merge for now.",
            evidence.hold.reason,
            "Comment `/dependabot-release`, or wait for the deadline.",
            f"{_day(evidence.hold.until)}. Automatic merging resumes after this date.",
        )
    blocker = _blocker(evidence, now > ci_deadline)
    if blocker is not None:
        return Decision("stop", review, "The automation will not merge this head.", *blocker, window)
    if evidence.pending:
        return Decision(
            "wait",
            review,
            "It merges this head once every check succeeds.",
            f"Waiting for {_names(evidence.pending)}.",
            "No action is needed.",
            f"{_minute(ci_deadline)}. Unfinished checks then stop the merge.",
        )
    if evidence.merge_state not in MERGEABLE_STATES:
        return Decision(
            "wait",
            review,
            "It merges this head once GitHub reports it as mergeable.",
            f"GitHub reports the merge state as `{evidence.merge_state}`.",
            "No action is needed.",
            f"{_minute(ci_deadline)}. This state then stops the merge.",
        )
    if not evidence.enabled:
        return Decision(
            "stop",
            review,
            "Automatic merging is paused; this head is ready.",
            "The repository variable `DEPENDABOT_MERGE_ENABLED` has paused automatic merging.",
            "Unset the variable or set it to `true`, or merge it by hand.",
            window,
        )
    return Decision(
        "merge",
        review,
        "Merging this head now.",
        "Every required check succeeded.",
        "No action is needed.",
        "None.",
    )


def _blocker(evidence: Evidence, ci_overdue: bool) -> tuple[str, str] | None:
    """Return why a maintainer must act and what would make the PR mergeable, or None if nothing stops it."""
    if evidence.changes_requested_by:
        return "A reviewer requested changes.", "Dismiss the review, or have the reviewer approve this PR."
    if evidence.problem is not None:
        return evidence.problem, "Comment `@dependabot recreate`, or review and merge it by hand."
    if evidence.failed:
        return f"These checks failed: {_names(evidence.failed)}.", _failure_remedy(evidence.failed)
    if evidence.merge_state == "dirty":
        return (
            "The PR conflicts with `master`.",
            "Dependabot normally rebases it by itself. If it does not, comment `@dependabot rebase`.",
        )
    if ci_overdue and evidence.pending:
        return (
            f"CI is still incomplete after {_hours(CI_GRACE)} hours: {_names(evidence.pending)}.",
            "Comment `@dependabot rebase`. An old base may not run the current checks.",
        )
    if ci_overdue and evidence.merge_state not in MERGEABLE_STATES:
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
            'Land the template first, as "Updating GHPages action pins" in CONTRIBUTING.md describes, '
            "then comment `@dependabot rebase`."
        )
    return (
        "Fix the cause on `master` and comment `@dependabot rebase`, or rerun the failed jobs if they are flaky. "
        "If this version must wait, comment `/dependabot-hold until=YYYY-MM-DD <reason>`."
    )


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
        if comment["user"]["login"] == security.BOT and (comment["body"] or "").startswith(marker):
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


def merge(api: security.GitHub, pr: Json, decision: Decision) -> Decision:
    """Merge exactly the evaluated head, and turn a refusal into a triage decision."""
    try:
        result = api.write(f"pulls/{pr['number']}/merge", {"sha": pr["head"]["sha"], "merge_method": "squash"}, "PUT")
    except security.GitHubError as error:
        if error.code not in MERGE_REFUSALS:
            raise
        reason = f"GitHub refused the merge with HTTP status {error.code}."
    else:
        if result.get("merged"):
            return Decision("merged", decision.review, "Merged this head.", decision.reason, "Already merged.", "None.")
        reason = "GitHub answered the merge request without merging."
    deadline = parse_time(pr["created_at"]) + DECISION_WINDOW
    return Decision(
        "stop",
        decision.review,
        "GitHub refused the merge of this head.",
        reason,
        "If the head moved, the next evaluation handles it. Otherwise check the token's permissions, or merge it by hand.",
        f"{_day(deadline)}. A maintainer decides by this date.",
    )


def command_authorizer(api: security.GitHub) -> Callable[[str], bool]:
    """Check repository permission, because `author_association` also admits members without write access."""
    known: dict[str, bool] = {}

    def can_command(login: str) -> bool:
        if login not in known:
            permission = api.get(f"collaborators/{quote(login, safe='')}/permission")["permission"]
            known[login] = permission in COMMAND_PERMISSIONS
        return known[login]

    return can_command


def evaluate(
    api: security.GitHub,
    number: int,
    root: Path,
    can_command: Callable[[str], bool],
    enabled: bool,
    now: dt.datetime,
) -> tuple[Json, list[Json], Decision]:
    pr = api.get(f"pulls/{number}")
    comments = api.pages(f"issues/{number}/comments")
    commits = api.pages(f"pulls/{number}/commits")
    files = api.pages(f"pulls/{number}/files")
    approved, changes = review_state(api.pages(f"pulls/{number}/reviews"))
    failed, pending, _ = security.ci_state(api, pr["head"]["sha"], root)
    # GitHub reports no push time for a head, so the latest commit date starts the CI clock.
    head_committed = max((parse_time(row["commit"]["committer"]["date"]) for row in commits), default=None)
    evidence = Evidence(
        created=parse_time(pr["created_at"]),
        head_committed=head_committed or parse_time(pr["created_at"]),
        merge_state=pr.get("mergeable_state"),
        problem=provenance_problem(pr, commits, api.repo) or change_problem(files),
        hold=active_hold(comments, can_command, now),
        approved_by=approved,
        changes_requested_by=changes,
        failed=failed,
        pending=pending,
        enabled=enabled,
    )
    return pr, comments, decide(evidence, now)


def ensure_post_merge(api: security.GitHub) -> None:
    """Start the master workflows that a GITHUB_TOKEN merge does not trigger, until each has a run."""
    sha = api.get("commits/master")["sha"]
    pulls = api.get(f"commits/{sha}/pulls")
    if not any(row["user"]["login"] == AUTHOR for row in pulls):
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
    can_command = command_authorizer(api)
    for row in api.pages("pulls?state=open&base=master"):
        number = row["number"]
        try:
            if (row.get("user") or {}).get("login") != AUTHOR:
                continue
            pr, comments, decision = evaluate(api, number, root, can_command, enabled, now)
            if decision.action == "merge":
                decision = merge(api, pr, decision)
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
    """Merge only when the variable is unset, empty or `true`: an emergency stop must not depend on its spelling."""
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
