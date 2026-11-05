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
# Bounds both unfinished checks and merge states that GitHub has computed, measured from the latest commit.
CI_GRACE = dt.timedelta(hours=2)
# Dependabot rebases a conflict by itself, and a rebase restarts the clock, so a merged sibling need not notify anyone.
REBASE_GRACE = dt.timedelta(days=1)
DECISION_WINDOW = dt.timedelta(days=7)
MAX_HOLD = dt.timedelta(days=30)
MAX_REASON_CHARS = 300
MAX_LISTED_CHECKS = 8
SHORT_SHA_CHARS = 10
MERGEABLE_STATES = {"clean", "has_hooks"}
# GitHub reports these while it recomputes mergeability, which a sibling merge can trigger.
UNCOMPUTED_STATES = {"unknown", None}
COMMAND_PERMISSIONS = {"admin", "write"}  # The endpoint reports the maintain role as write.
# 403: not permitted. 422: invalid merge. Other errors may be transient and fail the job instead.
MERGE_REFUSALS = {403, 422}
# 405: not mergeable, as when the base just moved. 409: the head moved. The next evaluation sees the new state.
MERGE_RETRIES = {405, 409}
# A malformed payload or a failed request for one PR must not stop the others; main() still fails the job.
OPERATIONAL_ERRORS = (OSError, RuntimeError, http.client.HTTPException, AttributeError, KeyError, TypeError, ValueError)
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
PIN_LINE = re.compile(r"([ \t]*(?:- )?uses:[ \t]+\w[\w./-]*@)[0-9a-f]{40}(?:[ \t]+#[^\x00-\x1f\x7f-\x9f\u2028\u2029]*)?")
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
        first = lines[0].strip().strip("`") if lines else ""
        released = RELEASE_COMMAND.fullmatch(first) is not None
        command = HOLD_COMMAND.fullmatch(first)
        if not released and command is None:
            continue
        author = _login(comment)
        if author is None or not can_command(author):
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


def review_state(reviews: list[Json], can_command: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """Return who approved and who requested changes, from each authorized user's latest verdict.

    Anyone can review a public repository, so a review without write access must not stop a security update.
    """
    latest = {}
    for review in reviews:
        login = _login(review)
        if login is not None and review["state"] != "COMMENTED":
            latest[login] = review["state"]
    approved = sorted(user for user, state in latest.items() if state == "APPROVED" and can_command(user))
    changes = sorted(user for user, state in latest.items() if state == "CHANGES_REQUESTED" and can_command(user))
    return approved, changes


def decide(evidence: Evidence, now: dt.datetime) -> Decision:
    """Choose the next action from the current evidence; every path but the last refuses to merge."""
    review = _review_line(evidence)
    window = _decision_deadline(evidence.created)
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
    blocker = _blocker(evidence, now)
    if blocker is not None:
        reason, mergeable_when = blocker
        return Decision("stop", review, "The automation will not merge this head.", reason, mergeable_when, window)
    paused = "Automatic merging is paused, so this head will not merge."
    resume = "Unset the variable `DEPENDABOT_MERGE_ENABLED` or set it to `true`, or merge it by hand."
    if evidence.merge_state == "dirty":
        return Decision(
            "wait",
            review,
            "It merges a new head once the conflict is gone." if evidence.enabled else paused,
            "The PR conflicts with `master`.",
            "Dependabot normally rebases it by itself. If it does not, comment `@dependabot rebase`.",
            f"{_minute(evidence.head_committed + REBASE_GRACE)}. A remaining conflict then stops the merge.",
        )
    if evidence.pending:
        return Decision(
            "wait",
            review,
            "It merges this head once every check succeeds." if evidence.enabled else paused,
            f"Waiting for {_names(evidence.pending)}.",
            "No action is needed." if evidence.enabled else resume,
            f"{_minute(ci_deadline)}. Unfinished checks then stop the merge.",
        )
    if evidence.merge_state not in MERGEABLE_STATES:
        settled = evidence.head_committed + (DECISION_WINDOW if evidence.merge_state in UNCOMPUTED_STATES else CI_GRACE)
        return Decision(
            "wait",
            review,
            "It merges this head once GitHub reports it as mergeable." if evidence.enabled else paused,
            f"GitHub reports the merge state as `{evidence.merge_state}`.",
            "No action is needed." if evidence.enabled else resume,
            f"{_minute(settled)}. This state then stops the merge.",
        )
    if not evidence.enabled:
        return Decision(
            "stop",
            review,
            paused,
            "The repository variable `DEPENDABOT_MERGE_ENABLED` has paused automatic merging.",
            resume,
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


def _blocker(evidence: Evidence, now: dt.datetime) -> tuple[str, str] | None:
    """Return why a maintainer must act and what would make the PR mergeable, or None if nothing stops it."""
    age = now - evidence.head_committed
    if evidence.changes_requested_by:
        return "A reviewer requested changes.", "Dismiss the review, or have the reviewer approve this PR."
    if evidence.problem is not None:
        return evidence.problem, "Comment `@dependabot recreate`, or review and merge it by hand."
    if evidence.failed:
        return f"These checks failed: {_names(evidence.failed)}.", _failure_remedy(evidence.failed)
    if evidence.merge_state == "dirty" and age > REBASE_GRACE:
        return "The PR still conflicts with `master`.", "Comment `@dependabot rebase`, or close this PR."
    if evidence.pending and age > CI_GRACE:
        return (
            f"CI is still incomplete after {_hours(CI_GRACE)} hours: {_names(evidence.pending)}.",
            "Comment `@dependabot rebase`. An old base may not run the current checks.",
        )
    limit = DECISION_WINDOW if evidence.merge_state in UNCOMPUTED_STATES else CI_GRACE
    if evidence.merge_state not in MERGEABLE_STATES | {"dirty"} and age > limit:
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


def merge(api: security.GitHub, pr: Json, decision: Decision, now: dt.datetime) -> Decision:
    """Merge exactly the evaluated head, and turn a refusal into a triage decision."""
    created = parse_time(pr["created_at"])
    window = _decision_deadline(created)
    try:
        result = api.write(f"pulls/{pr['number']}/merge", {"sha": pr["head"]["sha"], "merge_method": "squash"}, "PUT")
    except security.GitHubError as error:
        if error.code in MERGE_RETRIES and now <= created + DECISION_WINDOW:
            return Decision(
                "wait",
                decision.review,
                "The next evaluation tries again.",
                f"GitHub declined this merge attempt with HTTP status {error.code}.",
                "No action is needed.",
                window,
            )
        if error.code not in MERGE_REFUSALS | MERGE_RETRIES:
            raise
        reason = f"GitHub refused the merge with HTTP status {error.code}."
    else:
        if result.get("merged"):
            return Decision("merged", decision.review, "Merged this head.", decision.reason, "Already merged.", "None.")
        reason = "GitHub answered the merge request without merging."
    return Decision(
        "stop",
        decision.review,
        "GitHub refused the merge of this head.",
        reason,
        "Check the token's permissions, or merge it by hand.",
        window,
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
    approved, changes = review_state(api.pages(f"pulls/{number}/reviews"), can_command)
    # A Dependabot PR's read-only token cannot post checks, so unlike security.merge this does not re-authenticate them.
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
    can_command = command_authorizer(api)
    for row in api.pages("pulls?state=open&base=master"):
        number = row["number"]
        try:
            if _login(row) != AUTHOR:
                continue
            pr, comments, decision = evaluate(api, number, root, can_command, enabled, now)
            if decision.action == "merge":
                decision = merge(api, pr, decision, now)
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
