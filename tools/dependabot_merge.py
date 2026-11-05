"""Merge verified Dependabot PRs after full CI, and explain every other PR in a triage comment.

The controller is level-triggered: each run re-evaluates all open Dependabot PRs from GitHub's current state, so a missed
event only delays the next decision. It reads PR metadata as data and never checks out or executes PR code.
"""

import argparse
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools import security_automation as security

Json = security.Json

AUTHOR = "dependabot[bot]"
BRANCH_PREFIX = "dependabot/"
MAINTAINER = "NaokiHamada"
TRIAGE_MARKER = "<!-- dependabot-triage:v1 -->"
OVERDUE_MARKER = "<!-- dependabot-overdue:v1 -->"
CI_GRACE = dt.timedelta(hours=2)
DECISION_WINDOW = dt.timedelta(days=7)
MAX_HOLD = dt.timedelta(days=30)
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
DEPENDENCY_FILES = {
    "pyproject.toml",
    "uv.lock",
    "environment.yml",
    "docs/requirements.txt",
    ".pre-commit-config.yaml",
    "Dockerfile",
}
WORKFLOW_PATH = re.compile(r"\.github/workflows/[^/]+\.ya?ml")
PIN_LINE = re.compile(r"[+-]\s*(?:- )?uses:\s+[\w./-]+@[0-9a-f]{40}(?:\s+#.*)?")
HOLD_COMMAND = re.compile(r"/dependabot-hold(?:\s+(.*))?")
HOLD_ARGUMENTS = re.compile(r"until=(\d{4}-\d{2}-\d{2})\s+(\S.*)")
RELEASE_COMMAND = re.compile(r"/dependabot-release\s*")


@dataclass(frozen=True)
class Hold:
    """A maintainer's decision not to merge, in effect until the end of ``until`` (UTC)."""

    author: str
    reason: str
    until: dt.datetime


@dataclass(frozen=True)
class Decision:
    action: Literal["merge", "wait", "stop", "hold", "merged"]
    review: str
    decision: str
    reason: str
    unblock: str
    deadline: str


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def active_hold(comments: list[Json], now: dt.datetime) -> Hold | None:
    """Return the latest trusted hold command, which an expiry or a later release command ends.

    A malformed or too distant hold still stops the merge, because its author meant to stop it.
    """
    hold = None
    for comment in comments:
        if comment["author_association"] not in TRUSTED_ASSOCIATIONS:
            continue
        lines = comment["body"].splitlines()
        first = lines[0].strip() if lines else ""
        if RELEASE_COMMAND.fullmatch(first):
            hold = None
            continue
        command = HOLD_COMMAND.fullmatch(first)
        if command is None:
            continue
        author = comment["user"]["login"]
        created = parse_time(comment["created_at"])
        hold = _parse_hold(command[1] or "", author, created)
    return hold if hold is not None and now <= hold.until else None


def _parse_hold(arguments: str, author: str, created: dt.datetime) -> Hold:
    fallback = Hold(
        author,
        "The hold command is invalid. Use `/dependabot-hold until=YYYY-MM-DD <reason>`, at most 30 days ahead.",
        created + DECISION_WINDOW,
    )
    match = HOLD_ARGUMENTS.fullmatch(arguments.strip())
    if match is None:
        return fallback
    try:
        day = dt.date.fromisoformat(match[1])
    except ValueError:
        return fallback
    until = dt.datetime.combine(day, dt.time.max, dt.UTC)
    if until > created + MAX_HOLD:
        return fallback
    return Hold(author, match[2][:300], until)


def provenance_problem(pr: Json, commits: list[Json], repo: str) -> str | None:
    """Require a same-repository Dependabot PR whose every commit Dependabot signed."""
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
        if author != AUTHOR or commit["commit"]["verification"]["verified"] is not True:
            return f"Commit `{commit['sha'][:10]}` is not a signed Dependabot commit."
    return None


def change_problem(files: list[Json]) -> str | None:
    """Allow dependency manifests, and only action pins in workflows."""
    for row in files:
        path = row["filename"]
        if path in DEPENDENCY_FILES:
            continue
        if WORKFLOW_PATH.fullmatch(path) is None:
            return f"`{path}` is not a dependency file."
        patch = row.get("patch")
        if row["status"] != "modified" or patch is None:
            return f"`{path}` has a change that cannot be verified as an action pin update."
        # Split on newlines only, so a control character cannot hide the rest of a line.
        changed = [line for line in patch.split("\n") if line.startswith(("+", "-"))]
        if not changed or not all(PIN_LINE.fullmatch(line) for line in changed):
            return f"`{path}` changes more than action pins."
    return None


def decide(
    pr: Json,
    problem: str | None,
    hold: Hold | None,
    failed: list[str],
    pending: list[str],
    enabled: bool,
    now: dt.datetime,
) -> Decision:
    """Choose the next action from the current evidence; every path but one refuses to merge."""
    window = _day(parse_time(pr["created_at"]) + DECISION_WINDOW)
    unreviewed = "Not reviewed by a person; the automation evaluated this head."
    no_merge = "The automation will not merge this head."
    if hold is not None:
        return Decision(
            "hold",
            f"Reviewed by @{hold.author}.",
            "A maintainer decided not to merge for now.",
            hold.reason,
            "Comment `/dependabot-release`, or wait for the deadline.",
            f"{_day(hold.until)}. Automatic merging resumes after this date.",
        )
    if problem is not None:
        return Decision(
            "stop",
            unreviewed,
            no_merge,
            problem,
            "Comment `@dependabot recreate`, or review and merge it by hand.",
            f"{window}. A maintainer decides by this date.",
        )
    if failed:
        return _failed_decision(pr, failed, unreviewed, no_merge, window)
    state = pr.get("mergeable_state")
    if state == "dirty":
        return Decision(
            "stop",
            unreviewed,
            no_merge,
            "The PR conflicts with `master`.",
            "Dependabot normally rebases it by itself. If it does not, comment `@dependabot rebase`.",
            f"{window}. A maintainer decides by this date.",
        )
    if pending:
        waited = now - parse_time(pr["head"]["pushed_at"])
        listed = _names(pending)
        if waited <= CI_GRACE:
            return Decision(
                "wait",
                "Not reviewed yet; the automation is waiting for CI.",
                "It merges this head once every check succeeds.",
                f"Waiting for {listed}.",
                "No action is needed.",
                f"{_minute(parse_time(pr['head']['pushed_at']) + CI_GRACE)}. Unfinished checks then stop the merge.",
            )
        return Decision(
            "stop",
            unreviewed,
            no_merge,
            f"CI is still incomplete after {CI_GRACE.seconds // 3600} hours: {listed}.",
            "Comment `@dependabot rebase`. An old base may not run the current checks.",
            f"{window}. A maintainer decides by this date.",
        )
    if state not in {"clean", "has_hooks"}:
        return Decision(
            "wait",
            "Not reviewed yet; the automation is waiting for GitHub.",
            "It merges this head once GitHub reports it as mergeable.",
            f"GitHub reports the merge state as `{state}`.",
            "No action is needed unless this state persists.",
            f"{window}. A maintainer decides by this date.",
        )
    if not enabled:
        return Decision(
            "stop",
            unreviewed,
            "Automatic merging is paused; this head is ready.",
            "The repository variable `DEPENDABOT_MERGE_ENABLED` is `false`.",
            "Remove the variable, or merge it by hand.",
            f"{window}. A maintainer decides by this date.",
        )
    return Decision(
        "merge",
        unreviewed,
        "Merging this head now.",
        "Every required check succeeded.",
        "No action is needed.",
        "None.",
    )


def _failed_decision(pr: Json, failed: list[str], review: str, decision: str, window: str) -> Decision:
    if "pages-producer" in failed:
        unblock = (
            'Land the template first, as "Updating GHPages action pins" in CONTRIBUTING.md describes, '
            "then comment `@dependabot rebase`."
        )
    else:
        unblock = (
            "Fix the cause on `master` and comment `@dependabot rebase`, or rerun the failed jobs if they are flaky. "
            "If this version must wait, comment `/dependabot-hold until=YYYY-MM-DD <reason>`."
        )
    return Decision(
        "stop",
        review,
        decision,
        f"These checks failed: {_names(failed)}.",
        unblock,
        f"{window}. A maintainer decides by this date.",
    )


def _names(names: list[str]) -> str:
    unique = sorted(set(names))
    shown = ", ".join(f"`{name}`" for name in unique[:8])
    return shown if len(unique) <= 8 else f"{shown} and {len(unique) - 8} more"


def _day(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d UTC")


def _minute(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def render(decision: Decision, sha: str) -> str:
    """Render without a timestamp, so that an unchanged decision edits nothing."""
    return "\n".join(
        [
            TRIAGE_MARKER,
            f"**Dependabot triage** for `{sha[:10]}`",
            "",
            f"- **Review:** {decision.review}",
            f"- **Decision:** {decision.decision}",
            f"- **Reason:** {decision.reason}",
            f"- **Mergeable when:** {decision.unblock}",
            f"- **Deadline:** {decision.deadline}",
        ]
    )


def _own_comment(comments: list[Json], marker: str) -> Json | None:
    for comment in comments:
        if comment["user"]["login"] == security.BOT and comment["body"].startswith(marker):
            return comment
    return None


def upsert_triage(api: security.GitHub, number: int, comments: list[Json], body: str) -> None:
    comment = _own_comment(comments, TRIAGE_MARKER)
    if comment is None:
        api.write(f"issues/{number}/comments", {"body": body})
    elif comment["body"] != body:
        api.write(f"issues/comments/{comment['id']}", {"body": body}, "PATCH")


def escalate(api: security.GitHub, pr: Json, comments: list[Json], now: dt.datetime) -> None:
    """Notify the maintainer once when a PR stops, and once more when its decision is overdue."""
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
        result = api.write(
            f"pulls/{pr['number']}/merge",
            {"sha": pr["head"]["sha"], "merge_method": "squash"},
            "PUT",
        )
    except security.GitHubError as error:
        if error.code not in {403, 405, 409, 422}:
            raise
        result = {"merged": False, "code": error.code}
    if result.get("merged"):
        return Decision("merged", decision.review, "Merged this head.", decision.reason, "Already merged.", "None.")
    code = result.get("code", "without a merge")
    return Decision(
        "stop",
        decision.review,
        "GitHub refused the merge of this head.",
        f"The merge request ended with HTTP status {code}.",
        "If the head moved, the next evaluation handles it. Otherwise check the token's permissions, or merge it by hand.",
        decision.deadline if decision.deadline != "None." else "A maintainer decides at the next opportunity.",
    )


def evaluate(
    api: security.GitHub, number: int, root: Path, enabled: bool, now: dt.datetime
) -> tuple[Json, list[Json], Decision]:
    pr = api.get(f"pulls/{number}")
    comments = api.pages(f"issues/{number}/comments")
    commits = api.pages(f"pulls/{number}/commits")
    files = api.pages(f"pulls/{number}/files")
    pr["head"]["pushed_at"] = max(
        (row["commit"]["committer"]["date"] for row in commits),
        default=pr["created_at"],
    )
    problem = provenance_problem(pr, commits, api.repo) or change_problem(files)
    failed, pending, _ = security.ci_state(api, pr["head"]["sha"], root)
    hold = active_hold(comments, now)
    return pr, comments, decide(pr, problem, hold, failed, pending, enabled, now)


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
            api.write(f"actions/workflows/{workflow}/dispatches", {"ref": "master"})


def reconcile(api: security.GitHub, root: Path, enabled: bool, now: dt.datetime) -> list[str]:
    """Evaluate every open Dependabot PR; one PR's error never blocks another's decision."""
    errors = []
    for row in api.pages("pulls?state=open&base=master"):
        if row["user"]["login"] != AUTHOR:
            continue
        number = row["number"]
        try:
            pr, comments, decision = evaluate(api, number, root, enabled, now)
            if decision.action == "merge":
                decision = merge(api, pr, decision)
            upsert_triage(api, number, comments, render(decision, pr["head"]["sha"]))
            if decision.action == "stop":
                escalate(api, pr, comments, now)
        except (security.GitHubError, KeyError, TypeError, ValueError) as error:
            errors.append(f"PR #{number}: {type(error).__name__}: {error}")
    try:
        ensure_post_merge(api)
    except security.GitHubError as error:
        errors.append(f"Post-merge workflows: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    enabled = os.environ.get("DEPENDABOT_MERGE_ENABLED") != "false"
    errors = reconcile(security.GitHub(), args.root, enabled, dt.datetime.now(dt.UTC))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
