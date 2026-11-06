"""Merge release-please's release PR once its CI passes, so that merged fixes reach a release without a human.

A release PR merges a day after the newest releasable commit in the release, or at once when the release contains a
dependency or security fix. Like the Dependabot controller, each run re-evaluates the PR from GitHub's current state,
and the PR's content is read as data only. See "Release automation" in SECURITY.md.
"""

import argparse
import datetime as dt
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools import dependabot_merge as merge
from tools import pages_publish
from tools import security_automation as security

Json = security.Json

RELEASE_BRANCH = "release-please--branches--master--components--torch_bsf"
RELEASE_TITLE = re.compile(r"chore\(master\): release \d+\.\d+\.\d+")
RELEASE_LABEL = "autorelease: pending"
VERSION = r"\d+\.\d+\.\d+"
MANIFEST = ".release-please-manifest.json"
MANIFEST_LINE = rf'  "\.": "({VERSION})"'
# The only lines that release-please changes outside the changelog.
VERSION_LINES = {
    MANIFEST: re.compile(MANIFEST_LINE),
    "pyproject.toml": re.compile(rf'version = "{VERSION}"'),
}
CHANGELOG = "CHANGELOG.md"
# Release-please branches publish no documentation (tools/pages_build.py), so the GHPages build workflow, its `build`
# job and the publisher's check never report for them.
PAGES_BUILD = security.PAGES_PRODUCER.rsplit("/", 1)[-1]
WORKFLOWS = [workflow for workflow in security.WORKFLOWS if workflow != PAGES_BUILD]
DOCUMENTATION_CHECKS = frozenset({"build", pages_publish.CHECK_NAME})
# codecov/patch grades a push's own diff, not whether `master` works, so a failure there does not stop a release.
MASTER_EXCLUDED_CHECKS = DOCUMENTATION_CHECKS | {security.CODECOV_PATCH}
# The release workflow comes first, so that a failure to start the others cannot hold back the release.
RELEASE_WORKFLOW = "release-please-action.yml"
POST_MERGE = [RELEASE_WORKFLOW, *(workflow for workflow in security.POST_MERGE if workflow != RELEASE_WORKFLOW)]
QUIET = dt.timedelta(days=1)
# A commit subject that makes the release urgent: a dependency update or fix, or a security remediation. Dependabot
# writes `deps-dev` for dependencies it considers development ones.
URGENT = re.compile(r"(?:fix|feat)\((?:deps|deps-dev|security)\)!?:", re.IGNORECASE)
# A commit that release-please releases: a conventional commit of a type its python strategy shows in the changelog,
# or a conventional commit that breaks compatibility. Like release-please, types are read case-insensitively.
# tests/tools/release-please-fixture/releasable.json pins this against the engine.
CONVENTIONAL = re.compile(r"\w+(?:\([^)]*\))?!?:")
RELEASABLE = re.compile(r"(?:(?:feat|fix|perf|revert|deps|docs)(?:\([^)]*\))?!?|\w+(?:\([^)]*\))?!):", re.IGNORECASE)
BREAKING = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)
STATUS_MARKER = "<!-- release-merge:v1 -->"
STOP_MARKER = "<!-- release-merge-stop:v1"
# The GitHub App that opens and updates the release PR, named by the repository variable RELEASE_PLEASE_BOT.
APP_BOT_VARIABLE = "RELEASE_PLEASE_BOT"
APP_BOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\[bot\]")
APP_SETUP = "see Release App in SECURITY.md"


@dataclass(frozen=True)
class Evidence:
    """What the decision needs to know about the release PR's head."""

    problem: str | None
    draft: bool
    head_committed: dt.datetime
    # When the release last gained a releasable change on `master`. release-please rewrites the head daily for the
    # date in the changelog, so the head's own time does not show whether the release is still changing.
    last_change: dt.datetime
    # Releasable commits on `master` that the head lacks, and the newest time a commit missing from the head landed
    # on `master`, or None when no releasable commit is missing.
    unreleased: list[str]
    newest_unreleased: dt.datetime | None
    merge_state: str | None
    failed: list[str]
    pending: list[str]
    # Workflows whose run on this head waits for approval, as runs started by a GITHUB_TOKEN push do.
    awaiting_approval: list[str]
    master_failed: list[str]
    urgent: list[str]
    enabled: bool


@dataclass(frozen=True)
class Decision:
    action: Literal["merge", "wait", "stop", "hold", "merged"]
    outcome: str
    reason: str
    mergeable_when: str


def release_authors(app_bot: str | None) -> tuple[frozenset[str], str | None]:
    """Return the accounts release-please writes as, and an error if the App's bot variable is set but invalid.

    The accounts are GITHUB_TOKEN's bot, and the release App's bot when the variable names one.
    """
    bot = (app_bot or "").strip()
    if APP_BOT.fullmatch(bot):
        return frozenset({security.BOT, bot}), None
    error = f"The variable `{APP_BOT_VARIABLE}` must name the release App's bot, like `<app-slug>[bot]`." if bot else None
    return frozenset({security.BOT}), error


def _with_bot_hint(reason: str, login: str | None, authors: frozenset[str]) -> str:
    """Name the variable when a bot outside `authors` wrote it, which is likely the release App left unconfigured."""
    if login is None or login in authors or not login.endswith("[bot]"):
        return reason
    return (
        f"{reason} `{login}` is not an accepted release author. If it is the release App's bot, set the repository "
        f"variable `{APP_BOT_VARIABLE}` to it ({APP_SETUP})."
    )


def release_problem(pr: Json, commits: list[Json], files: list[Json], repo: str, authors: frozenset[str]) -> str | None:
    """Accept only release-please's own PR, which edits the version and the changelog and nothing else."""
    author = merge.author_login(pr)
    if (
        author not in authors
        or pr["head"]["repo"]["full_name"] != repo
        or pr["head"]["ref"] != RELEASE_BRANCH
        or pr["base"]["ref"] != "master"
    ):
        reason = "This is not release-please's release PR for `master`."
        return _with_bot_hint(reason, author, authors) if pr["head"]["ref"] == RELEASE_BRANCH else reason
    if RELEASE_TITLE.fullmatch(pr["title"]) is None:
        return "The title is not a release-please release title."
    if RELEASE_LABEL not in {label["name"] for label in pr.get("labels", [])}:
        return f"The PR lacks the `{RELEASE_LABEL}` label, without which release-please creates no release."
    return _commit_problem(commits, authors) or _file_problem(files)


def _commit_problem(commits: list[Json], authors: frozenset[str]) -> str | None:
    if not commits:
        return "The PR has no commits to verify."
    for row in commits:
        author = (row.get("author") or {}).get("login")
        committer = (row.get("committer") or {}).get("login")
        # GitHub signs API commits as web-flow, except that an App's own commits may be signed in the App's name.
        signers = {merge.COMMITTER} if author == security.BOT else {merge.COMMITTER, author}
        if author not in authors or committer not in signers or row["commit"]["verification"]["verified"] is not True:
            reason = f"Commit `{row['sha'][: merge.SHORT_SHA_CHARS]}` was not written and signed through release-please."
            return _with_bot_hint(reason, author, authors)
    return None


def _file_problem(files: list[Json]) -> str | None:
    for row in files:
        name = row["filename"]
        if row["status"] != "modified" or (name != CHANGELOG and name not in VERSION_LINES):
            return f"The PR changes `{name}`, which a release does not change."
        if name in VERSION_LINES:
            changed = [line[1:] for line in (row.get("patch") or "").splitlines() if merge.is_change(line)]
            if not changed or any(VERSION_LINES[name].fullmatch(line) is None for line in changed):
                return f"The PR changes `{name}` beyond its version."
    if previous_version(files) is None:
        return f"The PR does not show the previous version in `{MANIFEST}`."
    return None


def previous_version(files: list[Json]) -> str | None:
    for row in files:
        if row["filename"] == MANIFEST:
            for line in (row.get("patch") or "").splitlines():
                match = re.fullmatch("-" + MANIFEST_LINE, line)
                if match:
                    return match[1]
    return None


def compare_commits(api: security.GitHub, base: str, head: str) -> list[Json]:
    """Return every commit reachable from `head` but not from `base`."""
    commits: list[Json] = []
    total, page = None, 1
    while total is None or len(commits) < total:
        comparison = api.get(f"compare/{base}...{head}?per_page=100&page={page}")
        rows = comparison["commits"]
        if not rows:
            break
        commits.extend(rows)
        total, page = comparison["total_commits"], page + 1
    return commits


def subject(row: Json) -> str:
    return str(row["commit"]["message"]).split("\n", 1)[0]


def releasable(row: Json) -> bool:
    line = subject(row)
    if RELEASABLE.match(line):
        return True
    return CONVENTIONAL.match(line) is not None and BREAKING.search(row["commit"]["message"]) is not None


def committed(row: Json) -> dt.datetime:
    return merge.parse_time(row["commit"]["committer"]["date"])


def is_merge(row: Json) -> bool:
    return len(row.get("parents", [])) > 1


def landed(rows: list[Json]) -> dt.datetime | None:
    """Return the newest time a releasable change in `rows` landed on `master`.

    A squash or rebase merge commits at landing time. A merge commit does too, but the branch commits it brings in keep
    the times they were written, so its own time counts for them.
    """
    return max((committed(row) for row in rows if releasable(row) or is_merge(row)), default=None)


def decide(evidence: Evidence, now: dt.datetime) -> Decision:
    """Stop on anything wrong, wait for CI and GitHub within their limits, then merge when urgent or quiet."""
    if evidence.problem is not None:
        return Decision("stop", "Not merging this PR.", evidence.problem, "Merge or close it by hand.")
    if evidence.draft:
        return Decision("hold", "Held.", "The PR is a draft.", "Mark it ready for review.")
    if evidence.awaiting_approval:
        return Decision(
            "stop",
            "Not merging this head.",
            f"CI on this head waits for approval: {merge.format_names(evidence.awaiting_approval)}. release-please "
            "updated the PR with GITHUB_TOKEN instead of the release App.",
            f"Set up the release App ({APP_SETUP}), or approve and run the workflows.",
        )
    if evidence.failed:
        return Decision(
            "stop",
            "Not merging this head.",
            f"Checks failed: {merge.format_names(evidence.failed)}.",
            "Re-run the failed workflows on the release branch if the failure was transient. Otherwise fix `master`; "
            "a releasable fix updates this PR, and its new head is tested again.",
        )
    if evidence.master_failed:
        # The merge also ships the `master` commits that release-please did not add to this head.
        return Decision(
            "stop",
            "Not merging this head.",
            f"Checks failed on the head of `master`: {merge.format_names(evidence.master_failed)}.",
            "Fix `master`, or re-run its failed workflows if the failure was transient.",
        )
    return _waiting(evidence, now) or _merging(evidence, now)


def _waiting(evidence: Evidence, now: dt.datetime) -> Decision | None:
    if evidence.newest_unreleased is not None:
        # The merge would ship these commits outside the release, and release-please lists only later commits.
        update_deadline = evidence.newest_unreleased + merge.CI_GRACE
        reason = (
            f"`master` has releasable commits that this PR does not include yet: {merge.format_names(evidence.unreleased)}."
        )
        if now > update_deadline:
            return Decision(
                "stop",
                "Not merging this head.",
                reason,
                "Check the `release-pr` job of the Release workflow and re-run it, or merge the PR by hand.",
            )
        return Decision(
            "wait",
            "Waiting for release-please.",
            reason,
            f"The `release-pr` job updates the PR by {merge.format_minute(update_deadline)}.",
        )
    ci_deadline = evidence.head_committed + merge.CI_GRACE
    if evidence.pending:
        if now > ci_deadline:
            return Decision(
                "stop",
                "Not merging this head.",
                f"CI is still incomplete {merge.format_hours(merge.CI_GRACE)} hours after the head's commit: "
                f"{merge.format_names(evidence.pending)}.",
                "Re-run the unfinished workflows on this PR.",
            )
        return Decision(
            "wait",
            "Waiting for CI.",
            f"Unfinished checks: {merge.format_names(evidence.pending)}.",
            f"All checks succeed by {merge.format_minute(ci_deadline)}.",
        )
    if evidence.merge_state not in merge.MERGEABLE_STATES:
        state_deadline = evidence.head_committed + merge.merge_state_limit(evidence.merge_state)
        reason = f"GitHub reports the merge state `{evidence.merge_state}`."
        if now > state_deadline:
            return Decision("stop", "Not merging this head.", reason, "Resolve what blocks the merge, or merge it by hand.")
        return Decision(
            "wait",
            "Waiting for GitHub.",
            reason,
            f"GitHub reports the PR as mergeable by {merge.format_minute(state_deadline)}; release-please rebases it "
            "when `master` gets a releasable change.",
        )
    if not evidence.enabled:
        return Decision(
            "wait",
            "Paused.",
            "The repository variable `RELEASE_MERGE_ENABLED` has paused automatic release merges.",
            "Unset the variable `RELEASE_MERGE_ENABLED` or set it to `true`, or merge it by hand.",
        )
    return None


def _merging(evidence: Evidence, now: dt.datetime) -> Decision:
    if evidence.urgent:
        return Decision(
            "merge",
            "Merging now.",
            f"The release contains a dependency or security fix: {merge.format_names(evidence.urgent)}.",
            "Already mergeable.",
        )
    quiet_until = evidence.last_change + QUIET
    if now >= quiet_until:
        return Decision(
            "merge",
            "Merging now.",
            f"The release has had no new releasable commit for {merge.format_hours(QUIET)} hours.",
            "Already mergeable.",
        )
    return Decision(
        "wait",
        "Waiting for more changes.",
        "A release waits a day after its newest releasable commit, unless it contains a dependency or security fix.",
        f"No new releasable commit until {merge.format_minute(quiet_until)}.",
    )


def render(decision: Decision, sha: str) -> str:
    """Render without a timestamp, so that an unchanged decision edits nothing."""
    return "\n".join(
        [
            STATUS_MARKER,
            f"**Release automation** for `{sha[: merge.SHORT_SHA_CHARS]}`",
            "",
            f"- **Decision:** {decision.outcome}",
            f"- **Reason:** {decision.reason}",
            f"- **Mergeable when:** {decision.mergeable_when}",
        ]
    )


def notify_stop(api: security.GitHub, pr: Json, comments: list[Json]) -> None:
    """Mention the maintainer once for each stopped head, because editing the status comment notifies nobody."""
    marker = f"{STOP_MARKER} {pr['head']['sha']} -->"
    if merge.own_comment(comments, marker) is None:
        body = f"{marker}\n@{merge.MAINTAINER} The release automation stopped this release PR. See the status comment."
        api.write(f"issues/{pr['number']}/comments", {"body": body})


def master_failures(api: security.GitHub, root: Path) -> list[str]:
    """Return the failed CI of `master`'s head, ignoring checks that say nothing about the code a release ships.

    Other checks also land on that commit, such as GitHub's own Dependabot update jobs, and they do not count.
    """
    failed = security.ci_state(api, api.get("commits/master")["sha"], root, workflows=WORKFLOWS)[0]
    # ci_state reports every failed check on the commit, whatever is expected, so the filter has to be here.
    counted = (security.expected_checks(root) - MASTER_EXCLUDED_CHECKS) | set(WORKFLOWS)
    return [name for name in failed if name in counted]


def awaiting_approval(runs: dict[str, Json]) -> list[str]:
    return [workflow for workflow in WORKFLOWS if (runs.get(workflow) or {}).get("conclusion") == "action_required"]


def evaluate(
    api: security.GitHub,
    number: int,
    root: Path,
    enabled: bool,
    authors: frozenset[str],
    *,
    settling: frozenset[str] = frozenset(),
) -> tuple[Json, list[Json], Evidence]:
    """Read the PR, its checks and the checks of `master`'s head, and return what the decision needs."""
    pr = merge.read_pull(api, number, settling=settling)
    comments = api.pages(f"issues/{number}/comments")
    commits = api.pages(f"pulls/{number}/commits")
    files = api.pages(f"pulls/{number}/files")
    sha = pr["head"]["sha"]
    head = [row for row in commits if row["sha"] == sha]
    problem = release_problem(pr, commits, files, api.repo, authors) or (None if head else "The head commit is not listed.")
    head_committed = committed(head[0]) if head else merge.parse_time(pr["created_at"])
    last_change = head_committed
    unreleased: list[Json] = []
    newest_unreleased: dt.datetime | None = None
    failed: list[str] = []
    pending: list[str] = []
    master_failed: list[str] = []
    runs: dict[str, Json] = {}
    urgent: list[str] = []
    if problem is None:
        failed, pending, runs = security.ci_state(api, sha, root, workflows=WORKFLOWS, excluded_checks=DOCUMENTATION_CHECKS)
        master_failed = master_failures(api, root)
        version = previous_version(files)
        assert version is not None  # release_problem rejects a PR without it.
        try:
            in_release = compare_commits(api, f"v{version}", sha)
        except security.GitHubError as error:
            if error.code != 404:
                raise
            problem = f"The previous release tag `v{version}` does not exist."
        else:
            urgent = [subject(row) for row in in_release if URGENT.match(subject(row))]
            last_change = landed(in_release) or head_committed
            ahead = compare_commits(api, sha, "master")
            unreleased = [row for row in ahead if releasable(row)]
            if unreleased:
                # Any commit ahead may be the one that landed them, such as a merge commit.
                newest_unreleased = max(committed(row) for row in ahead)
    evidence = Evidence(
        problem,
        pr["draft"],
        head_committed,
        last_change,
        [subject(row) for row in unreleased],
        newest_unreleased,
        pr.get("mergeable_state"),
        failed,
        pending,
        awaiting_approval(runs),
        master_failed,
        urgent,
        enabled,
    )
    return pr, comments, evidence


def evaluate_after_codecov(
    api: security.GitHub, number: int, root: Path, enabled: bool, authors: frozenset[str], clock: merge.RunClock
) -> tuple[Json, list[Json], Evidence, Decision]:
    """Evaluate the PR, and when only codecov/patch keeps it from merging, wait for it.

    After the wait every input of the decision may have changed, so the PR is evaluated again from scratch.
    """
    pr, comments, evidence = evaluate(api, number, root, enabled, authors)
    decision = decide(evidence, clock.now)
    # A PR already stopped, for example by the CI deadline, gains nothing from the wait.
    if (
        decision.action == "wait"
        and merge.codecov_alone_stops(evidence, clock.now, decide)
        and merge.wait_for_codecov(
            api,
            pr["head"]["sha"],
            root,
            clock.codecov_deadline,
            workflows=WORKFLOWS,
            excluded_checks=DOCUMENTATION_CHECKS,
        )
    ):
        clock.advance()
        pr, comments, evidence = evaluate(api, number, root, enabled, authors, settling=frozenset({"unstable"}))
        decision = decide(evidence, clock.now)
    return pr, comments, evidence, decision


def merge_release(
    api: security.GitHub, pr: Json, comments: list[Json], decision: Decision, head_committed: dt.datetime, now: dt.datetime
) -> Decision:
    """Merge exactly the evaluated head and record it, or turn a refusal into a decision.

    Like the Dependabot controller, the next run retries a transient refusal within the decision window and stops on
    any other.
    """
    number = pr["number"]
    data = {"sha": pr["head"]["sha"], "merge_method": "squash", "commit_title": f"{pr['title']} (#{number})"}
    try:
        result = api.write(f"pulls/{number}/merge", data, "PUT")
    except security.GitHubError as error:
        reason = f"GitHub refused the merge with HTTP status {error.code}."
        if merge.refusal_action(error, head_committed, now) == "wait":
            return Decision("wait", "The next run tries again.", reason, "GitHub accepts the merge.")
        return Decision(
            "stop", "GitHub refused the merge of this head.", reason, "Check the token's permissions, or merge it by hand."
        )
    if not result.get("merged"):
        return Decision("stop", "GitHub did not merge this head.", "GitHub answered without merging.", "Merge it by hand.")
    merged = Decision("merged", "Merged this head.", decision.reason, "Already merged.")
    # The PR is closed now and no later run comments on it, so record the merge before a dispatch can fail.
    merge.upsert_comment(api, number, comments, STATUS_MARKER, render(merged, pr["head"]["sha"]))
    return merged


def start_release(api: security.GitHub) -> None:
    """Start the release and CI on `master`, which a merge with GITHUB_TOKEN does not."""
    for workflow in POST_MERGE:
        security.dispatch_workflow(api, workflow, "master")


def reconcile(api: security.GitHub, root: Path, enabled: bool, now: dt.datetime, authors: frozenset[str]) -> list[str]:
    """Evaluate the open release PR, if any, and return the errors that should fail the job."""
    errors: list[str] = []
    owner = api.repo.split("/")[0]
    started = time.monotonic()
    clock = merge.RunClock(now, started, started + merge.CODECOV_WAIT_SECONDS)
    merged = False
    for row in api.pages(f"pulls?state=open&base=master&head={owner}:{RELEASE_BRANCH}"):
        number = row["number"]
        try:
            pr, comments, evidence, decision = evaluate_after_codecov(api, number, root, enabled, authors, clock)
            # A PR closed or merged during the wait gets no comment, because no later run would correct it.
            if pr.get("state", "open") != "open":
                continue
            if decision.action == "merge":
                decision = merge_release(api, pr, comments, decision, evidence.head_committed, clock.now)
                if decision.action == "merged":
                    # Set before starting, so that a failed dispatch does not make recovery start the release again.
                    merged = True
                    start_release(api)
                    continue
            merge.upsert_comment(api, number, comments, STATUS_MARKER, render(decision, pr["head"]["sha"]))
            if decision.action == "stop":
                notify_stop(api, pr, comments)
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"PR #{number}: {type(error).__name__}: {error}")
    # Right after a merge, the dispatched runs are not listed yet, so recovery would start every workflow again.
    if not merged:
        try:
            merge.ensure_post_merge(
                api,
                owned=lambda row: merge.author_login(row) in authors and row["head"]["ref"] == RELEASE_BRANCH,
                workflows=POST_MERGE,
            )
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"Post-merge workflows: {type(error).__name__}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    enabled = merge.merging_enabled(os.environ.get("RELEASE_MERGE_ENABLED"))
    authors, variable_error = release_authors(os.environ.get(APP_BOT_VARIABLE))
    # Reconcile first: PRs that GITHUB_TOKEN wrote still merge while the variable is wrong.
    errors = reconcile(security.GitHub(), args.root, enabled, dt.datetime.now(dt.UTC), authors)
    if variable_error:
        errors.append(variable_error)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
