"""Merge release-please's release PR once its CI passes, so that merged fixes reach a release without a human.

A release PR merges once it has gone a day without an update, or at once when the release contains a dependency or
security fix. Like the Dependabot controller, each run re-evaluates the PR from GitHub's current state, and the PR's
content is read as data only. See "Release automation" in SECURITY.md.
"""

import argparse
import datetime as dt
import os
import re
import time
from dataclasses import dataclass, replace
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
# The release workflow comes first, so that a failure to start the others cannot hold back the release.
RELEASE_WORKFLOW = "release-please-action.yml"
POST_MERGE = [RELEASE_WORKFLOW, *(workflow for workflow in security.POST_MERGE if workflow != RELEASE_WORKFLOW)]
QUIET = dt.timedelta(days=1)
# A commit subject that makes the release urgent: a dependency update or fix, or a security remediation. Dependabot
# writes `deps-dev` for dependencies it considers development ones.
URGENT = re.compile(r"(?:fix|feat)\((?:deps|deps-dev|security)\)!?:")
STATUS_MARKER = "<!-- release-merge:v1 -->"
STOP_MARKER = "<!-- release-merge-stop:v1"


@dataclass(frozen=True)
class Evidence:
    """What the decision needs to know about the release PR's head."""

    problem: str | None
    draft: bool
    head_committed: dt.datetime
    # When CI last started on this head: the head's commit, or the latest dispatch of its CI.
    ci_started: dt.datetime
    merge_state: str | None
    failed: list[str]
    pending: list[str]
    master_failed: list[str]
    urgent: list[str]
    enabled: bool


@dataclass(frozen=True)
class Decision:
    action: Literal["merge", "wait", "stop", "hold", "merged"]
    outcome: str
    reason: str
    mergeable_when: str


def release_problem(pr: Json, commits: list[Json], files: list[Json], repo: str) -> str | None:
    """Accept only release-please's own PR, which edits the version and the changelog and nothing else."""
    if (
        merge.author_login(pr) != security.BOT
        or pr["head"]["repo"]["full_name"] != repo
        or pr["head"]["ref"] != RELEASE_BRANCH
        or pr["base"]["ref"] != "master"
    ):
        return "This is not release-please's release PR for `master`."
    if RELEASE_TITLE.fullmatch(pr["title"]) is None:
        return "The title is not a release-please release title."
    if RELEASE_LABEL not in {label["name"] for label in pr.get("labels", [])}:
        return f"The PR lacks the `{RELEASE_LABEL}` label, without which release-please creates no release."
    return _commit_problem(commits) or _file_problem(files)


def _commit_problem(commits: list[Json]) -> str | None:
    if not commits:
        return "The PR has no commits to verify."
    for row in commits:
        author = (row.get("author") or {}).get("login")
        committer = (row.get("committer") or {}).get("login")
        if author != security.BOT or committer != merge.COMMITTER or row["commit"]["verification"]["verified"] is not True:
            return f"Commit `{row['sha'][: merge.SHORT_SHA_CHARS]}` was not written and signed through release-please."
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


def urgent_subjects(api: security.GitHub, version: str, sha: str) -> list[str]:
    """Return the subjects of the urgent commits between the previous release tag and the head."""
    subjects: list[str] = []
    total, page = None, 1
    while total is None or len(subjects) < total:
        comparison = api.get(f"compare/v{version}...{sha}?per_page=100&page={page}")
        rows = comparison["commits"]
        if not rows:
            break
        subjects.extend(row["commit"]["message"].split("\n", 1)[0] for row in rows)
        total, page = comparison["total_commits"], page + 1
    return [subject for subject in subjects if URGENT.match(subject)]


def decide(evidence: Evidence, now: dt.datetime) -> Decision:
    """Stop on anything wrong, wait for CI and GitHub within their limits, then merge when urgent or quiet."""
    if evidence.problem is not None:
        return Decision("stop", "Not merging this PR.", evidence.problem, "Merge or close it by hand.")
    if evidence.draft:
        return Decision("hold", "Held.", "The PR is a draft.", "Mark it ready for review.")
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
    ci_deadline = evidence.ci_started + merge.CI_GRACE
    if evidence.pending:
        if now > ci_deadline:
            return Decision(
                "stop",
                "Not merging this head.",
                f"CI is still incomplete {merge.format_hours(merge.CI_GRACE)} hours after it started: "
                f"{merge.format_names(evidence.pending)}.",
                "Run the missing workflows on the release branch, or re-run them.",
            )
        return Decision(
            "wait",
            "Waiting for CI.",
            f"Unfinished checks: {merge.format_names(evidence.pending)}.",
            f"All checks succeed by {merge.format_minute(ci_deadline)}.",
        )
    if evidence.merge_state not in merge.MERGEABLE_STATES:
        state_deadline = evidence.ci_started + merge.merge_state_limit(evidence.merge_state)
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
    quiet_until = evidence.head_committed + QUIET
    if now >= quiet_until:
        return Decision(
            "merge",
            "Merging now.",
            f"The PR has had no update for {merge.format_hours(QUIET)} hours.",
            "Already mergeable.",
        )
    return Decision(
        "wait",
        "Waiting for more changes.",
        "A release waits until its PR has had no update for a day, unless it contains a dependency or security fix.",
        f"No update until {merge.format_minute(quiet_until)}.",
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


def ci_to_start(runs: dict[str, Json]) -> list[str]:
    """Return the CI workflows with no usable run on the head.

    release-please pushes with GITHUB_TOKEN, so the head's own `pull_request` runs end as `action_required` without
    running anything, and a dispatched run replaces them as the latest run.
    """
    return [
        workflow for workflow in WORKFLOWS if workflow not in runs or runs[workflow].get("conclusion") == "action_required"
    ]


def _ci_started(head_committed: dt.datetime, runs: dict[str, Json]) -> dt.datetime:
    dispatched = [merge.parse_time(run["created_at"]) for run in runs.values() if run.get("event") == "workflow_dispatch"]
    return max([head_committed, *dispatched])


def evaluate(
    api: security.GitHub,
    number: int,
    root: Path,
    enabled: bool,
    *,
    settling: frozenset[str] = frozenset(),
) -> tuple[Json, list[Json], Evidence, list[str]]:
    """Read the PR and its checks, and return the evidence and the CI workflows that still have to start."""
    pr = merge.read_pull(api, number, settling=settling)
    comments = api.pages(f"issues/{number}/comments")
    commits = api.pages(f"pulls/{number}/commits")
    files = api.pages(f"pulls/{number}/files")
    sha = pr["head"]["sha"]
    head = [row for row in commits if row["sha"] == sha]
    problem = release_problem(pr, commits, files, api.repo) or (None if head else "The head commit is not listed.")
    committed = merge.parse_time(head[0]["commit"]["committer"]["date"]) if head else merge.parse_time(pr["created_at"])
    failed: list[str] = []
    pending: list[str] = []
    master_failed: list[str] = []
    runs: dict[str, Json] = {}
    urgent: list[str] = []
    if problem is None:
        failed, pending, runs = security.ci_state(api, sha, root, workflows=WORKFLOWS, excluded_checks=DOCUMENTATION_CHECKS)
        master_failed = security.ci_state(api, api.get("commits/master")["sha"], root)[0]
        version = previous_version(files)
        assert version is not None  # release_problem rejects a PR without it.
        try:
            urgent = urgent_subjects(api, version, sha)
        except security.GitHubError as error:
            if error.code != 404:
                raise
            problem = f"The previous release tag `v{version}` does not exist."
    evidence = Evidence(
        problem,
        pr["draft"],
        committed,
        _ci_started(committed, runs),
        pr.get("mergeable_state"),
        failed,
        pending,
        master_failed,
        urgent,
        enabled,
    )
    return pr, comments, evidence, ci_to_start(runs) if problem is None else []


def _codecov_alone_stops(evidence: Evidence, now: dt.datetime) -> bool:
    """Whether the PR would merge once codecov/patch succeeds, which GitHub reports as `unstable` until then."""
    if not merge.only_codecov_unfinished(evidence.failed, evidence.pending):
        return False
    settled = "clean" if evidence.merge_state == "unstable" else evidence.merge_state
    return decide(replace(evidence, pending=[], merge_state=settled), now).action == "merge"


def evaluate_and_start_ci(
    api: security.GitHub, number: int, root: Path, enabled: bool, clock: merge.RunClock
) -> tuple[Json, list[Json], Evidence, Decision]:
    """Evaluate the PR, start any CI it lacks, and when only codecov/patch keeps it from merging, wait for it.

    After the wait every input of the decision may have changed, so the PR is evaluated again from scratch.
    """
    pr, comments, evidence, to_start = evaluate(api, number, root, enabled)
    # A draft is held, so CI waits until it is ready; a pause stops only merges, so CI still runs.
    if to_start and not evidence.draft:
        for workflow in to_start:
            security.dispatch_workflow(api, workflow, RELEASE_BRANCH, number)
        evidence = replace(evidence, ci_started=clock.now)
    decision = decide(evidence, clock.now)
    # A PR already stopped, for example by the CI deadline, gains nothing from the wait.
    if (
        decision.action == "wait"
        and _codecov_alone_stops(evidence, clock.now)
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
        pr, comments, evidence, _ = evaluate(api, number, root, enabled, settling=frozenset({"unstable"}))
        decision = decide(evidence, clock.now)
    return pr, comments, evidence, decision


def merge_release(api: security.GitHub, pr: Json, comments: list[Json], decision: Decision) -> Decision:
    """Merge exactly the evaluated head, record it, then start the release, which a GITHUB_TOKEN merge does not."""
    number = pr["number"]
    data = {"sha": pr["head"]["sha"], "merge_method": "squash", "commit_title": f"{pr['title']} (#{number})"}
    try:
        result = api.write(f"pulls/{number}/merge", data, "PUT")
    except security.GitHubError as error:
        if error.code not in merge.MERGE_REFUSALS | merge.MERGE_RETRIES:
            raise
        return Decision(
            "wait",
            "The next run tries again.",
            f"GitHub declined the merge with HTTP status {error.code}.",
            "GitHub accepts the merge; check the token's permissions if this persists.",
        )
    if not result.get("merged"):
        return Decision("stop", "GitHub did not merge this head.", "GitHub answered without merging.", "Merge it by hand.")
    merged = Decision("merged", "Merged this head.", decision.reason, "Already merged.")
    # The PR is closed now and no later run comments on it, so record the merge before a dispatch can fail.
    merge.upsert_comment(api, number, comments, STATUS_MARKER, render(merged, pr["head"]["sha"]))
    for workflow in POST_MERGE:
        security.dispatch_workflow(api, workflow, "master")
    return merged


def is_release_pr(row: Json) -> bool:
    return merge.author_login(row) == security.BOT and row["head"]["ref"] == RELEASE_BRANCH


def reconcile(api: security.GitHub, root: Path, enabled: bool, now: dt.datetime) -> list[str]:
    """Evaluate the open release PR, if any, and return the errors that should fail the job."""
    errors: list[str] = []
    owner = api.repo.split("/")[0]
    started = time.monotonic()
    clock = merge.RunClock(now, started, started + merge.CODECOV_WAIT_SECONDS)
    merged = False
    for row in api.pages(f"pulls?state=open&base=master&head={owner}:{RELEASE_BRANCH}"):
        number = row["number"]
        try:
            pr, comments, evidence, decision = evaluate_and_start_ci(api, number, root, enabled, clock)
            # A PR closed or merged during the wait gets no comment, because no later run would correct it.
            if pr.get("state", "open") != "open":
                continue
            if decision.action == "merge":
                decision = merge_release(api, pr, comments, decision)
                if decision.action == "merged":
                    merged = True
                    continue
            merge.upsert_comment(api, number, comments, STATUS_MARKER, render(decision, pr["head"]["sha"]))
            if decision.action == "stop":
                notify_stop(api, pr, comments)
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"PR #{number}: {type(error).__name__}: {error}")
    # Right after a merge, the dispatched runs are not listed yet, so recovery would start every workflow again.
    if not merged:
        try:
            merge.ensure_post_merge(api, owned=is_release_pr, workflows=POST_MERGE)
        except merge.OPERATIONAL_ERRORS as error:
            errors.append(f"Post-merge workflows: {type(error).__name__}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    enabled = merge.merging_enabled(os.environ.get("RELEASE_MERGE_ENABLED"))
    errors = reconcile(security.GitHub(), args.root, enabled, dt.datetime.now(dt.UTC))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
