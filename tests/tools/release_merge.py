"""Offline regressions for the release PR merge controller."""

import datetime as dt
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import dependabot_merge as merge
from tools import pages_build
from tools import release_merge as release
from tools import security_automation as security

ROOT = Path(__file__).resolve().parents[2]
REPO = "opthub-org/pytorch-bsf"
NOW = dt.datetime(2026, 11, 6, 12, tzinfo=dt.UTC)
SHA = "c" * 40
FIXTURE_ROOT = Path("/fixture")
COMPARE = f"compare/v0.24.0...{SHA}?per_page=100&page=1"
LISTING = f"pulls?state=open&base=master&head=opthub-org:{release.RELEASE_BRANCH}"
GREEN: tuple[list[str], list[str], dict[str, security.Json]] = ([], [], {w: {} for w in release.WORKFLOWS})


def release_pr(number=563, **overrides):
    pr = {
        "number": number,
        "state": "open",
        "title": "chore(master): release 0.25.0",
        "user": {"login": "github-actions[bot]"},
        "head": {"repo": {"full_name": REPO}, "ref": release.RELEASE_BRANCH, "sha": SHA},
        "base": {"ref": "master"},
        "labels": [{"name": "autorelease: pending"}],
        "draft": False,
        "created_at": "2026-10-29T12:29:44Z",
        "mergeable_state": "clean",
    }
    pr.update(overrides)
    return pr


def release_commit(sha=SHA, author="github-actions[bot]", committer="web-flow", verified=True, date="2026-11-06T11:00:00Z"):
    return {
        "sha": sha,
        "author": {"login": author},
        "committer": {"login": committer},
        "commit": {
            "message": "chore(master): release 0.25.0",
            "verification": {"verified": verified},
            "committer": {"date": date},
        },
    }


def release_files():
    return [
        {
            "filename": ".release-please-manifest.json",
            "status": "modified",
            "patch": '@@ -1,3 +1,3 @@\n {\n-  ".": "0.24.0"\n+  ".": "0.25.0"\n }',
        },
        {"filename": "CHANGELOG.md", "status": "modified", "patch": "@@ -1 +1,3 @@\n+## 0.25.0\n+* anything"},
        {
            "filename": "pyproject.toml",
            "status": "modified",
            "patch": '@@ -4,7 +4,7 @@\n [project]\n-version = "0.24.0"\n+version = "0.25.0"',
        },
    ]


def subjects(*lines):
    return {"total_commits": len(lines), "commits": [{"commit": {"message": line + "\n\nbody"}} for line in lines]}


def evidence(**overrides):
    values: dict[str, object] = {
        "problem": None,
        "draft": False,
        "head_committed": NOW - dt.timedelta(hours=1),
        "merge_state": "clean",
        "failed": [],
        "pending": [],
        "urgent": [],
        "enabled": True,
    }
    return release.Evidence(**{**values, **overrides})


def decide(now=NOW, **overrides):
    return release.decide(evidence(**overrides), now)


class DecisionTests(unittest.TestCase):
    def test_an_urgent_release_merges_at_once(self):
        decision = decide(urgent=["fix(deps): bump x (#1)"])
        self.assertEqual(decision.action, "merge")
        self.assertIn("fix(deps): bump x (#1)", decision.reason)

    def test_other_releases_wait_for_a_quiet_day(self):
        waiting = decide()
        self.assertEqual(waiting.action, "wait")
        self.assertIn(merge._minute(NOW - dt.timedelta(hours=1) + release.QUIET), waiting.mergeable_when)
        self.assertEqual(decide(head_committed=NOW - release.QUIET).action, "merge")
        self.assertEqual(decide(head_committed=NOW - release.QUIET + dt.timedelta(seconds=1)).action, "wait")

    def test_stops_and_holds_come_before_any_merge(self):
        cases: dict[str, list[dict[str, object]]] = {
            "stop": [{"problem": "no"}, {"failed": ["lint"]}],
            "hold": [{"draft": True}],
        }
        for action, variants in cases.items():
            for overrides in variants:
                with self.subTest(**overrides):
                    self.assertEqual(decide(urgent=["fix(deps): x"], **overrides).action, action)

    def test_unfinished_checks_wait_until_the_ci_deadline_and_then_stop(self):
        deadline = NOW - dt.timedelta(hours=1) + merge.CI_GRACE
        waiting = decide(pending=["lint"], urgent=["fix(deps): x"])
        self.assertEqual(waiting.action, "wait")
        self.assertIn(merge._minute(deadline), waiting.mergeable_when)
        self.assertEqual(decide(now=deadline, pending=["lint"]).action, "wait")
        late = decide(now=deadline + dt.timedelta(seconds=1), pending=["lint"])
        self.assertEqual(late.action, "stop")
        self.assertIn("still incomplete after 2 hours", late.reason)

    def test_an_unmergeable_state_or_a_pause_waits(self):
        for overrides, reason in (({"merge_state": "dirty"}, "`dirty`"), ({"enabled": False}, "RELEASE_MERGE_ENABLED")):
            with self.subTest(**overrides):
                decision = decide(urgent=["fix(deps): x"], head_committed=NOW - release.QUIET, **overrides)
                self.assertEqual(decision.action, "wait")
                self.assertIn(reason, decision.reason)


class ProblemTests(unittest.TestCase):
    def problem(self, pr=None, commits=None, files=None):
        commits = [release_commit()] if commits is None else commits
        return release.release_problem(pr or release_pr(), commits, release_files() if files is None else files, REPO)

    def test_release_pleases_own_pr_has_no_problem(self):
        self.assertIsNone(self.problem())

    def test_another_pr_is_outside_the_boundary(self):
        for overrides in (
            {"user": {"login": "dependabot[bot]"}},
            {"user": None},
            {"head": {"repo": {"full_name": "fork/pytorch-bsf"}, "ref": release.RELEASE_BRANCH, "sha": SHA}},
            {"head": {"repo": {"full_name": REPO}, "ref": "release-please--other", "sha": SHA}},
            {"base": {"ref": "develop"}},
        ):
            with self.subTest(overrides=overrides):
                self.assertIn("not release-please's release PR", self.problem(release_pr(**overrides)))

    def test_title_and_label_must_be_release_pleases(self):
        self.assertIn("title", self.problem(release_pr(title="chore(master): release 0.25.0 and more")))
        self.assertIn("autorelease: pending", self.problem(release_pr(labels=[{"name": "autorelease: tagged"}])))

    def test_every_commit_must_be_signed_through_release_please(self):
        self.assertIn("no commits", self.problem(commits=[]))
        for overrides in ({"author": "maintainer"}, {"committer": "maintainer"}, {"verified": False}):
            with self.subTest(**overrides):
                commits = [release_commit(sha="d" * 40), release_commit(**overrides)]
                self.assertIn(f"`{SHA[:10]}` was not written", self.problem(commits=commits))
        unknown = release_commit()
        unknown["author"] = None
        self.assertIn("was not written", self.problem(commits=[unknown]))

    def test_only_the_version_and_the_changelog_may_change(self):
        extra = [*release_files(), {"filename": "tools/release_merge.py", "status": "modified", "patch": "+x"}]
        self.assertIn("`tools/release_merge.py`", self.problem(files=extra))
        added = [{**row, "status": "added"} if row["filename"] == "CHANGELOG.md" else row for row in release_files()]
        self.assertIn("`CHANGELOG.md`", self.problem(files=added))
        for name, patch_text in (
            ("pyproject.toml", '-version = "0.24.0"\n+version = "0.25.0"\n+dependencies = ["evil"]'),
            ("pyproject.toml", ' version = "0.24.0"'),
            ("pyproject.toml", None),
            (".release-please-manifest.json", '-  ".": "0.24.0"\n+  "x": "0.25.0"'),
        ):
            with self.subTest(name=name, patch=patch_text):
                files = [{**row, "patch": patch_text} if row["filename"] == name else row for row in release_files()]
                self.assertIn(f"`{name}` beyond its version", self.problem(files=files))

    def test_the_previous_version_must_be_known(self):
        files = [row for row in release_files() if row["filename"] != release.MANIFEST]
        self.assertIn("previous version", self.problem(files=files))
        self.assertEqual(release.previous_version(release_files()), "0.24.0")


class UrgencyTests(unittest.TestCase):
    def test_only_dependency_and_security_fixes_are_urgent(self):
        api = Mock()
        lines = [
            "fix(deps): bump x from 1 to 2 (#1)",
            "fix(deps)!: require Python 3.11",
            "feat(security): add a check",
            "fix(security): remediate dependabot-99",
            "build(deps): bump actions/cache",
            "fix: unrelated",
            "docs: mention fix(deps): in prose",
            "fix(depsx): not a scope",
        ]
        api.get.return_value = subjects(*lines)
        self.assertEqual(release.urgent_subjects(api, "0.24.0", SHA), lines[:4])
        api.get.assert_called_once_with(COMPARE)

    def test_every_page_of_a_long_release_is_read(self):
        api = Mock()
        first = {"total_commits": 101, "commits": subjects(*["fix: x"] * 100)["commits"]}
        second = {"total_commits": 101, "commits": subjects("fix(deps): late")["commits"]}
        api.get.side_effect = [first, second]
        self.assertEqual(release.urgent_subjects(api, "0.24.0", SHA), ["fix(deps): late"])
        self.assertEqual(api.get.call_args_list[1].args[0], f"compare/v0.24.0...{SHA}?per_page=100&page=2")

    def test_an_empty_page_ends_the_listing(self):
        api = Mock()
        api.get.side_effect = [
            {"total_commits": 5, "commits": subjects("fix: x")["commits"]},
            {"total_commits": 5, "commits": []},
        ]
        self.assertEqual(release.urgent_subjects(api, "0.24.0", SHA), [])
        self.assertEqual(api.get.call_count, 2)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.repo = REPO
        self.pr = release_pr()
        self.comments: list[security.Json] = []
        self.master_pulls: list[security.Json] = []
        self.master_runs: list[security.Json] = []
        self.pages = {
            LISTING: [release_pr()],
            "issues/563/comments": self.comments,
            "pulls/563/commits": [release_commit()],
            "pulls/563/files": release_files(),
            f"actions/runs?head_sha={'e' * 40}": self.master_runs,
        }
        self.compare = subjects("feat: x", "fix(deps): bump y (#2)")
        self.api.pages.side_effect = lambda path, key=None: self.pages.get(path, [])
        self.api.get.side_effect = self.get
        self.api.write.side_effect = lambda path, data, method="POST": {"merged": True} if path.endswith("/merge") else {}
        ci_state = patch.object(security, "ci_state", return_value=GREEN)
        self.addCleanup(ci_state.stop)
        self.ci_state = ci_state.start()
        sleep = patch.object(merge.time, "sleep", side_effect=AssertionError("unexpected sleep"))
        self.addCleanup(sleep.stop)
        self.sleep = sleep.start()

    def get(self, path):
        if path == "commits/master":
            return {"sha": "e" * 40}
        if path == f"commits/{'e' * 40}/pulls":
            return self.master_pulls
        if path.startswith("compare/"):
            return self.compare
        return self.pr

    def run_once(self, enabled=True):
        return release.reconcile(self.api, FIXTURE_ROOT, enabled, NOW)

    def writes(self, path):
        return [call for call in self.api.write.call_args_list if call.args[0] == path]

    def dispatched(self, ref):
        return [
            call.args[0].split("/")[2]
            for call in self.api.write.call_args_list
            if call.args[0].endswith("/dispatches") and call.args[1]["ref"] == ref
        ]

    def status(self):
        bodies = [call.args[1]["body"] for call in self.writes("issues/563/comments")]
        return next(body for body in bodies if body.startswith(release.STATUS_MARKER))

    def test_an_urgent_release_merges_this_head_and_starts_the_release_first(self):
        self.assertEqual(self.run_once(), [])
        merges = self.writes("pulls/563/merge")
        self.assertEqual(len(merges), 1)
        self.assertEqual(
            merges[0].args[1:],
            ({"sha": SHA, "merge_method": "squash", "commit_title": "chore(master): release 0.25.0 (#563)"}, "PUT"),
        )
        self.assertEqual(self.dispatched("master"), release.POST_MERGE)
        self.assertEqual(self.dispatched("master")[0], "release-please-action.yml")
        self.assertIn("Merged this head.", self.status())
        self.assertIn("fix(deps): bump y (#2)", self.status())
        self.ci_state.assert_called_with(
            self.api, SHA, FIXTURE_ROOT, workflows=release.WORKFLOWS, excluded=release.DOCUMENTATION_CHECKS
        )

    def test_a_release_without_urgent_commits_waits_for_a_quiet_day(self):
        self.compare = subjects("feat: x", "build(deps): bump actions/cache (#3)")
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.writes("pulls/563/merge"), [])
        self.assertIn("No update until 2026-11-07 11:00 UTC", self.status())
        self.pages["pulls/563/commits"] = [release_commit(date="2026-11-05T12:00:00Z")]
        self.api.write.reset_mock()
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("pulls/563/merge")), 1)

    def test_missing_ci_starts_on_the_release_branch_once(self):
        self.ci_state.return_value = ([], ["lint"], {"python-package.yml": {}})
        self.assertEqual(self.run_once(), [])
        started = self.dispatched(release.RELEASE_BRANCH)
        self.assertEqual(started, [w for w in release.WORKFLOWS if w != "python-package.yml"])
        self.assertNotIn("sphinx-pages-build.yml", started)
        inputs = [
            call.args[1].get("inputs") for call in self.api.write.call_args_list if "python-package-conda" in call.args[0]
        ]
        self.assertEqual(inputs, [None])
        self.assertEqual(self.writes("pulls/563/merge"), [])
        self.assertIn("Waiting for CI.", self.status())

    def test_ci_still_unfinished_after_two_hours_stops_and_notifies_once_per_head(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.pages["pulls/563/commits"] = [release_commit(date="2026-11-06T09:59:00Z")]
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.dispatched(release.RELEASE_BRANCH), [])
        bodies = [call.args[1]["body"] for call in self.writes("issues/563/comments")]
        self.assertEqual(len(bodies), 2)
        notice = bodies[1]
        self.assertIn(f"@{merge.MAINTAINER}", notice)
        self.comments.extend(
            {"id": i, "body": body, "user": {"login": security.BOT}} for i, body in enumerate(bodies, start=1)
        )
        self.api.write.reset_mock()
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.api.write.call_args_list, [])

    def test_failed_checks_stop_the_merge(self):
        self.ci_state.return_value = (["lint"], [], {})
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.writes("pulls/563/merge"), [])
        self.assertIn("Checks failed: `lint`.", self.status())

    def test_a_pr_outside_the_boundary_is_never_checked_or_merged(self):
        self.pages["pulls/563/files"] = [*release_files(), {"filename": "setup.py", "status": "added", "patch": "+x"}]
        self.assertEqual(self.run_once(), [])
        self.ci_state.assert_not_called()
        self.assertEqual(self.writes("pulls/563/merge"), [])
        self.assertIn("`setup.py`", self.status())

    def test_a_head_missing_from_the_commit_list_stops(self):
        self.pages["pulls/563/commits"] = [release_commit(sha="d" * 40)]
        self.assertEqual(self.run_once(), [])
        self.ci_state.assert_not_called()
        self.assertIn("head commit is not listed", self.status())

    def test_a_draft_or_a_pause_merges_nothing_and_starts_no_ci(self):
        self.ci_state.return_value = ([], ["lint"], {})
        for pr, enabled in ((release_pr(draft=True), True), (release_pr(), False)):
            with self.subTest(draft=pr["draft"], enabled=enabled):
                self.pr = pr
                self.api.write.reset_mock()
                self.ci_state.return_value = GREEN
                self.assertEqual(self.run_once(enabled), [])
                self.assertEqual(self.writes("pulls/563/merge"), [])
                self.assertEqual(self.dispatched(release.RELEASE_BRANCH), [])

    def test_a_declined_merge_waits_and_other_errors_fail_the_job(self):
        for code in (403, 405, 409, 422):
            with self.subTest(code=code):
                self.api.write.reset_mock()
                self.api.write.side_effect = self.refuse(code)
                self.assertEqual(self.run_once(), [])
                self.assertIn(f"HTTP status {code}", self.status())
                self.assertEqual(self.dispatched("master"), [])
        self.api.write.side_effect = self.refuse(500)
        self.assertEqual(len(self.run_once()), 1)

    def refuse(self, code):
        def write(path, data, method="POST"):
            if path.endswith("/merge"):
                raise security.GitHubError("refused", code)
            return {}

        return write

    def test_a_merge_github_does_not_confirm_stops(self):
        self.api.write.side_effect = lambda path, data, method="POST": {"merged": False} if path.endswith("/merge") else {}
        self.assertEqual(self.run_once(), [])
        self.assertIn("GitHub did not merge this head.", self.status())
        self.assertEqual(self.dispatched("master"), [])

    def test_an_unchanged_status_edits_nothing_and_a_changed_one_is_edited(self):
        self.compare = subjects("feat: x")
        self.run_once()
        body = self.status()
        self.comments.append({"id": 9, "body": body, "user": {"login": security.BOT}})
        self.api.write.reset_mock()
        self.run_once()
        self.assertEqual(self.api.write.call_args_list, [])
        self.ci_state.return_value = (["lint"], [], {})
        self.run_once()
        self.assertEqual(len(self.writes("issues/comments/9")), 1)

    def test_a_run_waits_for_codecov_only_when_it_alone_stops_an_urgent_release(self):
        readings = iter(range(0, 10_000, merge.CODECOV_POLL_SECONDS))
        self.sleep.side_effect = None
        codecov: tuple[list[str], list[str], dict[str, security.Json]] = ([], [security.CODECOV_PATCH], {})
        self.ci_state.side_effect = [codecov, GREEN, GREEN]
        with patch.object(merge.time, "monotonic", side_effect=lambda: next(readings)):
            self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("pulls/563/merge")), 1)
        self.assertEqual(self.sleep.call_count, 1)
        self.ci_state.side_effect = None
        self.ci_state.return_value = codecov
        self.compare = subjects("feat: x")
        self.sleep.reset_mock()
        self.api.write.reset_mock()
        self.assertEqual(self.run_once(), [])
        self.sleep.assert_not_called()
        self.assertEqual(self.writes("pulls/563/merge"), [])

    def test_a_head_already_stopped_does_not_wait_for_codecov(self):
        self.pages["pulls/563/commits"] = [release_commit(date="2026-11-06T09:59:00Z")]  # Past the CI deadline.
        self.ci_state.return_value = ([], [security.CODECOV_PATCH], {})
        self.assertEqual(self.run_once(), [])
        self.sleep.assert_not_called()
        self.assertIn("still incomplete after 2 hours", self.status())

    def test_a_pr_closed_during_the_codecov_wait_gets_no_comment(self):
        readings = iter(range(0, 10_000, merge.CODECOV_POLL_SECONDS))
        self.sleep.side_effect = lambda seconds: self.pr.update(state="closed", mergeable_state=None)
        self.ci_state.side_effect = [([], [security.CODECOV_PATCH], {}), GREEN, GREEN]
        with patch.object(merge.time, "monotonic", side_effect=lambda: next(readings)):
            self.assertEqual(self.run_once(), [])
        self.assertEqual(self.api.write.call_args_list, [])

    def test_one_error_is_reported_and_post_merge_recovery_still_runs(self):
        self.api.pages.side_effect = lambda path, key=None: (
            (_ for _ in ()).throw(security.GitHubError("down", 502))
            if path == "issues/563/comments"
            else self.pages.get(path, [])
        )
        self.pr = {**self.pr, "merged_by": {"login": security.BOT}}
        self.master_pulls.append({"number": 563, "user": {"login": security.BOT}, "head": {"ref": release.RELEASE_BRANCH}})
        self.assertEqual(len(self.run_once()), 1)
        self.assertEqual(self.dispatched("master"), release.POST_MERGE)

    def test_a_post_merge_error_is_reported(self):
        self.pages[LISTING] = []
        self.api.get.side_effect = lambda path: (_ for _ in ()).throw(security.GitHubError("down", 502))
        self.assertEqual(self.run_once(), ["Post-merge workflows: GitHubError: down"])

    def test_post_merge_recovery_ignores_other_prs(self):
        self.pages[LISTING] = []
        self.pr = {**self.pr, "merged_by": {"login": security.BOT}}  # As the Dependabot controller merges.
        for user, ref in (("dependabot[bot]", "dependabot/pip/x"), ("maintainer", release.RELEASE_BRANCH)):
            with self.subTest(user=user):
                self.master_pulls[:] = [{"number": 7, "user": {"login": user}, "head": {"ref": ref}}]
                self.assertEqual(self.run_once(), [])
                self.assertEqual(self.api.write.call_args_list, [])
        self.master_pulls[0] = {"number": 563, "user": {"login": security.BOT}, "head": {"ref": release.RELEASE_BRANCH}}
        self.master_runs.append({"path": ".github/workflows/release-please-action.yml"})
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.dispatched("master"), [w for w in release.POST_MERGE if w != "release-please-action.yml"])


class MainTests(unittest.TestCase):
    def run_main(self, variable, errors):
        environment = {"GITHUB_REPOSITORY": REPO, "GH_TOKEN": "x", "RELEASE_MERGE_ENABLED": variable}
        with (
            patch.dict(os.environ, environment),
            patch("sys.argv", ["release_merge", "--root", "/fixture"]),
            patch.object(release, "reconcile", return_value=errors) as reconcile,
        ):
            release.main()
        return reconcile

    def test_main_passes_its_own_variable_to_the_controller(self):
        args = self.run_main("false", []).call_args.args
        self.assertEqual(args[1], FIXTURE_ROOT)
        self.assertIs(args[2], False)
        self.assertEqual(args[3].utcoffset(), dt.timedelta(0))
        with patch.dict(os.environ, {"DEPENDABOT_MERGE_ENABLED": "false"}):
            self.assertIs(self.run_main("", []).call_args.args[2], True)

    def test_main_fails_the_job_on_errors(self):
        with self.assertRaisesRegex(SystemExit, "PR #1: x\nPost-merge workflows: y"):
            self.run_main("", ["PR #1: x", "Post-merge workflows: y"])


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/dependabot-merge.yml").read_text()

    def test_release_branches_build_no_documentation_so_its_checks_are_not_required(self):
        with self.assertRaisesRegex(ValueError, "Excluded documentation branch"):
            pages_build.source_name(f"refs/heads/{release.RELEASE_BRANCH}")
        self.assertEqual(set(security.WORKFLOWS) - set(release.WORKFLOWS), {"sphinx-pages-build.yml"})
        producer = (ROOT / ".github/workflows/sphinx-pages-build.yml").read_text()
        self.assertIn("      - 'release-please**'", producer)
        self.assertRegex(producer, r"\njobs:\n(?:  #.*\n)*  build:\n")
        self.assertLessEqual(release.DOCUMENTATION_CHECKS, security.expected_checks(ROOT))

    def test_the_release_branch_and_previous_tag_match_the_release_configuration(self):
        config = json.loads((ROOT / "release-please-config.json").read_text())
        package = config["packages"]["."]
        self.assertEqual(release.RELEASE_BRANCH, f"release-please--branches--master--components--{package['package-name']}")
        self.assertTrue(package["include-v-in-tag"])
        self.assertFalse(package["include-component-in-tag"])
        manifest = json.loads((ROOT / release.MANIFEST).read_text())
        self.assertRegex(manifest["."], f"^{release.VERSION}$")

    def test_post_merge_starts_the_release_and_every_ci_workflow(self):
        self.assertEqual(release.POST_MERGE[0], "release-please-action.yml")
        self.assertEqual(sorted(release.POST_MERGE), sorted(security.POST_MERGE))
        source = (ROOT / ".github/workflows/release-please-action.yml").read_text()
        self.assertIn("  workflow_dispatch:", source)
        self.assertIn("if: github.ref == 'refs/heads/master'", source)

    def test_the_release_workflow_and_branch_start_the_controller(self):
        listed = re.search(r"workflows:\n((?:\s+- .*\n)+)", self.workflow)
        assert listed is not None
        name = re.search(r"^name: (.+)$", (ROOT / ".github/workflows/release-please-action.yml").read_text(), re.MULTILINE)
        assert name is not None
        self.assertIn(f"- {name[1].strip()}", listed[1])
        self.assertIn("github.event.workflow_run.name == 'Release'", self.workflow)
        self.assertIn("startsWith(github.event.workflow_run.head_branch, 'release-please--')", self.workflow)
        self.assertTrue(release.RELEASE_BRANCH.startswith("release-please--"))

    def test_the_release_step_runs_after_a_dependabot_failure_with_its_own_switch(self):
        self.assertRegex(
            self.workflow,
            r"      - name: Merge the release PR\n(?:        #.*\n)*        if: \$\{\{ !cancelled\(\) \}\}\n"
            r"        env:\n          GH_TOKEN: \$\{\{ github.token \}\}\n"
            r"          RELEASE_MERGE_ENABLED: \$\{\{ vars.RELEASE_MERGE_ENABLED \}\}\n"
            r"        run: python -m tools\.release_merge\n",
        )

    def test_only_distributed_dependencies_make_releasable_commits(self):
        config = (ROOT / ".github/dependabot.yml").read_text()
        prefixes = dict(
            re.findall(
                r'package-ecosystem: "([^"]+)".*?commit-message:\n\s+prefix: "([^"]+)"\n\s+include: "scope"', config, re.DOTALL
            )
        )
        self.assertEqual(prefixes, {"pip": "fix", "docker": "fix", "pre-commit": "build", "github-actions": "build"})
        for ecosystem, prefix in prefixes.items():
            with self.subTest(ecosystem=ecosystem):
                self.assertEqual(bool(release.URGENT.match(f"{prefix}(deps): bump x")), prefix == "fix")

    def test_repair_automation_cannot_edit_the_release_controller(self):
        for path in ("tools/release_merge.py", "tests/tools/release_merge.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)


class CiStateTests(unittest.TestCase):
    def test_excluded_checks_and_unlisted_workflows_are_not_expected(self):
        api = Mock()
        expected = security.expected_checks(ROOT)
        checks = [
            {"id": i, "name": name, "status": "completed", "conclusion": "success"}
            for i, name in enumerate(sorted(expected - release.DOCUMENTATION_CHECKS))
        ]
        runs = [
            {"id": i, "path": f".github/workflows/{w}", "status": "completed", "conclusion": "success"}
            for i, w in enumerate(release.WORKFLOWS)
        ]
        api.pages.side_effect = lambda path, key=None: {"check_runs": checks, "workflow_runs": runs}.get(key or "", [])
        self.assertEqual(security.ci_state(api, SHA, ROOT)[:2], ([], ["GHPages publish", "build", "sphinx-pages-build.yml"]))
        failed, pending, _ = security.ci_state(
            api, SHA, ROOT, workflows=release.WORKFLOWS, excluded=release.DOCUMENTATION_CHECKS
        )
        self.assertEqual((failed, pending), ([], []))


if __name__ == "__main__":
    unittest.main()
