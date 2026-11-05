"""Offline regressions for the Dependabot merge and triage controller."""

import datetime as dt
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import dependabot_merge as merge
from tools import security_automation as security

REPO = "opthub-org/pytorch-bsf"
NOW = dt.datetime(2026, 11, 5, 12, tzinfo=dt.UTC)
PIN_OLD = "-      - uses: actions/cache@" + "a" * 40 + " # v4"
PIN_NEW = "+      - uses: actions/cache@" + "b" * 40 + " # v6"


def make_pr(number=1, **overrides):
    pr = {
        "number": number,
        "user": {"login": merge.AUTHOR},
        "head": {
            "repo": {"full_name": REPO},
            "ref": "dependabot/pip/x",
            "sha": "c" * 40,
            "pushed_at": "2026-11-05T11:00:00Z",
        },
        "base": {"ref": "master"},
        "draft": False,
        "created_at": "2026-11-05T11:00:00Z",
        "mergeable_state": "clean",
        "assignees": [],
    }
    pr.update(overrides)
    return pr


def commit(author=merge.AUTHOR, verified=True):
    return {
        "sha": "d" * 40,
        "author": {"login": author},
        "commit": {"verification": {"verified": verified}, "committer": {"date": "2026-11-05T11:00:00Z"}},
    }


def comment(body, association="OWNER", login="maintainer", created="2026-11-05T10:00:00Z", ident=1):
    return {"id": ident, "body": body, "author_association": association, "user": {"login": login}, "created_at": created}


def decide(pr=None, problem=None, hold=None, failed=(), pending=(), enabled=True, now=NOW):
    return merge.decide(pr or make_pr(), problem, hold, list(failed), list(pending), enabled, now)


class DecisionTests(unittest.TestCase):
    def test_only_complete_evidence_merges(self):
        self.assertEqual(decide().action, "merge")

    def test_pending_checks_wait_and_then_stop(self):
        waiting = decide(pending=["lint"])
        self.assertEqual(waiting.action, "wait")
        self.assertIn("`lint`", waiting.reason)
        late = decide(pending=["lint"], now=NOW + dt.timedelta(hours=2))
        self.assertEqual(late.action, "stop")
        self.assertIn("incomplete", late.reason)

    def test_failure_stops_even_while_other_checks_run(self):
        decision = decide(failed=["lint"], pending=["docs"])
        self.assertEqual(decision.action, "stop")
        self.assertIn("`lint`", decision.reason)
        self.assertIn("2026-11-12", decision.deadline)

    def test_failed_pages_producer_points_to_the_template_procedure(self):
        self.assertIn("Updating GHPages action pins", decide(failed=["pages-producer"]).unblock)

    def test_conflict_and_unknown_merge_state_do_not_merge(self):
        self.assertEqual(decide(make_pr(mergeable_state="dirty")).action, "stop")
        for state in ("unknown", "blocked", "unstable", "behind", None):
            with self.subTest(state=state):
                self.assertEqual(decide(make_pr(mergeable_state=state)).action, "wait")

    def test_provenance_or_change_problem_outranks_green_checks(self):
        self.assertEqual(decide(problem="bad").action, "stop")

    def test_hold_outranks_everything_and_reports_who_and_until_when(self):
        hold = merge.Hold("maintainer", "breaks downstream", NOW + dt.timedelta(days=2))
        decision = decide(hold=hold, problem="bad", failed=["lint"])
        self.assertEqual(decision.action, "hold")
        self.assertIn("@maintainer", decision.review)
        self.assertIn("breaks downstream", decision.reason)
        self.assertIn("2026-11-07", decision.deadline)

    def test_disabled_automation_keeps_triage_without_merging(self):
        decision = decide(enabled=False)
        self.assertEqual(decision.action, "stop")
        self.assertIn("DEPENDABOT_MERGE_ENABLED", decision.reason)
        self.assertEqual(decide(enabled=False, pending=["lint"]).action, "wait")

    def test_every_triage_states_the_requested_fields(self):
        for decision in (decide(), decide(pending=["a"]), decide(failed=["a"]), decide(problem="b"), decide(enabled=False)):
            body = merge.render(decision, "c" * 40)
            for field in ("Review", "Decision", "Reason", "Mergeable when", "Deadline"):
                self.assertIn(f"**{field}:**", body)


class ProvenanceTests(unittest.TestCase):
    def problem(self, pr=None, commits=None):
        return merge.provenance_problem(pr or make_pr(), [commit()] if commits is None else commits, REPO)

    def test_signed_dependabot_pr_passes(self):
        self.assertIsNone(self.problem())

    def test_untrusted_identities_are_rejected(self):
        bad = {
            "author": make_pr(user={"login": "someone"}),
            "fork": make_pr(head={"repo": {"full_name": "x/y"}, "ref": "dependabot/a", "sha": "c" * 40}),
            "branch": make_pr(head={"repo": {"full_name": REPO}, "ref": "feature", "sha": "c" * 40}),
            "base": make_pr(base={"ref": "release"}),
            "draft": make_pr(draft=True),
        }
        for name, pr in bad.items():
            with self.subTest(name=name):
                self.assertIsNotNone(self.problem(pr))

    def test_foreign_unsigned_or_missing_commits_are_rejected(self):
        for name, commits in {
            "foreign": [commit(), commit(author="maintainer")],
            "unsigned": [commit(verified=False)],
            "web-flow only": [{**commit(), "author": None}],
            "none": [],
        }.items():
            with self.subTest(name=name):
                self.assertIsNotNone(self.problem(commits=commits))


class ChangeTests(unittest.TestCase):
    def workflow(self, patch_text, status="modified"):
        return {"filename": ".github/workflows/ci.yml", "status": status, "patch": patch_text}

    def test_dependency_files_and_pin_updates_pass(self):
        files = [
            {"filename": name, "status": "modified", "patch": "+anything"}
            for name in (
                "pyproject.toml",
                "uv.lock",
                "Dockerfile",
                ".pre-commit-config.yaml",
                "environment.yml",
                "docs/requirements.txt",
            )
        ]
        files.append(self.workflow(f"@@ -1 +1 @@\n{PIN_OLD}\n{PIN_NEW}"))
        self.assertIsNone(merge.change_problem(files))

    def test_other_paths_are_rejected(self):
        for path in ("tools/x.py", "torch_bsf/__init__.py", "docs/Dockerfile", ".github/dependabot.yml", "a/pyproject.toml"):
            with self.subTest(path=path):
                self.assertIsNotNone(merge.change_problem([{"filename": path, "status": "modified", "patch": "+x"}]))

    def test_workflow_changes_beyond_pins_are_rejected(self):
        for name, row in {
            "run step": self.workflow(f"{PIN_NEW}\n+      - run: curl evil"),
            "tag instead of sha": self.workflow("+      - uses: actions/cache@v6"),
            "control character": self.workflow(PIN_NEW + "\x0c+ run: curl evil\n" + PIN_OLD[:1] + "run: x"),
            "new file": self.workflow(PIN_NEW, status="added"),
            "no patch": {"filename": ".github/workflows/ci.yml", "status": "modified"},
            "empty patch": self.workflow("@@ -1 +1 @@"),
        }.items():
            with self.subTest(name=name):
                self.assertIsNotNone(merge.change_problem([row]))


class HoldTests(unittest.TestCase):
    def hold(self, *comments, now=NOW):
        return merge.active_hold(list(comments), now)

    def test_trusted_hold_applies_until_the_end_of_its_day(self):
        hold = self.hold(comment("/dependabot-hold until=2026-11-07 breaks downstream"))
        self.assertEqual((hold.author, hold.reason), ("maintainer", "breaks downstream"))
        at_end = dt.datetime(2026, 11, 7, 23, 59, tzinfo=dt.UTC)
        self.assertIsNotNone(self.hold(comment("/dependabot-hold until=2026-11-07 r"), now=at_end))
        after = dt.datetime(2026, 11, 8, tzinfo=dt.UTC)
        self.assertIsNone(self.hold(comment("/dependabot-hold until=2026-11-07 r"), now=after))

    def test_untrusted_authors_cannot_hold_or_release(self):
        self.assertIsNone(self.hold(comment("/dependabot-hold until=2026-11-07 r", association="NONE")))
        self.assertIsNotNone(
            self.hold(
                comment("/dependabot-hold until=2026-11-07 r"),
                comment("/dependabot-release", association="CONTRIBUTOR"),
            )
        )

    def test_release_ends_an_earlier_hold_only(self):
        hold, release = comment("/dependabot-hold until=2026-11-07 r"), comment("/dependabot-release")
        self.assertIsNone(self.hold(hold, release))
        self.assertIsNotNone(self.hold(release, hold))

    def test_invalid_hold_still_stops_the_merge(self):
        for body in (
            "/dependabot-hold",
            "/dependabot-hold until=soon r",
            "/dependabot-hold until=2026-13-40 r",
            "/dependabot-hold until=2026-11-07",
            "/dependabot-hold until=2027-11-07 too far",
        ):
            with self.subTest(body=body):
                self.assertIn("invalid", self.hold(comment(body)).reason)

    def test_text_that_merely_mentions_the_command_is_ignored(self):
        self.assertIsNone(self.hold(comment("Use /dependabot-hold until=2026-11-07 r")))
        self.assertIsNone(self.hold(comment("")))
        self.assertIsNone(self.hold(comment("/dependabot-holdall until=2026-11-07 r")))


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.repo = REPO
        self.pr = make_pr(number=7)
        self.comments: list[security.Json] = []
        self.pages = {
            "pulls?state=open&base=master": [make_pr(number=7), make_pr(number=8, user={"login": "someone"})],
            "issues/7/comments": self.comments,
            "pulls/7/commits": [commit()],
            "pulls/7/files": [{"filename": "pyproject.toml", "status": "modified", "patch": "+x"}],
        }
        self.api.pages.side_effect = lambda path, key=None: self.pages.get(path, [])
        self.api.get.side_effect = self.get
        self.api.write.return_value = {"merged": True}
        self.ci = patch.object(security, "ci_state", return_value=([], [], {}))
        self.ci_state = self.ci.start()
        self.addCleanup(self.ci.stop)

    def get(self, path):
        if path == "commits/master":
            return {"sha": "e" * 40}
        if path == f"commits/{'e' * 40}/pulls":
            return []
        return self.pr

    def run_once(self, enabled=True):
        return merge.reconcile(self.api, Path("."), enabled, NOW)

    def writes(self, path):
        return [call for call in self.api.write.call_args_list if call.args[0] == path]

    def test_ready_pr_merges_exactly_the_evaluated_head_and_reports_it(self):
        self.assertEqual(self.run_once(), [])
        merged = self.writes("pulls/7/merge")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].args[1], {"sha": "c" * 40, "merge_method": "squash"})
        self.assertIn("Merged this head", self.writes("issues/7/comments")[0].args[1]["body"])
        self.assertEqual(self.writes("pulls/8/merge"), [])

    def test_pending_pr_gets_an_immediate_comment_and_no_merge(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.run_once()
        self.assertEqual(self.writes("pulls/7/merge"), [])
        self.assertIn("Waiting for `lint`", self.writes("issues/7/comments")[0].args[1]["body"])

    def test_unchanged_triage_is_not_rewritten_and_changed_triage_is_edited(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.run_once()
        body = self.writes("issues/7/comments")[0].args[1]["body"]
        self.api.write.reset_mock()
        self.comments.append(comment(body, login=security.BOT, ident=99))
        self.run_once()
        self.api.write.assert_not_called()
        self.ci_state.return_value = ([], ["docs"], {})
        self.run_once()
        self.assertEqual(len(self.writes("issues/comments/99")), 1)
        self.assertEqual(self.writes("issues/7/comments"), [])

    def test_a_spoofed_triage_comment_is_not_edited(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.comments.append(comment(merge.TRIAGE_MARKER + "\nfake", association="NONE", login="someone"))
        self.run_once()
        self.assertEqual(len(self.writes("issues/7/comments")), 1)

    def test_stop_assigns_the_maintainer_and_overdue_notifies_once(self):
        self.ci_state.return_value = (["lint"], [], {})
        self.run_once()
        self.assertEqual(self.writes("issues/7/assignees")[0].args[1], {"assignees": [merge.MAINTAINER]})
        self.assertEqual(len(self.writes("issues/7/comments")), 1)
        self.pr["created_at"] = "2026-10-01T00:00:00Z"
        self.pr["assignees"] = [{"login": merge.MAINTAINER}]
        self.api.write.reset_mock()
        self.run_once()
        overdue = [c for c in self.writes("issues/7/comments") if merge.OVERDUE_MARKER in c.args[1]["body"]]
        self.assertEqual(len(overdue), 1)
        self.assertEqual(self.writes("issues/7/assignees"), [])
        self.comments.append(comment(overdue[0].args[1]["body"], login=security.BOT, ident=5))
        self.api.write.reset_mock()
        self.run_once()
        self.assertEqual([c for c in self.writes("issues/7/comments") if merge.OVERDUE_MARKER in c.args[1]["body"]], [])

    def test_refused_merge_becomes_a_triage_decision(self):
        def write(path, data, method="POST"):
            if path == "pulls/7/merge":
                raise security.GitHubError("refused", 403)
            return {}

        self.api.write.side_effect = write
        self.assertEqual(self.run_once(), [])
        self.assertIn("HTTP status 403", self.writes("issues/7/comments")[0].args[1]["body"])
        self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_one_pr_error_does_not_block_another(self):
        self.pages["pulls?state=open&base=master"] = [make_pr(number=6), make_pr(number=7)]

        def pages(path, key=None):
            if path == "issues/6/comments":
                raise security.GitHubError("down", 502)
            return self.pages.get(path, [])

        self.api.pages.side_effect = pages
        errors = self.run_once()
        self.assertEqual(len(errors), 1)
        self.assertIn("#6", errors[0])
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)

    def test_disabled_controller_does_not_merge(self):
        self.run_once(enabled=False)
        self.assertEqual(self.writes("pulls/7/merge"), [])
        self.assertIn("paused", self.writes("issues/7/comments")[0].args[1]["body"])


class PostMergeTests(unittest.TestCase):
    def api_for(self, author, merged_by, runs=()):
        api = Mock()
        sha = "e" * 40
        responses = {
            "commits/master": {"sha": sha},
            f"commits/{sha}/pulls": [{"number": 3, "user": {"login": author}}],
            "pulls/3": {"merged_by": {"login": merged_by}},
        }
        api.get.side_effect = responses.__getitem__
        api.pages.return_value = [{"path": f".github/workflows/{name}"} for name in runs]
        return api

    def test_token_merge_dispatches_only_missing_workflows(self):
        present = security.POST_MERGE[0]
        api = self.api_for(merge.AUTHOR, security.BOT, [present])
        merge.ensure_post_merge(api)
        started = [call.args[0] for call in api.write.call_args_list]
        self.assertEqual(len(started), len(security.POST_MERGE) - 1)
        self.assertNotIn(f"actions/workflows/{present}/dispatches", started)
        self.assertEqual(api.write.call_args.args[1], {"ref": "master"})

    def test_human_merge_or_other_commit_dispatches_nothing(self):
        for author, merged_by in ((merge.AUTHOR, "maintainer"), ("maintainer", security.BOT)):
            with self.subTest(author=author, merged_by=merged_by):
                api = self.api_for(author, merged_by)
                merge.ensure_post_merge(api)
                api.write.assert_not_called()


class ControlPlaneTests(unittest.TestCase):
    def test_repair_automation_cannot_edit_this_workflow_or_controller(self):
        for path in (".github/workflows/dependabot-merge.yml", "tools/dependabot_merge.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)


if __name__ == "__main__":
    unittest.main()
