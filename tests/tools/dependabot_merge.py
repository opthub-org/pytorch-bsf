"""Offline regressions for the Dependabot merge and triage controller."""

import datetime as dt
import http.client
import os
import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import dependabot_merge as merge
from tools import security_automation as security

ROOT = Path(__file__).resolve().parents[2]
REPO = "opthub-org/pytorch-bsf"
NOW = dt.datetime(2026, 11, 5, 12, tzinfo=dt.UTC)
CREATED = dt.datetime(2026, 11, 5, 11, tzinfo=dt.UTC)
OLD_SHA, NEW_SHA = "a" * 40, "b" * 40
PIN_OLD = f"-      - uses: actions/cache@{OLD_SHA} # v4"
PIN_NEW = f"+      - uses: actions/cache@{NEW_SHA} # v6"


def make_pr(number=1, **overrides):
    pr = {
        "number": number,
        "user": {"login": "dependabot[bot]"},
        "head": {"repo": {"full_name": REPO}, "ref": "dependabot/pip/x", "sha": "c" * 40},
        "base": {"ref": "master"},
        "draft": False,
        "created_at": "2026-11-05T11:00:00Z",
        "mergeable_state": "clean",
        "assignees": [],
    }
    pr.update(overrides)
    return pr


def commit(author="dependabot[bot]", committer="web-flow", verified=True, date="2026-11-05T11:00:00Z"):
    return {
        "sha": "d" * 40,
        "author": {"login": author},
        "committer": {"login": committer},
        "commit": {"verification": {"verified": verified}, "committer": {"date": date}},
    }


def comment(body, login="maintainer", created="2026-11-05T10:00:00Z", ident=1):
    return {"id": ident, "body": body, "user": {"login": login}, "created_at": created}


def make_evidence(**overrides):
    values = {
        "created": CREATED,
        "head_committed": CREATED,
        "merge_state": "clean",
        "problem": None,
        "hold": None,
        "approved_by": [],
        "changes_requested_by": [],
        "failed": [],
        "pending": [],
        "enabled": True,
    }
    return merge.Evidence(**{**values, **overrides})


def decide(now=NOW, **evidence):
    return merge.decide(make_evidence(**evidence), now)


ALLOW = lambda login: True  # noqa: E731


class DecisionTests(unittest.TestCase):
    def test_only_complete_evidence_merges(self):
        for state in ("clean", "has_hooks"):
            with self.subTest(state=state):
                self.assertEqual(decide(merge_state=state).action, "merge")

    def test_pending_checks_wait_through_the_grace_period_and_then_stop(self):
        waiting = decide(pending=["lint"])
        self.assertEqual(waiting.action, "wait")
        self.assertIn("`lint`", waiting.reason)
        self.assertEqual(decide(pending=["lint"], now=CREATED + merge.CI_GRACE).action, "wait")
        late = decide(pending=["lint"], now=CREATED + merge.CI_GRACE + dt.timedelta(seconds=1))
        self.assertEqual(late.action, "stop")
        self.assertIn("after 2 hours", late.reason)

    def test_failure_stops_even_while_other_checks_run(self):
        decision = decide(failed=["lint"], pending=["docs"])
        self.assertEqual(decision.action, "stop")
        self.assertIn("`lint`", decision.reason)
        self.assertIn("2026-11-12", decision.deadline)

    def test_each_stop_says_what_would_make_it_mergeable(self):
        for evidence, phrase in (
            ({"failed": ["pages-producer"]}, "Updating GHPages action pins"),
            ({"failed": ["lint"]}, "/dependabot-hold"),
            ({"problem": "bad"}, "@dependabot recreate"),
            ({"changes_requested_by": ["alice"]}, "Dismiss the review"),
            ({"enabled": False}, "DEPENDABOT_MERGE_ENABLED"),
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(phrase, decide(**evidence).mergeable_when)
        self.assertNotIn("Updating GHPages", decide(failed=["lint"]).mergeable_when)

    def test_conflict_waits_for_dependabot_and_then_stops(self):
        waiting = decide(merge_state="dirty")
        self.assertEqual(waiting.action, "wait")
        self.assertIn("@dependabot rebase", waiting.mergeable_when)
        self.assertEqual(decide(merge_state="dirty", now=CREATED + merge.REBASE_GRACE).action, "wait")
        late = decide(merge_state="dirty", now=CREATED + merge.REBASE_GRACE + dt.timedelta(seconds=1))
        self.assertEqual(late.action, "stop")

    def test_computed_unmergeable_state_waits_and_then_escalates(self):
        for state in ("blocked", "unstable", "behind"):
            with self.subTest(state=state):
                self.assertEqual(decide(merge_state=state).action, "wait")
                late = decide(merge_state=state, now=CREATED + merge.CI_GRACE + dt.timedelta(seconds=1))
                self.assertEqual(late.action, "stop")

    def test_uncomputed_state_does_not_escalate_until_the_decision_window(self):
        for state in ("unknown", None):
            with self.subTest(state=state):
                self.assertEqual(decide(merge_state=state, now=CREATED + dt.timedelta(days=1)).action, "wait")
                late = decide(merge_state=state, now=CREATED + merge.DECISION_WINDOW + dt.timedelta(seconds=1))
                self.assertEqual(late.action, "stop")

    def test_provenance_or_change_problem_outranks_green_checks(self):
        self.assertEqual(decide(problem="bad").action, "stop")

    def test_change_request_stops_and_is_reported(self):
        decision = decide(changes_requested_by=["alice"], approved_by=["bob"])
        self.assertEqual(decision.action, "stop")
        self.assertIn("@alice", decision.review)

    def test_approval_is_reported_and_absence_of_review_is_stated(self):
        self.assertIn("Approved by @bob", decide(approved_by=["bob"]).review)
        self.assertIn("Not reviewed by a person", decide().review)

    def test_hold_outranks_everything_and_reports_who_and_until_when(self):
        hold = merge.Hold("maintainer", "breaks downstream", NOW + dt.timedelta(days=2))
        decision = decide(hold=hold, problem="bad", failed=["lint"], changes_requested_by=["alice"])
        self.assertEqual(decision.action, "hold")
        self.assertIn("@maintainer", decision.review)
        self.assertIn("breaks downstream", decision.reason)
        self.assertIn("2026-11-07", decision.deadline)

    def test_disabled_automation_keeps_triage_without_merging(self):
        decision = decide(enabled=False)
        self.assertEqual(decision.action, "stop")
        self.assertIn("DEPENDABOT_MERGE_ENABLED", decision.reason)

    def test_paused_triage_never_promises_a_merge(self):
        for evidence in ({"pending": ["lint"]}, {"merge_state": "blocked"}, {"merge_state": "dirty"}):
            with self.subTest(evidence=evidence):
                decision = decide(enabled=False, **evidence)
                self.assertEqual(decision.action, "wait")
                self.assertIn("paused", decision.outcome)

    def test_long_check_lists_are_shortened(self):
        names = [f"check-{index}" for index in range(merge.MAX_LISTED_CHECKS + 1)]
        self.assertIn("and 1 more", decide(failed=names).reason)
        self.assertNotIn("more", decide(failed=names[:-1]).reason)

    def test_triage_renders_every_requested_field(self):
        decision = merge.Decision("stop", "review", "outcome", "reason", "mergeable", "deadline")
        self.assertEqual(
            merge.render(decision, "c" * 40),
            f"{merge.TRIAGE_MARKER}\n**Dependabot triage** for `{'c' * 10}`\n\n"
            "- **Review:** review\n- **Decision:** outcome\n- **Reason:** reason\n"
            "- **Mergeable when:** mergeable\n- **Deadline:** deadline",
        )


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

    def test_foreign_unsigned_spoofed_or_missing_commits_are_rejected(self):
        for name, commits in {
            "foreign": [commit(), commit(author="maintainer")],
            "unsigned": [commit(verified=False)],
            "spoofed author signed by someone else": [commit(committer="maintainer")],
            "no linked author": [{**commit(), "author": None}],
            "no linked committer": [{**commit(), "committer": None}],
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
            for name in ("pyproject.toml", "uv.lock", "Dockerfile", ".pre-commit-config.yaml", "environment.yml")
        ]
        files.append({"filename": "docs/requirements.txt", "status": "modified"})
        files.append(self.workflow(f"@@ -1 +1 @@\n{PIN_OLD}\n{PIN_NEW}"))
        self.assertIsNone(merge.change_problem(files))

    def test_nested_workflow_path_is_rejected(self):
        row = {"filename": ".github/workflows/sub/x.yml", "status": "modified", "patch": f"{PIN_OLD}\n{PIN_NEW}"}
        self.assertIsNotNone(merge.change_problem([row]))

    def test_adjacent_pins_update_as_one_block_or_in_separate_hunks(self):
        other_old, other_new = PIN_OLD.replace("cache", "other"), PIN_NEW.replace("cache", "other")
        block = f"{PIN_OLD}\n{other_old}\n{PIN_NEW}\n{other_new}"
        apart = f"{PIN_OLD}\n{PIN_NEW}\n@@ -9 +9 @@\n{other_old}\n{other_new}"
        for name, text in {"block": block, "apart": apart}.items():
            with self.subTest(name=name):
                self.assertIsNone(merge.change_problem([self.workflow(text)]))

    def test_real_pin_update_formats_pass(self):
        bare = f"-      uses: actions/cache@{OLD_SHA} # v4.3.0\n+      uses: actions/cache@{NEW_SHA} # v6.1.0"
        no_comment = f"-  - uses: o/a/b@{OLD_SHA}\n+  - uses: o/a/b@{NEW_SHA} # reviewed 2026-10-30"
        for name, text in {"no dash": bare, "subpath": no_comment, "two pins": f"{bare}\n{bare}"}.items():
            with self.subTest(name=name):
                self.assertIsNone(merge.change_problem([self.workflow(text)]))

    def test_other_paths_are_rejected(self):
        for path in ("tools/x.py", "torch_bsf/__init__.py", "docs/Dockerfile", ".github/dependabot.yml", "a/pyproject.toml"):
            with self.subTest(path=path):
                self.assertIsNotNone(merge.change_problem([{"filename": path, "status": "modified", "patch": "+x"}]))

    def test_no_files_and_non_edits_are_rejected(self):
        self.assertIsNotNone(merge.change_problem([]))
        for status in ("added", "removed", "renamed", "copied"):
            with self.subTest(status=status):
                self.assertIsNotNone(merge.change_problem([{"filename": "uv.lock", "status": status}]))
                self.assertIsNotNone(merge.change_problem([self.workflow(f"{PIN_OLD}\n{PIN_NEW}", status=status)]))

    def test_workflow_changes_beyond_pins_are_rejected(self):
        for name, text in {
            "run step": f"{PIN_NEW}\n+      - run: curl evil",
            "tag instead of sha": "-      - uses: actions/cache@v4\n+      - uses: actions/cache@v6",
            "added step": PIN_NEW,
            "removed step": PIN_OLD,
            "swapped action": f"{PIN_OLD}\n+      - uses: evil/cache@{NEW_SHA} # v6",
            "changed indentation": f"{PIN_OLD}\n+    - uses: actions/cache@{NEW_SHA} # v6",
            "local action": f"-  - uses: ./.github/a@{OLD_SHA}\n+  - uses: ./.github/a@{NEW_SHA}",
            "short hex ref": f"{PIN_OLD}\n+      - uses: actions/cache@deadbeef",
            "39 hex digits": f"{PIN_OLD}\n+      - uses: actions/cache@{NEW_SHA[:39]}",
            "step moved between jobs": f"@@ -10,3 +10,2 @@\n   lint:\n{PIN_OLD}\n@@ -50,2 +49,3 @@\n   publish:\n{PIN_NEW}",
            "step moved past context": f"{PIN_OLD}\n       - run: x\n{PIN_NEW}",
            "no changes": "@@ -1 +1 @@",
        }.items():
            with self.subTest(name=name):
                self.assertIsNotNone(merge.change_problem([self.workflow(text)]))
        self.assertIsNotNone(merge.change_problem([{"filename": ".github/workflows/ci.yml", "status": "modified"}]))

    def test_control_characters_cannot_hide_a_second_line(self):
        for char in ("\r", "\x0b", "\x0c", "\x1c", "\x85", "\u2028", "\u2029"):
            for name, line in {
                "in comment": f"{PIN_NEW}{char}        run: curl evil",
                "without comment": f"+      - uses: actions/cache@{NEW_SHA}{char}        run: curl evil",
            }.items():
                with self.subTest(char=repr(char), name=name):
                    self.assertIsNotNone(merge.change_problem([self.workflow(f"{PIN_OLD}\n{line}")]))


class ReviewTests(unittest.TestCase):
    def test_latest_verdict_of_each_user_counts(self):
        reviews = [
            {"state": "CHANGES_REQUESTED", "user": {"login": "alice"}},
            {"state": "COMMENTED", "user": {"login": "alice"}},
            {"state": "APPROVED", "user": {"login": "bob"}},
            {"state": "CHANGES_REQUESTED", "user": {"login": "carol"}},
            {"state": "APPROVED", "user": {"login": "carol"}},
            {"state": "CHANGES_REQUESTED", "user": {"login": "dave"}},
            {"state": "DISMISSED", "user": {"login": "dave"}},
            {"state": "COMMENTED", "user": {"login": "bot"}},
        ]
        self.assertEqual(merge.review_state(reviews, ALLOW), (["bob", "carol"], ["alice"]))

    def test_reviews_without_write_access_do_not_count(self):
        reviews: list[security.Json] = [
            {"state": "CHANGES_REQUESTED", "user": {"login": "stranger"}},
            {"state": "APPROVED", "user": {"login": "passerby"}},
            {"state": "CHANGES_REQUESTED", "user": None},
        ]
        self.assertEqual(merge.review_state(reviews, lambda login: False), ([], []))


class HoldTests(unittest.TestCase):
    def hold(self, *comments, now=NOW, allowed=ALLOW):
        return merge.active_hold(list(comments), allowed, now)

    def test_authorized_hold_applies_through_the_end_of_its_day(self):
        hold = self.hold(comment("/dependabot-hold until=2026-11-07 breaks downstream"))
        self.assertEqual((hold.author, hold.reason), ("maintainer", "breaks downstream"))
        body = "/dependabot-hold until=2026-11-07 r"
        self.assertIsNotNone(self.hold(comment(body), now=dt.datetime(2026, 11, 7, 23, 59, tzinfo=dt.UTC)))
        self.assertIsNone(self.hold(comment(body), now=dt.datetime(2026, 11, 8, tzinfo=dt.UTC)))

    def test_unauthorized_users_cannot_hold_or_release(self):
        deny_stranger = lambda login: login != "stranger"  # noqa: E731
        self.assertIsNone(self.hold(comment("/dependabot-hold until=2026-11-07 r", login="stranger"), allowed=deny_stranger))
        held = self.hold(
            comment("/dependabot-hold until=2026-11-07 r"),
            comment("/dependabot-release", login="stranger"),
            allowed=deny_stranger,
        )
        self.assertIsNotNone(held)

    def test_permission_is_not_queried_for_ordinary_comments(self):
        allowed = Mock(return_value=True)
        self.assertIsNone(self.hold(comment("LGTM"), allowed=allowed))
        allowed.assert_not_called()

    def test_release_ends_an_earlier_hold_only(self):
        hold, release = comment("/dependabot-hold until=2026-11-07 r"), comment("/dependabot-release")
        self.assertIsNone(self.hold(hold, release))
        self.assertIsNotNone(self.hold(release, hold))

    def test_a_later_hold_replaces_an_earlier_one(self):
        first, second = comment("/dependabot-hold until=2026-11-07 a"), comment("/dependabot-hold until=2026-11-20 b")
        self.assertEqual(self.hold(first, second).reason, "b")
        self.assertEqual(self.hold(second, first).reason, "a")

    def test_invalid_hold_still_stops_the_merge_for_the_decision_window(self):
        for body in (
            "/dependabot-hold",
            "/dependabot-hold until=soon r",
            "/dependabot-hold until=2026-13-40 r",
            "/dependabot-hold until=2026-11-07",
            "/dependabot-hold until=2027-11-07 too far",
            "/dependabot-hold until=2025-11-07 past year",
            "/DEPENDABOT-HOLD until=soon r",
            "/dependabot-hold: until=soon r",
            "/dependabot-hold-until=2026-11-07 r",
        ):
            with self.subTest(body=body):
                hold = self.hold(comment(body))
                self.assertIn("invalid", hold.reason)
                self.assertEqual(hold.until, merge._end_of_day(dt.date(2026, 11, 12)))
        self.assertIsNone(self.hold(comment("/dependabot-hold"), now=dt.datetime(2026, 11, 13, tzinfo=dt.UTC)))

    def test_a_past_date_does_not_cancel_an_active_hold(self):
        hold = self.hold(comment("/dependabot-hold until=2026-11-20 real"), comment("/dependabot-hold until=2025-11-20 typo"))
        self.assertIn("invalid", hold.reason)

    def test_hold_limit_is_thirty_days_inclusive(self):
        self.assertEqual(self.hold(comment("/dependabot-hold until=2026-12-05 r")).reason, "r")
        self.assertIn("invalid", self.hold(comment("/dependabot-hold until=2026-12-06 r")).reason)

    def test_hold_may_end_on_the_day_of_its_comment(self):
        self.assertEqual(self.hold(comment("/dependabot-hold until=2026-11-05 r")).reason, "r")

    def test_comment_from_a_deleted_account_is_ignored(self):
        self.assertIsNone(self.hold({**comment("/dependabot-hold until=2026-11-07 r"), "user": None}))

    def test_release_spelling_is_case_insensitive(self):
        self.assertIsNone(self.hold(comment("/dependabot-hold until=2026-11-07 r"), comment("/DEPENDABOT-RELEASE")))

    def test_near_miss_spellings_parse_and_long_reasons_are_cut(self):
        for body in (
            "/dependabot hold until=2026-11-07 r",
            "/dependabot_hold until=2026-11-07 r",
            "`/dependabot-hold until=2026-11-07 r`",
        ):
            with self.subTest(body=body):
                self.assertIsNotNone(self.hold(comment(body)))
        self.assertEqual(self.hold(comment("\n/Dependabot-Hold: until=2026-11-07 r\nmore")).reason, "r")
        long = "x" * (merge.MAX_REASON_CHARS + 1)
        self.assertEqual(len(self.hold(comment(f"/dependabot-hold until=2026-11-07 {long}")).reason), merge.MAX_REASON_CHARS)

    def test_text_that_merely_mentions_a_command_is_ignored(self):
        for body in ("Use /dependabot-hold until=2026-11-07 r", "", None, "/dependabot-holdall until=2026-11-07 r"):
            with self.subTest(body=body):
                self.assertIsNone(self.hold(comment(body)))
        held = comment("/dependabot-hold until=2026-11-07 r")
        self.assertIsNotNone(self.hold(held, comment("/dependabot-release now")))


class AuthorizerTests(unittest.TestCase):
    def test_only_write_access_can_command_and_lookups_are_cached(self):
        api = Mock()
        permissions = {"collaborators/w/permission": "write", "collaborators/a/permission": "admin"}
        api.get.side_effect = lambda path: {"permission": permissions.get(path, "read")}
        can_command = merge.command_authorizer(api)
        self.assertTrue(can_command("w"))
        self.assertTrue(can_command("a"))
        self.assertFalse(can_command("r"))
        self.assertFalse(can_command("r"))
        self.assertEqual(api.get.call_count, 3)

    def test_login_is_quoted(self):
        api = Mock()
        api.get.return_value = {"permission": "read"}
        merge.command_authorizer(api)("dependabot[bot]/x")
        api.get.assert_called_once_with("collaborators/dependabot%5Bbot%5D%2Fx/permission")


FIXTURE_ROOT = Path("/fixture")


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.repo = REPO
        self.pr = make_pr(number=7)
        self.comments: list[security.Json] = []
        self.master_pulls: list[security.Json] = []
        self.permission = "admin"
        self.pages = {
            "pulls?state=open&base=master": [make_pr(number=7), make_pr(number=8, user={"login": "someone"})],
            "issues/7/comments": self.comments,
            "pulls/7/commits": [commit()],
            "pulls/7/files": [{"filename": "pyproject.toml", "status": "modified", "patch": "+x"}],
            "pulls/7/reviews": [],
        }
        self.api.pages.side_effect = lambda path, key=None: self.pages.get(path, [])
        self.api.get.side_effect = self.get
        self.api.write.return_value = {"merged": True}
        ci = patch.object(security, "ci_state", return_value=([], [], {}))
        self.ci_state = ci.start()
        self.addCleanup(ci.stop)

    def get(self, path):
        if path == "commits/master":
            return {"sha": "e" * 40}
        if path == f"commits/{'e' * 40}/pulls":
            return self.master_pulls
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        return self.pr

    def run_once(self, enabled=True):
        return merge.reconcile(self.api, FIXTURE_ROOT, enabled, NOW)

    def writes(self, path):
        return [call for call in self.api.write.call_args_list if call.args[0] == path]

    def triage(self):
        return self.writes("issues/7/comments")[0].args[1]["body"]

    def assert_blocked(self, expected):
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.writes("pulls/7/merge"), [])
        self.assertIn(expected, self.triage())

    def test_ready_pr_merges_exactly_the_evaluated_head_and_reports_it(self):
        self.assertEqual(self.run_once(), [])
        self.ci_state.assert_called_once_with(self.api, "c" * 40, FIXTURE_ROOT)
        merged = self.writes("pulls/7/merge")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].args[1], {"sha": "c" * 40, "merge_method": "squash"})
        self.assertIn("Merged this head", self.triage())
        self.assertEqual(self.writes("pulls/8/merge"), [])
        self.assertEqual(self.writes("issues/7/assignees"), [])

    def test_each_gate_is_wired_into_the_evaluation(self):
        pin_patch = [{"filename": ".github/workflows/x.yml", "status": "modified", "patch": PIN_NEW}]
        cases = {
            "foreign commit": ("pulls/7/commits", [commit(author="someone")], "not a Dependabot commit"),
            "unsigned commit": ("pulls/7/commits", [commit(verified=False)], "not a Dependabot commit"),
            "other file": ("pulls/7/files", [{"filename": "a.py", "status": "modified"}], "not a dependency file"),
            "non-pin workflow": ("pulls/7/files", pin_patch, "more than action pins"),
            "change request": (
                "pulls/7/reviews",
                [{"state": "CHANGES_REQUESTED", "user": {"login": "alice"}}],
                "Changes requested by @alice",
            ),
            "hold": ("issues/7/comments", [comment("/dependabot-hold until=2026-11-07 wait for 2.0")], "wait for 2.0"),
        }
        for name, (path, value, expected) in cases.items():
            with self.subTest(name=name):
                self.api.write.reset_mock()
                with patch.dict(self.pages, {path: value}):
                    self.assert_blocked(expected)

    def test_merge_state_is_wired_into_the_evaluation(self):
        self.pr["mergeable_state"] = "blocked"
        self.assert_blocked("`blocked`")

    def test_approval_is_reported_after_the_merge(self):
        self.pages["pulls/7/reviews"] = [{"state": "APPROVED", "user": {"login": "alice"}}]
        self.run_once()
        self.assertIn("Approved by @alice", self.triage())

    def test_stranger_change_request_neither_stops_nor_is_mentioned(self):
        self.permission = "read"
        self.pages["pulls/7/reviews"] = [{"state": "CHANGES_REQUESTED", "user": {"login": "stranger"}}]
        self.run_once()
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)
        self.assertNotIn("stranger", self.triage())
        self.assertEqual(self.writes("issues/7/assignees"), [])

    def test_decision_deadline_survives_a_rebase(self):
        self.pages["pulls/7/commits"] = [commit(date="2026-11-05T11:30:00Z")]
        self.pr["created_at"] = "2026-11-01T00:00:00Z"
        self.ci_state.return_value = (["lint"], [], {})
        self.run_once()
        self.assertIn("2026-11-08 UTC", self.triage())

    def test_read_only_user_cannot_hold(self):
        self.permission = "read"
        self.comments.append(comment("/dependabot-hold until=2026-11-07 r", login="reader"))
        self.run_once()
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)

    def test_failed_permission_lookup_does_not_merge(self):
        self.comments.append(comment("/dependabot-hold until=2026-11-07 r"))
        get = self.api.get.side_effect

        def failing(path):
            if path.startswith("collaborators/"):
                raise security.GitHubError("down", 502)
            return get(path)

        self.api.get.side_effect = failing
        errors = self.run_once()
        self.assertIn("#7", errors[0])
        self.assertEqual(self.writes("pulls/7/merge"), [])

    def test_ci_clock_starts_at_the_latest_commit_of_a_rebased_pr(self):
        self.pr["created_at"] = "2026-10-01T00:00:00Z"
        self.pages["pulls/7/commits"] = [commit(date="2026-10-01T00:00:00Z"), commit(date="2026-11-05T11:30:00Z")]
        self.ci_state.return_value = ([], ["lint"], {})
        self.run_once()
        self.assertIn("Waiting for `lint`", self.triage())
        self.assertIn("2026-11-05 13:30 UTC", self.triage())

    def test_pending_pr_gets_an_immediate_comment_and_no_merge(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.assert_blocked("Waiting for `lint`")

    def test_unchanged_triage_is_not_rewritten_and_changed_triage_is_edited(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.run_once()
        body = self.triage()
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
        self.comments.append(comment(merge.TRIAGE_MARKER + "\nfake", login="someone"))
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

    def test_waiting_and_held_prs_do_not_notify(self):
        self.ci_state.return_value = ([], ["lint"], {})
        self.run_once()
        self.assertEqual(self.writes("issues/7/assignees"), [])
        self.ci_state.return_value = ([], [], {})
        self.comments.append(comment("/dependabot-hold until=2026-11-07 r"))
        self.api.write.reset_mock()
        self.run_once()
        self.assertEqual(self.writes("issues/7/assignees"), [])
        self.assertEqual(self.writes("pulls/7/merge"), [])

    def test_triage_and_overdue_comments_are_told_apart(self):
        self.ci_state.return_value = (["lint"], [], {})
        self.pr["created_at"] = "2026-10-01T00:00:00Z"
        self.comments.extend(
            [
                comment("Some other bot comment", login=security.BOT, ident=1),
                comment(merge.OVERDUE_MARKER + "\nolder", login=security.BOT, ident=2),
            ]
        )
        self.run_once()
        self.assertEqual(self.writes("issues/comments/1"), [])
        self.assertEqual(self.writes("issues/comments/2"), [])
        created = self.writes("issues/7/comments")
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].args[1]["body"].startswith(merge.TRIAGE_MARKER))

    def test_decision_is_not_overdue_at_its_exact_deadline(self):
        self.ci_state.return_value = (["lint"], [], {})
        self.pr["created_at"] = "2026-10-29T12:00:00Z"
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("issues/7/comments")), 1)

    def fail_merge(self, code):
        def write(path, data, method="POST"):
            if path == "pulls/7/merge":
                raise security.GitHubError("refused", code)
            return {}

        self.api.write.reset_mock()
        self.api.write.side_effect = write

    def test_refused_merge_becomes_a_triage_decision_with_a_deadline(self):
        for code in (403, 422):
            with self.subTest(code=code):
                self.fail_merge(code)
                self.assertEqual(self.run_once(), [])
                self.assertIn(f"HTTP status {code}", self.triage())
                self.assertIn("2026-11-12 UTC. A maintainer decides", self.triage())
                self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_moved_head_or_base_retries_without_notifying_until_overdue(self):
        for code in (405, 409):
            with self.subTest(code=code):
                self.fail_merge(code)
                self.pr["created_at"] = "2026-11-05T11:00:00Z"
                self.assertEqual(self.run_once(), [])
                self.assertIn("tries again", self.triage())
                self.assertEqual(self.writes("issues/7/assignees"), [])
                self.fail_merge(code)
                self.pr["created_at"] = "2026-10-01T00:00:00Z"
                self.assertEqual(self.run_once(), [])
                self.assertIn(f"HTTP status {code}", self.triage())
                self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_unconfirmed_merge_is_not_reported_as_merged(self):
        self.api.write.return_value = {"merged": False}
        self.run_once()
        self.assertIn("without merging", self.triage())

    def test_transient_merge_error_is_reported_without_a_triage_verdict(self):
        def write(path, data, method="POST"):
            if path == "pulls/7/merge":
                raise security.GitHubError("down", 502)
            return {}

        self.api.write.side_effect = write
        errors = self.run_once()
        self.assertIn("#7", errors[0])
        self.assertEqual(self.writes("issues/7/comments"), [])

    def test_one_pr_error_does_not_block_another(self):
        self.pages["pulls?state=open&base=master"] = [make_pr(number=6), make_pr(number=7)]
        for error in (
            security.GitHubError("down", 502),
            TimeoutError("slow"),
            RuntimeError("pages"),
            http.client.IncompleteRead(b""),
            KeyError("x"),
            TypeError("x"),
            ValueError("x"),
            AttributeError("x"),
        ):

            def pages(path, key=None, error=error):
                if path == "issues/6/comments":
                    raise error
                return self.pages.get(path, [])

            with self.subTest(error=type(error).__name__):
                self.api.write.reset_mock()
                self.api.pages.side_effect = pages
                errors = self.run_once()
                self.assertEqual(len(errors), 1)
                self.assertIn("#6", errors[0])
                self.assertEqual(len(self.writes("pulls/7/merge")), 1)

    def test_disabled_controller_does_not_merge(self):
        self.run_once(enabled=False)
        self.assertEqual(self.writes("pulls/7/merge"), [])
        self.assertIn("paused", self.triage())

    def test_reconcile_starts_post_merge_workflows_and_reports_their_failure(self):
        self.master_pulls.append({"number": 3, "user": {"login": merge.AUTHOR}})
        self.pr["merged_by"] = {"login": security.BOT}
        self.pages["pulls?state=open&base=master"] = []
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.api.write.call_args_list), len(security.POST_MERGE))
        self.api.write.side_effect = security.GitHubError("down", 500)
        self.assertIn("Post-merge workflows", self.run_once()[0])


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
        api.pages.assert_called_once_with(f"actions/runs?head_sha={'e' * 40}", "workflow_runs")
        started = [call.args[0] for call in api.write.call_args_list]
        self.assertEqual(len(started), len(security.POST_MERGE) - 1)
        self.assertNotIn(f"actions/workflows/{present}/dispatches", started)
        self.assertEqual(api.write.call_args.args[1]["ref"], "master")

    def test_human_merge_or_other_commit_dispatches_nothing(self):
        for author, merged_by in ((merge.AUTHOR, "maintainer"), ("maintainer", security.BOT)):
            with self.subTest(author=author, merged_by=merged_by):
                api = self.api_for(author, merged_by)
                merge.ensure_post_merge(api)
                api.write.assert_not_called()


class MainTests(unittest.TestCase):
    def test_only_unset_empty_or_true_enables_merging(self):
        for value in (None, "", " ", "true", "TRUE", " True "):
            with self.subTest(value=value):
                self.assertTrue(merge.merging_enabled(value))
        for value in ("false", "False", "FALSE", " false ", "0", "no", "off", "disabled"):
            with self.subTest(value=value):
                self.assertFalse(merge.merging_enabled(value))

    def run_main(self, variable, errors):
        environment = {"GITHUB_REPOSITORY": REPO, "GH_TOKEN": "x", "DEPENDABOT_MERGE_ENABLED": variable}
        with (
            patch.dict(os.environ, environment),
            patch("sys.argv", ["dependabot_merge", "--root", "/fixture"]),
            patch.object(merge, "reconcile", return_value=errors) as reconcile,
        ):
            merge.main()
        return reconcile

    def test_main_passes_the_variable_to_the_controller(self):
        args = self.run_main("FALSE", []).call_args.args
        self.assertEqual(args[1], FIXTURE_ROOT)
        self.assertIs(args[2], False)
        self.assertEqual(args[3].utcoffset(), dt.timedelta(0))
        self.assertIs(self.run_main("", []).call_args.args[2], True)

    def test_main_fails_the_job_on_errors(self):
        with self.assertRaisesRegex(SystemExit, "PR #1: x\nPR #2: y"):
            self.run_main("", ["PR #1: x", "PR #2: y"])


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/dependabot-merge.yml").read_text()

    def test_repair_automation_cannot_edit_this_workflow_or_controller(self):
        for path in (".github/workflows/dependabot-merge.yml", "tools/dependabot_merge.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_every_trigger_needed_for_immediate_and_recovered_runs_is_present(self):
        self.assertIn("types: [requested, completed]", self.workflow)
        self.assertRegex(self.workflow, r"schedule:\n(?:\s+#.*\n)*\s+- cron: '\*/15 \* \* \* \*'")

    def test_every_required_workflow_triggers_the_controller(self):
        listed = re.search(r"workflows:\n((?:\s+- .*\n)+)", self.workflow)
        assert listed is not None
        names = {line.strip().removeprefix("- ") for line in listed[1].splitlines()}
        for workflow in security.WORKFLOWS:
            with self.subTest(workflow=workflow):
                source = (ROOT / ".github/workflows" / workflow).read_text()
                match = re.search(r"^name: (.+)$", source, re.MULTILINE)
                assert match is not None
                self.assertIn(match[1].strip(), names)
        self.assertIn("GHPages publisher", names)

    def test_only_master_code_runs_with_write_permissions(self):
        self.assertIn("permissions: {}", self.workflow)
        self.assertIn("github.ref == 'refs/heads/master'", self.workflow)
        self.assertIn("ref: ${{ github.sha }}", self.workflow)
        self.assertIn("persist-credentials: false", self.workflow)
        self.assertRegex(self.workflow, r"run: python -m tools\.dependabot_merge\n")
        self.assertIn("DEPENDABOT_MERGE_ENABLED: ${{ vars.DEPENDABOT_MERGE_ENABLED }}", self.workflow)

    def test_job_settings_that_only_github_can_exercise_stay_in_place(self):
        for line in (
            "group: dependabot-merge",
            "cancel-in-progress: false",
            "workflow_dispatch:",
            "github.repository == 'opthub-org/pytorch-bsf'",
            "github.event.workflow_run.name == 'GHPages publisher'",
            "startsWith(github.event.workflow_run.head_branch, 'dependabot/')",
            "actions: write",
            "checks: read",
            "contents: write",
            "issues: write",
            "pull-requests: write",
            "statuses: read",
        ):
            with self.subTest(line=line):
                self.assertIn(line, self.workflow)

    def test_every_ecosystem_has_a_three_day_cooldown(self):
        config = (ROOT / ".github/dependabot.yml").read_text()
        self.assertEqual(config.count("package-ecosystem:"), config.count("cooldown:\n      default-days: 3\n"))


if __name__ == "__main__":
    unittest.main()
