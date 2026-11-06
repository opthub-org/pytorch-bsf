"""Offline regressions for the Dependabot merge and triage controller."""

import datetime as dt
import http.client
import json
import math
import os
import re
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

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


METADATA = """build(deps): bump x from 1 to 2

Release notes.

---
updated-dependencies:
- dependency-name: x
  dependency-version: 2.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] <support@github.com>
"""


def commit(author="dependabot[bot]", committer="web-flow", verified=True, date="2026-11-05T11:00:00Z", message=METADATA):
    return {
        "sha": "d" * 40,
        "author": {"login": author},
        "committer": {"login": committer},
        "commit": {"message": message, "verification": {"verified": verified}, "committer": {"date": date}},
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
        "version_advisories": [],
        "package_advisories": [],
        "lookup_error": None,
    }
    return merge.Evidence(**{**values, **overrides})


def decide(now=NOW, **evidence):
    return merge.decide(make_evidence(**evidence), now)


def allow(login):
    return True


class DecisionTests(unittest.TestCase):
    def test_failed_lookup_waits_with_a_deadline_and_then_stops(self):
        waiting = decide(lookup_error="ValueError: x")
        self.assertEqual(waiting.action, "wait")
        self.assertIn("advisory lookup failed", waiting.reason)
        self.assertIn(merge.format_minute(CREATED + merge.CI_GRACE), waiting.deadline)
        self.assertEqual(decide(lookup_error="ValueError: x", now=CREATED + merge.CI_GRACE).action, "wait")
        late = decide(lookup_error="ValueError: x", now=CREATED + merge.CI_GRACE + dt.timedelta(seconds=1))
        self.assertEqual(late.action, "stop")
        self.assertIn("more than 2 hours old", late.reason)
        self.assertIn("failed workflow run", late.mergeable_when)
        rebased = CREATED + dt.timedelta(hours=1)
        rebased_decision = decide(lookup_error="ValueError: x", head_committed=rebased)
        self.assertIn(merge.format_minute(rebased + merge.CI_GRACE), rebased_decision.deadline)

    def test_first_failure_on_an_older_head_stops_only_where_the_lookup_is_what_remains(self):
        old = CREATED + merge.CI_GRACE + dt.timedelta(seconds=1)
        self.assertEqual(decide(lookup_error="ValueError: x", now=old).action, "stop")
        conflict = decide(lookup_error="ValueError: x", merge_state="dirty", now=old)
        self.assertEqual((conflict.action, conflict.reason), ("wait", "The PR conflicts with `master`."))
        paused = decide(lookup_error="ValueError: x", enabled=False, now=old)
        self.assertEqual(paused.action, "wait")
        self.assertIn("paused", paused.outcome)
        self.assertIn("DEPENDABOT_MERGE_ENABLED", paused.reason)
        self.assertIn("None while paused", paused.deadline)

    def test_failed_lookup_never_merges_and_does_not_hide_other_decisions(self):
        self.assertEqual(decide(lookup_error="ValueError: x", failed=["lint"]).reason, decide(failed=["lint"]).reason)
        late = CREATED + merge.CI_GRACE + dt.timedelta(seconds=1)
        self.assertEqual(
            decide(lookup_error="ValueError: x", failed=["lint"], now=late).reason, decide(failed=["lint"], now=late).reason
        )
        self.assertEqual(decide(lookup_error="ValueError: x", changes_requested_by=["a"]).action, "stop")
        hold = merge.Hold("alice", "r", NOW + dt.timedelta(days=1))
        self.assertEqual(decide(lookup_error="ValueError: x", hold=hold).action, "hold")
        self.assertEqual(decide(lookup_error="ValueError: x", enabled=False).action, "wait")
        self.assertIn("advisory lookup failed", decide(lookup_error="ValueError: x", pending=["lint"]).reason)
        requested = decide(version_advisories=["G"], changes_requested_by=["a"])
        self.assertEqual(requested.reason, "A reviewer requested changes.")

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
                self.assertEqual(decide(merge_state=state, now=CREATED + merge.CI_GRACE).action, "wait")
                late = decide(merge_state=state, now=CREATED + merge.CI_GRACE + dt.timedelta(seconds=1))
                self.assertEqual(late.action, "stop")

    def test_uncomputed_state_does_not_escalate_until_the_decision_window(self):
        for state in ("unknown", None):
            with self.subTest(state=state):
                self.assertEqual(decide(merge_state=state, now=CREATED + merge.DECISION_WINDOW).action, "wait")
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
        self.assertEqual(decision.action, "wait")
        self.assertIn("paused", decision.outcome)
        self.assertIn("DEPENDABOT_MERGE_ENABLED", decision.reason)
        self.assertIn("DEPENDABOT_MERGE_ENABLED", decision.mergeable_when)
        self.assertIn("None while paused", decision.deadline)

    def test_paused_triage_never_promises_a_merge_and_says_how_to_resume(self):
        for evidence in (
            {"pending": ["lint"]},
            {"merge_state": "blocked"},
            {"merge_state": "dirty"},
            {"lookup_error": "ValueError: x"},
        ):
            with self.subTest(evidence=evidence):
                decision = decide(enabled=False, **evidence)
                self.assertEqual(decision.action, "wait")
                self.assertIn("paused", decision.outcome)
                self.assertIn("DEPENDABOT_MERGE_ENABLED", decision.mergeable_when)
        self.assertIn("@dependabot rebase", decide(enabled=False, merge_state="dirty").mergeable_when)

    def test_waiting_deadlines_name_the_time_their_own_limit_ends(self):
        for evidence, deadline in (
            ({"merge_state": "unknown"}, "2026-11-12 11:00 UTC. This state then stops the merge."),
            ({"merge_state": None}, "2026-11-12 11:00 UTC. This state then stops the merge."),
            ({"merge_state": "blocked"}, "2026-11-05 13:00 UTC. This state then stops the merge."),
            ({"merge_state": "dirty"}, "2026-11-06 11:00 UTC. A remaining conflict then stops the merge."),
            ({"pending": ["a"]}, "2026-11-05 13:00 UTC. Unfinished checks then stop the merge."),
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(decide(**evidence).deadline, deadline)

    def test_conflict_with_pending_checks_waits_for_the_rebase(self):
        evidence = {"merge_state": "dirty", "pending": ["build"]}
        self.assertEqual(decide(now=CREATED + merge.CI_GRACE + dt.timedelta(seconds=1), **evidence).action, "wait")
        self.assertEqual(decide(now=CREATED + merge.REBASE_GRACE + dt.timedelta(seconds=1), **evidence).action, "stop")

    def test_late_stops_say_what_would_make_them_mergeable(self):
        late = dt.timedelta(days=8)
        self.assertIn("@dependabot rebase", decide(pending=["a"], now=CREATED + late).mergeable_when)
        self.assertIn("@dependabot rebase", decide(merge_state="dirty", now=CREATED + late).mergeable_when)
        self.assertIn("merge it by hand", decide(merge_state="blocked", now=CREATED + late).mergeable_when)

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
            "verification unknown": [commit(verified=None)],
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
            for name in ("pyproject.toml", "uv.lock", "Dockerfile", ".pre-commit-config.yaml")
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
        yaml_row = {"filename": ".github/workflows/ci.yaml", "status": "modified", "patch": bare}
        self.assertIsNone(merge.change_problem([yaml_row]))

    def test_other_paths_are_rejected(self):
        for path in (
            "environment.yml",
            "tools/x.py",
            "torch_bsf/__init__.py",
            "docs/Dockerfile",
            ".github/dependabot.yml",
            "a/pyproject.toml",
        ):
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
            "reordered steps": (
                f"-  - uses: a/a@{OLD_SHA}\n-  - uses: b/b@{OLD_SHA}\n+  - uses: b/b@{NEW_SHA}\n+  - uses: a/a@{NEW_SHA}"
            ),
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


def metadata_block(*entries):
    lines = []
    for name, version in entries:
        lines.append(f"- dependency-name: {name}")
        if version is not None:
            lines.append(f"  dependency-version: {version}")
    return "---\nupdated-dependencies:\n" + "\n".join(lines) + "\n...\n"


def osv_response(raw):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = raw
    return response


# The encoded `@` of a query that names a version.
ENCODED_AT = "%40"


def api_answering_only_without_a_version(rows):
    api = Mock()
    api.request.side_effect = lambda path: rows if ENCODED_AT not in path else []
    return api


def branch_pr(segment):
    return make_pr(head={"repo": {"full_name": REPO}, "ref": f"dependabot/{segment}/x", "sha": "c" * 40})


PIP = branch_pr("pip")
ACTIONS = branch_pr("github_actions")


class GitHubAdvisoryFake:
    """Answer like GitHub: a query with a version matches literally, and one without it ignores case."""

    def __init__(self, advisories=()):
        self.advisories = advisories  # (id, filed name, affected version)
        self.queries = []

    def __call__(self, path):
        assert urlsplit(path).path == "/advisories"
        query = parse_qs(urlsplit(path).query)
        self.queries.append(query)
        name, _, version = query["affects"][0].partition("@")
        rows = []
        for ident, filed, affected in self.advisories:
            if query["type"][0] != "reviewed":
                continue
            if (filed == name and version == affected) or (not version and filed.lower() == name.lower()):
                rows.append({"ghsa_id": ident, "vulnerabilities": [{"package": {"name": filed}}]})
        return rows


class OsvFake:
    """Answer like OSV: names are normalized, and only named versions have advisories."""

    def __init__(self, advisories=None):
        self.advisories = advisories or {}
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        name = re.sub(r"[-_.]+", "-", body["package"]["name"]).lower()
        key = (name, body.get("version"))
        rows = [{"id": ident} for ident in self.advisories.get(key, [])]
        return {"vulns": rows} if rows else {}


SHA_WITH_DIGITS = "13e9c17417b29e9a276f73ec6ed7fb574137d3c3"


class AdvisoryTests(unittest.TestCase):
    def osv(self, advisories=None):
        fake = OsvFake(advisories)
        patcher = patch.object(merge, "_osv_query", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_real_metadata_is_parsed(self):
        self.assertEqual(merge.updated_dependencies(METADATA), [("x", "2.0.0")])
        two = "---\nupdated-dependencies:\n- dependency-name: a\n- dependency-name: 'b'\n  dependency-version: \"2\"\n...\n"
        self.assertEqual(merge.updated_dependencies(two), [("a", None), ("b", "2")])
        self.assertEqual(merge.updated_dependencies("no metadata"), [])

    def test_parser_ignores_a_version_before_any_name(self):
        orphan = "---\nupdated-dependencies:\n  dependency-version: 1\n- dependency-name: a\n...\n"
        self.assertEqual(merge.updated_dependencies(orphan), [("a", None)])

    def test_a_block_with_other_text_after_it_or_no_end_names_nothing(self):
        quoted = metadata_block(("x", "2.0.0"))
        for tail in (
            "\nMore release notes.\n",
            "Fixed in 2.0: a bug\n",
            "See:foo\n",
            "Note: see the changelog\n",
            "...\n",
            "Signed-off-by:x\n",
        ):
            with self.subTest(tail=tail):
                self.assertEqual(merge.updated_dependencies(quoted + tail), [])
        self.assertEqual(merge.updated_dependencies(quoted.removesuffix("...\n")), [])
        sign_off = "Signed-off-by: dependabot[bot] <support@github.com>"
        trailers = f"{quoted}\n{sign_off}\nCo-authored-by: a <a@b>\nCo-Authored-By: b <b@c>\n"
        self.assertEqual(merge.updated_dependencies(trailers), [("x", "2.0.0")])

    def test_a_block_must_start_on_its_own_lines(self):
        body = "- dependency-name: x\n  dependency-version: 2.0.0\n...\n"
        for head in (
            "> updated-dependencies:\n",
            "updated-dependencies:\n",
            "notes\nupdated-dependencies:\n",
            "---\n> updated-dependencies:\n",
            "---\n  updated-dependencies:\n",
            "  ---\nupdated-dependencies:\n",
        ):
            with self.subTest(head=head):
                self.assertEqual(merge.updated_dependencies(head + body), [])

    def test_an_indented_name_is_not_read(self):
        self.assertEqual(merge.updated_dependencies("---\nupdated-dependencies:\n  - dependency-name: x\n...\n"), [])

    def test_an_end_before_the_block_does_not_close_it(self):
        self.assertEqual(merge.updated_dependencies("...\n---\nupdated-dependencies:\n- dependency-name: x\n"), [])

    def test_an_over_indented_version_is_ignored(self):
        over_indented = "---\nupdated-dependencies:\n- dependency-name: x\n    dependency-version: 2\n...\n"
        self.assertEqual(merge.updated_dependencies(over_indented), [("x", None)])

    def test_quoted_release_notes_cannot_replace_the_last_block(self):
        fake = metadata_block(("harmless", "1"))
        self.assertEqual(merge.updated_dependencies(fake + METADATA), [("x", "2.0.0")])

    def test_lowercase_metadata_finds_an_advisory_filed_under_a_canonical_name(self):
        self.osv({("pyyaml", "5.3.1"): ["GHSA-a"]})
        found = merge.known_advisories(Mock(), PIP, [commit(message=metadata_block(("pyyaml", "5.3.1")))])
        self.assertEqual(found, (["GHSA-a"], []))

    def test_every_dependency_of_every_commit_is_looked_up(self):
        self.osv({("b", "2"): ["GHSA-b"], ("c", "3"): ["GHSA-c"]})
        commits = [commit(message=metadata_block(("a", "1"), ("b", "2"))), commit(message=metadata_block(("c", "3")))]
        self.assertEqual(merge.known_advisories(Mock(), PIP, commits)[0], ["GHSA-b", "GHSA-c"])

    def test_values_that_cannot_be_compared_ask_about_the_whole_package(self):
        for version in (
            None,
            "''",
            "null",
            "1!8.0.0",
            "abc1234",
            "a" * 40,
            SHA_WITH_DIGITS,
            "44rc1",
            "44.rc1",
            "v2a1",
            "41.0.0a1b2",
            "41.0.0.post1.a1",
            "41.0.0rc1rc2",
            "١.٢",
        ):
            with self.subTest(version=version):
                fake = self.osv({("x", None): ["GHSA-a"]})
                found = merge.known_advisories(Mock(), PIP, [commit(message=metadata_block(("x", version)))])
                self.assertEqual(found, ([], ["GHSA-a"]))
                self.assertEqual(fake.bodies, [{"package": {"name": "x", "ecosystem": "PyPI"}}])

    def test_version_shapes_are_compared_by_version(self):
        for version, sent in (
            ("2.0.0", "2.0.0"),
            ("v35", "35"),
            ("V35", "35"),
            ("2.0.0rc1", "2.0.0rc1"),
            ("2.0.0RC1", "2.0.0RC1"),
            ("1.0a1", "1.0a1"),
            ("1.0b2", "1.0b2"),
            ("1.0alpha1", "1.0alpha1"),
            ("1.0beta2", "1.0beta2"),
            ("1.0.a1", "1.0.a1"),
            ("1.0a.1", "1.0a.1"),
            ("1.0.rc.1", "1.0.rc.1"),
            ("1.0.post.2", "1.0.post.2"),
            ("3.1.dev.4", "3.1.dev.4"),
            ("1.0.post2", "1.0.post2"),
            ("3.1.dev4", "3.1.dev4"),
            ("41.0.0a1.post1.dev2", "41.0.0a1.post1.dev2"),
        ):
            with self.subTest(version=version):
                fake = self.osv()
                merge.known_advisories(Mock(), PIP, [commit(message=metadata_block(("x", version)))])
                self.assertEqual(fake.bodies[0]["version"], sent)

    def test_osv_pages_are_followed_and_withdrawn_entries_ignored(self):
        pages = [{"vulns": [{"id": "A"}], "next_page_token": "t"}, {"vulns": [{"id": "B"}, {"id": "C", "withdrawn": "2026"}]}]
        tokens = []

        def query(body):
            tokens.append(body.get("page_token"))
            return pages[len(tokens) - 1]

        with patch.object(merge, "_osv_query", query):
            self.assertEqual(merge._pip_advisories("x", "1"), {"A", "B"})
        self.assertEqual(tokens, [None, "t"])

    def test_endless_osv_pages_fail_instead_of_passing(self):
        with (
            patch.object(merge, "_osv_query", return_value={"next_page_token": "t"}),
            self.assertRaisesRegex(ValueError, "too many pages"),
        ):
            merge._pip_advisories("x", "1")

    def test_extras_are_not_part_of_the_package_name(self):
        fake = self.osv({("pyyaml", "5.3.1"): ["GHSA-a"]})
        self.assertEqual(merge._pip_advisories("pyyaml[libyaml]", "5.3.1"), {"GHSA-a"})
        self.assertEqual(fake.bodies[0]["package"]["name"], "pyyaml")

    def test_osv_request(self):
        response = osv_response(b'{"vulns": []}')
        with patch("urllib.request.urlopen", return_value=response) as opened:
            self.assertEqual(merge._osv_query({"package": {"name": "x"}}), {"vulns": []})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.osv.dev/v1/query")
        self.assertEqual(json.loads(request.data), {"package": {"name": "x"}})
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(opened.call_args.kwargs["timeout"], 60)
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(request.headers["User-agent"], "pytorch-bsf-dependabot-merge")

    def test_only_a_real_osv_answer_reads_as_no_advisories(self):
        response = osv_response(b"{}")
        with patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(merge._osv_query({}), {})
        response.__enter__.return_value.read.return_value = b'{"next_page_token": "t"}'
        with patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(merge._osv_query({}), {"next_page_token": "t"})
        for raw in (
            b"[]",
            b'{"message": "x"}',
            b'{"code": 3}',
            b'{"results": []}',
            b'{"vulns": {}}',
            b'{"vulns": ""}',
            b'{"vulns": null}',
            b'{"next_page_token": "t", "message": "x"}',
        ):
            with self.subTest(raw=raw):
                response.__enter__.return_value.read.return_value = raw
                with patch("urllib.request.urlopen", return_value=response), self.assertRaisesRegex(ValueError, "malformed"):
                    merge._osv_query({})

    def test_an_osv_answer_with_results_may_carry_new_fields(self):
        response = osv_response(b'{"vulns": [{"id": "G"}], "schema_version": "1"}')
        with patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(merge._osv_query({})["vulns"], [{"id": "G"}])

    def actions(self, name, version, advisories=()):
        fake = GitHubAdvisoryFake(advisories)
        api = Mock()
        api.request.side_effect = fake
        return merge.known_advisories(api, ACTIONS, [commit(message=metadata_block((name, version)))]), fake

    def test_subpath_action_finds_an_advisory_filed_on_its_repository(self):
        filed = [("GHSA-a", "github/codeql-action", "3.28.2")]
        self.assertEqual(self.actions("github/codeql-action/init", "3.28.2", filed)[0], (["GHSA-a"], []))
        self.assertEqual(self.actions("github/codeql-action/init", "4.0.0", filed)[0], ([], []))

    def test_subpath_sha_pin_asks_about_the_repository(self):
        filed = [("GHSA-a", "github/codeql-action", "3.28.2")]
        self.assertEqual(self.actions("github/codeql-action/init", "c" * 40, filed)[0], ([], ["GHSA-a"]))

    def test_sha_pin_finds_an_advisory_filed_only_on_the_subpath(self):
        self.assertEqual(self.actions("a/b/c", "d" * 40, [("GHSA-a", "a/b/c", "1.0")])[0], ([], ["GHSA-a"]))

    def test_subpath_finds_an_advisory_filed_in_mixed_case(self):
        self.assertEqual(self.actions("a/b/c", "1.0.0", [("GHSA-a", "A/B/C", "1.0.0")])[0], (["GHSA-a"], []))

    def test_moving_action_tags_ask_about_the_whole_package(self):
        filed = [("GHSA-a", "github/codeql-action", "3.28.2")]
        for tag in ("3", "v3", "V3", "3.28", "v3.28", "3.28a1", "٣.٢.١"):
            with self.subTest(tag=tag):
                self.assertEqual(self.actions("github/codeql-action", tag, filed)[0], ([], ["GHSA-a"]))
        for exact in ("3.28.2", "v3.28.2", "V3.28.2"):
            with self.subTest(exact=exact):
                self.assertEqual(self.actions("github/codeql-action", exact, filed)[0], (["GHSA-a"], []))

    def test_mixed_case_action_finds_an_advisory_filed_in_lowercase(self):
        self.assertEqual(self.actions("Azure/x", "1.0.0", [("GHSA-a", "azure/x", "1.0.0")])[0], (["GHSA-a"], []))

    def test_rows_without_package_data_do_not_fail_the_lookup(self):
        api = api_answering_only_without_a_version([{"ghsa_id": "GHSA-a", "vulnerabilities": None}, {"ghsa_id": "GHSA-b"}])
        self.assertEqual(merge._action_advisories(api, "a/b", "1.0.0"), set())

    def test_lowercase_action_finds_an_advisory_filed_in_mixed_case(self):
        filed = [("GHSA-a", "Azure/setup-kubectl", "0.0.1")]
        self.assertEqual(self.actions("azure/setup-kubectl", "0.0.1", filed)[0], (["GHSA-a"], []))

    def test_unrelated_filed_names_are_not_asked_about(self):
        api = api_answering_only_without_a_version(
            [{"ghsa_id": "GHSA-z", "vulnerabilities": [{"package": {"name": "other/thing"}}]}]
        )
        self.assertEqual(merge._action_advisories(api, "owner/repo", "1.0"), set())
        self.assertFalse(any("other" in call.args[0] for call in api.request.call_args_list))

    def test_a_sibling_repository_with_the_same_prefix_is_not_asked_about(self):
        api = api_answering_only_without_a_version(
            [{"ghsa_id": "GHSA-z", "vulnerabilities": [{"package": {"name": "o/repo-extra"}}]}]
        )
        self.assertEqual(merge._action_advisories(api, "o/repo", "1.0"), set())
        self.assertFalse(any("repo-extra" in call.args[0] for call in api.request.call_args_list))

    def test_every_filed_spelling_in_every_row_is_asked_about(self):
        rows = [
            {"ghsa_id": "G0", "vulnerabilities": [{"package": {"name": "o/other"}}]},
            {"ghsa_id": "G1", "vulnerabilities": [{"package": {"name": "o/other"}}, {"package": {"name": "A/B"}}]},
        ]

        def answer(path):
            if ENCODED_AT not in path:
                return rows
            return [{"ghsa_id": "G1"}] if f"A%2FB{ENCODED_AT}" in path else []

        api = Mock()
        api.request.side_effect = answer
        self.assertEqual(merge._action_advisories(api, "a/b", "1.0.0"), {"G1"})

    def test_a_filed_subpath_of_the_repository_is_asked_about(self):
        api = Mock()
        rows = [{"ghsa_id": "GHSA-z", "vulnerabilities": [{"package": {"name": "owner/repo/slim"}}]}]
        api.request.side_effect = lambda path: rows if ENCODED_AT not in path or f"slim{ENCODED_AT}" in path else []
        self.assertEqual(merge._action_advisories(api, "owner/repo", "1.0"), {"GHSA-z"})

    def test_actions_ask_github_for_both_advisory_types(self):
        _, fake = self.actions("owner/act", "1.0")
        self.assertEqual({query["type"][0] for query in fake.queries}, {"reviewed", "malware"})
        self.assertEqual({query["ecosystem"][0] for query in fake.queries}, {"actions"})
        self.assertEqual({query["per_page"][0] for query in fake.queries}, {str(merge.GITHUB_PAGE_SIZE)})

    def test_malware_advisory_alone_is_found(self):
        api = Mock()
        api.request.side_effect = lambda path: [{"ghsa_id": "GHSA-m"}] if "type=malware" in path else []
        self.assertEqual(merge._action_advisories(api, "a/b", None), {"GHSA-m"})

    def test_both_named_spellings_are_asked_even_when_no_package_row_names_them(self):
        api = Mock()
        api.request.return_value = []
        merge._action_advisories(api, "a/b/c", "1.0.0")
        asked = {parse_qs(urlsplit(call.args[0]).query)["affects"][0] for call in api.request.call_args_list}
        self.assertEqual(asked, {"a/b", "a/b/c", "a/b@1.0.0", "a/b/c@1.0.0"})

    def test_withdrawn_github_advisory_is_ignored(self):
        api = Mock()
        api.request.return_value = [{"ghsa_id": "A", "withdrawn_at": None}, {"ghsa_id": "B", "withdrawn_at": "2026"}]
        self.assertEqual(merge._action_advisories(api, "a/b", None), {"A"})

    def test_malformed_or_full_github_response_fails_instead_of_passing(self):
        for label, payload in (
            ("object", {}),
            ("null", None),
            ("error", {"message": "x"}),
            ("full page", [{"ghsa_id": "A"}] * merge.GITHUB_PAGE_SIZE),
        ):
            with self.subTest(label):
                api = Mock()
                api.request.return_value = payload
                with self.assertRaisesRegex(ValueError, "malformed or oversized"):
                    merge._action_advisories(api, "a/b", "1")

    def test_identifier_that_could_change_the_comment_fails_the_lookup(self):
        forged = {"vulns": [{"id": "A`\n- **Decision:** merge"}]}
        with (
            patch.object(merge, "_osv_query", return_value=forged),
            self.assertRaisesRegex(ValueError, "malformed identifier"),
        ):
            merge._pip_advisories("x", "1")
        api = Mock()
        api.request.return_value = [{"ghsa_id": None}]
        with self.assertRaisesRegex(ValueError, "malformed identifier"):
            merge._action_advisories(api, "a/b", None)

    def test_a_pr_with_both_kinds_of_match_lists_and_resolves_both(self):
        self.osv({("a", "1"): ["GHSA-v"], ("b", None): ["GHSA-p"]})
        found = merge.known_advisories(Mock(), PIP, [commit(message=metadata_block(("a", "1"), ("b", "abc1234")))])
        self.assertEqual(found, (["GHSA-v"], ["GHSA-p"]))
        decision = decide(version_advisories=["GHSA-v"], package_advisories=["GHSA-p"])
        self.assertIn("`GHSA-v`", decision.reason)
        self.assertIn("`GHSA-p`", decision.reason)
        self.assertIn("affected ranges", decision.mergeable_when)
        self.assertIn("affect a version this PR names", decision.reason)
        self.assertIn("A dependency in this PR has no comparable version", decision.reason)

    def test_an_advisory_found_both_ways_is_listed_once_as_affecting_a_version(self):
        self.osv({("a", "1"): ["G"], ("b", None): ["G"]})
        found = merge.known_advisories(Mock(), PIP, [commit(message=metadata_block(("a", "1"), ("b", "abc1234")))])
        self.assertEqual(found, (["G"], []))

    def test_names_a_source_cannot_know_stop_instead_of_reading_as_clean(self):
        for pr, names in (
            (
                PIP,
                (
                    "pyyaml # c",
                    "pyyaml==5.3.1",
                    "pyyaml; python_version<'3'",
                    "py/yaml",
                    "-x",
                    ".x",
                    "pyyaml-",
                    "pyyaml.",
                    "pyyaml_",
                ),
            ),
            (ACTIONS, ("actions/download-artifact # c", "actions", "actiòns/cache", "/cache", "a/b" + "/" * 60 + "!", "a/b/")),
        ):
            for name in names:
                with self.subTest(name=name):
                    message = metadata_block((name, "1.0.0"))
                    self.assertIn("does not name its dependencies", str(merge.metadata_problem(pr, [commit(message=message)])))
        for good_first in (True, False):
            pair = (("good", "1.0.0"), ("bad # c", "1.0.0"))
            message = metadata_block(*(pair if good_first else pair[::-1]))
            self.assertIn("does not name its dependencies", str(merge.metadata_problem(PIP, [commit(message=message)])))
        for pr, name in (
            (PIP, "pyyaml"),
            (PIP, "PyYAML[extra]"),
            (PIP, "jupyter_server.x-y"),
            (ACTIONS, "actions/cache"),
            (ACTIONS, "github/codeql-action/init"),
            (ACTIONS, "a.b/c_d/e/f"),
        ):
            with self.subTest(name=name):
                self.assertIsNone(merge.metadata_problem(pr, [commit(message=metadata_block((name, "1.0.0")))]))

    def test_ecosystems_without_data_are_not_queried_and_need_no_metadata(self):
        for segment in ("docker", "pre_commit"):
            with self.subTest(segment=segment):
                api = Mock()
                self.assertEqual(merge.known_advisories(api, branch_pr(segment), [commit()]), ([], []))
                api.request.assert_not_called()
                self.assertIsNone(merge.metadata_problem(branch_pr(segment), [commit(message="bump")]))

    def test_unknown_ecosystem_stops_and_cannot_read_as_clean(self):
        for segment in ("uv", "npm_and_yarn", "Pip", "group-all"):
            with self.subTest(segment=segment):
                self.assertIn(f"`{segment}`", str(merge.metadata_problem(branch_pr(segment), [commit()])))
                with self.assertRaisesRegex(KeyError, re.escape(segment)):
                    merge.known_advisories(Mock(), branch_pr(segment), [commit()])

    def test_missing_or_nameless_metadata_in_any_commit_is_a_problem(self):
        bad = commit(message="bump")
        null = commit(message=None)
        nameless = commit(message=metadata_block(("''", "1")))
        for commits in ([bad], [null], [nameless], [commit(), bad], [bad, commit()]):
            with self.subTest(commits=commits):
                self.assertIn("does not name its dependencies", str(merge.metadata_problem(PIP, commits)))
        self.assertIsNone(merge.metadata_problem(PIP, [commit()]))

    def test_advisory_stops_ahead_of_ci_and_lists_every_id(self):
        ids = [f"GHSA-{index}" for index in range(12)]
        for field in ("version_advisories", "package_advisories"):
            for evidence in ({"pending": ["lint"]}, {"failed": ["lint"]}, {"merge_state": "dirty"}):
                with self.subTest(field=field, evidence=evidence):
                    decision = decide(**{field: ids}, **evidence)
                    self.assertEqual(decision.action, "stop")
                    for ident in ids:
                        self.assertIn(f"`{ident}`", decision.reason)
                    self.assertNotIn("more", decision.reason)
        self.assertIn("unaffected version", decide(version_advisories=ids).mergeable_when)

    def test_package_wide_match_does_not_promise_that_a_newer_version_helps(self):
        decision = decide(package_advisories=["GHSA-a"])
        self.assertEqual(decision.action, "stop")
        self.assertIn("no comparable version", decision.reason)
        self.assertIn("`GHSA-a`", decision.reason)
        self.assertIn("affected ranges", decision.mergeable_when)
        self.assertNotIn("unaffected version", decision.mergeable_when)

    def test_hold_still_outranks_an_advisory(self):
        hold = merge.Hold("alice", "why", NOW)
        self.assertEqual(decide(version_advisories=["GHSA-a"], hold=hold).action, "hold")


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
        self.assertEqual(merge.review_state(reviews, allow), (["bob", "carol"], ["alice"]))

    def test_reviews_without_write_access_do_not_count(self):
        reviews: list[security.Json] = [
            {"state": "CHANGES_REQUESTED", "user": {"login": "stranger"}},
            {"state": "APPROVED", "user": {"login": "passerby"}},
            {"state": "CHANGES_REQUESTED", "user": None},
        ]
        self.assertEqual(merge.review_state(reviews, lambda login: False), ([], []))


class HoldTests(unittest.TestCase):
    def hold(self, *comments, now=NOW, allowed=allow):
        return merge.active_hold(list(comments), allowed, now)

    def test_authorized_hold_applies_through_the_end_of_its_day(self):
        hold = self.hold(comment("/dependabot-hold until=2026-11-07 breaks downstream"))
        self.assertEqual((hold.author, hold.reason), ("maintainer", "breaks downstream"))
        body = "/dependabot-hold until=2026-11-07 r"
        self.assertIsNotNone(self.hold(comment(body), now=dt.datetime(2026, 11, 7, 23, 59, tzinfo=dt.UTC)))
        self.assertIsNone(self.hold(comment(body), now=dt.datetime(2026, 11, 8, tzinfo=dt.UTC)))

    def test_unauthorized_users_cannot_hold_or_release(self):
        def deny_stranger(login):
            return login != "stranger"

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

    def test_reason_cannot_hide_the_rest_of_the_triage_comment(self):
        self.assertEqual(self.hold(comment("/dependabot-hold until=2026-11-07 <!-- hide")).reason, "&lt;!-- hide")

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
        can_command = merge.write_access_checker(api)
        self.assertTrue(can_command("w"))
        self.assertTrue(can_command("a"))
        self.assertFalse(can_command("r"))
        self.assertFalse(can_command("r"))
        self.assertEqual(api.get.call_count, 3)

    def test_unresolvable_login_has_no_access_but_other_errors_propagate(self):
        api = Mock()
        api.get.side_effect = security.GitHubError("missing", 404)
        self.assertFalse(merge.write_access_checker(api)("Copilot"))
        api.get.side_effect = security.GitHubError("down", 502)
        with self.assertRaises(security.GitHubError):
            merge.write_access_checker(api)("alice")

    def test_login_is_quoted(self):
        api = Mock()
        api.get.return_value = {"permission": "read"}
        merge.write_access_checker(api)("dependabot[bot]/x")
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
        osv = patch.object(merge, "_osv_query", return_value={})
        self.osv_query = osv.start()
        self.addCleanup(osv.stop)
        ci = patch.object(security, "ci_state", return_value=([], [], {}))
        self.ci_state = ci.start()
        self.addCleanup(ci.stop)
        # A test that reaches a wait must opt in to a clock, so that a mistake fails instead of sleeping for real.
        no_sleep = patch.object(merge.time, "sleep", side_effect=AssertionError("unexpected sleep"))
        no_sleep.start()
        self.addCleanup(no_sleep.stop)

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

    def stub_merge_states_and_sleep(self, *states):
        reads = iter(states)

        def get(path):
            return {**self.pr, "mergeable_state": next(reads)} if path == "pulls/7" else self.get(path)

        self.api.get.side_effect = get
        sleep = patch.object(merge.time, "sleep")
        self.addCleanup(sleep.stop)
        return sleep.start()

    def test_an_uncomputed_merge_state_is_read_again_before_deciding(self):
        sleep = self.stub_merge_states_and_sleep("unknown", None, "clean")
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(merge.UNCOMPUTED_DELAY_SECONDS)

    def test_a_state_that_stays_uncomputed_waits_after_a_bounded_number_of_reads(self):
        sleep = self.stub_merge_states_and_sleep(*["unknown"] * (merge.UNCOMPUTED_REREADS + 1))
        self.assert_blocked("GitHub reports the merge state as `unknown`.")
        self.assertEqual(sleep.call_count, merge.UNCOMPUTED_REREADS)

    def stub_clock(self):
        """Replace time with a clock that only sleep advances, and return the sleep mock."""
        clock = [0.0]
        sleep = patch.object(merge.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
        monotonic = patch.object(merge.time, "monotonic", side_effect=lambda: clock[0])
        self.addCleanup(sleep.stop)
        self.addCleanup(monotonic.stop)
        monotonic.start()
        return sleep.start()

    CODECOV_POLLS = math.ceil(merge.CODECOV_WAIT_SECONDS / merge.CODECOV_POLL_SECONDS)  # Polls in a full wait.

    def add_second_pr(self):
        self.pages["pulls?state=open&base=master"] = [make_pr(number=7), make_pr(number=9)]
        for path in ("commits", "files", "reviews"):
            self.pages[f"pulls/9/{path}"] = self.pages[f"pulls/7/{path}"]

    def codecov_then(self, *after):
        """Report codecov/patch as the only unfinished check, and GitHub's matching `unstable` state, then these states."""
        self.pr["mergeable_state"] = "unstable"
        states = iter([([], ["codecov/patch"], {}), *after])

        def ci_state(*_, **__):
            state = next(states)
            if not state[0] and not state[1]:
                self.pr["mergeable_state"] = "clean"
            return state

        self.ci_state.side_effect = ci_state

    def test_a_run_waits_for_codecov_and_merges_on_a_fresh_evaluation(self):
        sleep = self.stub_clock()
        self.codecov_then(([], [], {}), ([], [], {}))
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)
        sleep.assert_called_once_with(merge.CODECOV_POLL_SECONDS)
        self.assertEqual(self.ci_state.call_count, 3)

    def test_a_hold_posted_during_the_codecov_wait_stops_the_merge(self):
        self.stub_clock()
        self.codecov_then(([], [], {}), ([], [], {}))
        polls = self.ci_state.side_effect

        def poll_after_a_hold(*args):
            self.comments.append(comment("/dependabot-hold until=2026-11-07 breaks downstream"))
            return polls(*args)

        self.ci_state.side_effect = poll_after_a_hold
        self.assert_blocked("breaks downstream")

    def test_a_run_does_not_wait_for_codecov_while_anything_else_stops_the_merge(self):
        cases: dict[str, tuple[list[str], list[str], dict[str, security.Json]]] = {
            "other check unfinished": ([], ["codecov/patch", "lint"], {}),
            "check failed": (["lint"], ["codecov/patch"], {}),
        }
        for name, state in cases.items():
            with self.subTest(name):
                self.api.write.reset_mock()
                self.ci_state.reset_mock()
                self.ci_state.return_value = state
                sleep = self.stub_clock()
                self.run_once()
                self.assertEqual(self.writes("pulls/7/merge"), [])
                self.assertEqual(self.ci_state.call_count, 1)
                sleep.assert_not_called()
        with self.subTest("hold"):
            self.api.write.reset_mock()
            self.ci_state.reset_mock()
            self.ci_state.return_value = ([], ["codecov/patch"], {})
            self.comments.append(comment("/dependabot-hold until=2026-11-07 r"))
            sleep = self.stub_clock()
            self.run_once()
            sleep.assert_not_called()
        with self.subTest("merging disabled"):
            self.comments.clear()
            self.ci_state.reset_mock()
            sleep = self.stub_clock()
            merge.reconcile(self.api, FIXTURE_ROOT, False, NOW)
            sleep.assert_not_called()
        with self.subTest("conflict"):
            self.pr["mergeable_state"] = "dirty"
            sleep = self.stub_clock()
            self.run_once()
            sleep.assert_not_called()
        with self.subTest("failed advisory lookup"):
            self.pr["mergeable_state"] = "unstable"
            self.fail_lookup()
            sleep = self.stub_clock()
            self.run_once()
            sleep.assert_not_called()

    def test_a_pr_closed_during_the_codecov_wait_gets_no_comment(self):
        sleep = self.stub_clock()
        self.codecov_then(([], [], {}), ([], [], {}))
        polls = self.ci_state.side_effect

        def poll_then_close(*args, **options):
            state = polls(*args, **options)
            if self.ci_state.call_count > 1:  # The first poll, after the first evaluation.
                self.pr.update(state="closed", mergeable_state=None)
            return state

        self.ci_state.side_effect = poll_then_close
        self.assertEqual(self.run_once(), [])
        self.assertEqual(self.api.write.call_args_list, [])
        # No rereads of the merge state, which a closed PR never gets.
        self.assertEqual(sleep.call_count, 1)

    def test_a_merge_state_still_unstable_after_codecov_is_read_again(self):
        sleep = self.stub_clock()
        self.pr["mergeable_state"] = "unstable"
        self.ci_state.side_effect = [([], ["codecov/patch"], {}), ([], [], {}), ([], [], {})]
        states = iter(["unstable", "unstable", "clean"])
        get = self.get

        def settle(path):
            return {**self.pr, "mergeable_state": next(states)} if path == "pulls/7" else get(path)

        self.api.get.side_effect = settle
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [merge.CODECOV_POLL_SECONDS, merge.UNCOMPUTED_DELAY_SECONDS]
        )

    def test_the_clock_moves_forward_by_each_wait(self):
        # Two hours less a minute before the run, so only the moved clock passes the CI deadline after a full wait.
        self.pages["pulls/7/commits"] = [commit(date="2026-11-05T10:01:00Z")]
        self.stub_clock()
        self.ci_state.return_value = ([], ["codecov/patch"], {})
        self.assert_blocked(f"CI is still incomplete after {merge.format_hours(merge.CI_GRACE)} hours")

    def test_a_later_wait_moves_the_clock_only_by_its_own_length(self):
        self.add_second_pr()
        self.stub_clock()
        codecov: tuple[list[str], list[str], dict[str, security.Json]] = ([], ["codecov/patch"], {})
        done: tuple[list[str], list[str], dict[str, security.Json]] = ([], [], {})
        self.ci_state.side_effect = [
            *(codecov, done, done),  # PR 7: evaluation, one poll, evaluation; it merges.
            codecov,  # PR 9: evaluation, one poll into the run.
            *[codecov] * (self.CODECOV_POLLS - 1),  # PR 9: polls until the shared deadline.
            codecov,  # PR 9: evaluation after the wait.
        ]
        with patch.object(merge, "decide", wraps=merge.decide) as decide:
            self.assertEqual(self.run_once(), [])
        # Only decisions with codecov/patch unfinished: the wait gates' hypothetical decisions clear it, and so does
        # PR 7's last evaluation.
        seconds = [(call.args[1] - NOW).total_seconds() for call in decide.call_args_list if call.args[0].pending]
        self.assertEqual(seconds, [0, merge.CODECOV_POLL_SECONDS, merge.CODECOV_WAIT_SECONDS])

    def test_a_pr_already_stopped_does_not_wait(self):
        self.pages["pulls/7/commits"] = [commit(date="2026-11-05T09:00:00Z")]  # Older than the CI deadline.
        sleep = self.stub_clock()
        self.ci_state.return_value = ([], ["codecov/patch"], {})
        self.assert_blocked(f"CI is still incomplete after {merge.format_hours(merge.CI_GRACE)} hours")
        sleep.assert_not_called()

    def test_the_overdue_notice_uses_the_clock_after_the_wait(self):
        # Seven days less a minute before the run, so only the moved clock makes the decision overdue.
        self.pr["created_at"] = "2026-10-29T12:01:00Z"
        self.pages["pulls/7/commits"] = [commit(date="2026-11-05T10:01:00Z")]
        self.stub_clock()
        self.ci_state.return_value = ([], ["codecov/patch"], {})
        self.run_once()
        bodies = [call.args[1]["body"] for call in self.writes("issues/7/comments")]
        self.assertTrue(any(merge.OVERDUE_MARKER in body for body in bodies))

    def test_the_codecov_wait_reads_the_checks_it_is_given(self):
        sleep = patch.object(merge.time, "sleep")
        self.addCleanup(sleep.stop)
        sleep.start()
        with patch.object(merge.time, "monotonic", side_effect=[0, 0]):
            merge.wait_for_codecov(self.api, "c" * 40, FIXTURE_ROOT, 1, workflows=["a.yml"], excluded_checks=frozenset({"b"}))
        self.ci_state.assert_called_once_with(
            self.api, "c" * 40, FIXTURE_ROOT, workflows=["a.yml"], excluded_checks=frozenset({"b"})
        )

    def test_the_codecov_wait_ends_at_its_deadline(self):
        polls = self.CODECOV_POLLS
        sleep = self.stub_clock()
        self.ci_state.return_value = ([], ["codecov/patch"], {})
        self.assert_blocked("`codecov/patch`")
        self.assertEqual(sleep.call_count, polls)
        self.assertEqual(self.ci_state.call_count, polls + 2)

    def test_all_prs_of_a_run_share_one_codecov_wait(self):
        self.add_second_pr()
        polls = self.CODECOV_POLLS
        sleep = self.stub_clock()
        self.ci_state.return_value = ([], ["codecov/patch"], {})
        self.assertEqual(self.run_once(), [])
        # PR 7 uses up the wait and is evaluated twice. PR 9 is evaluated once, without a poll.
        self.assertEqual(sleep.call_count, polls)
        self.assertEqual(self.ci_state.call_count, polls + 2 + 1)
        self.assertEqual(self.writes("pulls/9/merge"), [])

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

    def test_known_advisory_stops_the_merge_and_notifies(self):
        self.osv_query.return_value = {"vulns": [{"id": "GHSA-a"}]}
        self.assert_blocked("GHSA-a")
        self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_package_wide_match_is_wired_into_the_evaluation(self):
        self.pages["pulls/7/commits"] = [commit(message=metadata_block(("x", None)))]
        self.osv_query.return_value = {"vulns": [{"id": "GHSA-a"}]}
        self.assert_blocked("no comparable version")

    def test_withdrawn_advisory_lets_the_next_evaluation_merge(self):
        self.osv_query.return_value = {"vulns": [{"id": "GHSA-a", "withdrawn": "2026-11-05T00:00:00Z"}]}
        self.run_once()
        self.assertEqual(len(self.writes("pulls/7/merge")), 1)

    def test_failed_advisory_lookup_merges_nothing_and_fails_the_job(self):
        for error in (urllib.error.URLError("down"), TimeoutError("slow"), ValueError("bad json"), KeyError("id")):
            with self.subTest(error=error):
                self.osv_query.side_effect = error
                self.api.write.reset_mock()
                errors = self.run_once()
                self.assertIn("#7", errors[0])
                self.assertIn("advisory lookup", errors[0])
                self.assertIn("advisory lookup failed", self.triage())
                self.assertEqual(self.writes("pulls/7/merge"), [])

    def test_unverified_commit_is_not_looked_up(self):
        self.pages["pulls/7/commits"] = [commit(verified=False)]
        self.assert_blocked("signed by GitHub")
        self.osv_query.assert_not_called()

    def test_commit_without_metadata_stops_with_triage(self):
        self.pages["pulls/7/commits"] = [commit(message="bump")]
        self.assert_blocked("does not name its dependencies")

    def use_action_pr(self):
        # A real branch: its later parts can name another ecosystem.
        self.pr["head"]["ref"] = "dependabot/github_actions/docker/build-push-action-6.0.0"
        self.pages["pulls/7/commits"] = [commit(message=metadata_block(("actions/cache", "6.1.0")))]

    def test_action_advisory_is_wired_into_the_evaluation(self):
        self.use_action_pr()
        self.api.request.return_value = [{"ghsa_id": "GHSA-a", "vulnerabilities": []}]
        self.assert_blocked("GHSA-a")

    def test_failed_action_lookup_merges_nothing_and_fails_the_job(self):
        self.use_action_pr()
        self.api.request.side_effect = security.GitHubError("down", 502)
        errors = self.run_once()
        self.assertIn("#7", errors[0])
        self.assertIn("advisory lookup", errors[0])
        self.assertIn("advisory lookup failed", self.triage())
        self.assertEqual(self.writes("pulls/7/merge"), [])

    def test_failed_lookup_is_explained_on_the_pr_without_the_error_text(self):
        self.osv_query.side_effect = ValueError("secret detail")
        errors = self.run_once()
        self.assertIn("advisory lookup", errors[0])
        self.assertIn("secret detail", errors[0])
        self.assertIn("advisory lookup failed", self.triage())
        self.assertNotIn("secret detail", self.triage())

    def fail_lookup(self):
        self.osv_query.side_effect = ValueError("down")

    def assert_failed_job_and_no_merge(self, errors):
        self.assertEqual(len(errors), 1)
        self.assertIn("advisory lookup", errors[0])
        self.assertEqual(self.writes("pulls/7/merge"), [])

    def assert_lookup_failure_shows_only_in_the_job(self):
        self.fail_lookup()
        self.assert_failed_job_and_no_merge(self.run_once())
        self.assertNotIn("lookup failed", self.triage())

    def test_failed_lookup_shows_only_in_the_job_during_a_hold(self):
        self.comments.append(comment("/dependabot-hold until=2026-11-07 r"))
        self.assert_lookup_failure_shows_only_in_the_job()

    def test_failed_lookup_shows_only_in_the_job_during_a_conflict(self):
        self.pr["mergeable_state"] = "dirty"
        self.assert_lookup_failure_shows_only_in_the_job()

    def test_failed_lookup_shows_only_in_the_job_after_a_failed_check(self):
        self.ci_state.return_value = (["lint"], [], {})
        self.assert_lookup_failure_shows_only_in_the_job()

    def test_failed_lookup_fails_the_job_while_paused_and_still_looks_up(self):
        self.fail_lookup()
        self.assert_failed_job_and_no_merge(self.run_once(enabled=False))
        self.osv_query.assert_called()
        self.assertNotIn("lookup failed", self.triage())

    def test_late_failed_lookup_stops_and_assigns_the_maintainer(self):
        self.fail_lookup()
        errors = merge.reconcile(self.api, FIXTURE_ROOT, True, CREATED + merge.CI_GRACE + dt.timedelta(seconds=1))
        self.assert_failed_job_and_no_merge(errors)
        self.assertEqual(len(self.writes("issues/7/assignees")), 1)
        self.assertIn("advisory lookup failed", self.triage())

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
                self.assertIn("token's permissions", self.triage())
                self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_moved_head_or_base_retries_without_notifying_until_the_head_is_old(self):
        for code in (405, 409):
            with self.subTest(code=code):
                self.fail_merge(code)
                self.pr["created_at"] = "2026-10-01T00:00:00Z"
                self.pages["pulls/7/commits"] = [commit(date="2026-10-29T12:00:00Z")]
                self.assertEqual(self.run_once(), [])
                self.assertIn("tries again", self.triage())
                self.assertEqual(self.writes("issues/7/assignees"), [])
                self.fail_merge(code)
                self.pages["pulls/7/commits"] = [commit(date="2026-10-29T11:59:59Z")]
                self.assertEqual(self.run_once(), [])
                self.assertIn(f"HTTP status {code}", self.triage())
                self.assertIn("@dependabot rebase", self.triage())
                self.assertEqual(len(self.writes("issues/7/assignees")), 1)

    def test_unconfirmed_merge_is_not_reported_as_merged(self):
        self.api.write.return_value = {"merged": False}
        self.run_once()
        self.assertIn("without merging", self.triage())

    def test_transient_merge_error_is_reported_without_a_triage_verdict(self):
        self.fail_merge(502)
        errors = self.run_once()
        self.assertIn("#7", errors[0])
        self.assertEqual(self.writes("issues/7/comments"), [])

    def test_deleted_accounts_do_not_stop_triage(self):
        self.comments.extend([{**comment("x"), "user": None}, comment(None, ident=2)])
        self.pages["pulls/7/reviews"] = [{"state": "APPROVED", "user": None}]
        self.pages["pulls?state=open&base=master"].append({"number": 9, "user": None})
        self.assertEqual(self.run_once(), [])
        self.assertEqual(len(self.writes("issues/7/comments")), 1)
        self.assertEqual(self.writes("pulls/9/merge"), [])

    def test_a_pr_error_does_not_stop_post_merge_workflows(self):
        self.pages["pulls?state=open&base=master"] = [make_pr(number=6)]
        self.master_pulls.append({"number": 3, "user": {"login": "dependabot[bot]"}})
        self.pr["merged_by"] = {"login": security.BOT}

        def pages(path, key=None):
            if path == "issues/6/comments":
                raise security.GitHubError("down", 502)
            return self.pages.get(path, [])

        self.api.pages.side_effect = pages
        self.assertEqual(len(self.run_once()), 1)
        self.assertEqual(len(self.api.write.call_args_list), len(security.POST_MERGE))

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
        self.pr["created_at"] = "2026-10-01T00:00:00Z"
        self.run_once(enabled=False)
        self.assertEqual(self.writes("pulls/7/merge"), [])
        self.assertIn("paused", self.triage())
        self.assertEqual(self.writes("issues/7/assignees"), [])
        self.assertEqual(len(self.writes("issues/7/comments")), 1)

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

    def test_missing_users_dispatch_nothing(self):
        for author_row, pull in (
            ({"number": 3, "user": None}, {}),
            ({"number": 3, "user": {"login": merge.AUTHOR}}, {"merged_by": None}),
        ):
            with self.subTest(author_row=author_row, pull=pull):
                api = Mock()
                sha = "e" * 40
                responses = {"commits/master": {"sha": sha}, f"commits/{sha}/pulls": [author_row], "pulls/3": pull}
                api.get.side_effect = responses.__getitem__
                merge.ensure_post_merge(api)
                api.write.assert_not_called()

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
        for path in (".github/workflows/dependabot-merge.yml", ".github/dependabot.yml", "tools/dependabot_merge.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_every_trigger_needed_for_immediate_and_recovered_runs_is_present(self):
        self.assertIn("types: [requested, completed]", self.workflow)
        self.assertRegex(self.workflow, r"schedule:\n(?:\s+#.*\n)*\s+- cron: '37 \* \* \* \*'")

    def test_other_branches_start_no_run(self):
        self.assertRegex(
            self.workflow,
            r"types: \[requested, completed\]\n(?:\s+#.*\n)*"
            r"    branches: \['dependabot/\*\*', 'release-please--\*\*', master\]\n",
        )

    def test_a_skipped_run_cannot_replace_a_pending_evaluation(self):
        # GitHub cancels a pending run that a newer run in its group replaces, even when the newer one skips its job.
        group = re.search(r"concurrency:\n  group: >-\n((?:    .*\n)+)", self.workflow)
        condition = re.search(r"    if: >-\n((?:      .*\n)+)", self.workflow)
        assert group is not None and condition is not None
        expected = " ".join(condition[1].split())
        skipped = "format('dependabot-merge-skipped-{0}', github.run_id)"
        self.assertEqual(" ".join(group[1].split()), "${{ (" + expected + ") && 'dependabot-merge' || " + skipped + " }}")

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

    def test_every_configured_ecosystem_is_classified_for_advisories(self):
        configured = set(re.findall(r'package-ecosystem: "([^"]+)"', (ROOT / ".github/dependabot.yml").read_text()))
        self.assertEqual(configured, {"pip", "pre-commit", "github-actions", "docker"})
        self.assertEqual(set(merge.ADVISORY_SOURCES), {"pip", "github_actions", "pre_commit", "docker"})
        self.assertIsNotNone(merge.ADVISORY_SOURCES["pip"])
        self.assertIsNotNone(merge.ADVISORY_SOURCES["github_actions"])
        self.assertIsNone(merge.ADVISORY_SOURCES["pre_commit"])
        self.assertIsNone(merge.ADVISORY_SOURCES["docker"])

    def test_every_ecosystem_has_a_three_day_cooldown(self):
        config = (ROOT / ".github/dependabot.yml").read_text()
        self.assertEqual(config.count("package-ecosystem:"), config.count("cooldown:\n      default-days: 3\n"))


if __name__ == "__main__":
    unittest.main()
