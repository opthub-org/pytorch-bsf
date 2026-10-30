"""Offline regressions for security repairs, privileged publication, and merge gates."""

import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import security_automation as security

REPO = "opthub-org/pytorch-bsf"
ROOT = Path(__file__).resolve().parents[2]


def pull_request(**overrides):
    result = {
        "number": 123,
        "state": "open",
        "draft": False,
        "user": {"login": security.BOT},
        "head": {"sha": "h" * 40, "ref": security.BRANCH_PREFIX + "abc", "repo": {"full_name": REPO}},
        "base": {"ref": "master"},
        "mergeable_state": "clean",
    }
    result.update(overrides)
    return result


def api_mock():
    api = Mock()
    api.repo = REPO
    api.pages.return_value = []
    return api


def check(name, state="success", ident=1):
    return {"name": name, "id": ident, "status": "completed" if state != "pending" else "in_progress", "conclusion": state}


class CheckEvidenceTests(unittest.TestCase):
    def test_missing_pending_skipped_and_neutral_checks_block(self):
        for rows in ([], [check("test", "pending")], [check("test", "skipped")], [check("test", "neutral")]):
            with self.subTest(rows=rows):
                self.assertEqual(security.check_evidence(rows, [], {"test"}), ([], ["test"]))

    def test_failed_new_run_supersedes_old_green(self):
        self.assertEqual(
            security.check_evidence([check("test", "failure", 2), check("test", "success", 1)], [], {"test"}),
            (["test"], ["test"]),
        )

    def test_latest_external_status_wins(self):
        rows = [
            {"id": 2, "context": "codecov/patch", "state": "pending"},
            {"id": 1, "context": "codecov/patch", "state": "success"},
        ]
        self.assertEqual(security.check_evidence([], rows, {"codecov/patch"}), ([], ["codecov/patch"]))

    def test_unexpected_failed_check_blocks(self):
        self.assertEqual(
            security.check_evidence([check("test"), check("extra", "failure")], [], {"test"}), (["extra"], ["extra"])
        )

    def test_complete_success_is_ready(self):
        self.assertEqual(security.check_evidence([check("test")], [], {"test"}), ([], []))

    def test_python_matrix_and_codecov_are_mandatory(self):
        expected = security.expected_checks(ROOT)
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for classifier in metadata["project"]["classifiers"]:
            if classifier.startswith("Programming Language :: Python :: 3."):
                self.assertIn(f"build ({classifier.rsplit(' ', 1)[-1]})", expected)
        self.assertTrue({"codecov/patch", "optional-deps (no-mlflow)"} <= expected)


class PatchBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.run_git("config", "user.name", "Security test")
        self.run_git("config", "user.email", "security@example.invalid")
        (self.root / "source.py").write_text("value = 1\n")
        workflow = self.root / ".github/workflows/test.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("steps:\n  - uses: owner/action@" + "a" * 40 + " # v1\n")
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "base")
        self.base = self.run_git("rev-parse", "HEAD").strip()

    def run_git(self, *args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=self.root, text=True)

    def test_regular_repair_and_new_regression_test_are_exported(self):
        (self.root / "source.py").write_text("value = 2\n")
        (self.root / "new_test.py").write_text("assert True\n")
        self.run_git("add", "new_test.py")
        paths, data = security.validate_changes(self.root, self.base)
        self.assertEqual(paths, ["new_test.py", "source.py"])
        self.assertIn(b"+value = 2", data)

    def test_control_plane_instructions_and_secret_paths_are_rejected(self):
        paths = [
            ".github/security/fix.md",
            "tools/security_automation.py",
            "tests/tools/security_automation.py",
            ".github/workflows/security-remediation.yml",
            ".github/workflows/security-policy.yml",
            "nested/AGENTS.md",
            ".codex/config.toml",
            ".claude/settings.json",
            "apiley.local",
            ".env.local",
        ]
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_only_existing_workflow_pin_can_change(self):
        target = self.root / ".github/workflows/test.yml"
        target.write_text(target.read_text().replace("a" * 40, "b" * 40).replace("# v1", "# v2"))
        self.assertEqual(security.validate_changes(self.root, self.base)[0], [".github/workflows/test.yml"])
        target.write_text(target.read_text() + "    if: false\n")
        with self.assertRaisesRegex(ValueError, "only update existing action SHA pins"):
            security.validate_changes(self.root, self.base)

    def test_unpinned_action_is_rejected(self):
        target = self.root / ".github/workflows/test.yml"
        target.write_text(target.read_text().replace("a" * 40, "v2"))
        with self.assertRaisesRegex(ValueError, "only update existing action SHA pins"):
            security.validate_changes(self.root, self.base)

    def test_symlink_and_binary_content_are_rejected(self):
        target = self.root / "source.py"
        target.unlink()
        target.symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "regular text files"):
            security.validate_changes(self.root, self.base)
        target.unlink()
        target.write_bytes(b"value\0")
        with self.assertRaisesRegex(ValueError, "Binary content"):
            security.validate_changes(self.root, self.base)

    def test_oversized_patch_is_rejected(self):
        (self.root / "source.py").write_text("x" * (security.MAX_PATCH_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "size limit"):
            security.validate_changes(self.root, self.base)

    def test_altered_task_artifact_is_rejected_before_any_github_operation(self):
        target = self.root / "task.json"
        target.write_text('{"mode":"fix"}')
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/security_automation.py"),
                "publish",
                "--task",
                str(target),
                "--task-digest",
                "0" * 64,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("trusted plan digest", result.stderr)

    def test_api_key_is_rejected_without_disclosing_it(self):
        (self.root / "source.py").write_text("sk-proj-" + "x" * 40)
        with self.assertRaisesRegex(ValueError, "API key") as error:
            security.validate_changes(self.root, self.base)
        self.assertNotIn("sk-proj", str(error.exception))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.api = api_mock()
        self.signal = {"key": "dependabot-1", "updated": "today", "severity": "high"}
        self.ident = security.fingerprint(self.signal)
        self.state: security.Json = {"tasks": {}}

    def call_plan(self, signal=None):
        with (
            patch.object(security, "load_state", return_value=(12, self.state)),
            patch.object(security, "save_state", return_value=12),
            patch.object(security, "base_sha", return_value="b" * 40),
        ):
            return security.plan(self.api, ROOT, [signal or self.signal])

    def test_attempt_is_reserved_before_running_ai(self):
        task = self.call_plan()
        self.assertEqual(task["mode"], "fix")
        self.assertEqual(task["attempts"], 1)
        self.assertEqual(self.state["tasks"][self.ident]["attempts"], 1)

    def test_daily_audit_is_kept_when_an_alert_takes_priority(self):
        audit = {"key": "audit", "updated": "today", "severity": "low"}
        with (
            patch.object(security, "load_state", return_value=(12, self.state)),
            patch.object(security, "save_state", return_value=12),
            patch.object(security, "base_sha", return_value="b" * 40),
        ):
            task = security.plan(self.api, ROOT, [self.signal, audit])
        self.assertEqual(task["signal"]["key"], "dependabot-1")
        queued = security.include_active_tasks(self.state, [])
        self.assertEqual(queued, [audit])

    def test_budget_exhaustion_blocks_and_new_advisory_resets(self):
        for _ in range(3):
            self.assertEqual(self.call_plan()["mode"], "fix")
        self.assertEqual(self.call_plan(), {"mode": "none"})
        self.assertEqual(self.call_plan(dict(self.signal, updated="new upstream fix"))["attempts"], 1)

    def test_fork_draft_and_spoofed_author_are_not_trusted(self):
        for pr in [
            pull_request(draft=True),
            pull_request(user={"login": "attacker"}),
            pull_request(head={"ref": security.BRANCH_PREFIX + "abc", "repo": {"full_name": "attacker/repo"}}),
        ]:
            with self.subTest(pr=pr):
                self.assertFalse(security.trusted_pr(pr, REPO))

    def test_rejected_review_routes_back_to_repair(self):
        task = {"pr": 123, "base": "b" * 40, "review_findings": ["unsafe loading"]}
        self.api.get.side_effect = [pull_request(), {"merge_base_commit": {"sha": task["base"]}}]
        with (
            patch.object(security, "ci_state", return_value=([], [], {})),
            patch.object(security, "review_threads", return_value=[]),
        ):
            self.assertTrue(security.choose_pr_mode(self.api, task, ROOT))
        self.assertEqual(task["mode"], "fix")

    def test_missing_ci_is_explicitly_dispatched_without_a_pat(self):
        task = {"pr": 123, "base": "b" * 40}
        self.api.get.side_effect = [pull_request(), {"merge_base_commit": {"sha": task["base"]}}]
        with (
            patch.object(security, "ci_state", return_value=([], ["lint"], {})),
            patch.object(security, "review_threads", return_value=[]),
        ):
            self.assertFalse(security.choose_pr_mode(self.api, task, ROOT))
        self.assertEqual(self.api.write.call_count, len(security.WORKFLOWS))

    def test_unfixed_alert_cannot_be_silenced_as_no_change(self):
        with patch.object(security, "update_task") as update:
            security.publish(self.api, {"signal": self.signal}, ROOT, {"status": "no_change", "summary": "no patch yet"})
        self.assertEqual(update.call_args.args[2]["result"], "blocked")
        self.api.write.assert_not_called()

    def test_audit_without_findings_finishes_without_a_pr(self):
        with patch.object(security, "update_task") as update:
            security.publish(self.api, {"signal": {"key": "audit"}}, ROOT, {"status": "no_change", "summary": "no findings"})
        self.assertEqual(update.call_args.args[2]["result"], "no_change")
        self.api.write.assert_not_called()

    def test_ci_registration_retries_are_bounded(self):
        task = {"head": "head", "branch": "branch"}
        for _ in range(5):
            security.retry_missing_ci(self.api, task, {})
        self.assertEqual(self.api.write.call_count, 3 * len(security.WORKFLOWS))
        self.assertIn("did not register", task["blocked"])


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.api = api_mock()
        self.task: security.Json = {
            "id": "task-id",
            "published_head": "h" * 40,
            "branch": security.BRANCH_PREFIX + "abc",
            "signal": {"key": "dependabot-1"},
            "summary": "Repair validated",
        }

    def test_published_ref_without_pr_is_resumed(self):
        self.api.get.return_value = {"object": {"sha": "h" * 40}}
        self.api.write.return_value = {"number": 123}
        with patch.object(security, "update_task") as update:
            security.restore_publication(self.api, self.task)
        self.assertEqual(self.api.write.call_args_list[0].args[0], "pulls")
        self.assertEqual(update.call_args.args[2]["pr"], 123)

    def test_existing_pr_is_adopted_without_duplicate(self):
        self.api.get.return_value = {"object": {"sha": "h" * 40}}
        self.api.pages.return_value = [pull_request(body="<!-- security-automation-task:task-id -->")]
        with patch.object(security, "update_task"):
            security.restore_publication(self.api, self.task)
        self.assertFalse(any(call.args[0] == "pulls" for call in self.api.write.call_args_list))

    def test_other_occupant_cannot_replace_reserved_branch(self):
        self.api.get.return_value = {"object": {"sha": "attacker-sha"}}
        with self.assertRaisesRegex(ValueError, "unexpected commit"):
            security.restore_publication(self.api, self.task)
        self.api.write.assert_not_called()

    def test_missing_ref_is_created_from_recorded_commit(self):
        self.api.get.side_effect = security.GitHubError("not found", 404)
        self.api.write.return_value = {"number": 123}
        with patch.object(security, "update_task"):
            security.restore_publication(self.api, self.task)
        self.api.write.assert_any_call(
            "git/refs", {"ref": "refs/heads/" + self.task["branch"], "sha": self.task["published_head"]}
        )


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.api = api_mock()
        self.api.get.return_value = pull_request()
        self.api.write.return_value = {"merged": True}
        self.task: security.Json = {"id": "id", "pr": 123, "head": "h" * 40, "base": "b" * 40}
        self.verdict = {"approved": True, "findings": [], "resolved_threads": [], "summary": "Safe repair"}

    def merge(self, **options):
        with (
            patch.object(security, "base_sha", side_effect=options.get("bases", [self.task["base"], self.task["base"]])),
            patch.object(security, "ci_state", return_value=options.get("ci", ([], [], {}))),
            patch.object(security, "review_threads", return_value=options.get("threads", [])),
            patch.object(security, "update_task") as update,
        ):
            security.merge(self.api, self.task, ROOT, self.verdict)
            return update

    def assert_no_merge(self):
        self.assertFalse(any(call.args[0].endswith("/merge") for call in self.api.write.call_args_list))

    def test_success_uses_exact_head_and_dispatches_post_merge_workflows(self):
        update = self.merge()
        self.api.write.assert_any_call("pulls/123/merge", {"sha": self.task["head"], "merge_method": "squash"}, "PUT")
        self.api.write.assert_any_call("actions/workflows/release-please-action.yml/dispatches", {"ref": "master"})
        update.assert_called_once()

    def test_changed_head_is_rejected(self):
        changed = pull_request()
        changed["head"]["sha"] = "new head"
        self.api.get.return_value = changed
        with self.assertRaisesRegex(ValueError, "head or base changed"):
            self.merge()
        self.assert_no_merge()

    def test_changed_base_is_rejected_before_merge(self):
        with self.assertRaisesRegex(ValueError, "immediately before merge"):
            self.merge(bases=[self.task["base"], "changed base"])
        self.assert_no_merge()

    def test_pending_codecov_cannot_merge(self):
        with self.assertRaisesRegex(ValueError, "merge gate"):
            self.merge(ci=([], ["codecov/patch"], {}))
        self.assert_no_merge()

    def test_unclean_merge_and_unresolved_thread_cannot_merge(self):
        self.api.get.return_value = pull_request(mergeable_state="unstable")
        with self.assertRaisesRegex(ValueError, "merge gate"):
            self.merge()
        self.api.get.return_value = pull_request()
        with self.assertRaisesRegex(ValueError, "resolve every open thread"):
            self.merge(threads=[{"id": "thread-1"}])
        self.assert_no_merge()

    def test_negative_review_goes_to_repair_without_any_merge(self):
        self.verdict.update(approved=False, findings=["unsafe input"])
        update = self.merge()
        self.assertEqual(update.call_args.args[2]["review_findings"], ["unsafe input"])
        self.assert_no_merge()

    def test_active_changes_requested_blocks_even_with_ai_approval(self):
        self.api.pages.return_value = [{"state": "CHANGES_REQUESTED", "user": {"login": "reviewer"}}]
        with self.assertRaisesRegex(ValueError, "change-request"):
            self.merge()
        self.assert_no_merge()

    def test_github_must_confirm_the_merge(self):
        self.api.write.return_value = {"merged": False}
        with self.assertRaisesRegex(ValueError, "did not confirm"):
            self.merge()


class ThreadPaginationTests(unittest.TestCase):
    def test_outdated_threads_on_later_pages_are_not_lost(self):
        api = api_mock()

        def page(nodes, more):
            return {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {"nodes": nodes, "pageInfo": {"hasNextPage": more, "endCursor": "cursor"}}
                        }
                    }
                }
            }

        api.request.side_effect = [
            page([{"id": "resolved", "isResolved": True}], True),
            page([{"id": "outdated", "isResolved": False}], False),
        ]
        self.assertEqual(security.review_threads(api, 123), [{"id": "outdated", "isResolved": False}])
        self.assertEqual(api.request.call_args.args[2]["variables"]["after"], "cursor")


class AlertPrivacyTests(unittest.TestCase):
    def test_public_signal_manifest_has_no_private_rule_or_location(self):
        api = api_mock()
        api.pages.side_effect = [
            [],
            [
                {
                    "number": 7,
                    "rule": {"severity": "high", "description": "private rule"},
                    "most_recent_instance": {"location": {"path": "private location"}},
                    "updated_at": "today",
                    "html_url": "https://example.invalid/7",
                }
            ],
        ]
        manifest = security.signals(api)
        self.assertEqual(set(manifest[0]), {"key", "severity", "updated", "revision"})
        self.assertNotIn("private", json.dumps(manifest))

    def test_detailed_evidence_is_only_written_to_ephemeral_prompt(self):
        api = api_mock()
        api.get.return_value = {"rule": {"description": "private diagnostic"}, "most_recent_instance": {}, "state": "open"}
        task = {"mode": "fix", "signal": {"key": "code-scanning-7"}}
        with tempfile.TemporaryDirectory() as scratch, patch.object(security, "GitHub", return_value=api):
            prompt = Path(scratch) / "prompt.md"
            security.prepare(task, ROOT, prompt)
            self.assertIn("private diagnostic", prompt.read_text())
        self.assertNotIn("private", json.dumps(task))


if __name__ == "__main__":
    unittest.main()
