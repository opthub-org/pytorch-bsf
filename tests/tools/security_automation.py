"""Offline regressions for security repairs, privileged publication, and merge gates."""

import base64
import datetime as dt
import json
import subprocess
import sys
import tempfile
import textwrap
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
        self.assertTrue({"coverage / coverage", "codecov/patch", "optional-deps (no-mlflow)"} <= expected)

    def test_dispatch_preserves_pr_identity_for_isolated_coverage_upload(self):
        api = api_mock()
        security.dispatch_ci(api, "automation/security/test", 123)
        api.write.assert_any_call(
            "actions/workflows/python-package.yml/dispatches",
            {"ref": "automation/security/test", "inputs": {"pr_number": "123"}},
        )
        self.assertEqual(api.write.call_count, len(security.WORKFLOWS))


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.run_git("config", "user.name", "Security test")
        self.run_git("config", "user.email", "security@example.invalid")
        self.run_git("config", "core.autocrlf", "false")
        (self.root / "source.py").write_text("value = 1\n", newline="\n")
        workflow = self.root / ".github/workflows/test.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("steps:\n  - uses: owner/action@" + "a" * 40 + " # v1\n", newline="\n")
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "base")
        self.base = self.run_git("rev-parse", "HEAD").strip()

    def run_git(self, *args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=self.root, text=True)


class PatchBoundaryTests(GitFixture):
    def test_regular_repair_and_new_regression_test_are_exported(self):
        (self.root / "source.py").write_text("value = 2\n", newline="\n")
        (self.root / "new_test.py").write_text("assert True\n", newline="\n")
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
            ".github/workflows/codeql-analysis.yml",
            "nested/AGENTS.md",
            ".codex/config.toml",
            ".claude/settings.json",
            "apiley.local",
            ".env.local",
        ]
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_new_modules_cannot_shadow_trusted_control_plane_imports(self):
        for path in (
            "tools/json.py",
            "tools/urllib/__init__.py",
            "tests/tools/json.py",
            "json.py",
            "pathlib.py",
            "urllib/__init__.py",
            "sitecustomize.py",
            "usercustomize.py",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "control-plane"):
                security.validate_path(path)

    def test_validation_configuration_and_common_credentials_are_protected(self):
        for path in [
            ".pre-commit-config.yaml",
            "codecov.yml",
            "pytest.ini",
            "tools/check_distribution.py",
            "tests/conftest.py",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".aws/credentials",
            "service-account.json",
        ]:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_dependency_updates_cannot_modify_pytest_or_coverage_configuration(self):
        target = self.root / "pyproject.toml"
        target.write_text(
            '[project]\ndependencies = ["demo>=1"]\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', newline="\n"
        )
        self.run_git("add", "pyproject.toml")
        self.run_git("commit", "-qm", "manifest")
        base = self.run_git("rev-parse", "HEAD").strip()
        target.write_text(target.read_text().replace("demo>=1", "demo>=2"), newline="\n")
        security.validate_changes(self.root, base)
        target.write_text(target.read_text().replace('["tests"]', '["tests/safe_only"]'), newline="\n")
        with self.assertRaisesRegex(ValueError, "Only dependency"):
            security.validate_changes(self.root, base)

    def test_alternate_validation_configs_cannot_override_trusted_policy(self):
        for name in (".pytest.ini", "pytest.toml", ".pytest.toml", "ruff.toml", ".ruff.toml", "mypy.ini", ".mypy.ini"):
            with self.subTest(name=name):
                target = self.root / name
                target.write_text("# override trusted configuration\n", newline="\n")
                self.run_git("add", name)
                with self.assertRaisesRegex(ValueError, "Validation policy"):
                    security.validate_changes(self.root, self.base)
                self.run_git("reset", "-q", "--", name)
                target.unlink()
        with self.assertRaisesRegex(ValueError, "Validation policy"):
            security.validate_path("tests/nested/ruff.toml")

    def test_nested_pyproject_cannot_override_lint_discovery(self):
        target = self.root / "pkg/pyproject.toml"
        target.parent.mkdir()
        target.write_text('[tool.ruff.lint]\nignore = ["F401"]\n', newline="\n")
        self.run_git("add", "pkg/pyproject.toml")
        with self.assertRaisesRegex(ValueError, "Validation policy"):
            security.validate_changes(self.root, self.base)

    def test_only_existing_workflow_pin_can_change(self):
        target = self.root / ".github/workflows/test.yml"
        target.write_text(target.read_text().replace("a" * 40, "b" * 40).replace("# v1", "# v2"), newline="\n")
        self.assertEqual(security.validate_changes(self.root, self.base)[0], [".github/workflows/test.yml"])
        target.write_text(target.read_text() + "    if: false\n", newline="\n")
        with self.assertRaisesRegex(ValueError, "only update existing action SHA pins"):
            security.validate_changes(self.root, self.base)

    def test_unpinned_action_is_rejected(self):
        target = self.root / ".github/workflows/test.yml"
        target.write_text(target.read_text().replace("a" * 40, "v2"), newline="\n")
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
        (self.root / "source.py").write_text("x" * (security.MAX_PATCH_BYTES + 1), newline="\n")
        with self.assertRaisesRegex(ValueError, "size limit"):
            security.validate_changes(self.root, self.base)

    def test_altered_task_artifact_is_rejected_before_any_github_operation(self):
        target = self.root / "task.json"
        target.write_text('{"mode":"fix"}', newline="\n")
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
        (self.root / "source.py").write_text("sk-proj-" + "x" * 40, newline="\n")
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
        self.assertEqual(self.call_plan(dict(self.signal, revision="new upstream fix"))["attempts"], 1)

    def test_still_open_alert_after_pr_finishes_is_visibly_blocked(self):
        for result in ("merged", "closed"):
            with self.subTest(result=result):
                entry: security.Json = {"signal": self.signal, "result": result, "pr": 123, "attempts": 1, "reviews": 1}
                self.state["tasks"][self.ident] = entry
                self.assertEqual(self.call_plan(), {"mode": "none"})
                self.assertIn("alert remains open", entry["blocked"])
                self.assertEqual((entry["attempts"], entry["reviews"], entry["result"]), (1, 1, result))
                security.save_state(self.api, 12, self.state, "No runnable repairs.")
                visible = self.api.write.call_args.args[1]["body"].split("<!--")[0]
                self.assertIn("alert remains open", visible)
                security.register_signals(self.state, [])
                self.assertEqual(self.state["tasks"], {})

    def test_fork_draft_and_spoofed_author_are_not_trusted(self):
        for pr in [
            pull_request(draft=True),
            pull_request(user={"login": "attacker"}),
            pull_request(head={"ref": security.BRANCH_PREFIX + "abc", "repo": {"full_name": "attacker/repo"}}),
        ]:
            with self.subTest(pr=pr):
                self.assertFalse(security.trusted_pr(pr, REPO))

    def test_rejected_review_routes_back_to_repair(self):
        task = {"pr": 123, "base": "b" * 40, "review_required": True}
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
            patch.object(security, "validate_changes", return_value=([], b"")),
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
        self.assertEqual(update.call_args.args[2]["post_merge_remaining"], [])

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
        self.assertTrue(update.call_args.args[2]["review_required"])
        self.assertNotIn("unsafe input", json.dumps(update.call_args.args[2]))
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


class CommittedPolicyTests(GitFixture):
    def commit(self):
        self.run_git("add", ".")
        self.run_git("commit", "-qm", "candidate")
        head = self.run_git("rev-parse", "HEAD").strip()
        self.run_git("checkout", "-q", self.base)
        return head

    def test_committed_bytes_are_checked_instead_of_trusted_working_tree(self):
        (self.root / "source.py").write_text("value = 2\n", newline="\n")
        head = self.commit()
        paths, patch_data = security.validate_changes(self.root, self.base, head)
        self.assertEqual(paths, ["source.py"])
        self.assertIn(b"+value = 2", patch_data)
        self.assertEqual((self.root / "source.py").read_text(), "value = 1\n")

    def test_renaming_a_protected_file_cannot_hide_its_deletion(self):
        protected = self.root / ".github/security/fix.md"
        protected.parent.mkdir()
        protected.write_text("Trusted security instructions\n", newline="\n")
        self.run_git("add", ".github/security/fix.md")
        self.run_git("commit", "-qm", "trusted policy")
        self.base = self.run_git("rev-parse", "HEAD").strip()
        self.run_git("mv", ".github/security/fix.md", "renamed.md")
        with self.assertRaisesRegex(ValueError, "control-plane"):
            security.validate_changes(self.root, self.base)
        head = self.commit()
        with self.assertRaisesRegex(ValueError, "control-plane"):
            security.validate_changes(self.root, self.base, head)

    def test_later_protected_commit_cannot_merge_even_with_green_ci_and_ai_approval(self):
        target = self.root / "pkg/pyproject.toml"
        target.parent.mkdir()
        target.write_text('[tool.ruff.lint]\nignore = ["F401"]\n', newline="\n")
        head = self.commit()
        self.assertFalse(target.exists())
        task = {"id": "id", "pr": 123, "base": self.base, "head": head}
        api = api_mock()
        pr = pull_request()
        pr["head"]["sha"] = head
        api.get.return_value = pr
        with (
            patch.object(security, "base_sha", return_value=self.base),
            patch.object(security, "ci_state", return_value=([], [], {})),
            self.assertRaisesRegex(ValueError, "Validation policy"),
        ):
            security.merge(api, task, self.root, {"approved": True, "findings": [], "resolved_threads": []})
        api.fetch_commit.assert_called_once_with(self.root, head)
        api.write.assert_not_called()

    def test_committed_symlinks_and_secret_content_are_rejected(self):
        target = self.root / "source.py"
        target.unlink()
        target.symlink_to("/etc/passwd")
        head = self.commit()
        with self.assertRaisesRegex(ValueError, "regular text files"):
            security.validate_changes(self.root, self.base, head)
        target.write_text("sk-proj-" + "x" * 40, newline="\n")
        head = self.commit()
        with self.assertRaisesRegex(ValueError, "API key"):
            security.validate_changes(self.root, self.base, head)

    def test_fetch_uses_scoped_header_without_secret_command_arguments(self):
        api = security.GitHub.__new__(security.GitHub)
        api.repo, api.token = REPO, "test-token"
        with patch.object(security.subprocess, "run") as run:
            api.fetch_commit(self.root, self.base)
        args, kwargs = run.call_args
        self.assertNotIn(api.token, " ".join(args[0]))
        self.assertEqual(args[0][-2:], [f"https://github.com/{REPO}.git", self.base])
        credential = base64.b64encode(b"x-access-token:test-token").decode()
        self.assertEqual(kwargs["env"]["GIT_CONFIG_VALUE_0"], "AUTHORIZATION: basic " + credential)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)


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

    def test_codeql_analysis_bookkeeping_does_not_reset_finding_budget(self):
        alert: security.Json = {
            "number": 7,
            "rule": {"severity": "high", "id": "query"},
            "most_recent_instance": {
                "commit_sha": "old",
                "location": {"path": "source.py", "start_line": 10},
                "message": {"text": "unsafe"},
            },
            "updated_at": "yesterday",
            "html_url": "https://example.invalid/7",
        }
        api = api_mock()
        api.pages.side_effect = [[], [alert]]
        old = security.signals(api)[0]
        alert["most_recent_instance"]["commit_sha"] = "new unrelated commit"
        alert["most_recent_instance"]["location"]["start_line"] = 12
        alert["updated_at"] = "today"
        api.pages.side_effect = [[], [alert]]
        new = security.signals(api)[0]
        self.assertEqual(security.fingerprint(old), security.fingerprint(new))


class PaginationTests(unittest.TestCase):
    def client(self):
        api = security.GitHub.__new__(security.GitHub)
        api.repo, api.token = REPO, "test-token"
        return api

    def test_repository_metadata_endpoint_has_no_trailing_slash(self):
        api = self.client()
        with patch.object(api, "request", return_value={"id": 381016316}) as request:
            self.assertEqual(api.get("")["id"], 381016316)
        request.assert_called_once_with(f"/repos/{REPO}")

    def test_cursor_link_is_followed_without_page_parameter(self):
        api = self.client()
        endpoint = f"/repos/{REPO}/dependabot/alerts"
        link = f'<https://api.github.com{endpoint}?after=cursor>; rel="next"'
        with patch.object(api, "_response", side_effect=[([{"number": 1}], link), ([{"number": 2}], "")]) as response:
            self.assertEqual(api.pages("dependabot/alerts?state=open"), [{"number": 1}, {"number": 2}])
        self.assertNotIn("&page=", response.call_args_list[0].args[0])
        self.assertEqual(response.call_args.args[0], endpoint + "?after=cursor")

    def test_cross_host_or_endpoint_link_is_never_authenticated(self):
        for destination in [
            "https://evil.invalid/alerts",
            "https://api.github.com/repos/elsewhere/secret",
            f"http://api.github.com/repos/{REPO}/dependabot/alerts",
            "https://api.github.com/repositories/999/dependabot/alerts",
            "https://api.github.com/repositories/381016316/issues",
            f"https://token@api.github.com/repos/{REPO}/dependabot/alerts",
        ]:
            api = self.client()
            api.repo_id = 381016316
            with patch.object(api, "_response", return_value=([], f'<{destination}>; rel="next"')) as response:
                with self.assertRaisesRegex(ValueError, "Unsafe GitHub pagination"):
                    api.pages("dependabot/alerts")
                self.assertEqual(response.call_count, 1)

    def test_verified_numeric_repository_alias_is_followed_and_cached(self):
        api = self.client()
        for resource in ("dependabot/alerts", "issues"):
            endpoint = "/repositories/381016316/" + resource
            link = f'<https://api.github.com{endpoint}?after=cursor>; rel="next"'
            with (
                patch.object(api, "get", return_value={"id": 381016316}) as metadata,
                patch.object(api, "_response", side_effect=[([{"id": 1}], link), ([{"id": 2}], "")]) as response,
            ):
                self.assertEqual(api.pages(resource), [{"id": 1}, {"id": 2}])
            self.assertEqual(response.call_args.args[0], endpoint + "?after=cursor")
            self.assertEqual(metadata.call_count, int(resource == "dependabot/alerts"))


class RealCIEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.api = api_mock()
        self.checks = [check(name, ident=index) for index, name in enumerate(security.expected_checks(ROOT), 1)]
        self.runs = [
            {"id": index, "path": ".github/workflows/" + workflow, "status": "completed", "conclusion": "success"}
            for index, workflow in enumerate(security.WORKFLOWS, 1)
        ]

        def pages(path, key=None):
            if "check-runs" in path:
                return self.checks
            if "actions/runs" in path:
                return self.runs
            return []

        self.api.pages.side_effect = pages

    def test_actual_aggregation_requires_all_checks_and_workflows(self):
        self.assertEqual(security.ci_state(self.api, "head", ROOT)[:2], ([], []))
        self.checks = [row for row in self.checks if row["name"] != "codecov/patch"]
        self.assertIn("codecov/patch", security.ci_state(self.api, "head", ROOT)[1])
        self.runs.pop()
        self.assertIn(security.WORKFLOWS[-1], security.ci_state(self.api, "head", ROOT)[1])

    def test_newer_failing_run_supersedes_previous_success(self):
        failed = dict(self.runs[0], id=1000, conclusion="failure")
        self.runs.append(failed)
        failures, pending, _ = security.ci_state(self.api, "head", ROOT)
        self.assertIn(security.WORKFLOWS[0], failures)
        self.assertIn(security.WORKFLOWS[0], pending)

    def test_unfinished_workflow_cannot_merge_with_green_named_checks(self):
        self.runs[0].update(status="in_progress", conclusion=None)
        self.api.get.return_value = pull_request()
        task = {"id": "id", "pr": 123, "head": "h" * 40, "base": "b" * 40}
        verdict = {"approved": True, "findings": [], "resolved_threads": [], "summary": "safe"}
        with patch.object(security, "base_sha", return_value=task["base"]), self.assertRaisesRegex(ValueError, "merge gate"):
            security.merge(self.api, task, ROOT, verdict)
        self.api.write.assert_not_called()


class PublicationIntegrationTests(GitFixture):
    def setUp(self):
        super().setUp()
        self.task: security.Json = {
            "id": "id",
            "signal": {"key": "dependabot-1"},
            "head": self.base,
            "base": self.base,
            "branch": security.BRANCH_PREFIX + "id",
            "attempts": 1,
        }
        self.state = {"tasks": {"id": dict(self.task)}}
        self.api = api_mock()
        self.writes: list[tuple[str, security.Json, str]] = []
        self.api.get.side_effect = self.get
        self.api.write.side_effect = self.write
        (self.root / "source.py").write_text("value = 2\n", newline="\n")

    def get(self, path):
        if path == "git/ref/heads/master":
            return {"object": {"sha": self.base}}
        if path.startswith("git/ref/heads/"):
            raise security.GitHubError("not found", 404)
        if path.startswith("git/commits/"):
            return {"tree": {"sha": "base-tree"}}
        if path.startswith("pulls/"):
            return {"head": {"sha": self.task["head"]}}
        raise AssertionError(path)

    def write(self, path, data, method="POST"):
        self.writes.append((path, json.loads(json.dumps(data)), method))
        replies = {
            "git/blobs": {"sha": "blob-sha"},
            "git/trees": {"sha": "new-tree"},
            "git/commits": {"sha": "c" * 40},
            "pulls": {"number": 123},
        }
        return replies.get(path, {})

    def publish(self):
        with (
            patch.object(security, "load_state", side_effect=lambda api: (12, self.state)),
            patch.object(security, "save_state", return_value=12),
        ):
            security.publish(self.api, self.task, self.root, {"status": "changed", "summary": "Tested repair"})

    def test_initial_repair_publishes_correct_tree_parent_ref_and_ci(self):
        self.publish()
        blob = next(data for path, data, _ in self.writes if path == "git/blobs")
        self.assertEqual(blob["encoding"], "base64")
        self.assertEqual(base64.b64decode(blob["content"]), b"value = 2\n")
        commit = next(data for path, data, _ in self.writes if path == "git/commits")
        tree = next(data for path, data, _ in self.writes if path == "git/trees")
        self.assertEqual(commit["parents"], [self.base])
        self.assertEqual(commit["tree"], "new-tree")
        self.assertEqual(tree["base_tree"], "base-tree")
        self.assertEqual(tree["tree"], [{"path": "source.py", "mode": "100644", "type": "blob", "sha": "blob-sha"}])
        self.assertIn(("git/refs", {"ref": "refs/heads/" + self.task["branch"], "sha": "c" * 40}, "POST"), self.writes)
        self.assertEqual(self.state["tasks"]["id"]["pr"], 123)
        self.assertEqual(sum(path.endswith("/dispatches") for path, _, _ in self.writes), len(security.WORKFLOWS))

    def test_new_deleted_and_executable_files_keep_bytes_and_modes(self):
        (self.root / "source.py").unlink()
        (self.root / "new_test.py").write_text("assert 2 == 2\n", newline="\n")
        script = self.root / "repair.sh"
        script.write_bytes(b"#!/bin/sh\nprintf 'repaired\\n'\n")
        script.chmod(0o755)
        self.run_git("add", "new_test.py", "repair.sh")
        self.publish()
        blobs = [data for path, data, _ in self.writes if path == "git/blobs"]
        self.assertTrue(all(blob["encoding"] == "base64" for blob in blobs))
        self.assertEqual([base64.b64decode(blob["content"]) for blob in blobs], [b"assert 2 == 2\n", script.read_bytes()])
        tree = next(data for path, data, _ in self.writes if path == "git/trees")
        # Windows chmod does not expose POSIX execute bits; publication runs on Linux.
        script_mode = "100644" if sys.platform == "win32" else "100755"
        self.assertEqual(
            tree["tree"],
            [
                {"path": "new_test.py", "mode": "100644", "type": "blob", "sha": "blob-sha"},
                {"path": "repair.sh", "mode": script_mode, "type": "blob", "sha": "blob-sha"},
                {"path": "source.py", "mode": "100644", "type": "blob", "sha": None},
            ],
        )

    def test_existing_repair_updates_ref_without_force_or_duplicate_pr(self):
        self.task.update(pr=123, head="h" * 40)
        self.publish()
        commit = next(data for path, data, _ in self.writes if path == "git/commits")
        self.assertEqual(commit["parents"], ["h" * 40])
        self.assertIn(("git/refs/heads/" + self.task["branch"], {"sha": "c" * 40, "force": False}, "PATCH"), self.writes)
        self.assertFalse(any(path == "pulls" for path, _, _ in self.writes))

    def test_stale_head_or_base_prevents_any_publication(self):
        self.task["pr"] = 123
        self.api.get.side_effect = lambda path: {"head": {"sha": "changed"}}
        with self.assertRaisesRegex(ValueError, "PR head changed"):
            self.publish()
        self.assertEqual(self.writes, [])
        self.task.pop("pr")
        self.api.get.side_effect = lambda path: {"object": {"sha": "new-base"}}
        with self.assertRaisesRegex(ValueError, "Master changed"):
            self.publish()
        self.assertEqual(self.writes, [])


class DurableRecoveryTests(unittest.TestCase):
    def test_candidate_tests_have_no_oidc_or_coverage_credentials(self):
        workflow = (ROOT / ".github/workflows/python-package.yml").read_text()
        build = workflow.split("\n  build:\n", 1)[1].split("\n  coverage:\n", 1)[0]
        lowest = workflow.split("\n  lowest:\n", 1)[1].split("\n  optional-deps:\n", 1)[0]
        for candidate in (build, lowest):
            self.assertNotIn("id-token:", candidate)
            self.assertNotIn("secrets.", candidate)
            self.assertNotIn("codecov/codecov-action@", candidate)
            self.assertIn("actions/upload-artifact@", candidate)
        self.assertIn("pytest-cov", lowest)
        self.assertIn("--cov-report=xml:lowest-coverage.xml", lowest)
        coverage = workflow.split("\n  coverage:\n", 1)[1].split("\n  distribution:\n", 1)[0]
        self.assertIn("uses: ./.github/workflows/security-coverage.yml", coverage)
        self.assertIn("needs: [python-support, lowest, build]", coverage)

    def test_credentialed_coverage_workflow_is_protected_and_reads_only_report_data(self):
        path = ".github/workflows/security-coverage.yml"
        with self.assertRaisesRegex(ValueError, "control-plane"):
            security.validate_path(path)
        workflow = (ROOT / path).read_text()
        self.assertIn("ref: master", workflow)
        self.assertIn("override_commit: ${{ inputs.commit }}", workflow)
        self.assertIn("plugins: noop", workflow)
        self.assertIn("disable_search: true", workflow)
        self.assertNotIn("pip install", workflow)
        self.assertNotIn("pytest", workflow)

    def test_coverage_artifact_is_validated_as_data_before_credentials_are_used(self):
        workflow = (ROOT / ".github/workflows/security-coverage.yml").read_text()
        code = textwrap.dedent(workflow.split("python3 -I - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "coverage"
            directory.mkdir()
            reports = [directory / name for name in ("coverage.xml", "lowest-coverage.xml")]
            payload = b"raise RuntimeError('Report content must never execute')\n"
            for report in reports:
                report.write_bytes(payload)

            def run():
                return subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    env={**security.os.environ, "RUNNER_TEMP": scratch},
                    capture_output=True,
                    check=False,
                ).returncode

            self.assertEqual(run(), 0)
            extra = directory / "injected.py"
            extra.write_bytes(b"raise RuntimeError('must never execute')")
            self.assertNotEqual(run(), 0)
            extra.unlink()
            for report in reports:
                with self.subTest(report=report.name):
                    report.unlink()
                    self.assertNotEqual(run(), 0)
                    report.symlink_to(ROOT / "pyproject.toml")
                    self.assertNotEqual(run(), 0)
                    report.unlink()
                    report.mkdir()
                    self.assertNotEqual(run(), 0)
                    report.rmdir()
                    with report.open("wb") as stream:
                        stream.truncate(20_000_001)
                    self.assertNotEqual(run(), 0)
                    report.write_bytes(payload)
            self.assertEqual(run(), 0)

    def test_technical_outage_recovers_after_cooldown_without_new_ai_budget(self):
        api = api_mock()
        task: security.Json = {"pr": 123, "attempts": 1, "reviews": 2}
        with patch.object(security, "choose_pr_mode", side_effect=security.GitHubError("private payload", 502)):
            for _ in range(3):
                self.assertFalse(security.prepare_task(api, task, ROOT, now=1000))
        self.assertEqual(task["recovery_failures"], 3)
        self.assertIn("HTTP 502", task["blocked"])
        self.assertNotIn("private payload", task["blocked"])
        with patch.object(security, "choose_pr_mode", return_value=True) as healthy:
            self.assertFalse(security.prepare_task(api, task, ROOT, now=1001))
            healthy.assert_not_called()
            self.assertTrue(security.prepare_task(api, task, ROOT, now=4600))
        self.assertEqual((task["attempts"], task["reviews"], task["recovery_failures"]), (1, 2, 0))
        self.assertNotIn("recovery_retry_after", task)
        self.assertEqual(task["blocked"], "")

    def test_successful_recovery_resets_consecutive_technical_failures(self):
        api = api_mock()
        task = {"pr": 123, "recovery_failures": 2}
        with patch.object(security, "choose_pr_mode", return_value=False):
            self.assertFalse(security.prepare_task(api, task, ROOT))
        self.assertEqual(task["recovery_failures"], 0)

    def test_failure_reports_remain_visible_while_other_work_runs(self):
        api = api_mock()
        task = {"signal": {"key": "dependabot-1"}, "blocked": "generate failed; see https://github.com/run/123"}
        security.save_state(api, 12, {"tasks": {"id": task}}, "Running another repair.")
        visible = api.write.call_args.args[1]["body"].split("<!--")[0]
        self.assertIn(task["blocked"], visible)
        self.assertIn("dependabot-1", visible)

    def test_obsolete_history_is_pruned_without_losing_dedup_or_recovery(self):
        today = dt.datetime.now(dt.UTC).date().isoformat()
        signal = {"key": "dependabot-open", "revision": "current"}
        ident = security.fingerprint(signal)
        state: security.Json = {
            "tasks": {
                str(number): {"signal": {"key": f"dependabot-{number}"}, "result": "merged", "pr": number}
                for number in range(400)
            }
        }
        keep = {
            ident: {"signal": signal, "result": "merged", "pr": 1000},
            "pr": {"signal": {"key": "closed-alert-pr"}, "pr": 1001, "result": "pending"},
            "publishing": {"signal": {"key": "closed-alert-publishing"}, "published_head": "head"},
            "dispatch": {"signal": {"key": "closed-alert-dispatch"}, "result": "merged", "post_merge_remaining": ["ci"]},
            "today": {"signal": {"key": "audit", "updated": today}, "result": "no_change"},
        }
        state["tasks"].update(keep)
        state["tasks"].update(
            blocked={"signal": {"key": "closed-alert-blocked"}, "result": "blocked"},
            exhausted={"signal": {"key": "closed-alert-exhausted"}, "attempts": 3},
            old_audit={"signal": {"key": "audit", "updated": "2000-01-01"}, "attempts": 3},
        )
        security.register_signals(state, [signal])
        self.assertEqual(state["tasks"], keep)

    def test_post_merge_dispatch_retries_only_remaining_work(self):
        api = api_mock()
        task = {
            "id": "id",
            "result": "merged",
            "signal": {"key": "dependabot-1"},
            "post_merge_remaining": list(security.POST_MERGE),
        }
        api.write.side_effect = [{}, security.GitHubError("transient", 502)]
        with patch.object(security, "update_task"):
            with self.assertRaises(security.GitHubError):
                security.finish_post_merge(api, task)
            self.assertEqual(task["post_merge_remaining"], security.POST_MERGE[1:])
            self.assertEqual(security.include_active_tasks({"tasks": {"id": task}}, []), [task["signal"]])
            api.write.side_effect = None
            api.write.reset_mock()
            security.finish_post_merge(api, task)
        self.assertEqual(task["post_merge_remaining"], [])
        self.assertEqual(api.write.call_count, len(security.POST_MERGE) - 1)

    def test_successful_merge_is_recovered_even_if_its_response_was_lost(self):
        api = api_mock()
        api.get.return_value = pull_request(state="closed", merged=True, merged_by={"login": security.BOT})
        task = {"id": "id", "pr": 123, "base": "base", "signal": {"key": "dependabot-1"}}
        with patch.object(security, "update_task"):
            self.assertFalse(security.choose_pr_mode(api, task, ROOT))
        self.assertEqual(task["result"], "merged")
        self.assertEqual(task["post_merge_remaining"], [])
        self.assertEqual(api.write.call_count, len(security.POST_MERGE))

    def test_failure_report_contains_only_phase_and_run_link(self):
        args = security.argparse.Namespace(command="report", phase="publish")
        task = {"id": "id", "run_id": "567"}
        with (
            patch.object(security, "GitHub", return_value=api_mock()),
            patch.object(security, "update_task") as update,
            patch.dict(security.os.environ, {"GITHUB_REPOSITORY": REPO}),
        ):
            security.run_task_command(args, task)
        message = update.call_args.args[2]["blocked"]
        self.assertEqual(message, f"publish failed; see https://github.com/{REPO}/actions/runs/567")

    def test_bad_publication_does_not_starve_other_findings(self):
        api = api_mock()
        signals = [{"key": "dependabot-1", "severity": "critical"}, {"key": "dependabot-2", "severity": "high"}]
        bad = security.fingerprint(signals[0])
        state: security.Json = {
            "tasks": {bad: {"signal": signals[0], "attempts": 1, "reviews": 0, "published_head": "reserved"}}
        }
        with (
            patch.object(security, "load_state", return_value=(12, state)),
            patch.object(security, "save_state", return_value=12),
            patch.object(security, "base_sha", return_value="base"),
            patch.object(security, "restore_publication", side_effect=ValueError("occupied")),
        ):
            task = security.plan(api, ROOT, signals)
        self.assertEqual(task["signal"]["key"], "dependabot-2")
        self.assertEqual(state["tasks"][bad]["recovery_failures"], 1)
        self.assertIn("recovery failed", state["tasks"][bad]["blocked"])

    def test_meaningful_revision_reuses_existing_pr_and_replenishes_budget(self):
        old = {"key": "dependabot-1", "revision": "old"}
        new = dict(old, revision="new")
        state: security.Json = {"tasks": {security.fingerprint(old): {"signal": old, "pr": 123, "attempts": 3, "reviews": 2}}}
        security.register_signals(state, [new])
        self.assertEqual(len(state["tasks"]), 1)
        entry = state["tasks"][security.fingerprint(new)]
        self.assertEqual((entry["pr"], entry["attempts"], entry["reviews"]), (123, 0, 0))

    def test_daily_audit_is_due_without_the_daily_cron_event(self):
        self.assertFalse(security.audit_due(dt.datetime(2026, 10, 30, 3, 16, tzinfo=dt.UTC)))
        self.assertTrue(security.audit_due(dt.datetime(2026, 10, 30, 3, 17, tzinfo=dt.UTC)))
        self.assertTrue(security.audit_due(dt.datetime(2026, 10, 30, 23, 59, tzinfo=dt.UTC)))

    def test_long_unresolved_threads_fail_closed(self):
        api = api_mock()
        api.request.return_value = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [
                                {
                                    "id": "long",
                                    "isResolved": False,
                                    "comments": {"nodes": [], "pageInfo": {"hasNextPage": True}},
                                }
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "comments exceed"):
            security.review_threads(api, 123)

    def test_disable_switch_guards_all_ai_and_mutation_jobs(self):
        workflow = (ROOT / ".github/workflows/security-remediation.yml").read_text()
        for name in ["generate", "publish", "review", "merge"]:
            # Job headers have four-space child indentation, unlike the next job.
            header = workflow.split("\n  " + name + ":\n", 1)[1].split("runs-on:", 1)[0]
            self.assertIn("vars.SECURITY_AUTOMATION_ENABLED != 'false'", header)


if __name__ == "__main__":
    unittest.main()
