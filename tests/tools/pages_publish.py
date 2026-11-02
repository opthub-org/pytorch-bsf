"""Offline trust-boundary, ordering and bare-Git regressions for Pages publishing."""

import base64
import copy
import hashlib
import io
import json
import stat
import subprocess
import tempfile
import unittest
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools import pages_publish as pages

SHA = "a" * 40
OTHER_SHA = "b" * 40
POLICY = b"trusted read-only builder\n"


def manifest(ref="refs/heads/master", files=None):
    files = {"index.html": b"current"} if files is None else files
    return {
        "schema": 1,
        "repository": pages.REPOSITORY,
        "run_id": 100,
        "run_attempt": 1,
        "source_ref": ref,
        "source_sha": SHA,
        "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()},
    }


def archive(record=None, files=None, extra=None):
    files = {"index.html": b"current"} if files is None else files
    record = manifest(files=files) if record is None else record
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("manifest.json", json.dumps(record))
        for name, data in files.items():
            handle.writestr("site/" + name, data)
        for name, data in extra or []:
            handle.writestr(name, data)
    return output.getvalue()


def run_record() -> dict[str, Any]:
    repo = {"id": 7, "full_name": pages.REPOSITORY}
    return {
        "id": 100,
        "run_attempt": 1,
        "repository": repo,
        "head_repository": repo,
        "workflow_id": 20,
        "path": pages.WORKFLOW_PATH,
        "name": pages.WORKFLOW_NAME,
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "head_sha": SHA,
        "head_branch": "master",
        "run_number": 3,
        "run_started_at": "2026-11-02T01:00:00Z",
    }


class API:
    def __init__(self, raw=None):
        self.run = run_record()
        self.repo = {"id": 7, "full_name": pages.REPOSITORY, "default_branch": "master"}
        self.raw = archive() if raw is None else raw
        self.definition = POLICY
        self.artifact: dict[str, Any] = {
            "id": 50,
            "name": "pages-100-1",
            "expired": False,
            "size_in_bytes": len(self.raw),
            "digest": "sha256:" + hashlib.sha256(self.raw).hexdigest(),
            "created_at": "2026-11-02T01:00:10Z",
            "workflow_run": {"id": 100, "repository_id": 7, "head_repository_id": 7, "head_sha": SHA, "head_branch": "master"},
        }
        self.refs: list[dict[str, Any]] = [{"ref": "refs/heads/master", "object": {"type": "commit", "sha": SHA}}]
        self.calls = []
        self.artifacts = [self.artifact]

    def request(self, path, method="GET", data=None, **kwargs):
        self.calls.append((path, method, data))
        if method == "POST" and path == "check-runs":
            return {"id": 999}
        if method == "PATCH":
            return {}
        static = {"": self.repo, "actions/artifacts/50": self.artifact, "actions/artifacts/50/zip": self.raw}
        if path in static:
            return static[path]
        if path.startswith("actions/runs/"):
            return copy.deepcopy(self.run)
        if path.startswith("actions/workflows/"):
            return {"id": 20, "path": pages.WORKFLOW_PATH}
        if path.startswith("contents/"):
            return {
                "type": "file",
                "encoding": "base64",
                "size": len(self.definition),
                "content": base64.b64encode(self.definition).decode(),
            }
        if path.startswith("git/matching-refs/"):
            return [r for r in self.refs if r["ref"].startswith("refs/" + path.rsplit("/", 2)[-2] + "/")]
        if path.startswith("git/tags/"):
            return {"object": {"type": "commit", "sha": SHA}}
        raise AssertionError(path)

    def pages(self, path, key=None):
        if path.endswith("/artifacts"):
            return self.artifacts
        return [r for r in self.refs if r["ref"].startswith("refs/" + path.rsplit("/", 2)[-2] + "/")]


def identity(value):
    return value


class MemoryPages:
    def __init__(self, old=None, before_push=None):
        self.old = old or []
        self.before_push = before_push
        self.pushed = False
        self.prepared = False

    def ledgers(self):
        return self.old

    def prepare(self, record, files, ledger):
        self.prepared = True
        self.ledger = ledger
        if self.before_push:
            self.before_push()
        return OTHER_SHA

    def push(self, commit):
        self.pushed = True


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.policy = patch.object(pages, "trusted_workflow", return_value=POLICY)
        self.policy.start()
        self.addCleanup(self.policy.stop)

    def test_success_creates_separate_check_at_producer_sha(self):
        api, target = API(), MemoryPages()
        result = pages.publish(api, {"workflow_run": copy.deepcopy(api.run)}, lambda: target)
        self.assertTrue(target.pushed)
        self.assertEqual(result["artifact_id"], 50)
        checks = [data for path, method, data in api.calls if path == "check-runs" and method == "POST"]
        self.assertEqual((checks[0]["name"], checks[0]["head_sha"]), ("GHPages publish", SHA))
        self.assertEqual(api.calls[-1][2]["conclusion"], "success")

    def test_canonical_repository_workflow_attempt_and_head_are_authenticated(self):
        mutations = [
            lambda r: r.update(run_attempt=2),
            lambda r: r.update(workflow_id=21),
            lambda r: r.update(head_sha=OTHER_SHA),
            lambda r: r.update(event="pull_request"),
            lambda r: r.update(path=".github/workflows/other.yml"),
            lambda r: r.update(conclusion="failure"),
            lambda r: r.update(head_repository={"id": 8, "full_name": "attacker/fork"}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                api = API()
                event = {"workflow_run": copy.deepcopy(api.run)}
                mutate(api.run)
                with self.assertRaises(ValueError):
                    pages.publish(api, event, lambda: self.fail("Unauthorized publication"))
                self.assertFalse(any(method != "GET" for _, method, _ in api.calls))

    def test_candidate_workflow_must_match_trusted_read_only_policy(self):
        api = API()
        api.definition = b"permissions: write-all\n"
        with self.assertRaisesRegex(ValueError, "trusted read-only"):
            pages.authenticate_run(api, {"workflow_run": api.run})

    def test_artifact_identity_digest_expiry_and_attempt_time_are_checked(self):
        changes: list[dict[str, Any]] = [
            {"id": 51},
            {"expired": True},
            {"digest": "sha256:" + "0" * 64},
            {"created_at": "2026-11-01T01:00:00Z"},
            {"size_in_bytes": pages.MAX_ARCHIVE_BYTES + 1},
            {"workflow_run": {"id": 101}},
        ]
        for change in changes:
            with self.subTest(change=change):
                api = API()
                api.artifact = {**api.artifact, **change}
                with self.assertRaises((ValueError, KeyError, AssertionError)):
                    pages.load_artifact(api, api.run, api.repo)

    def test_ambiguous_or_missing_artifact_is_rejected(self):
        for count in (0, 2):
            api = API()
            api.artifacts = [api.artifact] * count
            with self.assertRaisesRegex(ValueError, "exactly one"):
                pages.load_artifact(api, api.run, api.repo)

    def test_deleted_force_pushed_and_overlapping_refs_fail_before_push(self):
        mutations = [
            lambda api: api.refs.clear(),
            lambda api: api.refs[0]["object"].update(sha=OTHER_SHA),
            lambda api: api.refs.append({"ref": "refs/heads/master/child", "object": {"type": "commit", "sha": SHA}}),
        ]
        for mutate in mutations:
            api, target = API(), MemoryPages()
            event = {"workflow_run": copy.deepcopy(api.run)}
            target.before_push = partial(mutate, api)
            with self.assertRaises(ValueError):
                pages.publish(api, event, partial(identity, target))
            self.assertFalse(target.pushed)
            self.assertEqual(api.calls[-1][2]["conclusion"], "failure")

    def test_reverse_completion_same_sha_cannot_overwrite_newer_generation(self):
        api = API()
        old = {"source_ref": "refs/heads/master", "run_number": 4, "run_id": 101, "run_attempt": 1}
        target = MemoryPages([old])
        with self.assertRaisesRegex(ValueError, "newer generation"):
            pages.publish(api, {"workflow_run": api.run}, lambda: target)
        self.assertFalse(target.prepared)

    def test_retained_deleted_ref_namespace_is_preserved(self):
        api = API()
        target = MemoryPages([{"source_ref": "refs/heads/master/archived"}])
        with self.assertRaisesRegex(ValueError, "Retained publication"):
            pages.publish(api, {"workflow_run": api.run}, lambda: target)
        self.assertFalse(target.prepared)

    def test_moving_ref_after_push_marks_check_failed(self):
        api = API()
        target = MemoryPages()

        def push(commit):
            target.pushed = True
            api.refs.clear()

        with patch.object(target, "push", side_effect=push), self.assertRaises(ValueError):
            pages.publish(api, {"workflow_run": api.run}, lambda: target)
        self.assertTrue(target.pushed)
        self.assertEqual(api.calls[-1][2]["conclusion"], "failure")

    def test_annotated_release_tag_peels_to_source_commit(self):
        api = API()
        api.refs = [{"ref": "refs/tags/v1.2.3", "object": {"type": "tag", "sha": OTHER_SHA}}]
        pages.live_refs(api, manifest("refs/tags/v1.2.3"))
        api.refs.append({"ref": "refs/heads/v1.2.3/preview", "object": {"type": "commit", "sha": SHA}})
        with self.assertRaisesRegex(ValueError, "overlap"):
            pages.live_refs(api, manifest("refs/tags/v1.2.3"))


class ArchiveTests(unittest.TestCase):
    def test_valid_regular_files_and_optional_directories(self):
        raw = archive(extra=[("site/", b""), ("site/_static/", b"")])
        _, files = pages.validate_archive(raw)
        self.assertEqual(files, {"index.html": b"current"})

    def test_unsafe_names_duplicate_paths_and_unexpected_namespace(self):
        for name in (
            "../bad",
            "/bad",
            "site/../bad",
            "site/.git/config",
            "site/.GIT/config",
            "site/back\\slash",
            "other",
            "site/index.html",
            "site/nul\0suffix",
        ):
            with self.subTest(name=name), self.assertRaises((ValueError, zipfile.BadZipFile)):
                pages.validate_archive(archive(extra=[(name, b"evil")]))

    def test_symlinks_and_special_files_are_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR):
            output = io.BytesIO(archive())
            with zipfile.ZipFile(output, "a") as handle:
                entry = zipfile.ZipInfo("site/link")
                entry.create_system = 3
                entry.external_attr = (mode | 0o777) << 16
                handle.writestr(entry, b"../outside")
            with self.assertRaisesRegex(ValueError, "special"):
                pages.validate_archive(output.getvalue())

    def test_tampered_files_inventory_and_closed_schema(self):
        for change in (
            lambda r: r["files"]["index.html"].update(size=True),
            lambda r: r["files"]["index.html"].update(sha256="0" * 64),
            lambda r: r.update(extra="ignored"),
            lambda r: r.update(files={}),
        ):
            record = manifest()
            change(record)
            with self.assertRaises(ValueError):
                pages.validate_archive(archive(record=record))

    def test_file_prefix_collision_is_rejected_even_when_nonadjacent(self):
        files = {"index.html": b"x", "a": b"x", "a-b": b"x", "a/child": b"x"}
        with self.assertRaisesRegex(ValueError, "collision"):
            pages.validate_archive(archive(files=files))

    def test_bounded_compressed_expanded_file_and_entry_count(self):
        cases = [("MAX_ARCHIVE_BYTES", 10), ("MAX_SITE_BYTES", 10), ("MAX_FILE_BYTES", 2), ("MAX_FILES", 1)]
        for setting, limit in cases:
            with self.subTest(setting=setting), patch.object(pages, setting, limit), self.assertRaises(ValueError):
                pages.validate_archive(archive(extra=[("site/more", b"many")]))

    def test_destination_policy_preserves_common_branch_and_release_urls(self):
        self.assertEqual(pages.destination("refs/heads/fix/123-preview"), "fix/123-preview")
        self.assertEqual(pages.destination("refs/tags/v1.2.3"), "v1.2.3")
        for ref in (
            "refs/heads/CNAME",
            "refs/heads/index.html",
            "refs/heads/gh-pages",
            "refs/heads/release-please-test",
            "refs/tags/foo",
            "refs/pull/1/merge",
            "refs/heads/a..b",
            "refs/heads/a.lock",
            "refs/heads/.hidden",
        ):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                pages.destination(ref)

    def test_redirect_removes_authorization_header(self):
        request = urllib.request.Request("https://api.github.com/example", headers={"Authorization": "Bearer private"})
        result = pages.StripRedirectCredentials().redirect_request(
            request, None, 302, "Found", {}, "https://storage.example/archive"
        )
        assert result is not None
        self.assertIsNone(result.get_header("Authorization"))
        with self.assertRaises(ValueError):
            pages.StripRedirectCredentials().redirect_request(
                request, None, 302, "Found", {}, "http://storage.example/archive"
            )


class BareGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.source = self.root / "source"
        self.git("init", "--bare", "--quiet", str(self.remote))
        self.git("init", "--quiet", str(self.source))
        self.git("-C", str(self.source), "config", "user.name", "Test")
        self.git("-C", str(self.source), "config", "user.email", "test@example.invalid")
        for name in (
            "master/index.html",
            "master/legacy.html",
            "v1.2.3/index.html",
            "fix/archived/index.html",
            "index.html",
            ".nojekyll",
        ):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("legacy")
        self.git("-C", str(self.source), "add", ".")
        self.git("-C", str(self.source), "commit", "-qm", "baseline")
        self.git("-C", str(self.source), "push", "--quiet", str(self.remote), "HEAD:refs/heads/gh-pages")

    def git(self, *args):
        return subprocess.run(["git", *args], check=True, capture_output=True).stdout

    def publisher(self, name):
        remote = str(self.remote)

        class LocalPages(pages.GitPages):
            def git(self, *args, **kwargs):
                args = tuple(remote if value == f"https://github.com/{pages.REPOSITORY}.git" else value for value in args)
                return super().git(*args, **kwargs)

        return LocalPages(self.root / name, "not-a-real-token")

    def record(self, files, number=3):
        return {
            "source_ref": "refs/heads/master",
            "source_sha": SHA,
            "run_id": 100,
            "run_attempt": 1,
            "run_number": number,
            "artifact_id": 50,
            "digest": "sha256:" + "0" * 64,
            "files": sorted(files),
        }

    def show(self, name):
        return self.git("--git-dir", str(self.remote), "show", "refs/heads/gh-pages:" + name)

    def test_only_owned_files_change_and_legacy_previews_are_retained(self):
        first = self.publisher("first.git")
        files = {"index.html": b"new", "owned.html": b"owned"}
        first.push(first.prepare(manifest(files=files), files, self.record(files)))
        self.assertEqual(self.show("master/index.html"), b"new")
        self.assertEqual(self.show("master/legacy.html"), b"legacy")
        self.assertEqual(self.show("v1.2.3/index.html"), b"legacy")
        self.assertEqual(self.show("fix/archived/index.html"), b"legacy")
        self.assertEqual(self.show("index.html"), b"legacy")
        second = self.publisher("second.git")
        files = {"index.html": b"newer"}
        second.push(second.prepare(manifest(files=files), files, self.record(files, 4)))
        self.assertEqual(self.show("master/index.html"), b"newer")
        with self.assertRaises(subprocess.CalledProcessError):
            self.show("master/owned.html")
        self.assertEqual(self.show("master/legacy.html"), b"legacy")
        self.assertEqual(second.ledgers()[0]["files"], ["index.html", "owned.html"])

    def test_concurrent_write_rejects_stale_parent_without_losing_newer_pages(self):
        first, second = self.publisher("one.git"), self.publisher("two.git")
        newer = {"index.html": b"newer"}
        second.push(second.prepare(manifest(files=newer), newer, self.record(newer, 4)))
        stale = {"index.html": b"stale"}
        with self.assertRaisesRegex(RuntimeError, "branch may have changed"):
            first.push(first.prepare(manifest(files=stale), stale, self.record(stale)))
        self.assertEqual(self.show("master/index.html"), b"newer")

    def test_existing_root_file_cannot_be_replaced_by_branch_subdirectory(self):
        target = self.publisher("blocked.git")
        with self.assertRaises(ValueError):
            target.prepare(manifest("refs/heads/index.html/branch"), {"index.html": b"x"}, self.record({"index.html": b"x"}))

    def test_remote_rewind_to_ancestor_also_fails_expected_parent_lease(self):
        original = self.git("--git-dir", str(self.remote), "rev-parse", "refs/heads/gh-pages").decode().strip()
        first = self.publisher("first.git")
        files = {"index.html": b"current"}
        first.push(first.prepare(manifest(files=files), files, self.record(files)))
        pending = self.publisher("pending.git")
        commit = pending.prepare(manifest(files=files), files, self.record(files, 4))
        self.git("--git-dir", str(self.remote), "update-ref", "refs/heads/gh-pages", original)
        with self.assertRaises(RuntimeError):
            pending.push(commit)
        self.assertEqual(self.show("master/index.html"), b"legacy")


if __name__ == "__main__":
    unittest.main()
