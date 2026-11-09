"""Offline trust-boundary, ordering and bare-Git regressions for Pages publishing."""

import base64
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import stat
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

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

    def test_two_step_pin_update_publishes_only_exact_matches(self):
        old = b"- uses: actions/checkout@" + b"1" * 40 + b" # v1\n"
        new = b"- uses: actions/checkout@" + b"2" * 40 + b" # v2\n"
        # (stage, default-branch template, producer YAML at the pushed head, publication accepted)
        stages = [
            ("before the update and at the template PR head", old, old, True),
            ("workflow-only or combined PR head before the template lands", old, new, False),
            ("master or unsynced branch after the template lands", new, old, False),
            ("workflow PR head and master after both PRs land", new, new, True),
        ]
        for stage, template, producer, accepted in stages:
            with self.subTest(stage=stage), patch.object(pages, "trusted_workflow", return_value=template):
                api, target = API(), MemoryPages()
                api.definition = producer
                event = {"workflow_run": copy.deepcopy(api.run)}
                if accepted:
                    pages.publish(api, event, partial(identity, target))
                    self.assertTrue(target.pushed)
                    self.assertEqual(api.calls[-1][2]["conclusion"], "success")
                else:
                    with self.assertRaisesRegex(ValueError, "trusted read-only"):
                        pages.publish(api, event, partial(identity, target))
                    self.assertFalse(target.prepared)
                    # Rejected before any check is written: the head has no publish evidence.
                    self.assertFalse(any(method != "GET" for _, method, _ in api.calls))

    def _authenticate_paths(self, paths, workflow_path=pages.WORKFLOW_PATH):
        api = API()
        event = {"workflow_run": copy.deepcopy(api.run)}
        event["workflow_run"]["path"] = paths[0]
        original_request = api.request

        def request(path, *args, **kwargs):
            response = original_request(path, *args, **kwargs)
            if path == "actions/runs/100/attempts/1":
                response["path"] = paths[1]
            elif path == "actions/runs/100":
                response["path"] = paths[2]
            elif path.startswith("actions/workflows/"):
                response["path"] = workflow_path
            return response

        with patch.object(api, "request", side_effect=request):
            return pages.authenticate_run(api, event)

    def test_run_paths_accept_bare_ref_qualified_and_mixed_service_formats(self):
        bare = pages.WORKFLOW_PATH
        for paths in (
            (bare, bare, bare),
            (bare + "@main", bare + "@main", bare + "@main"),
            (bare + "@main", bare, bare + "@refs/heads/master"),
            (bare, bare + "@topic/branch", bare),
        ):
            with self.subTest(paths=paths):
                run, _ = self._authenticate_paths(paths)
                self.assertEqual(run["head_branch"], "master")
                self.assertEqual(run["head_sha"], SHA)

    def test_every_run_path_rejects_other_workflow_empty_qualifier_and_nonstrings(self):
        invalid_paths: tuple[Any, ...] = (".github/workflows/other.yml@main", pages.WORKFLOW_PATH + "@", None, 123, {})
        for invalid in invalid_paths:
            for index in range(3):
                paths: list[Any] = [pages.WORKFLOW_PATH] * 3
                paths[index] = invalid
                with self.subTest(invalid=invalid, index=index), self.assertRaises(ValueError):
                    self._authenticate_paths(paths)

    def test_workflow_metadata_still_requires_the_exact_bare_path(self):
        with self.assertRaisesRegex(ValueError, "Invalid producer workflow"):
            self._authenticate_paths([pages.WORKFLOW_PATH] * 3, workflow_path=pages.WORKFLOW_PATH + "@main")

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
    def test_transport_uses_repository_root_without_a_trailing_slash(self):
        client = pages.GitHub("not-a-real-token")
        with patch.object(client.opener, "open", side_effect=[io.BytesIO(b'{"id":7}'), io.BytesIO(b'{"id":100}')]) as opened:
            self.assertEqual(client.request(""), {"id": 7})
            self.assertEqual(client.request("actions/runs/100"), {"id": 100})
        urls = [call.args[0].full_url for call in opened.call_args_list]
        self.assertEqual(
            urls,
            [
                f"https://api.github.com/repos/{pages.REPOSITORY}",
                f"https://api.github.com/repos/{pages.REPOSITORY}/actions/runs/100",
            ],
        )

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


class LegacyOwnershipTests(unittest.TestCase):
    def test_reviewed_census_has_valid_nonoverlapping_full_ref_owners(self):
        owners = pages.trusted_legacy_owners()
        self.assertEqual(owners["master"], "refs/heads/master")
        self.assertEqual(owners["v0.24.0"], "refs/tags/v0.24.0")

    def test_malformed_policy_and_colliding_owners_fail_closed(self):
        baseline = {"schema": 1, "gh_pages_sha": SHA, "owners": {"master": "refs/heads/master"}, "reserved_owners": {}}
        changes: list[dict[str, Any]] = [
            {"schema": True},
            {"gh_pages_sha": "invalid"},
            {"owners": {}},
            {"owners": {"master": "refs/heads/other"}},
            {"reserved_owners": {"master": "refs/heads/master"}},
            {"reserved_owners": {"master/reserved": "refs/heads/master/reserved"}},
            {"owners": {"v1.2.3": "refs/tags/v1.2.3", "v1.2.3/child": "refs/heads/v1.2.3/child"}},
        ]
        for change in changes:
            raw = json.dumps({**baseline, **change}).encode()
            with (
                self.subTest(change=change),
                patch.object(Path, "read_bytes", return_value=raw),
                self.assertRaises(ValueError),
            ):
                pages.trusted_legacy_owners()

    def test_reserved_migration_namespace_is_distinct_from_observed_census(self):
        policy: dict[str, Any] = {
            "schema": 1,
            "gh_pages_sha": SHA,
            "owners": {"master": "refs/heads/master"},
            "reserved_owners": {"ci/migration": "refs/heads/ci/migration"},
        }
        with patch.object(Path, "read_bytes", return_value=json.dumps(policy).encode()):
            owners = pages.trusted_legacy_owners()
        self.assertNotIn("ci/migration", policy["owners"])
        self.assertEqual(owners["ci/migration"], "refs/heads/ci/migration")
        pages.check_namespace("refs/heads/ci/migration", ["ci/migration/index.html"], [], owners)
        with self.assertRaisesRegex(ValueError, "Trusted legacy"):
            pages.check_namespace("refs/heads/ci", ["ci/migration/index.html"], [], owners)

    def test_unknown_existing_prefix_requires_trusted_owner_or_same_ref_ledger(self):
        with self.assertRaisesRegex(ValueError, "no trusted owner"):
            pages.check_namespace("refs/heads/new", ["new/index.html"], [], {})
        pages.check_namespace("refs/heads/new", ["new/index.html"], [{"source_ref": "refs/heads/new"}], {})

    def test_known_tag_owner_cannot_be_replaced_by_same_named_branch(self):
        with self.assertRaisesRegex(ValueError, "Trusted legacy"):
            pages.check_namespace("refs/heads/v1.2.3", [], [], {"v1.2.3": "refs/tags/v1.2.3"})

    def test_unknown_nested_or_ancestor_sphinx_root_blocks_owned_publication(self):
        for marker in ("objects.inv", ".buildinfo"):
            for root in ("master/archived", "parent"):
                ref = "refs/heads/master" if root.startswith("master/") else "refs/heads/parent/child"
                owners = {pages.destination(ref): ref}
                with self.subTest(marker=marker, root=root), self.assertRaisesRegex(ValueError, "Unregistered"):
                    pages.check_namespace(ref, [root + "/" + marker], [], owners)
        pages.check_namespace(
            "refs/heads/master", ["master/objects.inv", "other/.buildinfo"], [], {"master": "refs/heads/master"}
        )


class BareGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        owner_policy = patch.object(
            pages,
            "trusted_legacy_owners",
            return_value={
                "master": "refs/heads/master",
                "v1.2.3": "refs/tags/v1.2.3",
                "fix/archived": "refs/heads/fix/archived",
            },
        )
        owner_policy.start()
        self.addCleanup(owner_policy.stop)
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

    def test_parent_branch_cannot_overwrite_archived_preview_exact_file(self):
        target = self.publisher("parent.git")
        files = {"index.html": b"parent", "archived/index.html": b"overwritten"}
        record = self.record(files)
        record["source_ref"] = "refs/heads/fix"
        with self.assertRaisesRegex(ValueError, "Trusted legacy"):
            target.prepare(manifest("refs/heads/fix", files), files, record)
        self.assertEqual(self.show("fix/archived/index.html"), b"legacy")
        self.assertEqual(target.git("write-tree"), target.git("rev-parse", target.base + "^{tree}"))

    def test_sphinx_root_added_after_census_blocks_owned_parent_before_mutation(self):
        marker = self.source / "master/late-preview/objects.inv"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"inventory")
        self.git("-C", str(self.source), "add", ".")
        self.git("-C", str(self.source), "commit", "-qm", "Late legacy preview")
        self.git("-C", str(self.source), "push", "--quiet", str(self.remote), "HEAD:refs/heads/gh-pages")
        target = self.publisher("late.git")
        files = {"index.html": b"new", "late-preview/objects.inv": b"overwritten"}
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            target.prepare(manifest(files=files), files, self.record(files))
        self.assertEqual(self.show("master/late-preview/objects.inv"), b"inventory")
        self.assertEqual(self.show("master/index.html"), b"legacy")


NOW = dt.datetime(2026, 11, 8, 12, 0, tzinfo=dt.UTC)
ROOT = Path(__file__).resolve().parents[2]


class SweepAPI(API):
    """Serve the producer run listing and the commit's check runs to the sweep."""

    def __init__(self):
        super().__init__()
        self.run.update(event="workflow_dispatch", triggering_actor={"login": pages.BOT})
        self.runs = [self.run]
        self.checks: list[dict[str, Any]] = []
        self.listed: list[str] = []

    def request(self, path, method="GET", data=None, **kwargs):
        if method == "POST" and path == "check-runs":
            # Record the check as GitHub would list it on the commit.
            self.checks.append({"head_sha": data["head_sha"], "external_id": data["external_id"]})
        if path.startswith("actions/runs/"):
            ident = int(path.split("/")[2])
            return copy.deepcopy(next(run for run in self.runs if run["id"] == ident))
        return super().request(path, method, data, **kwargs)

    def pages(self, path, key=None):
        self.listed.append(path)
        if key == "workflow_runs":
            return copy.deepcopy(self.runs)
        if key == "check_runs":
            return [check for check in self.checks if path.startswith(f"commits/{check['head_sha']}/")]
        return super().pages(path, key)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.policy = patch.object(pages, "trusted_workflow", return_value=POLICY)
        self.policy.start()
        self.addCleanup(self.policy.stop)

    def sweep(self, api: SweepAPI, factory: Callable[[], Any] | None = None) -> list[dict[str, Any]]:
        return pages.sweep(api, NOW, factory or MemoryPages)

    def test_bot_dispatched_run_without_a_publication_is_published(self):
        api = SweepAPI()
        records = self.sweep(api)
        self.assertEqual([(r["source_ref"], r["run_id"]) for r in records], [("refs/heads/master", 100)])
        checks = [data for path, method, data in api.calls if path == "check-runs" and method == "POST"]
        self.assertEqual(
            [(c["name"], c["head_sha"], c["external_id"]) for c in checks], [(pages.CHECK_NAME, SHA, "pages-100-1")]
        )

    def test_listing_is_bounded_to_successful_runs_within_the_artifact_lifetime(self):
        api = SweepAPI()
        self.sweep(api)
        listing = urllib.parse.urlsplit(api.listed[0])
        self.assertEqual(listing.path, "actions/workflows/sphinx-pages-build.yml/runs")
        query = urllib.parse.parse_qs(listing.query)
        self.assertEqual(query, {"status": ["success"], "created": [">=2026-11-02T12:00:00Z"]})
        checks = urllib.parse.parse_qs(urllib.parse.urlsplit(api.listed[1]).query)
        self.assertEqual(checks, {"check_name": [pages.CHECK_NAME], "filter": ["all"]})

    def test_runs_the_sweep_must_not_publish_are_left_alone(self):
        changes: list[dict[str, Any]] = [
            {"triggering_actor": {"login": "NaokiHamada"}},
            {"triggering_actor": None},
            {"event": "push"},
            {"conclusion": "failure"},
            {"status": "in_progress"},
            {"head_branch": None},
            {"run_number": True},
            {"run_attempt": 0},
        ]
        for change in changes:
            with self.subTest(change=change):
                api = SweepAPI()
                api.run.update(change)
                self.assertEqual(self.sweep(api), [])
                # The listing filter rejects the run before any authentication.
                self.assertFalse(any(path.startswith(("actions/runs/", "git/")) for path, _, _ in api.calls))

    def test_newer_run_that_did_not_succeed_does_not_hide_the_older_success(self):
        for change in ({"conclusion": "failure"}, {"status": "in_progress"}, {"event": "schedule"}):
            with self.subTest(change=change):
                api = SweepAPI()
                newer = copy.deepcopy(api.run)
                newer.update(id=110, run_number=4, **change)
                api.runs.append(newer)
                self.assertEqual([r["run_id"] for r in self.sweep(api)], [100])

    def test_newer_push_run_of_the_same_commit_supersedes_the_bot_run(self):
        api = SweepAPI()
        newer = copy.deepcopy(api.run)
        newer.update(id=110, run_number=4, event="push", triggering_actor={"login": "NaokiHamada"})
        api.runs.append(newer)
        self.assertEqual(self.sweep(api), [])
        self.assertFalse(any(path.startswith(("actions/runs/", "git/")) for path, _, _ in api.calls))

    def test_handled_attempt_is_never_published_again_even_after_a_failure(self):
        for conclusion in ("success", "failure"):
            with self.subTest(conclusion=conclusion):
                api = SweepAPI()
                api.checks.append({"head_sha": SHA, "external_id": "pages-100-1", "conclusion": conclusion})
                self.assertEqual(self.sweep(api), [])

    def test_check_of_another_run_or_attempt_does_not_count(self):
        api = SweepAPI()
        api.checks.append({"head_sha": SHA, "external_id": "pages-100-2"})
        api.checks.append({"head_sha": SHA, "external_id": "pages-99-1"})
        api.checks.append({"head_sha": SHA})
        self.assertEqual(len(self.sweep(api)), 1)

    def test_moved_or_deleted_ref_is_left_to_its_newer_build(self):
        for refs in ([{"ref": "refs/heads/master", "object": {"type": "commit", "sha": OTHER_SHA}}], []):
            with self.subTest(refs=refs):
                api = SweepAPI()
                api.refs = refs
                self.assertEqual(self.sweep(api), [])
                self.assertFalse(any(method == "POST" for _, method, _ in api.calls))

    def test_branch_and_tag_of_the_same_name_are_ambiguous(self):
        api = SweepAPI()
        api.refs.append({"ref": "refs/tags/master", "object": {"type": "commit", "sha": SHA}})
        self.assertEqual(self.sweep(api), [])

    def test_only_the_newest_run_of_each_ref_is_considered(self):
        api = SweepAPI()
        older = copy.deepcopy(api.run)
        older.update(id=90, run_number=2)
        api.runs.insert(0, older)
        api.runs.append({**copy.deepcopy(older), "id": 80, "run_number": 1})
        records = self.sweep(api)
        self.assertEqual([r["run_id"] for r in records], [100])

    def test_newer_run_that_a_user_dispatched_supersedes_the_bot_run(self):
        api = SweepAPI()
        newer = copy.deepcopy(api.run)
        newer.update(id=110, run_number=4, triggering_actor={"login": "NaokiHamada"})
        api.runs.append(newer)
        self.assertEqual(self.sweep(api), [])

    def test_release_tag_dispatched_by_release_please_is_published_at_its_tag(self):
        api = SweepAPI()
        api.run["head_branch"] = "v0.25.1"
        api.artifact["workflow_run"]["head_branch"] = "v0.25.1"
        api.raw = archive(manifest("refs/tags/v0.25.1"))
        api.artifact.update(size_in_bytes=len(api.raw), digest="sha256:" + hashlib.sha256(api.raw).hexdigest())
        api.refs.append({"ref": "refs/tags/v0.25.1", "object": {"type": "tag", "sha": "c" * 40}})
        records = self.sweep(api)
        self.assertEqual([r["source_ref"] for r in records], ["refs/tags/v0.25.1"])

    def test_one_failed_publication_does_not_stop_the_others(self):
        api = SweepAPI()
        second = copy.deepcopy(api.run)
        second.update(id=101, head_branch="fix")
        api.runs.append(second)
        api.refs.append({"ref": "refs/heads/fix", "object": {"type": "commit", "sha": SHA}})
        # "fix" comes first in name order and fails: its artifact belongs to master's run.
        published = []

        def factory():
            target = MemoryPages()
            published.append(target)
            return target

        with self.assertRaisesRegex(RuntimeError, "1 documentation publication"):
            self.sweep(api, factory)
        self.assertEqual([target.pushed for target in published], [True])
        failures = [data for path, method, data in api.calls if method == "PATCH" and data["conclusion"] == "failure"]
        self.assertEqual(len(failures), 1)

    def test_malformed_run_record_does_not_stop_the_others(self):
        api = SweepAPI()
        broken = copy.deepcopy(api.run)
        del broken["run_number"]
        broken.update(id=101, head_branch="fix")
        api.runs.insert(0, broken)
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual([r["run_id"] for r in self.sweep(api)], [100])
        self.assertIn("Skipping a malformed producer run record: 101", log.getvalue())

    def test_bad_ref_of_one_run_does_not_stop_the_others(self):
        api = SweepAPI()
        other = copy.deepcopy(api.run)
        other.update(id=101, head_branch="aaa")
        api.runs.append(other)
        api.refs.append({"ref": "refs/heads/aaa", "object": {"type": "tree", "sha": SHA}})
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual([r["run_id"] for r in self.sweep(api)], [100])
        self.assertIn("Skipping producer run 101", log.getvalue())

    def test_network_error_for_one_run_is_a_failure_that_does_not_stop_the_others(self):
        api = SweepAPI()
        other = copy.deepcopy(api.run)
        other.update(id=101, head_branch="aaa")
        api.runs.append(other)
        api.refs.append({"ref": "refs/heads/aaa", "object": {"type": "commit", "sha": SHA}})
        original = api.request

        def request(path, method="GET", data=None, **kwargs):
            if path.startswith("actions/runs/101/"):
                raise urllib.error.URLError("timeout")
            return original(path, method, data, **kwargs)

        api.request = request  # type: ignore[method-assign]
        published = []

        def factory():
            target = MemoryPages()
            published.append(target)
            return target

        with self.assertRaisesRegex(RuntimeError, "1 documentation publication"):
            self.sweep(api, factory)
        self.assertEqual([target.pushed for target in published], [True])

    def test_run_that_fails_authentication_is_skipped_without_a_check_or_a_failure(self):
        api = SweepAPI()
        api.definition = b"older producer\n"
        for _ in range(2):
            with contextlib.redirect_stderr(io.StringIO()) as log:
                self.assertEqual(self.sweep(api), [])
            self.assertIn("Skipping producer run 100", log.getvalue())
        self.assertFalse(any(method in {"POST", "PATCH"} for _, method, _ in api.calls))

    def test_api_error_during_authentication_fails_the_sweep_for_a_retry(self):
        api = SweepAPI()
        original = api.request

        def request(path, method="GET", data=None, **kwargs):
            if path.startswith("contents/"):
                raise pages.GitHubError(502)
            return original(path, method, data, **kwargs)

        api.request = request  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "1 documentation publication"):
            self.sweep(api)
        self.assertFalse(any(method == "POST" for _, method, _ in api.calls))

    def test_check_external_id_matches_the_one_publish_writes(self):
        api = SweepAPI()
        self.sweep(api)
        self.assertEqual(self.sweep(api), [])

    def test_window_ends_before_the_producer_artifacts_expire(self):
        template = yaml.safe_load((ROOT / ".github/pages-build-template.yml").read_text())
        (upload,) = [s for s in template["jobs"]["build"]["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
        self.assertLess(pages.SWEEP_WINDOW, dt.timedelta(days=int(upload["with"]["retention-days"])))

    def test_nothing_to_publish_reads_no_refs(self):
        api = SweepAPI()
        api.runs = []
        self.assertEqual(self.sweep(api), [])
        self.assertFalse(any(path.startswith("git/") for path, _, _ in api.calls))


class MainTests(unittest.TestCase):
    def environment(self, **values):
        base = {"GITHUB_REPOSITORY": pages.REPOSITORY, "GH_TOKEN": "token", "GITHUB_REF": "refs/heads/master"}
        return patch.dict("os.environ", {**base, **values}, clear=True)

    def test_sweep_runs_only_from_master_on_a_schedule_or_dispatch(self):
        for event in ("schedule", "workflow_dispatch"):
            with self.subTest(event=event), self.environment(GITHUB_EVENT_NAME=event):
                # The factory is called while the sweep runs, before its parent directory is removed.
                def fake_sweep(api, now, factory):
                    self.assertEqual(now.utcoffset(), dt.timedelta(0))
                    factory()
                    factory()

                with (
                    patch.object(pages, "sweep", side_effect=fake_sweep) as sweep,
                    patch.object(pages, "GitPages") as git_pages,
                ):
                    pages.main(["--sweep"])
                sweep.assert_called_once()
                first, second = (Path(call.args[0]) for call in git_pages.call_args_list)
                self.assertNotEqual(first, second)
                self.assertEqual(first.parent, second.parent)
        rejected = [
            {"GITHUB_EVENT_NAME": "workflow_run"},
            {"GITHUB_EVENT_NAME": "push"},
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REF": "refs/heads/feature"},
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REPOSITORY": "fork/pytorch-bsf"},
        ]
        for values in rejected:
            with self.subTest(values=values), self.environment(**values), patch.object(pages, "sweep") as sweep:
                with self.assertRaises(ValueError):
                    pages.main(["--sweep"])
                sweep.assert_not_called()

    def test_publisher_still_requires_its_workflow_run_context(self):
        with self.environment(GITHUB_EVENT_NAME="schedule"), patch.object(pages, "publish") as publish:
            with self.assertRaisesRegex(ValueError, "workflow_run"):
                pages.main([])
            publish.assert_not_called()

    def test_unknown_arguments_are_rejected(self):
        with self.environment(GITHUB_EVENT_NAME="schedule"), self.assertRaises(SystemExit):
            pages.main(["--sweep", "extra"])


class SweepWorkflowTests(unittest.TestCase):
    def load(self, name):
        return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)

    def test_sweep_is_a_trusted_default_branch_control_sharing_the_publisher_lock(self):
        workflow, publisher = self.load("sphinx-pages-sweep.yml"), self.load("sphinx-pages-publish.yml")
        self.assertEqual(set(workflow["on"]), {"schedule", "workflow_dispatch"})
        self.assertEqual(workflow["on"]["schedule"], [{"cron": "23,53 * * * *"}])
        self.assertEqual(workflow["on"]["workflow_dispatch"], "")
        self.assertEqual(workflow["permissions"], {})
        self.assertEqual(workflow["concurrency"], publisher["concurrency"])
        (job,) = workflow["jobs"].values()
        self.assertEqual(job["if"], "github.repository == 'opthub-org/pytorch-bsf' && github.ref == 'refs/heads/master'")
        self.assertEqual(job["permissions"], publisher["jobs"]["publish"]["permissions"])
        checkout, run = job["steps"]
        self.assertEqual(checkout, publisher["jobs"]["publish"]["steps"][0])
        self.assertEqual(checkout["with"], {"ref": "${{ github.sha }}", "persist-credentials": "false"})
        self.assertEqual(run["env"], {"GH_TOKEN": "${{ github.token }}"})
        self.assertEqual(run["run"], "python3 -I tools/pages_publish.py --sweep")

    def test_sweep_does_not_start_the_dependabot_merge_controller(self):
        name = self.load("sphinx-pages-sweep.yml")["name"]
        self.assertNotIn(name, self.load("dependabot-merge.yml")["on"]["workflow_run"]["workflows"])


if __name__ == "__main__":
    unittest.main()
