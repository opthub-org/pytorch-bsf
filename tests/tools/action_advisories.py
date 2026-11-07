"""Offline regressions for the check that files issues for advisories on the action pins of master's workflows."""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlsplit

from tools import action_advisories as advisories
from tools import security_automation as security

ROOT = Path(__file__).resolve().parents[2]
REPO = "opthub-org/pytorch-bsf"
SHA, OTHER_SHA, TAG_SHA = "a" * 40, "b" * 40, "c" * 40
BOT = security.BOT


def pin(name="actions/checkout", sha=SHA, tag="v7.0.1", path=".github/workflows/x.yml", line=1):
    return advisories.Pin(path, line, name, sha, tag)


class FakeGitHub(security.GitHub):
    """Answer like GitHub: an advisory query with a version matches the name literally, and one without it ignores case."""

    def __init__(self, database=(), tags=None, issues=(), failing=(), rows=None):
        self.repo = REPO
        self.database = database  # advisories in the Advisory Database: (id, name filed under, affected version)
        self.rows = rows  # the answer to every advisory query, instead of one built from `database`
        self.tags = tags or {}  # (repository in lowercase, tag) -> git object
        self.annotated = {}  # tag object SHA -> git object
        self.issues = list(issues)
        self.failing = failing  # substrings of paths that fail
        self.requests = []
        self.writes = []

    def request(self, path, method="GET", data=None):
        assert method == "GET"
        self.requests.append(path)
        if any(part in unquote(path) for part in self.failing):
            raise security.GitHubError("GitHub GET returned HTTP 502", 502)
        url = urlsplit(path)
        if url.path == "/advisories":
            return self.advisories(parse_qs(url.query))
        match = re.fullmatch(r"/repos/([^/]+/[^/]+)/git/ref/tags/(.+)", url.path)
        if match:
            target = self.tags.get((match[1].lower(), unquote(match[2])))
            if target is None:
                raise security.GitHubError("GitHub GET returned HTTP 404", 404)
            return {"object": target}
        match = re.fullmatch(r"/repos/[^/]+/[^/]+/git/tags/([0-9a-f]{40})", url.path)
        assert match, path
        return {"object": self.annotated[match[1]]}

    def advisories(self, query):
        assert query["ecosystem"] == ["actions"]
        if self.rows is not None:
            return self.rows
        name, _, version = query["affects"][0].partition("@")
        rows = []
        for ident, filed, affected in self.database:
            if query["type"][0] != "reviewed":
                continue
            if (filed == name and version == affected) or (not version and filed.lower() == name.lower()):
                rows.append({"ghsa_id": ident, "vulnerabilities": [{"package": {"name": filed}}]})
        return rows

    def pages(self, path, key=None):
        self.requests.append(path)
        assert path == "issues?state=all&creator=github-actions%5Bbot%5D", path
        if "issues" in self.failing:
            raise security.GitHubError("GitHub GET returned HTTP 502", 502)
        return self.issues

    def write(self, path, data, method="POST"):
        assert (path, method) == ("issues", "POST")
        if "fail" in data["title"]:
            raise security.GitHubError("GitHub POST returned HTTP 502", 502)
        self.writes.append(data)
        return {"number": len(self.writes)}


def commit_tag(repository="actions/checkout", tag="v7.0.1", sha=SHA):
    return {(repository, tag): {"sha": sha, "type": "commit"}}


class ParseTests(unittest.TestCase):
    def test_a_pin_gives_its_action_sha_and_exact_version(self):
        line = f"      - uses: actions/checkout@{SHA} # v7.0.1"
        self.assertEqual(advisories.parse_pin(line), ("actions/checkout", SHA, "v7.0.1"))
        self.assertEqual(advisories.parse_pin(f"        uses: a/b@{SHA}"), ("a/b", SHA, None))

    def test_a_subpath_is_kept_in_the_name(self):
        line = f"      - uses: github/codeql-action/init@{SHA} # v4.39.1"
        self.assertEqual(advisories.parse_pin(line), ("github/codeql-action/init", SHA, "v4.39.1"))
        self.assertEqual(pin("github/codeql-action/init").repository, "github/codeql-action")

    def test_a_comment_without_an_exact_version_gives_no_version(self):
        for comment in ("# reviewed 2026-10-30", "# v4", "# 4.39", "#", "# note v7.0.1", "# v7.0.1-beta"):
            with self.subTest(comment=comment):
                self.assertEqual(advisories.parse_pin(f"  - uses: a/b@{SHA} {comment}"), ("a/b", SHA, None))
        self.assertEqual(advisories.parse_pin(f"  - uses: a/b@{SHA} #v7.0.1 extra words"), ("a/b", SHA, "v7.0.1"))

    def test_lines_without_an_external_action_are_skipped(self):
        for line in (
            "    statuses: read",
            "      causes: x",
            "      pre-uses: x",
            "      reuses_x: y",
            "  # uses: a/b@v1",
            "    uses: ./.github/workflows/docker.yml # zizmor: ignore[self-repository] -- reason",
            "      - uses: ./local-action",
            "",
        ):
            with self.subTest(line=line):
                self.assertIsNone(advisories.parse_pin(line))

    def test_a_uses_line_in_any_other_form_raises(self):
        for line in (
            "      - uses: actions/checkout@v7",
            f"      - uses: actions/checkout@{SHA[:39]}",
            f"      - uses: 'actions/checkout@{SHA}'",
            f'      - uses: "actions/checkout@{SHA}"',
            "      - uses: docker://alpine:3.20",
            f"      - uses: checkout@{SHA}",
            f"      - uses: actions/../x@{SHA}",
            f"      - uses: actions/./x@{SHA}",
            f"      - uses: a/.@{SHA}",
            f"      - {{uses: actions/checkout@{SHA}}}",
            "      - uses:",
            "      - uses: ./x junk",
            f"      - uses: actions/checkout@{SHA} # v7.0.1\x0b",
            f"      - uses: actions/checkout@{SHA.upper()}",
            "      - name: x # uses: y",
            "        run: echo uses: foo",
            f'      - "uses": actions/checkout@{SHA}',
            f"      - 'uses': actions/checkout@{SHA}",
            '        "uses": evil/x@main',
            '      - {"uses": evil/x@main}',
            f"      - uses : actions/checkout@{SHA}",
        ):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "neither a local path nor an external action"):
                advisories.parse_pin(line)

    def test_trailing_blanks_and_a_crlf_line_end_do_not_hide_a_pin(self):
        for end in ("  ", "\t", "\r", " \r"):
            with self.subTest(end=end):
                line = f"      - uses: actions/checkout@{SHA} # v7.0.1{end}"
                self.assertEqual(advisories.parse_pin(line), ("actions/checkout", SHA, "v7.0.1"))
                self.assertEqual(advisories.parse_pin(f"      - uses: a/b@{SHA}{end}"), ("a/b", SHA, None))

    def test_every_pin_of_the_repository_is_read(self):
        pins, unreadable = advisories.read_pins(ROOT)
        self.assertEqual(unreadable, [])
        lines = 0
        for path in (ROOT / ".github/workflows").iterdir():
            for line in path.read_text().splitlines():
                if re.match(r"\s*(?:- )?uses:\s+(?!\./)", line):
                    lines += 1
        self.assertEqual(len(pins), lines)
        names = {row.name for row in pins}
        self.assertTrue({"github/codeql-action/init", "github/codeql-action/analyze", "pypa/gh-action-pypi-publish"} <= names)
        self.assertTrue(all(row.path.startswith(".github/workflows/") for row in pins))
        codex = [row for row in pins if row.name == "openai/codex-action"]
        self.assertTrue(codex)
        self.assertTrue(all(row.tag is None for row in codex))

    def workflows(self, files):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / ".github/workflows").mkdir(parents=True)
        for name, text in files.items():
            (root / ".github/workflows" / name).write_text(text)
        return root

    def test_pins_are_read_from_yml_and_yaml_files_only(self):
        root = self.workflows(
            {
                "a.yml": f"steps:\n  - uses: a/b@{SHA} # v1.0.0\n",
                "b.yaml": f"steps:\n  - uses: c/d@{SHA}\n",
                "notes.txt": "  - uses: e/f@v1\n",
            }
        )
        pins = advisories.read_pins(root)
        self.assertEqual(
            pins,
            (
                [
                    advisories.Pin(".github/workflows/a.yml", 2, "a/b", SHA, "v1.0.0"),
                    advisories.Pin(".github/workflows/b.yaml", 2, "c/d", SHA, None),
                ],
                [],
            ),
        )

    def test_an_unreadable_line_names_its_place_and_the_other_pins_are_still_read(self):
        root = self.workflows({"a.yml": f"steps:\n  - uses: a/b@v1\n  - uses: c/d@{SHA}\n"})
        pins, unreadable = advisories.read_pins(root)
        self.assertEqual(pins, [advisories.Pin(".github/workflows/a.yml", 3, "c/d", SHA, None)])
        self.assertEqual(len(unreadable), 1)
        self.assertRegex(unreadable[0], r"^\.github/workflows/a\.yml:2: A line with a `uses` key ")
        only_unreadable = self.workflows({"a.yml": "steps:\n  - uses: a/b@v1\n"})
        self.assertEqual(len(advisories.read_pins(only_unreadable)[1]), 1)

    def test_no_pins_at_all_raises(self):
        with self.assertRaisesRegex(ValueError, "No action pins"):
            advisories.read_pins(self.workflows({"a.yml": "steps:\n  - uses: ./x\n"}))


class LookupTests(unittest.TestCase):
    def test_a_tag_that_points_to_the_pin_asks_about_its_version(self):
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.1")], commit_tag())
        findings, errors = advisories.scan(api, [pin()])
        self.assertEqual((findings, errors), ({"GHSA-a": [advisories.Finding(pin(), "7.0.1")]}, []))
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.0")], commit_tag())
        self.assertEqual(advisories.scan(api, [pin()]), ({}, []))

    def test_a_capital_v_tag_asks_about_the_version_without_it(self):
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.1")], commit_tag(tag="V7.0.1"))
        upper = pin(tag="V7.0.1")
        self.assertEqual(advisories.scan(api, [upper]), ({"GHSA-a": [advisories.Finding(upper, "7.0.1")]}, []))

    def test_an_annotated_tag_is_followed_to_its_commit(self):
        api = FakeGitHub([("GHSA-a", "a/b", "1.0.0")], {("a/b", "v1.0.0"): {"sha": TAG_SHA, "type": "tag"}})
        api.annotated = {TAG_SHA: {"sha": SHA, "type": "commit"}}
        self.assertEqual(advisories.scan(api, [pin("a/b", tag="v1.0.0")])[0]["GHSA-a"][0].version, "1.0.0")

    def test_a_tag_on_another_commit_asks_about_any_version(self):
        database = [("GHSA-a", "actions/checkout", "6.0.0")]
        findings, errors = advisories.scan(FakeGitHub(database, commit_tag(sha=OTHER_SHA)), [pin()])
        self.assertEqual((findings, errors), ({"GHSA-a": [advisories.Finding(pin(), None)]}, []))
        findings, _ = advisories.scan(FakeGitHub(database), [pin()])  # the tag does not exist
        self.assertEqual(findings, {"GHSA-a": [advisories.Finding(pin(), None)]})

    def test_a_pin_without_a_version_asks_about_any_version_without_reading_tags(self):
        api = FakeGitHub([("GHSA-a", "openai/codex-action", "1.0.0")])
        findings, _ = advisories.scan(api, [pin("openai/codex-action", tag=None)])
        self.assertEqual(list(findings), ["GHSA-a"])
        self.assertFalse(any("/git/" in path for path in api.requests))

    def test_a_subpath_finds_advisories_on_its_repository_and_on_itself(self):
        tags = commit_tag("github/codeql-action", "v4.39.1")
        init, analyze = pin("github/codeql-action/init", tag="v4.39.1"), pin("github/codeql-action/analyze", tag="v4.39.1")
        database = [("GHSA-r", "github/codeql-action", "4.39.1"), ("GHSA-s", "github/codeql-action/init", "4.39.1")]
        findings, errors = advisories.scan(FakeGitHub(database, tags), [init, analyze])
        self.assertEqual(errors, [])
        self.assertEqual({row.pin.name for row in findings["GHSA-r"]}, {init.name, analyze.name})
        self.assertEqual([row.pin.name for row in findings["GHSA-s"]], [init.name])

    def test_a_name_in_another_case_finds_the_advisory(self):
        database = [("GHSA-a", "github/codeql-action", "4.39.1")]
        tags = commit_tag("github/codeql-action", "v4.39.1")
        upper = pin("GitHub/CodeQL-Action/init", tag="v4.39.1")
        self.assertEqual(list(advisories.scan(FakeGitHub(database, tags), [upper])[0]), ["GHSA-a"])
        database = [("GHSA-b", "Azure/x", "1.0.0")]
        lower = pin("azure/x", tag="v1.0.0")
        self.assertEqual(list(advisories.scan(FakeGitHub(database, commit_tag("azure/x", "v1.0.0")), [lower])[0]), ["GHSA-b"])

    def test_the_same_pin_is_looked_up_once_and_reported_at_every_line(self):
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.1")], commit_tag())
        findings, _ = advisories.scan(api, [pin(line=3), pin(line=9, path=".github/workflows/y.yml")])
        self.assertEqual([row.pin.line for row in findings["GHSA-a"]], [3, 9])
        self.assertEqual(sum("/git/ref/" in path for path in api.requests), 1)

    def test_a_failed_lookup_is_an_error_and_never_no_advisories(self):
        for failing in ("checkout/git/ref/", "affects=actions/checkout@", "type=malware"):
            with self.subTest(failing=failing):
                api = FakeGitHub([("GHSA-b", "c/d", "1.0.0")], commit_tag() | commit_tag("c/d", "v1.0.0"), failing=(failing,))
                findings, errors = advisories.scan(api, [pin(), pin("c/d", tag="v1.0.0")])
                self.assertEqual(len(errors), 1 if failing != "type=malware" else 2)
                self.assertRegex(errors[0], r"^actions/checkout@aaaaaaaaaa: GitHubError: GitHub GET returned HTTP 502$")
                if failing != "type=malware":
                    self.assertEqual(list(findings), ["GHSA-b"])

    def test_one_commit_with_different_comments_is_looked_up_once_per_comment(self):
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.1")], commit_tag())
        exact, untagged = pin(line=1), pin(tag=None, line=2)
        findings, errors = advisories.scan(api, [exact, untagged])
        self.assertEqual(errors, [])
        self.assertEqual(findings, {"GHSA-a": [advisories.Finding(exact, "7.0.1"), advisories.Finding(untagged, None)]})
        versioned = [query for query in api.requests if "affects=actions%2Fcheckout%40" in query]
        self.assertEqual(len(versioned), 2)  # both advisory types, for the exact pin only

    def tag_chain(self, length):
        """Return a tag `v1.0.0` that reaches the pinned commit through `length` annotated tags."""
        shas = [f"{n:040x}" for n in range(1, length + 1)]
        api = FakeGitHub([("GHSA-a", "a/b", "1.0.0")], {("a/b", "v1.0.0"): {"sha": shas[0], "type": "tag"}})
        api.annotated = {sha: {"sha": following, "type": "tag"} for sha, following in zip(shas, shas[1:], strict=False)}
        api.annotated[shas[-1]] = {"sha": SHA, "type": "commit"}
        return api

    def test_a_chain_of_annotated_tags_is_followed_up_to_the_bound(self):
        within = self.tag_chain(advisories.MAX_TAG_DEPTH - 1)
        self.assertEqual(advisories.scan(within, [pin("a/b", tag="v1.0.0")])[0]["GHSA-a"][0].version, "1.0.0")
        beyond = self.tag_chain(advisories.MAX_TAG_DEPTH)
        self.assertRegex(advisories.scan(beyond, [pin("a/b", tag="v1.0.0")])[1][0], "too many nested tags")

    def test_malformed_answers_are_errors(self):
        cases = {
            "malformed tag": {("actions/checkout", "v7.0.1"): {"sha": "x", "type": "commit"}},
            "neither a commit nor a tag": {("actions/checkout", "v7.0.1"): {"sha": SHA, "type": "tree"}},
            "missing type": {("actions/checkout", "v7.0.1"): {"sha": SHA}},
        }
        for name, tags in cases.items():
            with self.subTest(name=name):
                self.assertEqual(len(advisories.scan(FakeGitHub(tags=tags), [pin()])[1]), 1)
        api = FakeGitHub(tags={("actions/checkout", "v7.0.1"): {"sha": TAG_SHA, "type": "tag"}})
        api.annotated = {TAG_SHA: {"sha": TAG_SHA, "type": "tag"}}
        self.assertRegex(advisories.scan(api, [pin()])[1][0], "too many nested tags")
        oversized = FakeGitHub(tags=commit_tag(), rows=[{"ghsa_id": f"GHSA-{n}"} for n in range(100)])
        self.assertRegex(advisories.scan(oversized, [pin()])[1][0], "malformed or oversized")
        unsafe = FakeGitHub(tags=commit_tag(), rows=[{"ghsa_id": "GHSA-a <b>"}])
        self.assertRegex(advisories.scan(unsafe, [pin()])[1][0], "malformed identifier")

    def test_a_withdrawn_advisory_is_not_found(self):
        api = FakeGitHub(rows=[{"ghsa_id": "GHSA-a", "withdrawn_at": "2026-11-01T00:00:00Z"}])
        self.assertEqual(advisories.scan(api, [pin(tag=None)]), ({}, []))


def marker(ident, keys=()):
    return f"{advisories.MARKER_PREFIX}{ident}{''.join(' ' + key for key in keys)} -->"


def issue(ident, keys=(), login=BOT, state="closed", prefix="", **extra):
    body = prefix + marker(ident, keys) + "\nbody"
    return {"user": {"login": login}, "state": state, "title": advisories.TITLE.format(ident), "body": body, **extra}


CHECKOUT_KEY = f"actions/checkout@{SHA}"


class ReportTests(unittest.TestCase):
    def findings(self):
        return {
            "GHSA-a": [advisories.Finding(pin(line=4), "7.0.1"), advisories.Finding(pin("O/C", tag=None, line=2), None)],
            "GHSA-b": [advisories.Finding(pin("o/d", tag="v1.0.0", sha=OTHER_SHA), None)],
        }

    def test_each_new_advisory_gets_one_issue_that_notifies_the_maintainer(self):
        api = FakeGitHub()
        self.assertEqual(advisories.report(api, self.findings()), (["GHSA-a", "GHSA-b"], []))
        first, second = api.writes
        self.assertEqual(first["title"], "Advisory GHSA-a affects an action pinned in a workflow")
        self.assertEqual(first["assignees"], ["NaokiHamada"])
        body = first["body"]
        self.assertTrue(body.startswith(f"<!-- action-advisory:v1 GHSA-a {CHECKOUT_KEY} o/c@{SHA} -->\n@NaokiHamada "))
        self.assertIn("[GHSA-a](https://github.com/advisories/GHSA-a)", body)
        self.assertIn(f"https://github.com/{REPO}/blob/master/SECURITY.md#action-advisories-after-merge", body)
        self.assertNotIn("earlier issue", body)
        lines = [line for line in body.splitlines() if line.startswith("- ")]
        self.assertEqual(
            lines,
            [
                "- `.github/workflows/x.yml:2` `O/C@aaaaaaaaaa`: its comment names no exact version, "
                "so any advisory on the action counts.",
                "- `.github/workflows/x.yml:4` `actions/checkout@aaaaaaaaaa`: version `7.0.1` is affected.",
            ],
        )
        self.assertIn("the tag `v1.0.0` in its comment is not this commit", second["body"])

    def test_the_marker_of_a_filed_issue_reads_back_as_filed(self):
        api = FakeGitHub()
        advisories.report(api, self.findings())
        filed = [{"user": {"login": BOT}, "title": row["title"], "body": row["body"]} for row in api.writes]
        self.assertEqual(advisories.report(FakeGitHub(issues=filed), self.findings()), ([], []))

    def test_pins_already_reported_for_an_advisory_are_not_filed_again(self):
        for state in ("open", "closed"):
            with self.subTest(state=state):
                api = FakeGitHub(issues=[issue("GHSA-a", [CHECKOUT_KEY, f"o/c@{SHA}"], state=state)])
                self.assertEqual(advisories.report(api, self.findings()), (["GHSA-b"], []))
                self.assertEqual([row["title"].split()[1] for row in api.writes], ["GHSA-b"])

    def test_a_pin_that_no_issue_reported_gets_an_issue_for_itself_only(self):
        api = FakeGitHub(issues=[issue("GHSA-a", [CHECKOUT_KEY]), issue("GHSA-b", [f"o/d@{OTHER_SHA}"])])
        self.assertEqual(advisories.report(api, self.findings()), (["GHSA-a"], []))
        (body,) = [row["body"] for row in api.writes]
        self.assertTrue(body.startswith(f"<!-- action-advisory:v1 GHSA-a o/c@{SHA} -->\n"))
        self.assertNotIn("actions/checkout", body)
        self.assertIn("An earlier issue for this advisory reports other pins.", body)

    def test_a_marker_below_text_that_the_maintainer_added_still_counts(self):
        api = FakeGitHub(issues=[issue("GHSA-a", [CHECKOUT_KEY, f"o/c@{SHA}"], prefix="Triage note.\n\n")])
        self.assertEqual(advisories.report(api, self.findings())[0], ["GHSA-b"])

    def test_only_this_checks_own_issues_count_as_filed(self):
        keys = [CHECKOUT_KEY, f"o/c@{SHA}"]
        deleted = issue("GHSA-a", keys) | {"user": None}
        pull = issue("GHSA-b", [f"o/d@{OTHER_SHA}"], pull_request={})
        # The status issue quotes alert text as JSON, which could carry a marker.
        quoted = json.dumps({"summary": marker("GHSA-a", keys)})
        state = {"user": {"login": BOT}, "title": security.STATE_TITLE, "body": f"<!-- {security.STATE_MARKER}\n{quoted}\n-->"}
        retitled = issue("GHSA-b", [f"o/d@{OTHER_SHA}"]) | {"title": advisories.TITLE.format("GHSA-a")}
        empty = {"user": {"login": BOT}, "title": advisories.TITLE.format("GHSA-a"), "body": None}
        api = FakeGitHub(issues=[issue("GHSA-a", keys, login="someone"), pull, deleted, state, retitled, empty])
        self.assertEqual(advisories.report(api, self.findings())[0], ["GHSA-a", "GHSA-b"])

    def test_no_finding_reads_no_issues(self):
        api = FakeGitHub()
        self.assertEqual(advisories.report(api, {}), ([], []))
        self.assertEqual(api.requests, [])

    def test_a_failed_filing_is_an_error_and_the_others_are_still_filed(self):
        findings = self.findings()
        findings["GHSA-fail"] = findings.pop("GHSA-a")
        api = FakeGitHub()
        filed, errors = advisories.report(api, findings)
        self.assertEqual(filed, ["GHSA-b"])
        self.assertEqual(errors, ["GHSA-fail: filing the issue failed: GitHubError: GitHub POST returned HTTP 502"])

    def test_a_failed_issue_listing_files_nothing_and_is_an_error(self):
        api = FakeGitHub(failing=("issues",))
        self.assertEqual(
            advisories.report(api, self.findings()),
            ([], ["Reading the filed issues failed, so nothing was filed: GitHubError: GitHub GET returned HTTP 502"]),
        )
        self.assertEqual(api.writes, [])


class RunTests(unittest.TestCase):
    def run_with(self, api, pins, unreadable=()):
        with (
            patch.object(advisories, "read_pins", return_value=(pins, list(unreadable))),
            patch("builtins.print") as printed,
        ):
            return advisories.run(api, ROOT), printed

    def test_run_reports_found_advisories_and_returns_every_error(self):
        api = FakeGitHub([("GHSA-a", "actions/checkout", "7.0.1")], commit_tag(), failing=("c/d",))
        errors, printed = self.run_with(api, [pin(), pin("c/d", tag="v1.0.0")], ["x.yml:3: unreadable"])
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0], "x.yml:3: unreadable")
        self.assertTrue(errors[1].startswith("c/d@"))
        self.assertEqual([row["title"].split()[1] for row in api.writes], ["GHSA-a"])
        printed.assert_called_once_with(
            "Checked 2 action pins: 1 advisories found, 1 newly filed (GHSA-a), "
            "1 lines unreadable, 1 lookups failed, 0 filing errors."
        )

    def test_run_returns_failed_filings(self):
        errors, printed = self.run_with(FakeGitHub(rows=[{"ghsa_id": "GHSA-fail"}]), [pin(tag=None)])
        self.assertEqual(errors, ["GHSA-fail: filing the issue failed: GitHubError: GitHub POST returned HTTP 502"])
        printed.assert_called_once_with(
            "Checked 1 action pins: 1 advisories found, 0 newly filed, 0 lines unreadable, 0 lookups failed, 1 filing errors."
        )

    def test_a_failed_issue_listing_keeps_the_lookup_errors(self):
        api = FakeGitHub(rows=[{"ghsa_id": "GHSA-x"}], failing=("c/d", "issues"))
        errors, _ = self.run_with(api, [pin(tag=None), pin("c/d", tag=None)])
        self.assertEqual(len(errors), 2)
        self.assertTrue(errors[0].startswith("c/d@"))
        self.assertTrue(errors[1].startswith("Reading the filed issues failed"))

    def test_main_fails_the_job_on_errors(self):
        environment = {"GITHUB_REPOSITORY": REPO, "GH_TOKEN": "x"}
        with (
            patch.dict(os.environ, environment),
            patch("sys.argv", ["action_advisories", "--root", "/fixture"]),
            patch.object(advisories, "run", return_value=["a: x", "b: y"]) as run,
        ):
            with self.assertRaisesRegex(SystemExit, "^a: x\nb: y$"):
                advisories.main()
            self.assertEqual(run.call_args.args[1], Path("/fixture"))
            run.return_value = []
            advisories.main()


class WorkflowTests(unittest.TestCase):
    path = ".github/workflows/security-action-advisories.yml"

    def setUp(self):
        self.workflow = (ROOT / self.path).read_text()

    def test_repair_automation_cannot_edit_the_check(self):
        for path in (self.path, "tools/action_advisories.py", "tests/tools/action_advisories.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require maintainer review"):
                security.validate_path(path)

    def test_the_check_runs_on_a_schedule_and_when_master_changes(self):
        self.assertRegex(self.workflow, r"schedule:\n(?:\s+#.*\n)*\s+- cron: '41 \* \* \* \*'")
        self.assertRegex(self.workflow, r"push:\n    branches: \[master\]\n")
        self.assertIn("workflow_dispatch:", self.workflow)

    def test_only_master_code_runs_with_the_issue_permission(self):
        for line in (
            "permissions: {}",
            "github.repository == 'opthub-org/pytorch-bsf' && github.ref == 'refs/heads/master'",
            "ref: ${{ github.sha }}",
            "persist-credentials: false",
            "contents: read",
            "issues: write",
            "cancel-in-progress: false",
            "timeout-minutes:",
        ):
            with self.subTest(line=line):
                self.assertIn(line, self.workflow)
        self.assertEqual(self.workflow.count(": write"), 1)
        self.assertRegex(self.workflow, r"run: python -m tools\.action_advisories\n")

    def test_the_tests_run_in_the_security_policy_check(self):
        policy = (ROOT / ".github/workflows/security-policy.yml").read_text()
        self.assertIn("python -m unittest discover -s tests/tools -p action_advisories.py -v", policy)


if __name__ == "__main__":
    unittest.main()
