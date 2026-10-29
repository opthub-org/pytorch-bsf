"""Exercise lifecycle transitions without network or numerical dependencies."""

import contextlib
import copy
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("update_python_support", ROOT / "tools/update_python_support.py")
assert SPEC is not None and SPEC.loader is not None
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def release(status, first, end):
    return {"status": status, "first_release": first, "end_of_life": end}


def release_data():
    return {
        "3.9": release("end-of-life", "2020-10-05", "2025-10-31"),
        "3.10": release("end-of-life", "2021-10-04", "2026-10-01"),
        "3.11": release("security", "2022-10-24", "2027-10"),
        "3.12": release("security", "2023-10-02", "2028-10"),
        "3.13": release("security", "2024-10-07", "2029-10"),
        "3.14": release("bugfix", "2025-10-07", "2030-10"),
        "3.15": release("bugfix", "2026-10-09", "2031-10"),
        "3.16": release("feature", "2027-10-06", "2032-10"),
    }


class PythonLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        for name in ("pyproject.toml", "environment.yml", "README.md", "CONTRIBUTING.md", "docs/installation.rst"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copytree(ROOT / ".github/workflows", self.root / ".github/workflows")
        # Keep the fixture baseline independent of future automated support updates.
        project = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        current = re.search(r'requires-python = ">=(3\.\d+)"', project)[1]
        project = project.replace(f'requires-python = ">={current}"', 'requires-python = ">=3.10"')
        project = re.sub(r'target-version = "py\d+"', 'target-version = "py310"', project)
        project = re.sub(r'python_version = "3\.\d+"', 'python_version = "3.10"', project)
        project = re.sub(
            r'(?:  "Programming Language :: Python :: 3\.\d+",\n)+',
            "".join(f'  "Programming Language :: Python :: 3.{minor}",\n' for minor in range(10, 15)),
            project,
        )
        (self.root / "pyproject.toml").write_text(project, encoding="utf-8")
        for name in ("README.md", "CONTRIBUTING.md", "docs/installation.rst"):
            path = self.root / name
            text = path.read_text(encoding="utf-8").replace(f"Python {current}", "Python 3.10")
            text = re.sub(r"`python_version = 3\.\d+`", "`python_version = 3.10`", text)
            text = re.sub(r'`target-version = "py\d+"`', '`target-version = "py310"`', text)
            text = re.sub(r"once on Python 3\.\d+\.", "once on Python 3.14.", text)
            path.write_text(text, encoding="utf-8")
        environment = self.root / "environment.yml"
        environment.write_text(
            re.sub(r"- python=3\.\d+", "- python=3.14", environment.read_text(encoding="utf-8")), encoding="utf-8"
        )
        self.data = release_data()
        self.today = date(2026, 10, 29)

    def apply_plan(self, data=None, today=None):
        changes, title, summary = updater.plan_updates(self.root, data or self.data, today or self.today)
        for path, content in changes.items():
            path.write_text(content, encoding="utf-8")
        return changes, title, summary

    def ci_outputs(self):
        output = self.root / "ci-output"
        with (
            patch.object(sys, "argv", ["updater", "--root", str(self.root), "--ci-output", str(output)]),
            patch.object(updater, "urlopen", side_effect=AssertionError("CI must not access the network")),
        ):
            updater.main()
        return dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())

    def test_eol_and_new_stable_update_all_surfaces(self):
        changes, title, summary = self.apply_plan()
        self.assertIn("ci(python)!:", title)
        self.assertIn("BREAKING CHANGE", summary)
        project = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.11"', project)
        self.assertNotIn('Python :: 3.10"', project)
        self.assertIn('Python :: 3.15"', project)
        self.assertIn('target-version = "py311"', project)
        self.assertIn('python_version = "3.11"', project)
        configured = self.ci_outputs()
        self.assertEqual(configured["minimum"], "3.11")
        self.assertEqual(configured["latest"], "3.15")
        self.assertEqual(json.loads(configured["matrix"]), ["3.11", "3.12", "3.13", "3.14", "3.15"])
        self.assertIn("- python=3.14", (self.root / "environment.yml").read_text(encoding="utf-8"))
        self.assertEqual(configured["integrations"], "3.14")
        self.assertEqual(json.loads(configured["integration-matrix"]), ["3.11", "3.12", "3.13", "3.14"])
        contributing = (self.root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn('`target-version = "py311"`', contributing)
        self.assertIn("once on Python 3.14.", contributing)
        for name in ("README.md", "CONTRIBUTING.md", "docs/installation.rst"):
            self.assertIn("Python 3.11", (self.root / name).read_text(encoding="utf-8"))
        self.assertFalse(any(path.suffix in {".yaml", ".yml"} and path.parent.name == "workflows" for path in changes))
        original = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(
            project.split("dependencies = [", 1)[1].split("[tool.pytest", 1)[0],
            original.split("dependencies = [", 1)[1].split("[tool.pytest", 1)[0],
        )
        self.assertNotIn(self.root / "uv.lock", changes)  # The workflow uses uv, never text editing, for the lock.

    def test_configured_ci_versions_are_offline_and_numeric(self):
        configured = self.ci_outputs()
        self.assertEqual(configured["minimum"], "3.10")
        self.assertEqual(configured["latest"], "3.14")
        self.assertEqual(json.loads(configured["matrix"]), ["3.10", "3.11", "3.12", "3.13", "3.14"])

    def test_eol_conda_python_is_raised_to_a_supported_version(self):
        self.apply_plan()
        environment = self.root / "environment.yml"
        environment.write_text(environment.read_text(encoding="utf-8").replace("python=3.14", "python=3.11"), encoding="utf-8")
        self.data["3.11"]["status"] = "end-of-life"
        self.apply_plan(today=date(2027, 11, 1))
        configured = self.ci_outputs()
        self.assertEqual(configured["minimum"], "3.12")
        self.assertEqual(configured["integrations"], "3.12")
        self.assertIn("python=3.12", environment.read_text(encoding="utf-8"))

    def test_conda_pin_can_advance_when_optional_dependencies_are_ready(self):
        self.apply_plan()
        environment = self.root / "environment.yml"
        environment.write_text(environment.read_text(encoding="utf-8").replace("python=3.14", "python=3.15"), encoding="utf-8")
        configured = self.ci_outputs()
        self.assertEqual(configured["integrations"], "3.15")
        self.assertEqual(json.loads(configured["integration-matrix"]), json.loads(configured["matrix"]))

    def test_newer_pythons_still_enter_the_matrix_with_a_conda_pin(self):
        self.apply_plan()
        self.data["3.16"]["status"] = "bugfix"
        self.apply_plan(today=date(2027, 10, 10))
        configured = self.ci_outputs()
        self.assertEqual(configured["latest"], "3.16")
        self.assertIn("3.16", json.loads(configured["matrix"]))
        self.assertNotIn("3.16", json.loads(configured["integration-matrix"]))
        self.assertEqual(configured["integrations"], "3.14")

    def test_invalid_project_support_cannot_reach_ci_outputs(self):
        for project in ('requires-python = ">=3.10"', 'requires-python = ">=3.10,<4"'):
            with self.subTest(project=project), self.assertRaises(ValueError):
                updater.configured_versions(project)

    def test_second_run_has_no_changes(self):
        self.apply_plan()
        changes, title, _ = updater.plan_updates(self.root, self.data, self.today)
        self.assertEqual(changes, {})
        self.assertNotIn("!", title)

    def test_no_transition_does_not_raise_minimum_or_add_prerelease(self):
        self.data["3.10"]["status"] = "security"
        self.data["3.15"]["status"] = "prerelease"
        changes, title, _ = updater.plan_updates(self.root, self.data, date(2026, 9, 30))
        self.assertEqual(changes, {})
        self.assertNotIn("!", title)

    def test_new_release_alone_is_not_a_breaking_change(self):
        self.data["3.10"]["status"] = "security"
        self.data["3.10"]["end_of_life"] = "2026-12-01"
        _, title, _ = self.apply_plan()
        self.assertNotIn("!", title)
        self.assertIn('requires-python = ">=3.10"', (self.root / "pyproject.toml").read_text(encoding="utf-8"))

    def test_future_stable_date_is_not_treated_as_released(self):
        self.data["3.16"]["status"] = "bugfix"
        self.assertEqual(updater.supported_versions(self.data, self.today)[-1], "3.15")

    def test_eol_boundary_and_month_only_schedule(self):
        self.data["3.10"]["status"] = "security"
        self.assertIn("3.10", updater.supported_versions(self.data, date(2026, 9, 30)))
        self.assertNotIn("3.10", updater.supported_versions(self.data, date(2026, 10, 1)))
        self.assertIn("3.11", updater.supported_versions(self.data, date(2027, 10, 30)))
        self.assertNotIn("3.11", updater.supported_versions(self.data, date(2027, 10, 31)))

    def test_next_year_and_multiple_eols(self):
        self.apply_plan()
        self.data["3.11"]["status"] = "end-of-life"
        self.data["3.12"]["status"] = "end-of-life"
        self.data["3.16"]["status"] = "bugfix"
        _, title, _ = self.apply_plan(today=date(2028, 11, 1))
        self.assertIn("Python 3.13 through 3.16", title)

    def test_higher_existing_minimum_is_preserved(self):
        self.apply_plan()
        path = self.root / "pyproject.toml"
        path.write_text(path.read_text(encoding="utf-8").replace('>=3.11"', '>=3.12"'), encoding="utf-8")
        for name in ("README.md", "CONTRIBUTING.md", "docs/installation.rst"):
            path = self.root / name
            path.write_text(path.read_text(encoding="utf-8").replace("Python 3.11", "Python 3.12"), encoding="utf-8")
        _, title, _ = self.apply_plan()
        self.assertIn("Python 3.12 through 3.15", title)
        self.assertNotIn("!", title)

    def test_invalid_or_incomplete_data_produces_no_writes(self):
        malformed = [None, [], {}, {"bad": {}}, {"3.14": {"status": "unknown"}}]
        missing = copy.deepcopy(self.data)
        del missing["3.12"]
        stale = copy.deepcopy(self.data)
        stale["3.14"]["status"] = "prerelease"
        stale["3.15"]["status"] = "prerelease"
        malformed.extend([missing, stale])
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for data in malformed:
            with self.subTest(data=data), self.assertRaises((ValueError, KeyError, TypeError)):
                updater.plan_updates(self.root, data, self.today)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_missing_edit_location_fails_before_writing(self):
        path = self.root / "environment.yml"
        path.write_text("dependencies: []\n", encoding="utf-8")
        project = (self.root / "pyproject.toml").read_bytes()
        with self.assertRaisesRegex(ValueError, "pinned Python 3 version"):
            updater.plan_updates(self.root, self.data, self.today)
        self.assertEqual((self.root / "pyproject.toml").read_bytes(), project)

    def test_cli_preview_and_apply_are_reproducible(self):
        data_file = self.root / "release-cycle.json"
        data_file.write_text(json.dumps(self.data), encoding="utf-8")
        summary_file = self.root / "summary.md"
        output_file = self.root / "output"
        arguments = [
            "update_python_support.py",
            "--root",
            str(self.root),
            "--data-file",
            str(data_file),
            "--summary-file",
            str(summary_file),
            "--github-output",
            str(output_file),
        ]
        project = (self.root / "pyproject.toml").read_bytes()
        with patch.object(sys, "argv", [*arguments, "--check"]), patch.object(updater, "date", wraps=date) as clock:
            clock.today.return_value = self.today
            with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as result:
                updater.main()
        self.assertEqual(result.exception.code, 1)
        self.assertIn('requires-python = ">=3.11"', output.getvalue())
        self.assertEqual((self.root / "pyproject.toml").read_bytes(), project)
        with patch.object(sys, "argv", arguments), patch.object(updater, "date", wraps=date) as clock:
            clock.today.return_value = self.today
            with contextlib.redirect_stdout(io.StringIO()):
                updater.main()
        self.assertIn("changed=true", output_file.read_text(encoding="utf-8"))
        self.assertIn(updater.SOURCE, summary_file.read_text(encoding="utf-8"))

    def test_cli_fetch_failure_leaves_checkout_unchanged(self):
        before = (self.root / "pyproject.toml").read_bytes()
        with (
            patch.object(sys, "argv", ["updater", "--root", str(self.root)]),
            patch.object(updater, "urlopen", side_effect=TimeoutError),
            self.assertRaises(TimeoutError),
        ):
            updater.main()
        self.assertEqual(before, (self.root / "pyproject.toml").read_bytes())

    def test_entrypoint_help_requires_no_installed_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/update_python_support.py"), "--help"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--check", result.stdout)


if __name__ == "__main__":
    unittest.main()
