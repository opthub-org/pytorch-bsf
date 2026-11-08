"""Compare and synchronize the ranges of environment.yml and pyproject.toml without numerical dependencies."""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sync_environment", ROOT / "tools/sync_environment.py")
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

PROJECT_TEXT = """\
[project]
name = "pytorch-bsf"
dependencies = [
  "torch>=2.3,<3",
  "numpy>=1.23.2,<3",
  "PyYAML>=6,<7",
]

[project.optional-dependencies]
adaptive = ["scipy>=1.9.2,<2"]
plot = ["matplotlib>=3.6,<4", "scipy>=1.9.2,<2"]
all = ["pytorch-bsf[adaptive,plot]"]
develop = ["pytest", "mypy>=1"]
"""

ENVIRONMENT_TEXT = """\
name: pytorch-bsf
channels:
- conda-forge
dependencies:
# Keep these ranges in sync with pyproject.toml.
- python=3.14
- "pytorch>=2.3,<3"
- numpy>=1.23.2,<3  # unquoted
- 'pyyaml>=6,<7'
- "scipy>=1.9.2,<2"
- "matplotlib-base>=3.6,<4"
- pip
- pip:
  - -e .
"""


class DriftTests(unittest.TestCase):
    def test_repository_files_can_be_read(self):
        # Only readable, not free of drift: Dependabot PRs and master may have drift that CI tolerates.
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
        ranges, _ = sync.project_ranges(project)
        self.assertIn("matplotlib-base", ranges)
        self.assertIn("pytorch", sync.environment_entries(environment))

    def test_fixture_agrees_with_renamed_packages_and_any_clause_order(self):
        self.assertEqual(sync.drift(PROJECT_TEXT, ENVIRONMENT_TEXT), [])
        reordered = ENVIRONMENT_TEXT.replace('"pytorch>=2.3,<3"', '"pytorch <3,>=2.3"')
        self.assertEqual(sync.drift(PROJECT_TEXT, reordered), [])

    def test_range_change_of_a_renamed_package_is_drift(self):
        for name, conda, old, new in (
            ("matplotlib", "matplotlib-base", ">=3.6,<4", ">=3.6,<5"),
            ("torch", "pytorch", ">=2.3,<3", ">=2.3,<4"),
            ("PyYAML", "pyyaml", ">=6,<7", ">=6.0.1,<7"),
        ):
            with self.subTest(name=name):
                project = PROJECT_TEXT.replace(f'"{name}{old}"', f'"{name}{new}"')
                self.assertEqual(
                    sync.drift(project, ENVIRONMENT_TEXT),
                    [f"environment.yml has `{conda}` {old}, but pyproject.toml has {new}"],
                )

    def test_every_extra_but_develop_counts(self):
        project = PROJECT_TEXT.replace('adaptive = ["scipy>=1.9.2,<2"]', 'adaptive = ["scipy>=1.9.2,<2", "sympy>=1.12,<2"]')
        self.assertEqual(sync.drift(project, ENVIRONMENT_TEXT), ["environment.yml lacks `sympy>=1.12,<2` of pyproject.toml"])

    def test_undeclared_conda_package_is_drift(self):
        environment = ENVIRONMENT_TEXT.replace("- pip\n", '- pip\n- "pytest>=8"\n')
        self.assertEqual(
            sync.drift(PROJECT_TEXT, environment), ["environment.yml has `pytest`, which pyproject.toml does not declare"]
        )

    def test_missing_range_is_drift(self):
        environment = ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', "scipy")
        self.assertEqual(
            sync.drift(PROJECT_TEXT, environment), ["environment.yml has `scipy` no range, but pyproject.toml has >=1.9.2,<2"]
        )

    def test_entries_of_the_pip_section_and_other_keys_are_ignored(self):
        environment = ENVIRONMENT_TEXT.replace("  - -e .\n", "  - -e .\n  - unrelated>=1\nvariables:\n  - not-a-dependency\n")
        self.assertEqual(sync.drift(PROJECT_TEXT, environment), [])

    def test_unwritable_requirements_are_errors(self):
        for requirement, message in (
            ('"numpy~=1.23"', "conda cannot read the range"),
            ("\"numpy>=1.23; python_version<'3.13'\"", "cannot read"),
            ('"numpy[extra]>=1.23"', "has extras"),
            ('"numpy @ https://example.com/numpy.whl"', "cannot read"),
        ):
            with self.subTest(requirement=requirement):
                project = PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', requirement)
                with self.assertRaisesRegex(ValueError, message):
                    sync.drift(project, ENVIRONMENT_TEXT)

    def test_extras_with_different_ranges_are_drift_and_nothing_else(self):
        project = (
            PROJECT_TEXT.replace('adaptive = ["scipy>=1.9.2,<2"]', 'adaptive = ["scipy>=1.9.2,<3"]')
            .replace('"scipy>=1.9.2,<2"]', '"scipy>=1.10,<2"]')
            .replace("all = ", 'sparse = ["scipy>=1.11,<2"]\nall = ')
        )
        # Three ranges, none of them the one in environment.yml: only the conflict is reported, once.
        environment = ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy>=1.9,<2"')
        self.assertEqual(sync.drift(project, environment), ["pyproject.toml has different ranges of `scipy`"])

    def test_unexpected_project_structures_are_errors(self):
        for project, message in (
            ("", "no `\\[project\\]` table"),
            ('[project]\ndependencies = ["numpy"]\n', "no `\\[project\\]` table"),
            ('[project]\nname = "x"\ndependencies = "numpy"\n', "`dependencies` is not a list of strings"),
            ('[project]\nname = "x"\noptional-dependencies = 1\n', "`optional-dependencies` is not a table"),
            ('[project]\nname = "x"\n[project.optional-dependencies]\nplot = [1]\n', "`optional-dependencies.plot` is not"),
        ):
            with self.subTest(project=project), self.assertRaisesRegex(ValueError, message):
                sync.drift(project, ENVIRONMENT_TEXT)

    def test_indented_lists_are_read(self):
        environment = re.sub(r"^(- |  - )", r"  \1", ENVIRONMENT_TEXT, flags=re.MULTILINE)
        self.assertIn('  - "pytorch>=2.3,<3"\n', environment)
        self.assertIn("    - -e .\n", environment)
        self.assertEqual(sync.drift(PROJECT_TEXT, environment), [])
        project = PROJECT_TEXT.replace('"torch>=2.3,<3"', '"torch>=2.3,<4"')
        synchronized = sync.synchronize(project, environment)
        self.assertEqual(synchronized, environment.replace('"pytorch>=2.3,<3"', '"pytorch>=2.3,<4"'))

    def test_trailing_blanks_are_kept(self):
        environment = ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"\n', '"scipy>=1.9.2,<2"  \n')
        self.assertEqual(sync.drift(PROJECT_TEXT, environment), [])
        project = PROJECT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy>=1.10,<2"')
        self.assertEqual(sync.synchronize(project, environment), environment.replace("1.9.2", "1.10"))

    def test_unreadable_environments_are_errors(self):
        for case, environment, message in (
            ("no dependencies", "name: x\n", "no top-level `dependencies:`"),
            ("twice", ENVIRONMENT_TEXT.replace("- pip\n", "- pip\n- numpy>=1\n"), "listed twice"),
            ("no blank", ENVIRONMENT_TEXT.replace("- pip\n", "- pip\n-numpy\n"), "cannot read"),
            ("brackets", ENVIRONMENT_TEXT.replace("- pip\n", "- pip\n- numpy[version='>=1']\n"), "cannot read"),
            ("channel", ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', '"conda-forge::scipy>=1.9.2,<2"'), "cannot read"),
            ("build after blank", ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy 1.9.2 py_0"'), "cannot read"),
            ("build after =", ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy=1.9.2=py_0"'), "cannot read"),
            ("build after range", ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy>=1.9.2,<2=*_0"'), "cannot read"),
            ("comment without blank", ENVIRONMENT_TEXT.replace('"scipy>=1.9.2,<2"', "scipy>=1.9.2,<2#c"), "cannot read"),
            ("less indented", ENVIRONMENT_TEXT.replace("- python=3.14\n", "  - python=3.14\n"), "cannot read"),
        ):
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, message):
                sync.drift(PROJECT_TEXT, environment)


class SynchronizeTests(unittest.TestCase):
    def test_rewrites_only_the_drifted_ranges_and_keeps_quotes_and_comments(self):
        project = (
            PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"')
            .replace('"PyYAML>=6,<7"', '"PyYAML>=6,<8"')
            .replace('"matplotlib>=3.6,<4"', '"matplotlib>=3.6,<5"')
        )
        expected = (
            ENVIRONMENT_TEXT.replace("- numpy>=1.23.2,<3  # unquoted", '- "numpy>=1.23.2,<4"  # unquoted')
            .replace("'pyyaml>=6,<7'", "'pyyaml>=6,<8'")
            .replace('"matplotlib-base>=3.6,<4"', '"matplotlib-base>=3.6,<5"')
        )
        synchronized = sync.synchronize(project, ENVIRONMENT_TEXT)
        self.assertEqual(synchronized, expected)
        self.assertEqual(sync.drift(project, synchronized), [])

    def test_keeps_agreeing_ranges_as_written(self):
        reordered = ENVIRONMENT_TEXT.replace('"pytorch>=2.3,<3"', '"pytorch <3,>=2.3"')
        self.assertEqual(sync.synchronize(PROJECT_TEXT, reordered), reordered)

    def test_keeps_line_endings(self):
        project = PROJECT_TEXT.replace('"scipy>=1.9.2,<2"', '"scipy>=1.10,<2"')
        crlf = ENVIRONMENT_TEXT.replace("\n", "\r\n")
        self.assertEqual(sync.synchronize(project, crlf), crlf.replace('"scipy>=1.9.2,<2"', '"scipy>=1.10,<2"'))

    def test_leaves_packages_whose_extras_disagree_to_a_person(self):
        project = PROJECT_TEXT.replace('adaptive = ["scipy>=1.9.2,<2"]', 'adaptive = ["scipy>=1.9.2,<3"]')
        self.assertEqual(sync.synchronize(project, ENVIRONMENT_TEXT), ENVIRONMENT_TEXT)

    def test_leaves_missing_and_undeclared_packages_to_a_person(self):
        project = PROJECT_TEXT.replace('"PyYAML>=6,<7",', '"PyYAML>=6,<7",\n  "sympy>=1.12,<2",')
        environment = ENVIRONMENT_TEXT.replace("- pip\n", '- pip\n- "pytest>=8"\n')
        self.assertEqual(sync.synchronize(project, environment), environment)


class CommandTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.write(PROJECT_TEXT, ENVIRONMENT_TEXT)

    def write(self, project, environment):
        (self.root / "pyproject.toml").write_text(project, encoding="utf-8")
        (self.root / "environment.yml").write_bytes(environment.encode("utf-8"))

    def run_main(self, *args, github=False):
        output = io.StringIO()
        # Keep PATH and the rest, which git needs; patch.dict restores GITHUB_ACTIONS afterwards.
        with contextlib.redirect_stdout(output), patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}):
            if not github:
                del os.environ["GITHUB_ACTIONS"]
            code = sync.main(["--root", str(self.root), *args])
        return code, output.getvalue()

    def commit(self):
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is not installed")
        for command in (
            ["init", "--quiet"],
            ["add", "pyproject.toml", "environment.yml"],
            [
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "--no-verify",
                "-m",
                "base",
            ],
        ):
            subprocess.run([git, *command], cwd=self.root, check=True, capture_output=True)

    def test_passes_without_drift(self):
        self.assertEqual(self.run_main(), (0, "environment.yml has the ranges of pyproject.toml\n"))

    def test_fails_on_drift_with_an_annotation(self):
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"'), ENVIRONMENT_TEXT)
        code, output = self.run_main(github=True)
        self.assertEqual(code, 1)
        self.assertIn(
            "::error file=environment.yml::environment.yml has `numpy` >=1.23.2,<3, but pyproject.toml has >=1.23.2,<4\n",
            output,
        )
        self.assertIn("python tools/sync_environment.py --write", output)

    def test_annotation_escapes_special_characters(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}):
            sync.report(["100% a\nb\rc"], "warning")
        self.assertEqual(output.getvalue(), "::warning file=environment.yml::100%25 a%0Ab%0Dc\n")

    def test_allow_drift_warns(self):
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"'), ENVIRONMENT_TEXT)
        code, output = self.run_main("--allow-drift")
        self.assertEqual(code, 0)
        self.assertEqual(output, "warning: environment.yml has `numpy` >=1.23.2,<3, but pyproject.toml has >=1.23.2,<4\n")

    def test_errors_fail_even_when_drift_is_allowed(self):
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy~=1.23"'), ENVIRONMENT_TEXT)
        code, output = self.run_main("--allow-drift")
        self.assertEqual(code, 1)
        self.assertIn("error: pyproject.toml: conda cannot read the range of `numpy~=1.23`", output)

    def test_write_fixes_ranges(self):
        project = PROJECT_TEXT.replace('"torch>=2.3,<3"', '"torch>=2.3,<4"')
        self.write(project, ENVIRONMENT_TEXT)
        code, output = self.run_main("--write")
        self.assertEqual(code, 0)
        self.assertEqual(output, "Updated the ranges in environment.yml\nenvironment.yml has the ranges of pyproject.toml\n")
        environment = (self.root / "environment.yml").read_text(encoding="utf-8")
        self.assertEqual(environment, ENVIRONMENT_TEXT.replace('"pytorch>=2.3,<3"', '"pytorch>=2.3,<4"'))

    def test_write_fails_on_drift_it_cannot_fix(self):
        self.write(PROJECT_TEXT, ENVIRONMENT_TEXT.replace("- pip\n", '- pip\n- "pytest>=8"\n'))
        code, output = self.run_main("--write")
        self.assertEqual(code, 1)
        self.assertIn("error: environment.yml has `pytest`, which pyproject.toml does not declare", output)

    def test_base_tolerates_only_the_drift_it_already_had(self):
        drifted = PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"')
        self.write(drifted, ENVIRONMENT_TEXT)
        self.commit()
        code, output = self.run_main("--base", "HEAD")
        self.assertEqual(code, 0)
        self.assertTrue(output.startswith("warning: environment.yml has `numpy`"))

        self.write(drifted.replace('"torch>=2.3,<3"', '"torch>=2.3,<4"'), ENVIRONMENT_TEXT)
        code, output = self.run_main("--base", "HEAD")
        self.assertEqual(code, 1)
        self.assertIn("warning: environment.yml has `numpy`", output)
        self.assertIn("error: environment.yml has `pytorch` >=2.3,<3, but pyproject.toml has >=2.3,<4", output)

    def test_base_that_cannot_be_read_tolerates_nothing(self):
        self.commit()
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"'), ENVIRONMENT_TEXT)
        code, output = self.run_main("--base", "no-such-revision")
        self.assertEqual(code, 1)
        self.assertIn("warning: cannot read the drift of no-such-revision, so all drift fails", output)
        self.assertIn("error: environment.yml has `numpy`", output)

    def test_base_with_unreadable_files_tolerates_nothing(self):
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy~=1.23"'), ENVIRONMENT_TEXT)
        self.commit()
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"'), ENVIRONMENT_TEXT)
        code, output = self.run_main("--base", "HEAD")
        self.assertEqual(code, 1)
        self.assertIn("warning: cannot read the drift of HEAD, so all drift fails: pyproject.toml: conda cannot read", output)
        self.assertIn("error: environment.yml has `numpy`", output)

    def test_base_without_git_tolerates_nothing(self):
        self.write(PROJECT_TEXT.replace('"numpy>=1.23.2,<3"', '"numpy>=1.23.2,<4"'), ENVIRONMENT_TEXT)
        with patch.object(sync.subprocess, "run", side_effect=FileNotFoundError("no git")):
            code, output = self.run_main("--base", "HEAD")
        self.assertEqual(code, 1)
        self.assertIn("warning: cannot read the drift of HEAD, so all drift fails: no git", output)


if __name__ == "__main__":
    unittest.main()
