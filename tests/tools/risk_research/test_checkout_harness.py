"""Keep isolated research subprocesses bound to the intended clean checkout."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.risk_research.checkout_harness import checkout_state, run_python


def _git(checkout, *arguments):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *arguments],
        cwd=checkout,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Harness Test",
            "GIT_AUTHOR_EMAIL": "harness@example.invalid",
            "GIT_COMMITTER_NAME": "Harness Test",
            "GIT_COMMITTER_EMAIL": "harness@example.invalid",
        },
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def checkout(tmp_path):
    directory = tmp_path / "checkout"
    directory.mkdir()
    (directory / "probe.py").write_text("VALUE = 'child checkout'\n", encoding="utf-8")
    _git(directory, "init")
    _git(directory, "add", "probe.py")
    _git(directory, "commit", "-m", "Test source")
    return directory


def test_parent_absolute_pythonpath_is_replaced_and_child_origin_is_correct(checkout, tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "probe.py").write_text("raise AssertionError('imported parent checkout')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(parent))
    state = checkout_state(checkout)
    result = run_python(
        checkout,
        ["-c", "import json,os,probe; print(json.dumps([probe.VALUE,probe.__file__,os.environ['PYTHONPATH']]))"],
        modules=("probe",),
        expected_head=state.head,
        expected_tree=state.tree,
        no_site=True,
    )
    assert result.returncode == 0, result.stderr
    value, origin, pythonpath = json.loads(result.stdout)
    assert value == "child checkout" and Path(origin).resolve() == checkout / "probe.py"
    assert pythonpath == "." and checkout_state(checkout) == state


def test_relative_pythonpath_follows_a_nested_child_checkout(checkout, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "probe.py").write_text("VALUE = 'nested checkout'\n", encoding="utf-8")
    source = """
import subprocess,sys
result = subprocess.run([sys.executable,'-S','-c','import probe; print(probe.VALUE)'],
                        cwd=sys.argv[1],capture_output=True,text=True,check=True)
print(result.stdout,end='')
"""
    result = run_python(checkout, ["-c", source, str(nested)], modules=("probe",), no_site=True)
    assert result.returncode == 0 and result.stdout == "nested checkout\n"


@pytest.mark.parametrize("expected", ["head", "tree"])
def test_wrong_committed_baseline_is_rejected_before_execution(checkout, expected):
    with pytest.raises(ValueError, match="expected"):
        run_python(
            checkout,
            ["-c", "raise AssertionError('must not execute')"],
            modules=("probe",),
            expected_head="0" * 40 if expected == "head" else None,
            expected_tree="0" * 40 if expected == "tree" else None,
        )


def test_initial_dirty_or_untracked_source_is_rejected(checkout):
    (checkout / "untracked.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean before"):
        run_python(checkout, ["-c", "pass"], modules=("probe",))


def test_installed_module_origin_outside_checkout_is_rejected(checkout):
    with pytest.raises(RuntimeError, match="module import escaped"):
        run_python(checkout, ["-c", "pass"], modules=("json",))


def test_changed_source_is_detected_even_when_command_fails(checkout):
    with pytest.raises(RuntimeError, match="clean status changed"):
        run_python(
            checkout,
            ["-c", "from pathlib import Path; Path('probe.py').write_text('changed'); raise SystemExit(2)"],
            modules=("probe",),
        )


def test_head_change_without_file_changes_is_detected(checkout):
    source = """
import subprocess
subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false',
                '-c','user.name=Harness Test','-c','user.email=harness@example.invalid',
                'commit','--allow-empty','-m','Changed head'],check=True,capture_output=True)
"""
    with pytest.raises(RuntimeError, match="HEAD, tree"):
        run_python(checkout, ["-c", source], modules=("probe",))


def test_failure_and_timeout_keep_the_checkout_clean(checkout):
    state = checkout_state(checkout)
    result = run_python(checkout, ["-c", "raise SystemExit(7)"], modules=("probe",))
    assert result.returncode == 7 and checkout_state(checkout) == state
    with pytest.raises(subprocess.TimeoutExpired):
        run_python(checkout, ["-c", "import time; time.sleep(5)"], modules=("probe",), timeout=0.2)
    assert checkout_state(checkout) == state


@pytest.mark.parametrize("modules", [(), ("",), ("probe..wrong",), ("bad-name",), ("probe.class",), (None,)])
def test_missing_or_invalid_import_origin_modules_are_rejected(checkout, modules):
    with pytest.raises(ValueError, match="nonempty tuple"):
        run_python(checkout, ["-c", "raise AssertionError('must not execute')"], modules=modules)


def test_nested_path_is_not_a_checkout_root(checkout):
    nested = checkout / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="repository root"):
        checkout_state(nested)


@pytest.mark.parametrize(
    "arguments",
    [[], ["-c"], ["-m", "../outside"], ["-I", "-c", "pass"], ["-P", "-c", "pass"], ["-S", "-c", "pass"]],
)
def test_only_explicit_module_or_code_modes_are_supported(checkout, arguments):
    with pytest.raises(ValueError, match="-m or -c"):
        run_python(checkout, arguments, modules=("probe",))


def test_external_script_cannot_change_the_checked_import_search_path(checkout, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "probe.py").write_text("VALUE = 'external import'\n", encoding="utf-8")
    script = external / "check.py"
    script.write_text("import probe; print(probe.VALUE)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="-m or -c"):
        run_python(checkout, [str(script)], modules=("probe",))
