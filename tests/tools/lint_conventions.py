"""Tests for tools/lint_conventions.py, the checker of the conventions that Ruff cannot check."""

import textwrap

import pytest

from tools import lint_conventions
from tools.lint_conventions import check_source, is_test_file, main


def codes(source: str, *, is_test: bool = False) -> list[tuple[int, str]]:
    return [(v.line, v.code) for v in check_source(textwrap.dedent(source), is_test=is_test)]


# BSF001: console-facing strings are ASCII


@pytest.mark.parametrize(
    "source",
    [
        'print("Bézier")',
        'print("x", "Bézier")',
        'print("x", sep="é")',
        'print(f"{n} Bézier simplices")',
        "print(f\"{'Bézier'}\")",
        'print("a" + "é")',
        'print("%s é" % name)',
        'print("{} é".format(name))',
        'print("x" if flag else "é")',
        'raise ValueError("not a Bézier simplex")',
        'raise ValueError(f"{index!r} is not a control point of a Bézier simploid")',
        'raise errors.ConvergenceError("Bézier")',
        'raise RuntimeWarning("é")',
        'raise BaseException("é")',
        'warnings.warn("Bézier")',
        'warnings.warn(message="Bézier", category=UserWarning)',
        'warn("Bézier")',
        'logger.debug("é")',
        'logger.info("é")',
        'logger.warning("é")',
        'logging.error("é")',
        'logger.exception("é")',
        'logger.critical("é")',
        'logger.log(logging.INFO, "é")',
        'parser.error("é")',
        'ArgumentParser(description="Bézier fitting")',
        'argparse.ArgumentParser(epilog="é")',
        'ArgumentParser(usage="é")',
        'parser.add_argument("--degree", help="degree of the Bézier simplex")',
        'commands.add_parser("fit", help="é")',
        'commands.add_parser("fit", description="é")',
        'print("B\\xe9zier")',
    ],
)
def test_non_ascii_console_strings_are_reported(source):
    assert codes(source) == [(1, "BSF001")]


@pytest.mark.parametrize(
    "source",
    [
        '"""A Bézier simplex."""',
        "# Bézier",
        'name = "Bézier"',
        "print(name)",
        'print(table["Bézier"])',
        "print(f\"{table['Bézier']}\")",
        'print("Bezier", end="\\n")',
        'raise ValueError("not a Bezier simplex")',
        'raise ValueError(f"{index!r} is not a Bezier simploid")',
        'pytest.raises(ValueError, match="Bézier")',
        'error_message("é")',
        'ArgumentParser(prog="é")',
        'parser.add_argument("--é", help="ASCII")',
        'plot(title="Bézier")',
        'warnings.warn("ASCII", category=UserWarning, stacklevel=2)',
        'raise ValueError(b"\\xe9")',
    ],
)
def test_ascii_and_non_console_strings_are_accepted(source):
    assert codes(source) == []


def test_reported_line_is_the_line_of_the_character():
    source = """\
        raise ValueError(
            f"The objective values have the shape {shape}, "
            "but linear_scalarization needs a Bézier simplex."
        )
        """
    assert codes(source) == [(3, "BSF001")]


def test_each_offending_literal_is_reported():
    assert codes('print("é")\nraise ValueError("é", "ü")\n') == [(1, "BSF001"), (2, "BSF001"), (2, "BSF001")]


# BSF002: assert_close in tests passes rtol and atol


@pytest.mark.parametrize(
    "source",
    [
        "torch.testing.assert_close(a, b)",
        "torch.testing.assert_close(a, b, rtol=0)",
        "torch.testing.assert_close(a, b, atol=1e-12)",
        "assert_close(a, b, check_dtype=False)",
        "testing.assert_close(a, b, msg='x')",
    ],
)
def test_assert_close_without_both_tolerances_is_reported_in_tests(source):
    assert codes(source, is_test=True) == [(1, "BSF002")]
    assert codes(source, is_test=False) == []


@pytest.mark.parametrize(
    "source",
    [
        "torch.testing.assert_close(a, b, rtol=0, atol=0)",
        "torch.testing.assert_close(a, b, atol=8 * U, rtol=0)",
        "torch.testing.assert_close(a, b, **tolerances)",
        "np.testing.assert_allclose(a, b)",
        "assert_closed(a, b)",
    ],
)
def test_assert_close_with_both_tolerances_is_accepted(source):
    assert codes(source, is_test=True) == []


def test_missing_tolerances_are_named():
    (violation,) = check_source("torch.testing.assert_close(a, b, rtol=0)", "tests/t.py", is_test=True)
    assert "without atol," in violation.message
    (violation,) = check_source("torch.testing.assert_close(a, b)", "tests/t.py", is_test=True)
    assert "without rtol and atol," in violation.message


# Suppression comments


@pytest.mark.parametrize(
    "source",
    [
        'print("é")  # lint-conventions: ignore[BSF001] the reason',
        'print("é")  #lint-conventions:ignore[BSF001]',
        'print("é")  # lint-conventions: ignore[BSF002, BSF001]',
        'print(\n    "é"  # lint-conventions: ignore[BSF001]\n)',
        'print(\n    "é"\n)  # lint-conventions: ignore[BSF001]',
    ],
)
def test_suppression_comment_on_a_line_of_the_node(source):
    assert codes(source) == []


@pytest.mark.parametrize(
    "source",
    [
        'print("é")  # lint-conventions: ignore[BSF002]',
        'print("é")  # noqa',
        'print("é # lint-conventions: ignore[BSF001]")',
        '# lint-conventions: ignore[BSF001]\nprint("é")',
        'print("x", "é")  # lint-conventions: ignore[BSF001X]',
    ],
)
def test_suppression_needs_the_code_in_a_comment_on_the_node(source):
    assert [code for _, code in codes(source)] == ["BSF001"]


def test_suppression_of_assert_close():
    source = """\
        torch.testing.assert_close(a, b)  # lint-conventions: ignore[BSF002] default tolerance on purpose
        torch.testing.assert_close(
            a,
            b,  # lint-conventions: ignore[BSF002]
        )
        torch.testing.assert_close(a, b)  # lint-conventions: ignore[BSF001]
        """
    assert codes(source, is_test=True) == [(6, "BSF002")]


# Messages and the command line


def test_messages_give_location_code_reason_fix_and_section():
    (ascii_violation,) = check_source('\n\nprint("Bézier")\n', "torch_bsf/module.py")
    text = str(ascii_violation)
    assert text.startswith("torch_bsf/module.py:3: BSF001 non-ASCII character '\\xe9' in a console-facing string")
    assert "LANG=C" in text and "Fix: use ASCII only" in text and "CONTRIBUTING.md" in text
    assert text.isascii()
    (tolerance_violation,) = check_source("assert_close(a, b)", "tests/module.py", is_test=True)
    text = str(tolerance_violation)
    assert text.startswith("tests/module.py:1: BSF002 assert_close without rtol and atol")
    assert "Fix: pass rtol=0, atol=0 for an exact comparison" in text and "U = 2**-53" in text
    assert "CONTRIBUTING.md" in text and "Tolerances" in text
    assert text.isascii()


def test_is_test_file():
    assert is_test_file(lint_conventions.TESTS / "module.py")
    assert is_test_file(lint_conventions.TESTS / "tools" / "module.py")
    assert not is_test_file(lint_conventions.ROOT / "torch_bsf" / "module.py")
    assert not is_test_file(lint_conventions.ROOT / "tools" / "tests.py")


def test_main_reports_violations_and_exit_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lint_conventions, "TESTS", tmp_path / "tests")
    (tmp_path / "tests").mkdir()
    clean = tmp_path / "clean.py"
    clean.write_text('"""Bézier."""\nprint("Bezier")\n', encoding="utf-8")
    module = tmp_path / "module.py"
    module.write_text('raise ValueError("Bézier")\nassert_close(a, b)\n', encoding="utf-8")
    test = tmp_path / "tests" / "test.py"
    test.write_text("assert_close(a, b)\n", encoding="utf-8")
    latin = tmp_path / "latin.py"
    latin.write_bytes('# -*- coding: latin-1 -*-\nprint("Bézier")\n'.encode("latin-1"))

    assert main([str(clean)]) == 0
    assert capsys.readouterr().out == ""
    assert main([]) == 0
    assert main([str(clean), str(module), str(test), str(latin)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert [line.split(" ")[:2] for line in out] == [
        [f"{module}:1:", "BSF001"],
        [f"{test}:1:", "BSF002"],
        [f"{latin}:2:", "BSF001"],
    ]


def test_main_reports_syntax_errors(tmp_path, capsys):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n", encoding="utf-8")
    assert main([str(broken)]) == 1
    assert capsys.readouterr().out.startswith(f"{broken}:1: cannot parse:")
