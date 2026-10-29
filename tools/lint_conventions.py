"""Check the conventions of CONTRIBUTING.md that Ruff cannot check.

The ``lint-conventions`` hook of ``.pre-commit-config.yaml`` runs this script on the committed Python files,
and CI runs it on all files with ``pre-commit run --all-files``. To run it by hand::

    python tools/lint_conventions.py torch_bsf/*.py tests/*.py

It uses only the standard library and reports each violation on one line with the file, line, rule code,
reason and fix. Rules:

BSF001
    A string literal that reaches the console must be ASCII, so that it reads correctly under ``LANG=C``
    (write "Bezier", not "Bézier"). Checked: the arguments of ``print``, of the ``logging`` methods
    (``debug``, ``info``, ``warning``, ``error``, ``exception``, ``critical``, ``log``), of ``warn``
    (``warnings.warn``) and of calls whose name ends with ``Error``, ``Warning`` or ``Exception``; the
    ``description``, ``epilog``, ``usage`` and ``help`` keywords of ``ArgumentParser``, ``add_parser`` and
    ``add_argument``. A message is followed through f-strings (their constant parts and replacement fields),
    ``+`` and ``%``, conditional expressions and ``"...".format(...)``.
BSF002
    In ``tests/``, every ``assert_close`` call (``torch.testing.assert_close``) passes both ``rtol`` and
    ``atol`` as keywords, because its default tolerance is not derived from the computation under test.
    Exact comparisons pass ``rtol=0, atol=0``. A ``**kwargs`` argument counts as explicit.

A comment ``# lint-conventions: ignore[BSF002]`` (codes separated by commas) on any line of the call
suppresses those rules for that call; say why in the same comment.
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

_SPELLING = 'CONTRIBUTING.md, Coding Conventions, "Bezier" spelling section'
_TOLERANCES = "CONTRIBUTING.md, Writing Tests, Property-based tests of numerical code, Tolerances"

_LOGGING = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
_EXCEPTION_SUFFIXES = ("Error", "Warning", "Exception")
_PRINT_KEYWORDS = {"sep", "end"}
_WARN_KEYWORDS = {"message"}
_ARGPARSE_KEYWORDS = {
    "ArgumentParser": {"description", "epilog", "usage"},
    "add_parser": {"description", "epilog", "usage", "help"},
    "add_argument": {"help"},
}
_SUPPRESSION = re.compile(r"#\s*lint-conventions:\s*ignore\[([A-Z0-9,\s]+)\]")


@dataclass(frozen=True)
class Violation:
    """A violation of a rule at a line of a file."""

    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _call_name(call: ast.Call) -> str | None:
    """Returns the name of the called function or method (``warn`` for ``warnings.warn(...)``)."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _console_arguments(call: ast.Call) -> Iterator[ast.expr]:
    """Yields the arguments of a call whose strings reach the console."""
    name = _call_name(call)
    if name is None:
        return
    keywords = _ARGPARSE_KEYWORDS.get(name, set())
    if name == "print" or name in _LOGGING or name.endswith(_EXCEPTION_SUFFIXES):
        yield from call.args
        keywords = _PRINT_KEYWORDS if name == "print" else _WARN_KEYWORDS if name == "warn" else set()
    for keyword in call.keywords:
        if keyword.arg in keywords:
            yield keyword.value


def _message_strings(node: ast.expr) -> Iterator[tuple[ast.Constant, str]]:
    """Yields the string literals, and their values, of an expression that become part of the resulting text."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node, node.value
    elif isinstance(node, ast.JoinedStr):
        for value in node.values:
            yield from _message_strings(value)
    elif isinstance(node, ast.FormattedValue):
        yield from _message_strings(node.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        yield from _message_strings(node.left)
        yield from _message_strings(node.right)
    elif isinstance(node, ast.IfExp):
        yield from _message_strings(node.body)
        yield from _message_strings(node.orelse)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        yield from _message_strings(node.func.value)
        for argument in node.args:
            yield from _message_strings(argument)


def _line_of(character: str, node: ast.expr, source: str) -> int:
    """Returns the line of the first occurrence of a character in the source of a node, or the node's first line."""
    segment = ast.get_source_segment(source, node) or ""
    index = segment.find(character)  # -1 if the character is written as an escape sequence
    return node.lineno + segment.count("\n", 0, index) if index >= 0 else node.lineno


def _check_ascii(call: ast.Call, source: str, path: str) -> Iterator[Violation]:
    """BSF001: console-facing string literals are ASCII."""
    for argument in _console_arguments(call):
        for constant, text in _message_strings(argument):
            character = next((c for c in text if not c.isascii()), None)
            if character is not None:
                yield Violation(
                    path,
                    _line_of(character, constant, source),
                    "BSF001",
                    f"non-ASCII character {ascii(character)} in a console-facing string, which is garbled or "
                    "raises an encoding error under LANG=C. Fix: use ASCII only, e.g. 'Bezier' without the accent. "
                    f"See {_SPELLING}.",
                )


def _check_tolerance(call: ast.Call, path: str) -> Iterator[Violation]:
    """BSF002: ``assert_close`` passes both ``rtol`` and ``atol``."""
    if _call_name(call) != "assert_close":
        return
    keywords = {keyword.arg for keyword in call.keywords}
    missing = [name for name in ("rtol", "atol") if name not in keywords]
    if missing and None not in keywords:  # None is the name of a **kwargs argument
        yield Violation(
            path,
            call.lineno,
            "BSF002",
            f"assert_close without {' and '.join(missing)}, so it falls back to a default tolerance that is not "
            "derived from the computation. Fix: pass rtol=0, atol=0 for an exact comparison, or derive both from "
            f"the unit roundoff U = 2**-53 and write the derivation in a comment. See {_TOLERANCES}.",
        )


def _suppressions(source: str) -> dict[int, set[str]]:
    """Maps line numbers to the rule codes suppressed by a comment on that line."""
    suppressed: dict[int, set[str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        match = _SUPPRESSION.search(token.string) if token.type == tokenize.COMMENT else None
        if match:
            suppressed.setdefault(token.start[0], set()).update(code.strip() for code in match.group(1).split(","))
    return suppressed


def check_source(source: str, path: str = "<string>", *, is_test: bool = False) -> list[Violation]:
    """Checks the source code of a Python file.

    Parameters
    ----------
    source : str
        The source code.
    path : str
        The file name used in the reported violations.
    is_test : bool
        Whether the file is a test, to which BSF002 applies.

    Returns
    -------
    list[Violation]
        The violations not suppressed by a comment, sorted by line.

    Raises
    ------
    SyntaxError
        If the source code cannot be parsed.
    """
    tree = ast.parse(source, filename=path)
    suppressed = _suppressions(source)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        found = list(_check_ascii(node, source, path))
        if is_test:
            found += _check_tolerance(node, path)
        lines = range(node.lineno, (node.end_lineno or node.lineno) + 1)
        violations += [v for v in found if not any(v.code in suppressed.get(line, ()) for line in lines)]
    return sorted(violations, key=lambda v: (v.line, v.code))


def is_test_file(path: Path) -> bool:
    """Returns whether a file is in the ``tests/`` directory of the repository."""
    return path.resolve().is_relative_to(TESTS)


def main(argv: Sequence[str] | None = None) -> int:
    """Checks the given files, prints the violations and returns 1 if there are any, 0 otherwise."""
    parser = argparse.ArgumentParser(description="Check the conventions of CONTRIBUTING.md that Ruff cannot check.")
    parser.add_argument("files", nargs="*", type=Path, help="Python files to check")
    arguments = parser.parse_args(argv)
    failed = False
    for path in arguments.files:
        with tokenize.open(path) as file:  # honors a PEP 263 encoding declaration
            source = file.read()
        try:
            violations = check_source(source, str(path), is_test=is_test_file(path))
        except SyntaxError as error:
            print(f"{path}:{error.lineno}: cannot parse: {error.msg}")
            failed = True
            continue
        for violation in violations:
            print(violation)
        failed = failed or bool(violations)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
