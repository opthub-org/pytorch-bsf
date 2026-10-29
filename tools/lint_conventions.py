"""Check the conventions of CONTRIBUTING.md that Ruff cannot check.

The ``lint-conventions`` hook of ``.pre-commit-config.yaml`` runs this script on the committed Python files,
and CI runs it on all files with ``pre-commit run --all-files``. To run it by hand::

    pre-commit run lint-conventions --all-files

It uses only the standard library and reports each violation on one line with the file, line, rule code,
reason and fix. Rules:

BSF001
    A string literal that reaches the console must be ASCII, so that it reads correctly under ``LANG=C``
    (write "Bezier", not "Bézier"). Checked: the arguments of ``print`` (with ``sep`` and ``end``), of the
    ``logging`` methods (``debug``, ``info``, ``warning``, ``warn``, ``error``, ``exception``, ``critical``,
    ``fatal``, ``log``, with ``msg``), of ``warnings.warn`` (with ``message``), of ``add_argument_group``,
    and of calls whose name ends with ``Error``, ``Warning`` or ``Exception`` (with all keywords); the
    ``description``, ``epilog`` and ``usage`` keywords of ``ArgumentParser`` and ``add_parser``, the
    ``help`` keyword of ``add_parser`` and ``add_argument``, and the ``title``, ``description`` and ``help``
    keywords of ``add_subparsers``. A message is followed through f-strings (their constant parts and
    replacement fields), ``+`` and ``%`` (including a tuple or dict of ``%`` arguments), conditional
    expressions and ``"...".format(...)`` (with its keywords). Not checked: ``SystemExit``, ``sys.exit`` and
    ``assert`` messages, format specs of f-string fields, and calls of anything but a name or attribute.
BSF002
    In ``tests/``, every ``assert_close`` call (``torch.testing.assert_close``) passes both ``rtol`` and
    ``atol`` as keywords other than ``None``, because its default tolerance is not derived from the
    computation under test. Exact comparisons pass ``rtol=0, atol=0``. A ``**kwargs`` argument counts as
    explicit.

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

# Calls whose positional arguments reach the console, mapped to the keywords that also do
_MESSAGE_CALLS = {
    "print": {"sep", "end"},
    **dict.fromkeys(["debug", "info", "warning", "error", "exception", "critical", "fatal", "log"], {"msg"}),
    "warn": {"message", "msg"},  # warnings.warn and the deprecated logging.warn
    "add_argument_group": {"title", "description"},
}
_EXCEPTION_SUFFIXES = ("Error", "Warning", "Exception")
# Calls of which only some keywords reach the console
_ARGPARSE_KEYWORDS = {
    "ArgumentParser": {"description", "epilog", "usage"},
    "add_parser": {"description", "epilog", "usage", "help"},
    "add_argument": {"help"},
    "add_subparsers": {"title", "description", "help"},
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
    """Return the name of the called function or method (``warn`` for ``warnings.warn(...)``)."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _console_arguments(call: ast.Call) -> Iterator[ast.expr]:
    """Yield the arguments of a call whose strings reach the console."""
    name = _call_name(call)
    if name is None:
        return
    if name.endswith(_EXCEPTION_SUFFIXES):
        yield from call.args
        yield from (keyword.value for keyword in call.keywords)
        return
    if name in _MESSAGE_CALLS:
        yield from call.args
    keywords = _MESSAGE_CALLS.get(name) or _ARGPARSE_KEYWORDS.get(name, set())
    yield from (keyword.value for keyword in call.keywords if keyword.arg in keywords)


def _text_parts(node: ast.expr) -> list[ast.expr]:
    """Return the subexpressions of an expression whose text becomes part of the resulting text."""
    if isinstance(node, ast.JoinedStr):
        return list(node.values)
    if isinstance(node, ast.FormattedValue):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return [node.body, node.orelse]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [node.left, node.right]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.right, ast.Tuple):
            return [node.left, *node.right.elts]
        if isinstance(node.right, ast.Dict):
            return [node.left, *node.right.values]
        return [node.left, node.right]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return [node.func.value, *node.args, *(keyword.value for keyword in node.keywords)]
    return []


def _message_strings(node: ast.expr) -> Iterator[tuple[ast.Constant, str]]:
    """Yield the string literals, and their values, of an expression that become part of the resulting text."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node, node.value
    else:
        for part in _text_parts(node):
            yield from _message_strings(part)


def _line_of(character: str, node: ast.expr, source: str) -> int:
    """Return the line of the first occurrence of a character in the source of a node, or the node's first line."""
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


def _is_none(node: ast.expr) -> bool:
    """Return whether an expression is the literal ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def _check_tolerance(call: ast.Call, path: str) -> Iterator[Violation]:
    """BSF002: ``assert_close`` passes both ``rtol`` and ``atol``."""
    if _call_name(call) != "assert_close":
        return
    keywords = {keyword.arg for keyword in call.keywords if not _is_none(keyword.value)}
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
    """Map line numbers to the rule codes suppressed by a comment on that line."""
    suppressed: dict[int, set[str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        match = _SUPPRESSION.search(token.string) if token.type == tokenize.COMMENT else None
        if match:
            suppressed.setdefault(token.start[0], set()).update(code.strip() for code in match.group(1).split(","))
    return suppressed


def check_source(source: str, path: str = "<string>", *, is_test: bool = False) -> list[Violation]:
    """Check the source code of a Python file.

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
    ValueError
        If the source code contains a null byte (Python 3.11 and earlier; later versions raise SyntaxError).
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
    """Return whether a file is in the ``tests/`` directory of the repository."""
    return path.resolve().is_relative_to(TESTS)


def main(argv: Sequence[str] | None = None) -> int:
    """Check the given files, print the violations and return 1 if there are any, 0 otherwise."""
    parser = argparse.ArgumentParser(description="Check the conventions of CONTRIBUTING.md that Ruff cannot check.")
    parser.add_argument("files", nargs="*", type=Path, help="Python files to check")
    arguments = parser.parse_args(argv)
    failed = False
    for path in arguments.files:
        try:
            with tokenize.open(path) as file:  # honors a PEP 263 encoding declaration
                source = file.read()
            violations = check_source(source, str(path), is_test=is_test_file(path))
        except (OSError, SyntaxError, ValueError) as error:  # ValueError: undecodable bytes or, before 3.12, null bytes
            if isinstance(error, SyntaxError):
                line, reason = error.lineno, error.msg
            else:
                line, reason = None, error.strerror if isinstance(error, OSError) else str(error)
            print(f"{path}:{line or 1}: cannot parse: {reason}")
            failed = True
            continue
        for violation in violations:
            print(violation)
        failed = failed or bool(violations)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
