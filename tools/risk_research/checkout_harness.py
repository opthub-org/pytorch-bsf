"""Run research checks in a clean checkout without inheriting its parent's imports.

This is a validation harness, not a sandbox for untrusted Python code. Outputs
must be outside the checkout so its HEAD, tree and complete clean status can
be checked before and after each command, including unsuccessful commands.
"""

import keyword
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckoutState:
    """Record the committed revision and working-tree state used by a check."""

    head: str
    tree: str
    status: str


def checkout_state(checkout: Path) -> CheckoutState:
    """Read HEAD, its tree and tracked, untracked and ignored paths."""
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", *arguments], cwd=checkout, env=environment, text=True
        ).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != checkout.resolve():
        raise ValueError("checkout must be the repository root")
    return CheckoutState(
        git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}"), git("status", "--porcelain", "-uall", "--ignored=matching")
    )


def _module_name(name: object) -> bool:
    return type(name) is str and all(part.isidentifier() and not keyword.iskeyword(part) for part in name.split("."))


def _validate_arguments(arguments: list[str]) -> None:
    if (
        type(arguments) is not list
        or len(arguments) < 2
        or any(type(value) is not str for value in arguments)
        or arguments[0] not in ("-m", "-c")
        or (arguments[0] == "-m" and not _module_name(arguments[1]))
    ):
        raise ValueError("validation supports only explicit -m or -c commands without Python path flags")


def run_python(
    checkout: Path,
    arguments: list[str],
    *,
    modules: tuple[str, ...],
    executable: str = sys.executable,
    expected_head: str | None = None,
    expected_tree: str | None = None,
    timeout: float = 60,
    no_site: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run Python after checking import origins, then require an unchanged checkout.

    ``PYTHONPATH=.`` is deliberately relative: a nested child that changes its
    checkout must not keep importing this checkout through an absolute path.
    Only ``-m module`` and ``-c code`` modes are supported so their initial
    search path agrees with the import preflight. Python path flags and script
    filenames are rejected. A nonzero return code is returned normally;
    timeout and source changes raise.
    """
    if type(modules) is not tuple or not modules or any(not _module_name(name) for name in modules):
        raise ValueError("modules must be a nonempty tuple of valid dotted module names")
    _validate_arguments(arguments)
    checkout = checkout.resolve()
    before = checkout_state(checkout)
    if before.status:
        raise ValueError("validation checkout must be clean before execution")
    if expected_head is not None and before.head != expected_head:
        raise ValueError("validation checkout HEAD differs from the expected revision")
    if expected_tree is not None and before.tree != expected_tree:
        raise ValueError("validation checkout tree differs from the expected tree")
    environment = {
        **os.environ,
        "PYTHONPATH": ".",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    prefix = [executable, *(["-S"] if no_site else [])]
    origin_check = """
import importlib
import sys
from pathlib import Path
root = Path.cwd().resolve()
for name in sys.argv[1:]:
    origin = getattr(importlib.import_module(name), '__file__', None)
    if origin is None or not Path(origin).resolve().is_relative_to(root):
        raise RuntimeError('module import escaped validation checkout: ' + name)
"""
    try:
        preflight = subprocess.run(
            [*prefix, "-c", origin_check, *modules],
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if preflight.returncode:
            raise RuntimeError("validation import-origin check failed: " + preflight.stderr.strip())
        return subprocess.run(
            [*prefix, *arguments], cwd=checkout, env=environment, capture_output=True, text=True, timeout=timeout, check=False
        )
    finally:
        if checkout_state(checkout) != before:
            raise RuntimeError("validation checkout HEAD, tree or clean status changed")
