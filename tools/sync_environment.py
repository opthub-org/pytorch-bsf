"""Check that environment.yml declares the dependency ranges of pyproject.toml, and copy them over with --write.

Every requirement of ``[project].dependencies`` and of the extras, except ``develop``, needs a conda dependency of the
same name (``CONDA_NAMES`` maps the names that differ on conda-forge) and the same clauses, in any order. Conda
dependencies that pyproject.toml does not declare are drift too, except ``CONDA_ONLY``. Requirements with markers or
extras, or clauses that conda cannot read, are errors, as is a package whose extras declare different ranges.

With ``--base REV``, only drift that REV did not have fails; the rest is reported as a warning, so drift that is already
on master does not fail unrelated PRs. ``--allow-drift`` reports every drift as a warning. GitHub Actions annotates the
reports. The script needs only the standard library, like the other development scripts.
"""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "pyproject.toml"
ENVIRONMENT = "environment.yml"
# Extras that environment.yml does not install: the development tools come from the Conda workflow or pip.
EXCLUDED_EXTRAS = frozenset({"develop"})
# conda-forge names that differ from the PyPI name. matplotlib-base is matplotlib without the Qt GUI backend.
CONDA_NAMES = {"torch": "pytorch", "matplotlib": "matplotlib-base"}
# Conda dependencies that pyproject.toml has no requirement for.
CONDA_ONLY = frozenset({"python", "pip"})
REQUIREMENT = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(\[[^\]]*\])?\s*(.*?)\s*")
# The PEP 440 operators that a conda match spec reads the same way; `~=` and `===` are not among them.
CLAUSE = re.compile(r"(?:==|!=|<=|>=|<|>)[A-Za-z0-9.*+!-]+")
DEPENDENCIES = re.compile(r"dependencies:[ \t]*(?:#.*)?")
# One entry of a block sequence: indentation, then a quoted or plain scalar, then an optional comment.
ENTRY = re.compile(r"""(?P<indent>[ \t]*)- (?P<quote>["']?)(?P<value>.*?)(?P=quote)(?P<comment>[ \t]+#.*)?""")
# A conda match spec of a name and an optional range, such as `numpy>=1.23.2,<3` or `python=3.14`.
MATCH_SPEC = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*([^\[\]\s]*)\s*")


def normalize(name: str) -> str:
    """Return the PEP 503 normalized name, which conda-forge also uses."""
    return re.sub(r"[-_.]+", "-", name).lower()


def clauses(specifier: str) -> list[str]:
    """Split a version specifier into its clauses, without blanks."""
    return [clause for clause in re.sub(r"\s+", "", specifier).split(",") if clause]


def project_ranges(text: str) -> dict[str, list[str]]:
    """Return the clauses of every requirement that environment.yml must declare, by conda name.

    Raises
    ------
    ValueError
        If a requirement cannot be written in environment.yml, or extras declare different ranges of one package.
    """
    project = tomllib.loads(text)["project"]
    own = normalize(project["name"])
    requirements = list(project.get("dependencies", []))
    for extra, values in project.get("optional-dependencies", {}).items():
        if extra not in EXCLUDED_EXTRAS:
            requirements.extend(values)
    ranges: dict[str, list[str]] = {}
    for requirement in requirements:
        match = REQUIREMENT.fullmatch(requirement)
        if match is None or ";" in requirement or "@" in requirement:
            raise ValueError(f"{PROJECT}: cannot read `{requirement}`; write a name and a version range")
        name = normalize(match[1])
        if name == own:
            continue  # The `all` extra refers to the other extras.
        if match[2]:
            raise ValueError(f"{PROJECT}: `{requirement}` has extras, which environment.yml cannot install")
        required = clauses(match[3])
        if not all(CLAUSE.fullmatch(clause) for clause in required):
            raise ValueError(f"{PROJECT}: conda cannot read the range of `{requirement}`; use ==, !=, <, <=, > or >=")
        conda = CONDA_NAMES.get(name, name)
        if conda in ranges and sorted(ranges[conda]) != sorted(required):
            raise ValueError(f"{PROJECT}: `{match[1]}` has different ranges in different extras")
        ranges[conda] = required
    return ranges


def environment_entries(text: str) -> dict[str, tuple[int, list[str]]]:
    """Return the line index and clauses of every top-level conda dependency, by normalized name.

    The pip section and its entries are skipped. Only block sequences, as environment.yml uses, are read.

    Raises
    ------
    ValueError
        If the dependencies are missing, an entry cannot be read, or a package is listed twice.
    """
    entries: dict[str, tuple[int, list[str]]] = {}
    for index, value in _dependency_values(text.splitlines()):
        match = MATCH_SPEC.fullmatch(value)
        if match is None:
            raise ValueError(f"{ENVIRONMENT}:{index + 1}: cannot read `{value}`")
        name = normalize(match[1])
        if name in entries:
            raise ValueError(f"{ENVIRONMENT}:{index + 1}: `{match[1]}` is listed twice")
        entries[name] = (index, clauses(match[2]))
    return entries


def _dependency_values(lines: list[str]) -> Iterator[tuple[int, str]]:
    """Yield the line index and scalar of every entry of the top-level dependencies, except `- pip:`."""
    start = next((index for index, line in enumerate(lines) if DEPENDENCIES.fullmatch(line)), None)
    if start is None:
        raise ValueError(f"{ENVIRONMENT}: no top-level `dependencies:`")
    indent = None
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] not in " \t-":
            return  # The next top-level key.
        depth = len(line) - len(line.lstrip())
        if indent is not None and depth > indent:
            continue  # A line of a nested section, such as the entries of `pip:`.
        entry = ENTRY.fullmatch(line)
        if entry is None or (indent is not None and depth != indent):
            raise ValueError(f"{ENVIRONMENT}:{index + 1}: cannot read `{line.strip()}`")
        indent = depth
        if not entry["value"].rstrip().endswith(":"):  # `- pip:` opens the pip section.
            yield index, entry["value"]


def drift(project: str, environment: str) -> list[str]:
    """Describe every difference between the ranges of pyproject.toml and environment.yml, in a stable order."""
    ranges = project_ranges(project)
    entries = environment_entries(environment)
    problems = []
    for name, required in sorted(ranges.items()):
        if name not in entries:
            problems.append(f"{ENVIRONMENT} lacks `{name}{','.join(required)}` of {PROJECT}")
        elif sorted(entries[name][1]) != sorted(required):
            declared = ",".join(entries[name][1]) or "no range"
            problems.append(f"{ENVIRONMENT} has `{name}` {declared}, but {PROJECT} has {','.join(required)}")
    for name in sorted(entries.keys() - ranges.keys() - CONDA_ONLY):
        problems.append(f"{ENVIRONMENT} has `{name}`, which {PROJECT} does not declare")
    return problems


def synchronize(project: str, environment: str) -> str:
    """Return environment.yml with the ranges of pyproject.toml on its existing lines.

    Missing and undeclared packages are left alone: where they belong, and whether they belong, needs a person.
    """
    ranges = project_ranges(project)
    entries = environment_entries(environment)
    lines = environment.splitlines(keepends=True)
    for name, (index, declared) in entries.items():
        required = ranges.get(name)
        if required is None or sorted(declared) == sorted(required):
            continue
        line = lines[index]
        entry = ENTRY.fullmatch(line.rstrip("\r\n"))
        assert entry is not None  # environment_entries() read this line.
        match = MATCH_SPEC.fullmatch(entry["value"])
        assert match is not None
        quote = entry["quote"] or '"'  # Quoted, because a plain scalar must not start with `>` or contain ` #`.
        newline = line[len(line.rstrip("\r\n")) :]
        lines[index] = f"{entry['indent']}- {quote}{match[1]}{','.join(required)}{quote}{entry['comment'] or ''}{newline}"
    return "".join(lines)


def read_revision(root: Path, revision: str, name: str) -> str:
    """Return a file as it is in a git revision."""
    result = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, check=True, capture_output=True)
    return result.stdout.decode("utf-8")


def known_drift(root: Path, revision: str) -> set[str]:
    """Return the drift of a git revision; a revision whose files cannot be read has none to tolerate."""
    try:
        return set(drift(read_revision(root, revision, PROJECT), read_revision(root, revision, ENVIRONMENT)))
    except (subprocess.CalledProcessError, UnicodeDecodeError, ValueError) as error:
        report([f"cannot read the drift of {revision}, so all drift fails: {error}"], "warning")
        return set()


def report(problems: list[str], level: str) -> None:
    """Print problems, as GitHub annotations when running in GitHub Actions."""
    for problem in problems:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            escaped = problem.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::{level} file={ENVIRONMENT}::{escaped}")
        else:
            print(f"{level}: {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="copy the ranges of pyproject.toml into environment.yml")
    group.add_argument("--base", metavar="REV", help="fail only on drift that this git revision does not have")
    group.add_argument("--allow-drift", action="store_true", help="report drift as warnings and exit 0")
    args = parser.parse_args(argv)
    path = args.root / ENVIRONMENT
    # Bytes keep the line endings, which --write must not change.
    project = (args.root / PROJECT).read_bytes().decode("utf-8")
    environment = path.read_bytes().decode("utf-8")
    try:
        if args.write:
            synchronized = synchronize(project, environment)
            if synchronized != environment:
                path.write_bytes(synchronized.encode("utf-8"))
                print(f"Updated the ranges in {ENVIRONMENT}")
            environment = synchronized
        problems = drift(project, environment)
    except ValueError as error:
        report([str(error)], "error")
        return 1
    known = known_drift(args.root, args.base) if args.base is not None and problems else set()
    tolerated = [problem for problem in problems if args.allow_drift or problem in known]
    failed = [problem for problem in problems if problem not in tolerated]
    report(tolerated, "warning")
    report(failed, "error")
    if failed:
        print(f"Run `python tools/sync_environment.py --write`, or edit {ENVIRONMENT} by hand; see CONTRIBUTING.md.")
    elif not problems:
        print(f"{ENVIRONMENT} has the ranges of {PROJECT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
