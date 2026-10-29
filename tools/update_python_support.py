"""Synchronize Python support with the official release-cycle API.

Only Python 3 bugfix/security branches are eligible. Run with --check to preview
changes without writing them, or --data-file for a reproducible offline run.
"""

import argparse
import calendar
import difflib
import json
import re
from datetime import date
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://peps.python.org/api/release-cycle.json"
ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"feature", "prerelease", "bugfix", "security", "end-of-life"}


def version_key(value: str) -> tuple[int, ...]:
    return tuple(map(int, value.split(".")))


def release_date(value: str, *, end: bool = False) -> date:
    """Resolve month-only schedules conservatively, as the official chart does."""
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        return date(year, month, calendar.monthrange(year, month)[1] if end else 1)
    return date.fromisoformat(value)


def supported_versions(data: dict[str, dict[str, str]], today: date) -> list[str]:
    if not isinstance(data, dict) or not data:
        raise ValueError("Release-cycle data must be a nonempty mapping")
    supported = []
    for version, details in data.items():
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
            raise ValueError(f"Invalid Python version: {version!r}")
        if not isinstance(details, dict) or details.get("status") not in STATUSES:
            raise ValueError(f"Unknown release status for Python {version}")
        first = release_date(details["first_release"])
        end = release_date(details["end_of_life"], end=True)
        if first >= end:
            raise ValueError(f"Invalid release dates for Python {version}")
        if version.startswith("3.") and details["status"] in {"bugfix", "security"} and first <= today < end:
            supported.append(version)
    if not supported:
        raise ValueError("No supported stable Python 3 versions in release-cycle data")
    return sorted(supported, key=version_key)


def replace_required(text: str, pattern: str, replacement: str, *, count: int = 1) -> str:
    updated, found = re.subn(pattern, lambda _: replacement, text, flags=re.MULTILINE)
    if found != count:
        raise ValueError(f"Expected {count} matches for {pattern!r}, found {found}")
    return updated


def configured_versions(project: str) -> list[str]:
    minimum = re.search(r'^requires-python = ">=(3\.\d+)"$', project, re.MULTILINE)
    declared = re.findall(r'  "Programming Language :: Python :: (3\.\d+)",', project)
    if minimum is None or minimum[1] not in declared:
        raise ValueError("Expected a Python 3 lower bound and matching classifiers")
    return sorted({version for version in declared if version_key(version) >= version_key(minimum[1])}, key=version_key)


def integration_versions(versions: list[str], environment: str) -> list[str]:
    """Retain the Conda Python while supported; raise it when it reaches EOL."""
    match = re.search(r"^- python=(3\.\d+)$", environment, re.MULTILINE)
    if match is None:
        raise ValueError("Expected a pinned Python 3 version in environment.yml")
    candidates = [version for version in versions if version_key(version) >= version_key(match[1])]
    if not candidates:
        raise ValueError("Refusing to downgrade the Conda Python version")
    return [version for version in versions if version_key(version) <= version_key(candidates[0])]


def support_notes(versions: list[str], integrations: str, *, rst: bool) -> str:
    ticks = "``" if rst else "`"
    notes = (
        f"CI covers Python {versions[0]} through {versions[-1]}.\n"
        f"The optional {ticks}mlflow{ticks} and {ticks}all{ticks} extras are tested\n"
        f"on Python {versions[0]} through {integrations}."
    )
    newer = [version for version in versions if version_key(version) > version_key(integrations)]
    if newer:
        command = 'python -m pip install "pytorch-bsf[lightning,sparse,sklearn,plot]"'
        notes += f"\n\nFor Python {', '.join(newer)}, install the other integrations with"
        notes += f"::\n\n    {command}" if rst else f"\n`{command}`."
    return notes


def update_documentation(content: str, name: str, old_minimum: str, versions: list[str], integrations: str) -> str:
    minimum = versions[0]
    if f"Python {old_minimum}" not in content:
        raise ValueError(f"Missing Python minimum in {name}")
    content = content.replace(f"Python {old_minimum}", f"Python {minimum}")
    if name == "CONTRIBUTING.md":
        content = replace_required(content, r"`python_version = 3\.\d+`", f"`python_version = {minimum}`")
        content = replace_required(
            content, r'`target-version = "py\d+"`', f'`target-version = "py{minimum.replace(".", "")}"`'
        )
        content = replace_required(content, r"once on Python 3\.\d+\.", f"once on Python {integrations}.")
    else:
        rst = name.endswith(".rst")
        start, end = (
            (".. python-support-start", ".. python-support-end")
            if rst
            else ("<!-- python-support:start -->", "<!-- python-support:end -->")
        )
        notes = support_notes(versions, integrations, rst=rst)
        content = replace_required(content, rf"{re.escape(start)}[\s\S]*?{re.escape(end)}", f"{start}\n\n{notes}\n\n{end}")
    return content


def plan_updates(root: Path, data: dict[str, dict[str, str]], today: date) -> tuple[dict[Path, str], str, str]:
    """Validate the complete input and all edit locations before writing anything."""
    available = supported_versions(data, today)
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    declared = configured_versions(project)
    old_minimum = declared[0]
    # Missing rows and stale/partial responses must never silently drop support.
    if any(version not in data for version in declared):
        raise ValueError("Release-cycle data is missing declared Python versions")
    versions = [version for version in available if version_key(version) >= version_key(old_minimum)]
    if not versions or version_key(versions[-1]) < max(map(version_key, declared)):
        raise ValueError("Refusing to downgrade the latest supported Python version")
    minimum, latest = versions[0], versions[-1]
    project = replace_required(project, r'^requires-python = ".*"$', f'requires-python = ">={minimum}"')
    classifiers = "".join(f'  "Programming Language :: Python :: {version}",\n' for version in versions)
    project = replace_required(project, r'(?:  "Programming Language :: Python :: 3\.\d+",\n)+', classifiers)
    project = replace_required(project, r'^target-version = "py\d+"$', f'target-version = "py{minimum.replace(".", "")}"')
    project = replace_required(project, r'^python_version = "3\.\d+"$', f'python_version = "{minimum}"')
    proposed = {root / "pyproject.toml": project}

    path = root / "environment.yml"
    environment = path.read_text(encoding="utf-8")
    integrations = integration_versions(versions, environment)[-1]
    proposed[path] = replace_required(environment, r"^- python=3\.\d+$", f"- python={integrations}")
    for name in ("README.md", "CONTRIBUTING.md", "docs/installation.rst"):
        path = root / name
        proposed[path] = update_documentation(path.read_text(encoding="utf-8"), name, old_minimum, versions, integrations)

    changes = {path: content for path, content in proposed.items() if path.read_text(encoding="utf-8") != content}
    breaking = "!" if minimum != old_minimum else ""
    title = f"ci(python){breaking}: support Python {minimum} through {latest}"
    summary = (
        f"Synchronize Python support with the [official release-cycle data]({SOURCE}).\n\n"
        f"- Minimum Python: {old_minimum} → {minimum}\n"
        f"- Supported stable Python versions: {', '.join(versions)}\n"
        f"- Full integrations and Conda: Python {integrations}\n"
        "- Synchronize package metadata, Ruff/mypy, CI, Conda and installation documentation.\n"
        "- Refresh uv.lock; dependency lower bounds remain governed by the lowest-dependency CI job.\n\n"
        "Only released bugfix/security branches are included. Prereleases are excluded. "
        "Review CI results for dependency availability and compatibility before merging.\n"
        "\nIf checks have not started after this bot creates or updates the PR, close and reopen it "
        "from your GitHub account to trigger the existing `reopened` CI workflows. "
        "Repeat after subsequent bot updates if needed. If GitHub shows pending workflow approval, "
        "select **Approve workflows to run**. Configure `DEPENDENCY_UPDATE_TOKEN` for automatic PR CI.\n"
    )
    if breaking:
        summary += f"\nBREAKING CHANGE: Python {old_minimum} is no longer supported because it has reached end of life.\n"
    return changes, title, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--check", action="store_true", help="Print the proposed diff and exit 1 if updates are needed")
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument(
        "--ci-output", type=Path, help="Write configured CI versions to GITHUB_OUTPUT without network or edits"
    )
    args = parser.parse_args()
    if args.ci_output:
        versions = configured_versions((args.root / "pyproject.toml").read_text(encoding="utf-8"))
        integrations = integration_versions(versions, (args.root / "environment.yml").read_text(encoding="utf-8"))
        with args.ci_output.open("a", encoding="utf-8") as output:
            output.write(f"minimum={versions[0]}\nlatest={versions[-1]}\nmatrix={json.dumps(versions)}\n")
            output.write(f"integrations={integrations[-1]}\nintegration-matrix={json.dumps(integrations)}\n")
        return
    if args.data_file:
        data = json.loads(args.data_file.read_text(encoding="utf-8"))
    else:
        with urlopen(SOURCE, timeout=30) as response:
            data = json.load(response)
    changes, title, summary = plan_updates(args.root, data, date.today())
    for path, content in changes.items():
        if args.check:
            print(
                "".join(
                    difflib.unified_diff(
                        path.read_text(encoding="utf-8").splitlines(True),
                        content.splitlines(True),
                        fromfile=str(path),
                        tofile=str(path),
                    )
                ),
                end="",
            )
        else:
            path.write_text(content, encoding="utf-8")
    if args.summary_file:
        args.summary_file.write_text(summary, encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"title={title}\nchanged={str(bool(changes)).lower()}\n")
    print(f"{len(changes)} file(s) {'need updating' if args.check else 'updated'}")
    if args.check and changes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
