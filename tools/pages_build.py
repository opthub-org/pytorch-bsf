"""Freeze the triggering ref, build its documentation and package only that build."""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_FILES = 20_000
MAX_SITE_BYTES = 245_000_000
MAX_FILE_BYTES = 20_000_000
MAX_MANIFEST_BYTES = 5_000_000
# Must equal DOCS_VERSION_VARIABLE in docs/conf.py, which reads it.
DOCS_VERSION_VARIABLE = "PYTORCH_BSF_DOCS_VERSION"


def source_name(ref: str) -> str:
    """Keep the existing version URLs, rejecting reserved publication paths."""
    if ref.startswith("refs/heads/"):
        name = ref.removeprefix("refs/heads/")
        if name.startswith(("gh-pages", "release-please")):
            raise ValueError("Excluded documentation branch")
    elif ref.startswith("refs/tags/"):
        name = ref.removeprefix("refs/tags/")
        if not re.fullmatch(r"v\d+\.\d+\.\d+", name):
            raise ValueError("Only release tags have documentation URLs")
    else:
        raise ValueError("Expected a branch or release tag")
    parts = PurePosixPath(name).parts
    if (
        not parts
        or name.startswith("/")
        or any(part in (".", "..") or part.startswith(".") for part in parts)
        or "\\" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("Unsafe documentation path")
    if str(PurePosixPath(name)) != name or parts[0] in {"README.md", "index.html"}:
        raise ValueError("Reserved documentation path")
    return name


def freeze(ref: str, sha: str) -> None:
    """Build the event's commit even if checkout fetched a newer remote ref."""
    source_name(ref)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid source SHA")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != sha:
        raise ValueError("Checkout does not match the triggering source SHA")
    subprocess.run(["git", "update-ref", ref, sha], check=True)


def build_html(ref: str, build: Path) -> None:
    """Build only the triggering ref's HTML into its version root under ``build``.

    ``docs/conf.py`` reads the version name from ``DOCS_VERSION_VARIABLE`` for the
    version menu and source links.
    """
    name = source_name(ref)
    environment = dict(os.environ, **{DOCS_VERSION_VARIABLE: name})
    subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", "docs", str(build / name)], check=True, env=environment)


def inventory(source: Path) -> dict[str, dict[str, object]]:
    """Hash regular documentation files within the publication size budget."""
    files: dict[str, dict[str, object]] = {}
    total = 0
    for path in sorted(source.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Documentation contains a nonregular file")
        relative = path.relative_to(source).as_posix()
        if any(part in {".git", ".gitattributes", ".gitmodules"} for part in path.relative_to(source).parts):
            raise ValueError("Documentation contains a Git control path")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Documentation file is too large")
        data = path.read_bytes()
        total += len(data)
        if len(files) >= MAX_FILES or total > MAX_SITE_BYTES or len(data) > MAX_FILE_BYTES:
            raise ValueError("Documentation artifact is too large")
        files[relative] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def package(build: Path, output: Path, metadata: dict[str, object]) -> None:
    """Copy regular files, recording byte digests without including other refs."""
    name = source_name(str(metadata["source_ref"]))
    source = build / name
    if not source.is_dir() or source.is_symlink() or not (source / "index.html").is_file():
        raise ValueError("The triggering ref was not built")
    if output.exists():
        raise ValueError("Artifact staging directory already exists")
    files = inventory(source)
    # Reserve ownership atomically. A competing creator's directory must never
    # enter our cleanup block when this exclusive creation fails.
    output.mkdir(parents=True)
    try:
        (output / "site").mkdir()
        for relative in files:
            target = output / "site" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (source / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != files[relative]["sha256"]:
                raise ValueError("Documentation changed during packaging")
            target.write_bytes(data)
        manifest = (json.dumps(dict(metadata, schema=1, files=files), sort_keys=True) + "\n").encode()
        if len(manifest) > MAX_MANIFEST_BYTES:
            raise ValueError("Documentation manifest is too large")
        (output / "manifest.json").write_bytes(manifest)
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    """Read event provenance directly from the runner environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="pin the triggering ref to the event commit")
    mode.add_argument("--html", action="store_true", help="build the triggering ref's HTML")
    mode.add_argument("--output", type=Path, help="package the built ref into this new directory")
    parser.add_argument(
        "--build",
        type=Path,
        default=Path("docs/_build"),
        help="documentation build root: --html writes <build>/<ref name> and --output reads it",
    )
    args = parser.parse_args()
    ref, sha = os.environ["GITHUB_REF"], os.environ["GITHUB_SHA"]
    if args.freeze:
        freeze(ref, sha)
    elif args.html:
        build_html(ref, args.build)
    else:
        package(
            args.build,
            args.output,
            {
                "repository": os.environ["GITHUB_REPOSITORY"],
                "run_id": int(os.environ["GITHUB_RUN_ID"]),
                "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                "source_ref": ref,
                "source_sha": sha,
            },
        )


if __name__ == "__main__":
    main()
