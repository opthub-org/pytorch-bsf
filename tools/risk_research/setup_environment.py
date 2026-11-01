"""Install the optional SageMath research environment and pinned Ore algebra."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ORE_URL = "https://github.com/mkauers/ore_algebra.git"
ORE_COMMIT = "afbe4dad50bd63f5b2ce112193555d6f9740e0d3"
REPOSITORY = Path(__file__).resolve().parents[2]


def install(root: Path, conda: str) -> None:
    """Create an isolated environment, install Ore algebra, and run smoke checks."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / "sage-env"
    source = root / "ore-algebra-source"
    environment = os.environ.copy()
    environment.update(
        CONDA_PKGS_DIRS=str(root / "conda-pkgs"),
        CONDA_ENVS_PATH=str(root / "envs"),
        DOT_SAGE=str(root / "sage-state"),
        MPLCONFIGDIR=str(root / "matplotlib"),
        CONDA_ALWAYS_YES="true",
        CONDA_CHANNEL_PRIORITY="strict",
    )
    action = "update" if (prefix / "conda-meta").is_dir() else "create"
    subprocess.run(
        [conda, "env", action, "--prefix", str(prefix), "--file", str(Path(__file__).with_name("environment.yml"))],
        env=environment,
        check=True,
    )
    if not source.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", ORE_URL, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", ORE_URL, ORE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", ORE_COMMIT], check=True)
    # Rebuild generated extensions with the pinned Cython ABI. In particular,
    # cached C files from another Cython version must not survive a retry.
    shutil.rmtree(source / "build", ignore_errors=True)
    for pyx in (source / "src" / "ore_algebra" / "analytic").glob("*.pyx"):
        pyx.with_suffix(".c").unlink(missing_ok=True)
    subprocess.run(
        [
            conda,
            "run",
            "--no-capture-output",
            "--prefix",
            str(prefix),
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(source),
        ],
        env=environment,
        check=True,
    )
    explicit = subprocess.run(
        [conda, "list", "--prefix", str(prefix), "--explicit"], env=environment, check=True, capture_output=True, text=True
    )
    (root / "conda-explicit.txt").write_text(explicit.stdout, encoding="utf-8")
    (root / "ore-source.json").write_text(
        json.dumps({"url": ORE_URL, "commit": ORE_COMMIT}, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [
            conda,
            "run",
            "--no-capture-output",
            "--prefix",
            str(prefix),
            "python",
            str(Path(__file__).with_name("sage_smoke.py")),
            "--output",
            str(root / "sage-smoke.json"),
        ],
        env=environment,
        check=True,
    )


def main() -> None:
    """Parse installation options and install the research environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY / ".research" / "skeleton-risk")
    parser.add_argument("--conda", default=shutil.which("conda"))
    args = parser.parse_args()
    if not args.conda:
        parser.error("Conda is required; install Miniforge or pass --conda PATH.")
    install(args.root, args.conda)


if __name__ == "__main__":
    main()
