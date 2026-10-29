"""Build and exercise distributions from tracked sources, without checkout imports.

Run once per CI build with the runtime dependencies, setuptools and wheel installed.
Builds and wheel installations stay inside a temporary directory.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = """
import sys
from pathlib import Path
import torch
import torch_bsf
import torch_bsf._kernels
import torch_bsf.integrations
import torch_bsf.model_selection

installed = Path(sys.argv[1]).resolve()
assert Path(torch_bsf.__file__).resolve().is_relative_to(installed)
assert not any(m.split('.')[0] in ('lightning', 'mlflow') for m in sys.modules)
params = torch.tensor([[1., 0.], [0.5, 0.5], [0., 1.]], dtype=torch.float64)
values = torch.tensor([[0.], [0.5], [1.]], dtype=torch.float64)
space = torch_bsf.BezierSpace.simplex(dimension=1, degree=1)
result = torch_bsf.fit(space, torch_bsf.Samples(params, values))
result.require_success()
torch.testing.assert_close(result.model(params), values)
query = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float64)
torch.testing.assert_close(result.model(query), torch.tensor([[0.75], [0.25]], dtype=torch.float64))
assert not any(m.split('.')[0] in ('lightning', 'mlflow') for m in sys.modules)
"""


def run(arguments: list[str], directory: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=directory, env=env, check=True)


def clean_source(destination: Path) -> None:
    """Copy tracked working-tree contents, excluding caches and editable artifacts."""
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, tracked):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    # Namespace discovery must reject arbitrary root directories and similarly
    # prefixed packages, while preserving future namespace children of torch_bsf.
    for package in ("unrelated_namespace", "torch_bsf_extra", "torch_bsf/future_namespace"):
        module = destination / package / "probe.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("DISTRIBUTION_PROBE = True\n")


def build(source: Path, kind: str, destination: Path) -> Path:
    destination.mkdir(parents=True)
    run(
        [sys.executable, "-c", f"import setuptools.build_meta as b; b.build_{kind}({str(destination)!r})"],
        source,
    )
    return next(destination.glob("*.whl" if kind == "wheel" else "*.tar.gz"))


def check_wheel(wheel: Path, source: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        roots = {name.split("/")[0] for name in names}
        metadata = {name for name in roots if name.endswith(".dist-info")}
        assert len(metadata) == 1, roots
        assert roots == {"torch_bsf", *metadata}, roots
        info = metadata.pop()
        assert archive.read(f"{info}/top_level.txt").decode().split() == ["torch_bsf"]
        assert archive.read(f"{info}/METADATA").startswith(b"Metadata-Version:")
        assert f"{info}/WHEEL" in names
        assert f"{info}/RECORD" in names
        assert any(name.endswith("/LICENSE") and archive.read(name) == (source / "LICENSE").read_bytes() for name in names)
        required = {str(path.relative_to(source)) for path in (source / "torch_bsf").rglob("*.py")}
        assert required | {"torch_bsf/py.typed"} <= names, required - names
    # --target avoids modifying the test interpreter or its editable installation.
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(destination), str(wheel)], wheel.parent)
    env = {**os.environ, "PYTHONPATH": str(destination)}
    run([sys.executable, "-c", SMOKE, str(destination)], wheel.parent, env)
    run([sys.executable, "-m", "torch_bsf", "--help"], wheel.parent, env)
    print(f"Validated {wheel}: package contents, installed imports, fit/predict and CLI")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bsf-distribution-") as scratch:
        root = Path(scratch)
        source = root / "source"
        clean_source(source)
        wheel = build(source, "wheel", root / "direct-wheel")
        check_wheel(wheel, source, root / "direct-install")
        sdist_source = root / "sdist-source"
        clean_source(sdist_source)
        sdist = build(sdist_source, "sdist", root / "sdist")
        unpacked = root / "unpacked"
        # This archive was just built from local tracked sources by setuptools.
        with tarfile.open(sdist) as archive:
            archive.extractall(unpacked, filter="data")
        rebuilt = build(next(unpacked.iterdir()), "wheel", root / "sdist-wheel")
        check_wheel(rebuilt, source, root / "sdist-install")


if __name__ == "__main__":
    main()
