---
name: setup-env
description: Set up or repair the pytorch-bsf development environment (conda env, editable install, dev/test/docs tools) and verify it. Use when the user asks to set up / 環境構築 / fix the environment, or when pytest, mypy, ruff or sphinx is missing.
---

# setup-env

Make the `pytorch-bsf` conda environment able to run every check CI runs, with `torch_bsf` installed in editable mode from **this** checkout. Tell the user what you installed or changed.

Each Bash call starts a fresh shell, so do not rely on `conda activate`. Run everything through the environment's own interpreter:

```bash
PY="$(conda info --base)/envs/pytorch-bsf/bin/python"   # Windows: .../envs/pytorch-bsf/python.exe
```

## 1. Inspect the current state

```bash
conda --version
conda env list
"$PY" --version
"$PY" -c "import torch_bsf; print(torch_bsf.__file__)"
for m in torch lightning mlflow scipy sklearn matplotlib pytest pytest_cov pytest_randomly pytest_benchmark mypy ruff pre_commit sphinx sphinx_rtd_theme sphinx_design sphinxcontrib.bibtex; do
  "$PY" -c "import $m" 2>/dev/null && echo "ok $m" || echo "MISSING $m"
done
test -f "$(git rev-parse --git-path hooks)/pre-commit" && echo "ok git hook" || echo "MISSING git hook"
```

Decide what is needed from the result; skip steps that are already satisfied.

## 2. Create the conda environment (only if it does not exist)

```bash
conda env create -f environment.yml
```

This installs Python, PyTorch, Lightning, MLflow and `torch_bsf` in editable mode. It takes several minutes; run it in the background if possible.

If the environment exists but runtime dependencies are missing or out of date, run `conda env update -n pytorch-bsf -f environment.yml`. Never pass `--prune` (it would remove the dev tools installed in step 3), and never remove or recreate an existing environment without asking the user.

If conda is not installed, see "Without conda" below.

## 3. Install development, optional and docs dependencies

```bash
"$PY" -m pip install -e ".[develop,all]" $(grep -vE '^\s*\.\s*$|^\s*#|^\s*$' docs/requirements.txt)
```

`docs/requirements.txt` starts with `.` (this package, for the docs CI). Passing it with `-r` together with `-e .` makes pip fail with `ResolutionImpossible`, and installing it alone would replace the editable install, so the `.` line is filtered out.

- `develop`: pytest, mypy, ruff, pre-commit and type stubs.
- `all`: scipy, scikit-learn, matplotlib. Without these, parts of the test suite are skipped and mypy may report missing imports.
- `docs/requirements.txt`: Sphinx and extensions needed for `sphinx-build -b doctest`.

## 4. Install the git hook

```bash
"$PY" -m pre_commit install
```

This makes every `git commit` run Ruff (`ruff check --fix` and `ruff format`) with the version pinned in `.pre-commit-config.yaml`. The hook goes into the shared hooks directory of the clone (`git rev-parse --git-path hooks`), so one install covers the main checkout and all its worktrees, including `.claude/worktrees/`; skip this step if the check in step 1 printed `ok git hook`. If a commit is aborted because a hook changed files, review the changes, stage them and commit again.

## 5. Make sure the editable install points to this checkout

Check from outside the repository, since the current directory shadows the installed package:

```bash
(cd /tmp && "$PY" -c "import torch_bsf; print(torch_bsf.__file__)")
"$PY" -m pip show pytorch-bsf | grep -i editable
```

The path must be inside the current repository (or worktree). If it points elsewhere (another clone or worktree, or `site-packages`), re-run step 3 from this directory and tell the user that the environment now follows this checkout.

## 6. Verify

```bash
"$PY" -c "import torch, torch_bsf; print(torch.__version__, torch_bsf.__version__)"
"$PY" -m pre_commit run --all-files
"$PY" -m ruff check .
"$PY" -m mypy torch_bsf
"$PY" -m pytest tests/test_init.py -q
```

- These are smoke checks. Run the full `"$PY" -m pytest` and `"$PY" -m sphinx -b doctest docs docs/_build/doctest` only when the user asks or a task needs them.
- A failure caused by the environment (import error, missing package) means setup is not done: fix it and re-verify. A failure in the code itself is not an environment problem; report it without fixing.

## Without conda

Use a virtual environment in `.venv/` (git-ignored) with Python 3.10 or later:

```bash
python3 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e ".[develop,all]" -r docs/requirements.txt
```

Then run steps 4 and 6. Do not reuse or delete an existing broken `venv/` directory; report it to the user instead.

## Report

Tell the user:
- the environment path and Python / torch / torch_bsf versions,
- what was created, updated or installed (including whether the git hook was installed),
- the verification results, and anything still failing.
