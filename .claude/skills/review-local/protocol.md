# Reviewer protocol

Every `review-*` agent in `.claude/agents/` follows this protocol. Your agent file defines your aspect; this file defines how to review and how to report.

## Inputs

The caller gives you:

- the repository path (a worktree) and the base ref, usually `origin/master`;
- the issue number and a path to a context file with the issue body and the acceptance criteria;
- a scratch directory for temporary files;
- the Python interpreter of the `pytorch-bsf` environment. Commands below write it as `$PY`: each Bash call starts a fresh shell, so substitute the path, or prefix the command with `PY=<path>`.

## Procedure

1. Read the context file, then the diff: `git diff <base>...HEAD --stat` and `git diff <base>...HEAD`.
2. Read the changed files in full where your aspect needs it, and the code they call or that calls them (`grep -rn`). Read `CONTRIBUTING.md` sections relevant to your aspect.
3. Check every item of your aspect's checklist against the changed lines. Unchanged code is out of scope unless the change makes it wrong (e.g. a caller of a function whose contract changed).
4. Where a claim can be checked cheaply, check it: run a few lines of Python with `PYTHONPATH=. "$PY" -c '...'` from the worktree root, or a single test with `PYTHONPATH=. "$PY" -m pytest -o addopts="" -p no:cacheprovider -q -p no:randomly <file>::<test>`. `PYTHONPATH=.` makes `torch_bsf` come from the worktree, not from the checkout the environment was installed from. `-o addopts=""` drops the reports that `pyproject.toml` writes to fixed paths in the worktree (`.coverage`, `coverage.xml`, `htmlcov/`, `pytest.xml`): other reviewers run in parallel in the same worktree and would overwrite them. For coverage, write it to your scratch directory: `COVERAGE_FILE=<scratch>/.coverage` and `--cov=torch_bsf --cov-report=xml:<scratch>/coverage.xml`. Keep each command under a minute and the whole review under about ten minutes. Write temporary files only under the scratch directory you are given, never in the repository (git-ignored test artifacts such as `pytest.xml` are fine).

## Rules

- **Do not modify the repository**: no edits, no `git add`/`commit`/`checkout`/`stash`, no formatting. You report; the caller fixes.
- **Stay in your aspect.** Another reviewer covers each other aspect. Report a finding outside your aspect only if it is a blocker, under "Outside my aspect".
- **Evidence or nothing.** Each finding needs a concrete location and a concrete reason: a failing input, a quoted line that contradicts another, a command and its output. Do not report "consider…" without a reason the current code is worse.
- **No style opinions the tools already enforce** (ruff, mypy) unless the tools are configured not to catch them.
- Prefer few correct findings to many speculative ones. If unsure, say so with `confidence: medium`; drop anything below that.

## Severity

| Severity | Meaning |
|----------|---------|
| `blocker` | Wrong results, crash, data loss, security hole, an unmet acceptance criterion, or a CI failure. Must be fixed before the PR. |
| `major` | Should be fixed before the PR: a reachable unhandled edge case, untested new behavior, docs that contradict the code, an avoidable large slowdown, an undocumented breaking change. |
| `minor` | Worth fixing, but the PR is correct without it. |
| `nit` | Cosmetic. Report at most three. |

## Report

Return only this, in English:

```text
ASPECT: <your aspect>
VERDICT: findings | no findings | not applicable
CHECKED: <one line: what you examined, e.g. "torch_bsf/interpolation.py refine(), tests/interpolation.py, docs/advanced/interpolation.rst">

1. [<severity>, confidence: high|medium] <path>:<line> — <one-line title>
   Evidence: <input/output, quoted lines, or command + result>
   Fix: <concrete suggestion>

Outside my aspect:
- <blocker only, same format; omit the section if empty>
```

Use `not applicable` with a one-line reason when the diff has nothing your aspect covers (e.g. a docs-only change for the performance reviewer).
