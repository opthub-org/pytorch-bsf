---
name: work-on-issue
description: Implement a GitHub issue of opthub-org/pytorch-bsf end to end - read the issue, branch, implement, test, commit, and open a PR that closes it. Use when the user says "Issue #N をやって", "work on #N", "fix #N", or similar. Args - issue number.
---

# work-on-issue

Take issue `$ARGUMENTS` (a number, with or without `#`) from reading to an open PR. Invoking this skill authorizes pushing the branch and opening the PR; it does **not** authorize merging, closing issues, or commenting on GitHub.

Follow CONTRIBUTING.md throughout (commit rules, coding conventions, tests, "Bezier" vs "Bézier"). Write code, commits and the PR in English; talk to the user in their language.

## 1. Read the issue

```bash
gh issue view <N> -R opthub-org/pytorch-bsf --json number,title,body,labels,state,comments
gh pr list -R opthub-org/pytorch-bsf -s all --search "<N> in:body" --json number,title,state,headRefName
```

- If the issue is closed, or an open PR already addresses it, stop and ask the user.
- Read the comments too; later comments may override the body.

Issues filed with the forms in `.github/ISSUE_TEMPLATE/` have `### <Field>` sections. Extract:

| Field | Use |
|-------|-----|
| Change type / label | Branch prefix and commit type: `bug` → `fix`, `documentation` → `docs`, feature form → its **Change type** |
| Acceptance criteria | The definition of done. Every item must be satisfied and reported in the PR |
| Backward compatibility | Whether public API may break; if allowed, use `!` and a `BREAKING CHANGE:` footer |
| Undecided design points | `Decide and explain…` → decide and record in the PR; `Ask in this issue…` → ask the user before implementing |
| Scope / Target / Suspected cause | Where to start; never change anything listed as out of scope |
| Steps to reproduce | Reproduce the bug **before** fixing it, then turn it into a regression test |
| Notes for the implementer | Constraints and decisions already made |

Older or free-form issues (e.g. a title only) have no sections: infer the intent from the title, body and code. If the intent, the expected result, or a compatibility decision is genuinely ambiguous, ask the user with concrete options instead of guessing.

## 2. Prepare the branch

```bash
git status --porcelain
git fetch origin
git switch -c <type>/<N>-<short-kebab-desc> origin/master
```

- If tracked files have uncommitted changes, stop and ask the user how to handle them. Untracked files may stay but must never be staged.
- Example branch names: `feat/313-unify-fit-kfold`, `fix/456-nan-in-output`, `docs/309-elastic-net-formulation`.

## 3. Check the environment

Use the `pytorch-bsf` conda environment through its interpreter, `PY="$(conda info --base)/envs/pytorch-bsf/bin/python"`, and run every Python tool in steps 4–5 as `"$PY" -m <tool>`. Confirm the tools are available and `torch_bsf` is imported from this checkout:

```bash
(cd /tmp && "$PY" -c "import torch_bsf, pytest, mypy, ruff, sphinx; print(torch_bsf.__file__)")
```

If anything is missing or `torch_bsf` comes from elsewhere, follow the `setup-env` skill (`.claude/skills/setup-env/SKILL.md`) first.

## 4. Investigate and implement

- Read the code and tests around the scope before editing. Follow existing patterns in neighboring code.
- For bugs: write the failing test first, confirm it fails, then fix.
- Public API changes require type hints, NumPy-style docstrings with runnable `Examples`, and updates to the Sphinx docs under `docs/`.
- Tests go in the file mirroring the module (`torch_bsf/foo.py` → `tests/foo.py`).
- Keep the diff focused on the issue. Note unrelated problems for the PR description instead of fixing them.

## 5. Verify (same checks as CI)

Run all of these and fix every failure:

```bash
"$PY" -m ruff check --fix <changed .py files>
"$PY" -m ruff format <changed .py files>
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" -m mypy torch_bsf tests
"$PY" -m pytest
"$PY" -m sphinx -b doctest docs docs/_build/doctest
```

- Format only the files you changed, to keep the diff clean.
- If CLI code (`torch_bsf/__main__.py`, `_cli_args.py`) changed, also run `PATH="$(dirname "$PY"):$PATH" bash examples/quickstart/run.sh`.
- If docs changed, also run `"$PY" -m sphinx -b html docs docs/_build/html` and check for new warnings.
- Then walk through each acceptance criterion and confirm it is met with evidence (test name, command output, doc page). An unmet criterion means the work is not done.

## 6. Commit

- Stage files explicitly (`git add <paths>`), never `git add -A`. Do not commit generated files (`pytest.xml`, `coverage.xml`, `htmlcov/`, `docs/_build/`, `lightning_logs/`).
- Conventional Commits, English, imperative, first line under 72 characters:
  `<type>(<scope>): <subject>`. Add `!` and a `BREAKING CHANGE:` footer for breaking changes.
- One or a few logical commits are fine.

## 7. Sync with master

```bash
git fetch origin
git rebase origin/master
```

If master moved, resolve conflicts and re-run step 5.

## 8. Open the PR

```bash
git push -u origin <branch>
gh pr create -R opthub-org/pytorch-bsf --base master --head <branch> \
  --title "<type>(<scope>): <subject>" --body-file <scratchpad>/pr-body.md
```

The title follows Conventional Commits because release-please builds the CHANGELOG from it. Write the body to a file in the scratchpad directory:

```markdown
Closes #<N>

## Summary
<What changed and why, in a few sentences.>

## Changes
- <Grouped list of notable changes, with file paths>

## Acceptance criteria
- [x] <criterion from the issue> — <evidence: test name / command / doc page>

## Design decisions
<Choices made for undecided points, alternatives considered. Omit if none.>

## Breaking changes
<What breaks and how to migrate. Omit if none.>

## Verification
- `python -m pytest`: <N passed>
- `python -m mypy torch_bsf tests`: <result>
- `ruff check` / `ruff format --check`: <result>
- `sphinx-build -b doctest`: <result>

## Out of scope / follow-ups
<Unrelated issues noticed. Omit if none.>
```

## 9. Report

Give the user the PR URL, a short summary, any design decisions made, and anything left unverified or skipped (with the reason). If a check could not be made to pass, do not open the PR; report the failure output instead.
