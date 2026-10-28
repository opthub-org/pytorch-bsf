---
name: review-local
description: Review the current branch of pytorch-bsf locally before opening a PR - run one specialized reviewer agent per aspect (requirements, design, correctness, numerics, edge cases, tests, docs, compatibility, …) in parallel, verify their findings, and fix them. Use when the user asks for a local review / ローカルレビュー / PR 前レビュー, and as step 8 of work-on-issue. Args - optional issue number and base ref.
---

# review-local

Review the commits of the current branch against `origin/master` (or the base ref given in `$ARGUMENTS`, after the optional issue number) with one specialized reviewer per aspect, running in parallel, so that the PR needs as few CI and review round trips as possible. This skill fixes findings locally; it does not push or comment on GitHub.

## Aspects

Each reviewer is a read-only agent in `.claude/agents/review-<aspect>.md`; all follow `protocol.md` in this directory.

| Aspect | Agent | Run |
|--------|-------|-----|
| Requirements and scope (acceptance criteria, scope creep, completeness) | `review-requirements` | always |
| Design and API | `review-design` | always |
| Correctness and logic | `review-correctness` | always |
| Edge cases and input validation | `review-edge-cases` | always |
| Test quality | `review-tests` | always |
| Documentation consistency | `review-docs` | always |
| Project conventions | `review-conventions` | always |
| Readability and maintainability | `review-readability` | always |
| Numerical robustness | `review-numerics` | floating-point code changed |
| PyTorch and framework integration | `review-frameworks` | torch, Lightning, sklearn or MLflow code changed |
| Performance | `review-performance` | library code under `torch_bsf/` changed |
| Backward compatibility | `review-compatibility` | public API, CLI, file formats or `MLproject` changed |
| Security and safety | `review-security` | I/O, deserialization, subprocess, workflows or new dependencies changed |
| Reproducibility and side effects | `review-reproducibility` | randomness, global state or argument mutation involved |
| Dependencies and packaging | `review-packaging` | imports of third-party packages, `pyproject.toml`, `environment.yml` or package data changed |

For a change limited to docs, skills or CI, the "always" set may drop `review-correctness` and `review-edge-cases` when there is no code to run; say so in the summary. When in doubt, run the reviewer: an unneeded one answers `not applicable` cheaply.

## 1. Prepare the context

```bash
BASE=origin/master             # or the base ref given in the arguments
git fetch origin
git status --porcelain          # commit or set aside tracked changes first: reviewers read HEAD
git diff "$BASE"...HEAD --stat
```

- The branch must be committed: reviewers see `git diff $BASE...HEAD`. Untracked files are ignored.
- Create a scratch directory (`<scratchpad>/review-<issue number>`, or the branch name with `/` replaced by `-`), and write `context.md` there with: the repository path, the base ref, the branch, the issue number, title and full body (`gh issue view <N> -R opthub-org/pytorch-bsf --json title,body,comments`), and one paragraph on what the change intends. Without an issue, describe the intent and acceptance criteria yourself.
- Choose the aspects from the table using `git diff --stat` and the content of the diff.

## 2. Run the reviewers in parallel

Launch every chosen reviewer **in a single message** with one Agent call each (`subagent_type: review-<aspect>`), so they run concurrently. Give each the same brief:

```text
Review branch <branch> of the pytorch-bsf worktree at <path> against <BASE> for your aspect.
Context (issue, acceptance criteria, intent): <scratch>/context.md
Scratch directory for temporary files: <scratch>/<aspect>
Python ($PY): <path to the pytorch-bsf env python>
Follow <path>/.claude/skills/review-local/protocol.md and return its report format only.
```

Do not pass your own opinion of the change: an anchored reviewer tends to confirm it.

If the `review-*` agent types are not available (agents are loaded when a session starts, so a session that predates them does not see them), launch `general-purpose` agents instead and start the brief with "Your role definition is `<worktree>/.claude/agents/review-<aspect>.md` (ignore its frontmatter); read it and follow it."

Reviewers that run experiments can take several minutes; the slowest usually sets the wall-clock time. Start consolidating the reports that have arrived while waiting for the rest.

## 3. Consolidate and verify

1. Check that `git status --porcelain` still shows the clean tree of step 1: reviewers are read-only by protocol, not by their tool list (they need Bash for `git diff` and experiments). If a reviewer changed a file, inspect the change and revert it.
2. Collect the reports, including the "Outside my aspect" sections. Merge duplicates (same location and cause) and keep the highest severity; a finding raised independently by several reviewers is more likely real, but still verify it.
3. **Verify every `blocker` and `major` yourself** before acting: read the cited lines, run the reproducer or write one. Mark each finding `confirmed`, `rejected` (with the reason, e.g. "input is rejected earlier at validator.py:42") or `uncertain`. Reviewers are wrong sometimes; do not "fix" a non-bug.
4. Write the consolidated list to `<scratch>/findings.md`, sorted by severity, with the verdict of each.

## 4. Fix

- Fix every confirmed `blocker` and `major`. Fix `minor` and `nit` when cheap and within the issue's scope; otherwise defer them with a reason (and an issue if they are worth tracking).
- A finding outside the issue's scope that is a real problem: file an issue with the `create-issue` skill and list it in the PR's "Out of scope / follow-ups".
- A design question that the issue leaves to the user ("Ask in this issue…"): ask the user instead of deciding.
- After fixing, rerun the checks of work-on-issue step 5 and commit the fixes.

## 5. Re-review

If step 4 changed any file, rerun in parallel the reviewers that reported a fixed finding, plus every conditional reviewer whose trigger the fixes touch, with the same brief plus "Previous findings: <scratch>/findings.md; check that they are fixed and that the fixes introduced no new problems". Process their reports through steps 3 and 4 again. Repeat until a round has no confirmed `blocker` or `major`, at most two re-review rounds; if one remains then, report it to the user instead of opening the PR. Skip this step when step 4 changed nothing.

## 6. Record the result

Add a `## Local review` section to the PR body:

```markdown
## Local review
Aspects: requirements, design, correctness, edge cases, tests, docs, conventions, readability, numerics, performance (N reviewers, R rounds)

| Severity | Found | Fixed | Rejected | Deferred |
|----------|-------|-------|----------|----------|
| blocker | 0 | 0 | 0 | 0 |
| major | 2 | 2 | 0 | 0 |
| minor | 3 | 1 | 1 | 1 |

- Fixed: <one line per blocker/major>
- Rejected: <finding — reason>
- Deferred: <finding — issue link or reason>
```

When run on its own (not from work-on-issue), report the same table and lists to the user.

## Without Claude Code

Other agents and humans can run the same review: use each `.claude/agents/review-<aspect>.md` body together with `protocol.md` as the prompt of one reviewer, run them in parallel, and follow steps 3–6.
