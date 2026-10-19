---
name: create-issue
description: Create a GitHub issue on opthub-org/pytorch-bsf with gh. Use when the user asks to file / 起票 / 作成 an issue, report a bug, or record a feature or docs idea as an issue. Args - free-form description of the issue.
---

# create-issue

Turn the user's request (`$ARGUMENTS` or the conversation) into a GitHub issue on `opthub-org/pytorch-bsf`.

## 1. Gather context

- If the request mentions code (functions, modules, docs pages, errors), read the relevant files first so the issue names real symbols and paths (e.g. `torch_bsf.fit_kfold()`, `docs/...`).
- For a bug, collect: reproduction steps or snippet, expected vs. actual behavior, error message/traceback, and environment (Python / torch / torch_bsf versions, OS) when available. Do not invent missing facts; ask or leave them out.

## 2. Check for duplicates

```bash
gh issue list -R opthub-org/pytorch-bsf -s all -L 20 --search "<keywords>"
```

If a likely duplicate exists, show it to the user and ask whether to proceed, comment there instead, or stop.

## 3. Choose the template and label

The issue must be implementable by an AI agent told only "Issue #N をやって", so the body follows the issue forms in `.github/ISSUE_TEMPLATE/`. Pick one:

| Kind | Template | Label |
|------|----------|-------|
| Bug | `bug_report.yml` | `bug` |
| Feature, API change, refactor, tests, perf, CI, tooling, deps | `feature_request.yml` | `enhancement` |
| Docs, README, docstrings, examples | `documentation.yml` | `documentation` |

Add `good first issue` / `help wanted` only when the user asks.

## 4. Draft title and body

Read the chosen template and reproduce its structure exactly as GitHub renders a submitted form:

- One `### <label>` heading per field, in the template's order, using the field's `label` verbatim (English).
- Dropdown fields: the chosen option text verbatim (e.g. `Must keep backward compatibility`).
- Fields with `render:` → wrap the content in a fenced block with that language.
- Optional fields with nothing to say → `_No response_`. Required fields must have real content; if the user's request does not cover one (e.g. acceptance criteria, backward compatibility), propose a value and point it out in step 5.

Field contents are written in the user's language (usually Japanese). Title: short, in the user's language, no type prefix.

Quality bar for an implementer with no other context:

- **Acceptance criteria** are concrete, checkable `- [ ]` items (behavior, tests, docs), not restatements of the goal.
- **Scope / Target / Suspected cause** name real paths and symbols found in step 1.
- **Change type** (feature template) matches CONTRIBUTING.md commit types, since it drives the branch name and commit prefix.
- Decisions the user already made (e.g. "後方互換性は気にせず刷新して") go into the dropdowns and **Notes for the implementer**.

## 5. Confirm, then create

Creating an issue is outward-facing. Show the user the title, labels, and body, and create it only after they approve (skip confirmation only if they explicitly said to file it without asking).

Write the body to a file in the scratchpad directory to avoid shell-quoting problems, then:

```bash
gh issue create -R opthub-org/pytorch-bsf \
  --title "<title>" \
  --label "<label>" \
  --body-file "<scratchpad>/issue-body.md"
```

Add `--assignee @me` only if the user asks.

## 6. Report

Return the issue URL printed by `gh`. If creation fails (auth, unknown label), show the error and fix the cause rather than retrying blindly.
