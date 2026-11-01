---
name: work-on-issue
description: Implement a GitHub issue of opthub-org/pytorch-bsf end to end - read the issue, branch, implement, test, commit, and open a PR that closes it. Use when the user says "Issue #N をやって", "work on #N", "fix #N", or similar. Args - issue number.
---

# work-on-issue

Take issue `$ARGUMENTS` (a number, with or without `#`) from reading to an open PR. Invoking this skill authorizes pushing the branch, opening the PR, posting `@codex review` requests on that PR, and replying to and resolving review threads on that PR, including threads on older commits. Handle those threads proactively as described in step 10. Merging, closing issues, and commenting elsewhere on GitHub still require explicit user authorization.

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
SKIP=typos "$PY" -m pre_commit run --all-files    # the CI lint job: Ruff and numpydoc-validation
"$PY" -m pre_commit run typos --files <changed files>  # local only; report false positives in the PR
"$PY" -m mypy torch_bsf tests
"$PY" -m pytest --cov=torch_bsf --cov-report=xml  # the reports are not in addopts; diff_cover reads coverage.xml
git fetch origin master && "$PY" -m diff_cover.diff_cover_tool coverage.xml --compare-branch=origin/master
"$PY" -m sphinx -b doctest docs docs/_build/doctest
```

- The Ruff hooks fix and format files in place; review what they changed before committing. They leave unchanged files alone, because `master` already passes them.
- `diff_cover` reports the changed lines that no test runs, as the `codecov/patch` check does; cover them with tests.
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

## 8. Review locally

Follow the `review-local` skill (`.claude/skills/review-local/SKILL.md`) with this issue number. Do not open the PR while a confirmed blocker or major finding remains; report it to the user instead.

## 9. Open the PR

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
- `SKIP=typos pre-commit run --all-files` / `pre-commit run typos --files <changed files>`: <result>
- `sphinx-build -b doctest`: <result>

## Local review
<Aspects run, and the findings table and lists from step 6 of the review-local skill.>

## Out of scope / follow-ups
<Unrelated issues noticed. Omit if none.>
```

## 10. Wait for CI and handle PR review threads

After opening the PR and after every push, record the current head SHA and inspect the Codex Review Summary and reviews. If that exact head has no running or completed Codex review, proactively request one with a top-level `@codex review` PR comment; do not assume a push triggers a review. This request is authorized on the PR this skill opened. Avoid duplicate requests for the same head: inspect existing request comments before posting, and after an ambiguous API timeout, fetch comments to check whether the request was created before retrying. Then wait for CI and the Codex review of the current head commit. Record the head SHA and check that the results refer to that SHA; an older review is not evidence for the new head. If Codex has not reviewed it yet, keep waiting or report the pending review; do not claim it passed. Codex may post a review or a top-level PR comment, so inspect both. A successful CI check alone does not establish that the Codex review finished.

```bash
PR=<PR-number>
REPO=opthub-org/pytorch-bsf
gh pr view "$PR" -R "$REPO" --json headRefOid,reviews,statusCheckRollup
gh api --paginate "repos/$REPO/issues/$PR/comments"
# When this head has neither a running/completed review nor an existing request:
HEAD_SHA=$(gh pr view "$PR" -R "$REPO" --json headRefOid --jq .headRefOid) || exit 1
gh pr comment "$PR" -R "$REPO" --body "@codex review

Please review head $HEAD_SHA."
# GitHub can take a moment to register checks after a PR opens or a push.
# Abort on API errors; zero registered checks means pending, not failure.
CHECK_COUNT=0
CHECK_ATTEMPT=0
while [ "$CHECK_COUNT" -eq 0 ] && [ "$CHECK_ATTEMPT" -lt 12 ]; do
  CHECK_COUNT=$(gh pr view "$PR" -R "$REPO" --json statusCheckRollup \
    --jq '.statusCheckRollup | length') || exit 1
  CHECK_ATTEMPT=$((CHECK_ATTEMPT + 1))
  if [ "$CHECK_COUNT" -eq 0 ]; then sleep 10; fi
done
if [ "$CHECK_COUNT" -eq 0 ]; then
  echo "CI pending: no checks registered within 2 minutes; inspect workflow approval and triggers."
else
  gh pr checks "$PR" -R "$REPO" --watch
fi
gh api --paginate "repos/$REPO/pulls/$PR/reviews"
gh api --paginate "repos/$REPO/issues/$PR/comments"
```

After requesting review, verify that Codex acknowledges it and that the Review Summary identifies the recorded head as running or completed. Posting a request alone does not establish that review started. If there is no acknowledgement or an access/usage error, report it explicitly and stop retrying; do not repeatedly post requests. If the head changes, inspect and request review for the new head using the same procedure.

The registration loop makes at most 12 attempts, 10 seconds apart. If checks remain unregistered after 2 minutes (for example, workflows need maintainer approval), stop waiting and report CI as pending with the reason. Never treat an empty check list as success. If the head changes during registration or watch, restart the check for the new head.

List **all** review threads, including outdated ones, with their root comment's REST ID and resolution state. `--paginate` and `pageInfo` ensure PRs with more than 100 threads are covered; `comments(first: 1)` fetches only the root needed for a reply.

```bash
gh api graphql --paginate -F pr="$PR" -f query='
  query($pr: Int!, $endCursor: String) {
    repository(owner: "opthub-org", name: "pytorch-bsf") {
      pullRequest(number: $pr) {
        headRefOid
        reviewThreads(first: 100, after: $endCursor) {
          nodes {
            id isResolved isOutdated path line
            comments(first: 1) {
              nodes { databaseId body url commit { oid } }
            }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
  }'
```

For each unresolved thread:

1. Read the whole conversation before acting (`gh api --paginate "repos/$REPO/pulls/$PR/comments"` includes replies; match `in_reply_to_id` to the root's `databaseId`). Verify the finding against the current code. Fix it and run the relevant checks, or explain why no change is appropriate. Do not silently dismiss a finding.
2. Commit and push any fix. Reply **in that thread** with what changed and the full fixing commit SHA, or the concrete reason it was not fixed. For a thread on an older commit, point to the commit that already addressed it. Use the root comment's numeric `databaseId` as `COMMENT_ID`, not the GraphQL thread ID. Write the reply to a file to preserve newlines and avoid shell interpolation of review text:

   ```bash
   COMMENT_ID=<root-comment-databaseId>
   gh api "repos/$REPO/pulls/$PR/comments/$COMMENT_ID/replies" \
     -F body=@<scratchpad>/review-reply.md
   ```

   For a short literal reply, `-f body='Addressed in <full-SHA>: <explanation>.'` also works.

3. Resolve only after the thread has been fixed or answered with a substantive reason. A pending investigation or a promise to fix later is not an answer. Use its GraphQL `id` as `THREAD_ID`:

   ```bash
   THREAD_ID=<review-thread-node-id>
   gh api graphql -f threadId="$THREAD_ID" -f query='
     mutation($threadId: ID!) {
       resolveReviewThread(input: {threadId: $threadId}) {
         thread { id isResolved }
       }
     }'
   ```

After every push, repeat the review-request procedure, CI/head-review check and thread listing: bot reviewers may open new threads, and fixed threads on older commits may still be unresolved. Re-list threads after mutations to verify replies and resolution. If the head changes while waiting, repeat for the new SHA. Report any remaining CI failure, pending review or unanswered thread explicitly; do not merge or close the issue without explicit user authorization.

## 11. Report

Give the user the PR URL, a short summary, any design decisions made, the local review result, and anything left unverified or skipped (with the reason). If a check could not be made to pass, do not open the PR; report the failure output instead.
