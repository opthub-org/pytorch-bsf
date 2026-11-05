# Security maintenance

The **Security remediation** workflow handles Dependabot alerts and CodeQL
findings on `master`, including dependency fixes that need source or test
changes. It also runs a daily AI audit of executable code, model loading,
filesystem operations, subprocesses, dependencies, Docker, and CI boundaries.
The audit opens a patch only for a confirmed vulnerability with a reproducible
attack path. It does not guarantee that every vulnerability will be found.

## Automatic repair and merge

The workflow checks current alerts every five minutes, after validation workflows
finish, on manual dispatch, and on `repository_dispatch` with event type
`security-alert`. A dispatch is only a wake-up signal: the workflow reads fresh
GitHub alerts and ignores its payload. An external webhook receiver is not
required. GitHub can delay or skip scheduled executions under load, so five
minutes is a polling interval, not a remediation-time guarantee.

CodeQL runs daily and on pull requests and pushes to `master`, with the
`security-extended` query suite. A proactive Codex audit runs daily at 03:17 UTC
(12:17 JST). **Run workflow → audit** starts the same audit manually.

Every poll checks whether today's audit is due, so a replaced or delayed cron
event cannot lose the daily audit. Critical and high-severity alerts are selected
first. One finding is repaired per workflow run. Each finding is identified by its
current advisory data, so an upstream patch or changed advisory starts a new repair
budget. The automation:

1. Reserves one of at most three repair attempts in the **Security automation
   status** issue before calling the AI.
2. Runs Codex on a runner with a read-only GitHub token. Its OpenAI API key is
   isolated by the official Action's `drop-sudo` strategy and API proxy. Package
   updates can reach the registries listed in `.github/security/config.toml`.
3. Passes the patch as untrusted data to a fresh runner. That runner executes
   only trusted controller code, validates the changed files, and uses the GitHub
   API to create or update a PR on `automation/security/*`. Interrupted branch/PR
   publication is resumed without spending another AI attempt.
4. Explicitly dispatches PyTest, Conda, release-configuration, CodeQL, and security
   policy validation. This avoids relying on PR CI triggered by `GITHUB_TOKEN`,
   which may otherwise be suppressed or await approval.
5. Requires every expected test/matrix job, documentation and distribution checks,
   CodeQL, and `codecov/patch` to succeed on the current head. Missing, pending,
   skipped, or neutral required checks cannot authorize a merge. A failed check
   returns the patch to the repair loop, except that **pages-producer** failing on a
   producer change stops the task as blocked, because only a maintainer can land the
   template.
   Test runners have no repository secrets or OIDC permission. Coverage is passed
   as report data to the protected `security-coverage.yml` workflow on a fresh
   runner; only that runner receives Codecov authentication. It checks out trusted
   `master`, validates the report file, disables plugins, and explicitly identifies
   the tested commit and PR without executing candidate code.
6. Runs an independent Codex review in a fresh read-only session. Findings return
   to the repair loop. Each revision has at most three review attempts. Open
   review threads are resolved only when the independent reviewer explicitly
   verifies the current code addresses every one.
7. Rechecks the head, base, CI, review threads, and merge state immediately before
   a squash merge with the expected head SHA. It also reads the exact committed
   diff as Git objects and reapplies the publication policy, including additions
   pushed after the original patch. Candidate code is never executed by this check.
   Active change-request reviews still block merging. GitHub's repository rules
   remain in force.
8. Explicitly dispatches validation, Pages, and Release after merging, since
   Actions-token merges do not reliably trigger downstream push workflows.
   Pending dispatches are persisted individually and retried without another AI
   call. A failed publication is isolated so other findings can still proceed.

Technical recovery failures reset after a successful recovery. Three consecutive
publication or PR-evidence recovery failures pause that task for one hour before
another automatic retry; this does not replenish its AI attempt budget. Pending
post-merge dispatches continue retrying on subsequent runs. Status records retain
currently open alerts and unfinished PR/publication/post-merge work. Closed
findings without pending work and old completed audits are removed to bound
history growth. A genuinely
reopened alert receives a new repair budget.
If the same alert remains open after its repair PR is merged or closed, the status issue
explicitly marks it blocked pending alert clearance or investigation of post-merge
validation. A merged PR alone never marks an alert cleared or dismisses it. This
verification state does not reset the finding's AI budget.

Repository and alert text are untrusted evidence, not instructions. Secrets never
enter the task JSON. Public status records and the task manifest contain finding IDs and
opaque revisions; detailed alert evidence is fetched only while preparing the
ephemeral AI prompt. Patch artifacts contain proposed repairs and their summaries;
PRs and Actions logs have the repository's visibility. The AI jobs have no
repository write token; publication and
merge jobs never execute candidate code. The publisher rejects changes to the
automation itself, its tests and prompts, repository instructions, validation
configuration/helpers (including the trusted `tools/` and `tests/tools/` trees),
the CodeQL scanner workflow,
root modules that shadow Python's standard library or startup hooks,
common credential paths and key-file extensions, symlinks,
binaries, and patches larger than 750 KB. File-content checks recognize OpenAI
key patterns; they are not a complete secret scanner. Root `pyproject.toml` changes
are limited to dependency and Python support declarations; nested manifests require
maintainer review. Other workflow changes are limited to existing immutable
action SHA pins. Publishing workflow-file changes
may additionally require an App/PAT with GitHub's workflows permission; without
it, the operation stops and is recorded in the status issue.

## Secret scanning

[Betterleaks](https://github.com/betterleaks/betterleaks) checks staged changes in
the `pre-commit` git hook. The `secret-scan` job of the **Security policy** workflow
checks the full history of the checked-out commit, including merge commits, on pull
requests and pushes to `master`. On a pull request that commit is the merge with
`master`, so a finding on another branch cannot fail it. The job fails on a merge of
three or more parents, because Betterleaks cannot read what such a merge changes.

Betterleaks reads diffs. It does not scan commit messages, files that git treats as
binary, or branches without a pull request. It redacts each finding's own secret, but
the log still shows the commit message and any other text on the reported line, so
treat every credential on that line as exposed.

If the git hook blocked the commit, nothing was recorded: remove the secret from the
file, stage it again and commit. Revoke the secret if it may be exposed elsewhere.

For a real secret in a commit:

1. Revoke or rotate it first. Rewriting history does not undo the exposure of a pushed
   credential, and removing it in a later commit does not clear the job.
2. If the commit is not on `master`, rewrite the branch to drop the secret and
   force-push.
3. If the commit is on `master`, add its `Fingerprint` from the job's log to
   `.betterleaksignore`, and state in the PR that the credential was revoked. Rewriting
   the history of `master` is a maintainer's decision.

For a false positive, give the reason in the PR and use one of these:

- Add `# betterleaks:allow` to the line in the commit that introduces it. If that
  commit is already on the branch, amend or rebase it, as a later commit does not clear
  an earlier one.
- In a file without comments, such as JSON or CSV, add the `<path>:<rule>:<line>` entry
  to `.betterleaksignore`. The git hook prints it as `Fingerprint`. The job's log prints
  `<commit>:<path>:<rule>:<line>`, so drop the commit from that value; it may show the
  path of a conflict resolved in a merge as `b/<path>`, which the entry must repeat. The
  entry applies to that path, rule and line in every commit, so if the line moves, add
  an entry with the line number that the job reports.
- If the commit is on `master`, add its `Fingerprint` from the job's log to
  `.betterleaksignore`.

Do not add an entry that starts with a commit that is not on `master`: a rebase or
squash merge gives the commit a new SHA, and a line that a pull request's merge commit
changes is reported under that merge's SHA, which changes whenever the merge is
recreated. The security automation cannot change `.betterleaksignore` or the scanner's
configuration files.

## Operation and limits

The repository's Actions Secret **OPENAI_API_KEY** powers repair and independent
review. Its value must never be committed or printed. GitHub Actions must be
allowed to create pull requests. No persistent GitHub PAT is required for code,
manifest, or lockfile repairs; the workflow uses job-scoped `GITHUB_TOKEN`s.
The observer uses `read-all`, including Dependabot-alert read access; it has no
write permissions and receives no OpenAI secret.

Set the repository Actions variable **SECURITY_AUTOMATION_ENABLED** to `false` to
stop subsequent repair, publication, review, and merge jobs. Also cancel active
workflow runs to stop a job that is already executing. Optionally set **SECURITY_AUTOMATION_MODEL** to a
model available to the API project; otherwise the Codex Action uses its default.
Repair and review jobs have 40- and 30-minute timeouts. Only a finding's changed
upstream data replenishes its automatic attempt budget. API/quota failures,
incompatible upstream fixes, CI registration failures, and exhausted attempts
remain visible in the status issue and Actions runs; they are never treated as
successful repairs or dismissed alerts.
Unresolved review threads with more than 100 comments stop automatically rather
than approve a truncated discussion. Detailed negative-review diagnostics are
discarded instead of copied into public state; a subsequent repair independently
re-examines current code, alerts, and CI evidence.

This automation covers repository changes it can validate. It does not revoke
leaked credentials, alter cloud permissions, patch deployed infrastructure, or
automatically accept breaking product decisions. Secret-scanning alerts require
additional GitHub App/PAT access and provider-specific revocation integrations;
they are not fetched or forwarded to the AI. Complex workflow migrations and
changes to the automation's trust boundary require maintainer intervention.
Existing Dependabot security updates remain enabled. The repair pipeline may
open its own PR when Dependabot's patch is insufficient or unavailable. Dependabot
PRs follow the policy below.

## Dependabot auto-merge

The **Dependabot merge** workflow merges every Dependabot PR once all checks of
its exact head succeed, including major updates in every ecosystem. The maintainer
accepts that a compatible-looking update may break a release: a release that does
not work is safer than vulnerable software that keeps running, and anyone who needs
the old behavior can downgrade at their own risk.

CI cannot detect a malicious release. To reduce that risk, `cooldown` in
`.github/dependabot.yml` delays version updates by three days, so that a bad release
can be withdrawn first.

It merges only a same-repository PR whose commits Dependabot signed, and which
changes dependency manifests or only the action pins of workflows. Anything else stops.
Every open PR has one triage comment with the review state, the decision, the reason,
what would make it mergeable, and the deadline. A stopped PR is assigned to the
maintainer, who is notified again if the decision becomes overdue after seven days.

To keep a PR from merging, comment on it:

```text
/dependabot-hold until=YYYY-MM-DD <reason>
```

The hold ends after that date, at most 30 days ahead, or when someone comments
`/dependabot-release`. Only an owner, member or collaborator can use these commands,
and a run may take about 15 minutes to read them. A malformed hold still stops the merge.

Set the repository Actions variable **DEPENDABOT_MERGE_ENABLED** to `false` to stop
merging while triage continues.

The controller tests run without API access:

```bash
python -m unittest discover -s tests/tools -p security_automation.py -v
```
