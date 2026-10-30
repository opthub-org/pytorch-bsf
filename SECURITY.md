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

Critical and high-severity alerts are selected first. One finding is repaired per
workflow run. Each finding is identified by its current advisory data, so an
upstream patch or changed advisory starts a new repair budget. The automation:

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
   returns the patch to the repair loop.
6. Runs an independent Codex review in a fresh read-only session. Findings return
   to the repair loop. Each revision has at most three review attempts. Open
   review threads are resolved only when the independent reviewer explicitly
   verifies the current code addresses every one.
7. Rechecks the head, base, CI, review threads, and merge state immediately before
   a squash merge with the expected head SHA. Active change-request reviews still
   block merging. GitHub's repository rules remain in force.
8. Explicitly dispatches validation, Pages, and Release after merging, since
   Actions-token merges do not reliably trigger downstream push workflows.

Repository and alert text are untrusted evidence, not instructions. Secrets never
enter the task JSON. Public status records and the task manifest contain finding IDs and
opaque revisions; detailed alert evidence is fetched only while preparing the
ephemeral AI prompt. Patch artifacts contain proposed repairs and their summaries;
PRs and Actions logs have the repository's visibility. The AI jobs have no
repository write token; publication and
merge jobs never execute candidate code. The publisher rejects changes to the
automation itself, its tests and prompts, repository instructions, secret files,
symlinks, binaries, and patches larger than 750 KB. Other workflow changes are
limited to existing immutable action SHA pins. Publishing workflow-file changes
may additionally require an App/PAT with GitHub's workflows permission; without
it, the operation stops and is recorded in the status issue.

## Operation and limits

The repository's Actions Secret **OPENAI_API_KEY** powers repair and independent
review. Its value must never be committed or printed. GitHub Actions must be
allowed to create pull requests. No persistent GitHub PAT is required for code,
manifest, or lockfile repairs; the workflow uses job-scoped `GITHUB_TOKEN`s.
The observer uses `read-all`, including Dependabot-alert read access; it has no
write permissions and receives no OpenAI secret.

Set the repository Actions variable **SECURITY_AUTOMATION_ENABLED** to `false` to
stop new repairs and merges. Optionally set **SECURITY_AUTOMATION_MODEL** to a
model available to the API project; otherwise the Codex Action uses its default.
Repair and review jobs have 40- and 30-minute timeouts. Only a finding's changed
upstream data replenishes its automatic attempt budget. API/quota failures,
incompatible upstream fixes, CI registration failures, and exhausted attempts
remain visible in the status issue and Actions runs; they are never treated as
successful repairs or dismissed alerts.

This automation covers repository changes it can validate. It does not revoke
leaked credentials, alter cloud permissions, patch deployed infrastructure, or
automatically accept breaking product decisions. Secret-scanning alerts require
additional GitHub App/PAT access and provider-specific revocation integrations;
they are not fetched or forwarded to the AI. Complex workflow migrations and
changes to the automation's trust boundary require maintainer intervention.
Existing Dependabot security updates remain enabled. The repair pipeline may
open its own PR when Dependabot's patch is insufficient or unavailable; ordinary
version-update PRs are not auto-merged by this policy.

The controller tests run without API access:

```bash
python -m unittest discover -s tests/tools -p security_automation.py -v
```
