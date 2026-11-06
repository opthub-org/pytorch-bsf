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

The **Dependabot merge** workflow merges every Dependabot PR once all checks of its
exact head succeed and no known advisory stops it, including major updates in every
ecosystem. No person or AI reviews it first. The maintainer accepts that a
compatible-looking update may break a release: a release that does not work is safer
than vulnerable software that keeps running, and anyone who needs the old behavior can
downgrade at their own risk.

CI cannot detect a malicious release. To reduce that risk, `cooldown` in
`.github/dependabot.yml` delays version updates by three days, so that a bad release
can be withdrawn first. Security updates skip the delay. Dependabot's Python updater
skips yanked releases; this is not confirmed for the other ecosystems.

The controller also looks up the version that each commit's Dependabot metadata names,
on every evaluation of a PR that passes the provenance, change and metadata checks:
pip packages in [OSV](https://osv.dev), and GitHub Actions in the GitHub Advisory
Database, including malware advisories in both. A known advisory stops the merge, and
the triage comment names it unless a hold or an earlier stop reason takes precedence. A
failed lookup merges nothing and fails the job. The triage comment says that the lookup
failed, unless a hold, another stop reason, a conflict or a pause comes first; then only
the failed job shows it. A pip or Actions commit whose metadata is missing or names a
dependency in an unexpected form stops the merge, as does an ecosystem with no
classification here. When the metadata names a commit SHA, no version or anything else
that cannot be compared, any advisory on the package stops the merge, and a maintainer
must check its affected range. An action tag with fewer than three numbers, such as
`v3`, counts as not comparable, because it can move to any 3.x release.

Neither the cooldown nor the advisory lookup catches:

- a bad release that nobody has reported and no registry has withdrawn within the three
  days, including an unnoticed malicious release, which merges once CI passes;
- a problem reported only elsewhere, such as in an upstream issue, unless a registry
  withdraws the release or a maintainer holds the PR;
- known advisories on pre-commit hooks and Docker images, which have no advisory source
  here;
- transitive dependencies, or actions nested in a composite action;
- other versions that a pip range allows, because only the named version is looked up;
- unreviewed GitHub advisories for actions, which carry no package data;
- a commit without Dependabot's metadata whose quoted text ends in a block of the same
  shape, because the text alone cannot prove who wrote it; the lookup then uses the quoted
  names;
- action advisories filed under a name other than `owner/repo[/path]`, such as a URL or
  an `@scope/name`;
- an advisory published after the merge. Dependabot alerts report this for pip packages
  only: GitHub creates no alerts for SHA-pinned actions, and none for pre-commit hooks
  or Docker images, so nothing here reports it for those.

The remaining exposure is accepted: action pins also update in privileged workflows,
including this one, the security automation and PyPI publishing, and a malicious action
runs with their permissions once it reaches `master`.

It merges only a same-repository PR whose commits Dependabot authored and GitHub
signed, and which edits dependency manifests or only updates existing action pins in
workflows. Any other change, or a change request from a user with write access, stops the
merge. Anyone can review a public repository, so other reviews do not count. Every open PR
has one triage comment with the review state, the decision, the reason, what would
make it mergeable, and the deadline. A stopped PR is assigned to the maintainer, who
gets one more comment if the decision becomes overdue after seven days. Measured from
the head's latest commit, unfinished checks, blocked merge states and a failed advisory
lookup stop the merge after two hours, a conflict that Dependabot has not rebased after a
day, and a merge state that GitHub has not computed after seven days. No failure time is
stored, so one failed lookup can stop an older head at once, unless it conflicts or
merging is paused; a later successful lookup clears the stop.

To keep a PR from merging, comment on it, with the command on the first line. A review
body does not count:

```text
/dependabot-hold until=YYYY-MM-DD <reason>
```

The hold ends after that date in UTC, at most 30 days ahead, or when someone comments
`/dependabot-release`. Only a user with write access can use these commands. A comment
starts no run, but every run reads the commands while it evaluates a PR, so a hold posted
before an evaluation starts stops its merge. A malformed hold, such as one with a past date,
stops the merge for seven days.

A run evaluates the PRs when a workflow that reports a required check starts or finishes
on a Dependabot branch, when the GHPages publisher runs, and on a schedule; the
[release controller](#release-automation) adds Release runs and the release branch. Runs
of other workflows on `master` skip, and other branches start no run. Codecov reports
`codecov/patch` without starting a run, so when only that check keeps a PR from merging,
a run waits for it and then evaluates the PR again from scratch. The wait ends five
minutes after the run started, a limit that all PRs of the run share. Each
run evaluates every open Dependabot PR, so a merge or triage comment usually follows CI
within minutes.

Some changes take effect only at the next run: a deadline passing, a hold expiring or
being released, a review, an advisory published while a PR waits, and a `codecov/patch`
result that arrives after that wait. The schedule asks for a run every hour, but
GitHub delays and drops scheduled runs. From 2026-11-01 to 2026-11-05, a workflow here
scheduled every five minutes and once a day had only 25 scheduled runs, 2.5 to 7.3 hours
apart. These changes can therefore take hours, and GitHub guarantees no upper bound. To
apply them at once, run the workflow by hand.

Set the repository Actions variable **DEPENDABOT_MERGE_ENABLED** to `false` to stop
merging while triage continues. Any value other than an empty one or `true`, in any case,
also stops it.

The controller tests run without API access:

```bash
python -m unittest discover -s tests/tools -p security_automation.py -v
python -m unittest discover -s tests/tools -p dependabot_merge.py -v
python -m unittest discover -s tests/tools -p release_merge.py -v
```

## Release automation

A merged fix helps nobody until it is released, so release-please's release PR also
merges without a human, in a second step of the **Dependabot merge** workflow
(`tools/release_merge.py`). This includes releases with breaking changes: a release
contains every change since the previous one, so holding back a breaking change would
hold back the fixes behind it too. Anyone who needs the old behavior can pin the
previous version.

Pip and Docker updates are `fix(deps)` commits, which release-please releases.
Pre-commit and GitHub Actions updates are `build(deps)` commits, which change nothing
that is distributed, so they alone create no release PR.

The release PR merges once the checks of its exact head succeed and:

- the release contains a commit with a conventional header (see below) of type `fix` or
  `feat` and the scope `deps`, `deps-dev` (which Dependabot writes for what it considers
  development dependencies) or `security`, with or without `!`; such a release merges at
  once; or
- a day has passed since the newest releasable commit in the release landed on `master`,
  which gathers a series of changes into one release. The head's own time does not
  count, because release-please rewrites the head each day for the date in the
  changelog. A squash or rebase merge lands at its own time; commits brought in by a
  merge commit land when the merge commit does.

A conventional header is read from the subject and from each unindented body line, such
as the PR title in a merge commit's body. A commit is releasable when one of its headers
has a type that release-please releases (`feat`, `fix`, `perf`, `revert`, `deps` or
`docs`, in any case), or when its subject header breaks compatibility (`!` or a
`BREAKING CHANGE` footer); `tests/tools/release-please-fixture/releasable.json` pins
these cases against release-please. This reading only approximates release-please, so
it sets when a release merges, never what it holds.

When `master` has commits that the head lacks, the controller waits until the
`release-pr` job of the Release workflow has succeeded on the head of `master`, because
a stale head would ship releasable commits without a changelog entry, and release-please
lists only commits after the release. That run, not the controller, decides which
commits the release needs, so no commit message is parsed for this. Release runs on
`master` never overlap and start in push order, so an older one cannot update the PR
after it. Re-running an older run cancels a newer one that waits. The controllers start
such a cancelled run again after their own merges (see below), and that run in turn
cancels the re-run if it still waits, so re-run an older run only while no Release run
waits.

The controller merges only release-please's own PR for `master`: opened by
`github-actions[bot]` or the release App's bot from
`release-please--branches--master--components--torch_bsf` with a release title and the
`autorelease: pending` label, made of commits that one of those accounts wrote and that
GitHub signed (as `web-flow`, or as the App's bot for the App's own commits), and
changing only the version in `pyproject.toml` and `.release-please-manifest.json`, and
`CHANGELOG.md`.
Anything else stops the merge. The release App, described [below](#release-app), opens
and updates the PR so that its CI runs.

The GHPages build and publisher exclude release-please branches, so their checks are not
required there; the PyTest `docs` job still builds the documentation. A merge with
`GITHUB_TOKEN` starts no workflow, so after merging, the controller starts the Release
workflow on `master`, which creates the tag and the GitHub release and publishes to PyPI
and GHCR, and then the CI workflows. If starting them fails, a later run starts the
missing ones. The same recovery follows the Dependabot controller's merges. A later run
also starts Release again when every Release run on the head of `master` was cancelled
before any job ran, as the workflow's concurrency group cancels a waiting run that a
newer run replaces. A run cancelled after a job started is left alone, because a person
may have stopped it. A person who cancels a run that is still waiting cannot be told
apart, so to stop a release, cancel the run after one of its jobs starts, or disable
the workflow. Release is started again at most `MAX_REDISPATCHES` times for one head
(see `tools/dependabot_merge.py`); after that, the controller job fails until someone
starts Release by hand or `master` moves.

The release controller runs when Release starts or finishes on `master`, when a CI
workflow starts or finishes on the release branch, and on the same schedule as the
Dependabot controller, and it waits for `codecov/patch` in the same way. The PR keeps
one status comment with the decision, the reason and what would make it mergeable. The
controller stops the merge, and mentions the maintainer once for each stopped head, on:

- CI on the release head that waits for approval, which means the release App is not
  set up;
- failed checks on the release head;
- failed CI on the head of `master`. A merge also ships the `master` commits that
  release-please did not add to the release PR, such as `build(deps)` or `ci` commits,
  and those were tested only in their own PRs. Only the CI workflows and their expected
  checks count, not the documentation checks, `codecov/patch` or other checks on that
  commit, such as GitHub's own Dependabot update jobs;
- a `release-pr` job on the head of `master` that did not succeed, or that has not
  finished two hours after that head landed, while the release head lacks commits of
  `master`;
- checks still unfinished two hours after the head's commit;
- a merge state other than mergeable two hours after the head's commit, or seven days
  while GitHub has not computed it. A conflict usually clears when release-please
  rebases the PR for the next releasable change;
- a missing tag for the previous release, or a PR outside the limits above;
- a merge that GitHub refuses with HTTP status 403 or 422, or with 405 or 409 seven
  days after the head's commit (until then the next run tries again), or that GitHub
  answers without merging.

To keep a release from merging, mark its PR as a draft. Set the repository Actions
variable **RELEASE_MERGE_ENABLED** to `false` to stop all release merges; as with
`DEPENDABOT_MERGE_ENABLED`, any value other than an empty one or `true` stops them.

### Release App

A release PR that release-please updates with `GITHUB_TOKEN` gets `pull_request` runs
that wait for approval (`action_required`). Runs started any other way, for example by
hand, do not count as the PR's required checks, so GitHub blocks the merge. The Release
workflow therefore opens and updates the release PR with a GitHub App's token, and
creates releases with `GITHUB_TOKEN` as before, so that tag and release events start no
second publication. The PR update runs in its own job, `release-pr`, which no publishing
job needs, so that a failure there, such as a revoked key, cannot keep a created release
from being published. Without the App, `GITHUB_TOKEN` still opens the PR, and the
controller stops it, saying that CI waits for approval. To set the App up:

1. Create a GitHub App owned by the organization, with no webhook, and repository
   permissions **Contents: Read and write** and **Pull requests: Read and write**
   (Metadata: Read-only is added automatically). Install it on this repository only.
2. Generate a private key. Add the App's client ID as the repository Actions variable
   **RELEASE_PLEASE_APP_CLIENT_ID** and the key as the secret
   **RELEASE_PLEASE_APP_PRIVATE_KEY**.
3. Add the App's bot account, `<app-slug>[bot]`, as the variable **RELEASE_PLEASE_BOT**,
   so that the controller accepts the PRs and commits it writes. A value of another form
   fails the controller's job, and a PR by a bot that the variable does not name stops
   with a reason that names the variable.

The key can push to any branch of this repository, including `master` unless the branch
rules stop it, so keep it only in that secret, and rotate it if it leaks.
