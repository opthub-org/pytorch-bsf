# Contributing Guide

This document is a guide for contributing to pytorch-bsf.

## Table of Contents

1. [Development Policy](#development-policy)
2. [Setting Up the Development Environment](#setting-up-the-development-environment)
3. [First Things to Try](#first-things-to-try)
4. [Development Workflow](#development-workflow)
5. [Commit Rules](#commit-rules)
6. [Coding Conventions](#coding-conventions)
7. [Writing Tests](#writing-tests)
8. [Repository Structure](#repository-structure)

---

## Development Policy

pytorch-bsf is a Python library for Bézier Simploid Fitting (formerly Bézier Simplex Fitting). Development follows these principles:

- **Simple API**: Keep the API minimal so users can get started with a single call to `torch_bsf.fit()`.
- **Backward compatibility**: Before 1.0, deliberate API consolidation may break compatibility in a minor release. Document migrations and use a breaking Conventional Commit. After 1.0, reserve breaking changes for major releases.
- **Type safety**: Add type hints to all public APIs (PEP 561 compliant; `py.typed` marker is included).
- **Test coverage**: Every new feature or bug fix must be accompanied by tests.
- **Documentation sync**: Changes to the public API must include updates to the Sphinx docs and docstrings.

### Dependency versions

pytorch-bsf is a library installed alongside other packages, so `[project].dependencies` in `pyproject.toml` declares the widest ranges that work, not the latest versions:

- **Lower bound**: the oldest version that passes the tests. The `lowest` job of `.github/workflows/python-package.yml` installs every lower bound on Python 3.11 (`uv pip install --resolution lowest-direct '.[lightning,mlflow]'`, so the bounds of the optional `lightning` and `mlflow` are checked too) and runs pytest, so a bound that does not work fails CI. Raise a lower bound only when the code needs a newer feature or a bug fix, and say why in the PR.
- **Upper bound**: below the next major version (`<3`). Widen it after checking that the new major version works.
- **Dependabot** uses `versioning-strategy: increase-if-necessary` for pip: it leaves a range alone while it admits a new release and opens a PR when a release falls outside it. Review any proposed lower-bound change against the policy above. It also checks pre-commit hooks, GitHub Actions and the Docker base image weekly.
- Declare only packages that `torch_bsf` imports. Do not pin transitive dependencies to work around a bug in another package; if one is unavoidable, add a comment with the upstream issue.
- Keep `environment.yml` in sync with `pyproject.toml`.

### Python lifecycle updates

Security alerts have a separate [automatic repair and merge policy](SECURITY.md).
It covers Dependabot, CodeQL, and a daily source security audit, with independent
AI review and exact-head CI gates before merging.

The **Dependency lifecycle** workflow runs every Monday at 04:23 UTC (13:23 JST),
and can also be run from Actions with **Run workflow**. After this workflow is
merged into `master`, its first run catches up with any existing EOLs or releases.
It uses the [official Python release-cycle API](https://peps.python.org/api/release-cycle.json),
the same source used by the Python Developer's Guide.

The updater removes Python 3 versions at EOL and adds released versions in the
`bugfix` or `security` phases. It excludes prereleases, never lowers the existing
minimum, and rejects incomplete or invalid upstream data. It synchronizes
`requires-python`, classifiers, Ruff/mypy targets, the lowest/latest CI jobs,
the test matrix and coverage upload, Conda, and installation documentation, then
runs `uv lock`. It keeps package dependency lower bounds under the policy above;
package releases, Actions and Docker updates are handled by Dependabot. EOL
monitoring currently covers Python; other products need their own lifecycle policy.

Changes go into one PR on `automation/python-support`; repeated runs update that
PR and runs without changes open no PR. Raising the Python minimum uses a breaking
Conventional Commit. The PR requires compatibility review and passing CI, since
new Python releases can precede dependency wheels or Conda packages. A failed
lock refresh is reported in the PR and must be fixed before merging.

Repository **Settings → Actions → General → Workflow permissions** must allow
GitHub Actions to create pull requests. The built-in `GITHUB_TOKEN` needs no new
secret. If no CI runs appear after the bot creates or updates a PR, close and
reopen that PR from a maintainer's GitHub account. This human `reopened` event
starts the PyTest, Conda and CodeQL workflows; repeat after later bot updates if
checks do not start. If GitHub instead shows pending workflow approval, select
**Approve workflows to run**. Generated PR bodies include these instructions.
For fully automatic CI, optionally set `DEPENDENCY_UPDATE_TOKEN` to an App/PAT
token with repository contents and pull-request write access.
The CI workflows read their minimum, latest and matrix versions directly from
the package metadata, so generated PRs do not need to edit workflow files.

`environment.yml` also selects the Python used for full optional integrations,
documentation and Codecov. Keep that version while it is supported, and advance
it after checking dependency compatibility. Optional dependency wheel constraints
can lag new Python releases; newer matrix entries test the other integrations.
The README and installation guide track the full-integration range and list the
newer Python versions that should use the non-MLflow extras.
The updater raises the Conda Python automatically if it reaches EOL, so this
compatibility pin cannot retain an EOL interpreter. CI then exposes any remaining
dependency incompatibility in the generated PR.

Preview the generated patch without changing files:

```bash
python tools/update_python_support.py --check
```

The preview exits 1 if changes are needed. Use `--data-file path/to/release-cycle.json`
for an offline preview. Run the updater's tests without installing numerical dependencies:

```bash
python -m unittest discover -s tests/tools -p update_python_support.py -v
```

---

## Setting Up the Development Environment

### Prerequisites

- Python 3.11 or later
- [Conda](https://docs.conda.io/) or pip

### With Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate pytorch-bsf
```

### With pip

```bash
# Clone the repository
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# Install in editable mode with development dependencies
pip install -e ".[develop]"
```

The `[develop]` extras include:

| Package | Purpose |
|---------|---------|
| `pytest`, `pytest-cov`, `pytest-randomly` | Running tests |
| `diff-cover` | Coverage of the lines a branch changes (as `codecov/patch` reports it) |
| `hypothesis` | Property-based tests |
| `mypy` + type stubs | Static type checking |
| `ruff` | Linting, import sorting and code formatting |
| `pre-commit` | Running Ruff, numpydoc-validation, `tools/lint_conventions.py` and typos on `git commit` |

With Conda, install them into the environment with `pip install -e ".[develop]"` as well.

MLflow is an optional extra (`pip install -e ".[develop,mlflow]"`, also included in `all`). The CLI records experiments only when `fit --track` is requested, even when MLflow is installed. Without the extra, ordinary fitting and prediction still work and the MLflow tests are skipped; the `optional-deps` CI job covers `minimal` (no extras), `no-lightning` and `no-mlflow` configurations.

### Install the git hook

Install the [pre-commit](https://pre-commit.com/) hook so that Ruff, numpydoc, `tools/lint_conventions.py` and typos check your changes (and Ruff formats them) on every commit:

```bash
pre-commit install
```

Run this once per clone. The hook is installed in the shared `.git/hooks` directory, so it also runs in every `git worktree` of the clone (including `.claude/worktrees/`). If a hook fixes or reformats files, the commit is aborted; review the changes, `git add` them and commit again.

---

## First Things to Try

After setting up the environment, run the following to verify everything works.

### 1. Run the tests

```bash
pytest tests/
```

All tests should pass.

To see which changed lines the tests miss, as the `codecov/patch` check on a PR does, write a
coverage report and compare it with `master`:

```bash
pytest tests/ --cov=torch_bsf --cov-report=xml
git fetch origin master
diff-cover coverage.xml --compare-branch=origin/master
```

### 2. Try the CLI

The repository includes sample data (`params.csv`, `values.csv`):

```bash
python -m torch_bsf fit --params=params.csv --values=values.csv --degree=3 --save=model.bsf
python -m torch_bsf predict --model=model.bsf --params=params.csv --save=predictions.csv
```

Run from the repository root to use the bundled CSV files. Fitting writes the
reusable model archive; prediction writes one output row per parameter row.

### 3. Run the quickstart script

```bash
bash examples/quickstart/run.sh
```

### 4. Run with MLflow

Install the `mlflow` extra first. To record a local CLI run explicitly:

```bash
python -m torch_bsf fit --params=params.csv --values=values.csv --degree=3 --save=model.bsf --track
```

The MLflow project entry point also supplies `--track`:

```bash
mlflow run . --entry-point main \
  -P params=params.csv \
  -P values=values.csv \
  -P degree=3
```

### Keeping these examples runnable

`tests/test_contributor_examples.py` executes the two CLI commands in step 2
against the bundled CSV files in a temporary directory, then checks the saved
model and prediction shape. It also executes the Python examples in the bug and
feature Issue forms. These checks run with the normal pytest suite; update the
examples and their behavioral assertions together when the API changes.
The optional tracking commands are covered by the MLflow adapter/CLI tests,
and `examples/quickstart/run.sh` runs separately in CI. Sphinx examples are
checked by the existing doctest build.

---

## Development Workflow

### Filing issues

Issues are written so that a developer or an AI agent can implement them and open a PR without asking for more context. Use one of the issue forms:

| Form | Use for | Label |
|------|---------|-------|
| **Bug report** | Incorrect behavior | `bug` |
| **Feature / improvement** | New features, API changes, refactoring, tests, performance, CI, tooling, dependencies | `enhancement` |
| **Documentation** | Sphinx docs, README, docstrings, examples | `documentation` |

- Fill in **Acceptance criteria** with concrete, checkable items; project-wide checks (tests, lint, type check, doctest, docs build) are implied.
- Name the files, modules or APIs involved when you know them, and state what is out of scope.
- Record decisions already made (backward compatibility, design choices) so the implementer does not have to guess.
- Link related issues or PRs to keep discussions focused.
- When filing from the command line (`gh issue create --body-file`), keep the same `### <Field>` headings as the form.

### Creating branches

Branch off from `master`:

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

Use a branch name that combines the commit type, issue number, and a short description:

| Type | Example |
|------|---------|
| New feature | `feat/123-add-new-sampler` |
| Bug fix | `fix/456-fix-nan-in-output` |
| Documentation | `docs/789-update-contributing` |
| Refactoring | `refactor/101-simplify-validator` |
| CI / build | `ci/102-update-workflow` |

### Local review before a PR

CI and PR review take a while per round trip, so review the branch locally first. With [Claude Code](https://claude.com/claude-code), run `/review-local` (it is also step 8 of `/work-on-issue`): it starts one read-only reviewer agent per aspect in parallel, verifies their findings and fixes them. The reviewers are defined in `.claude/agents/review-<aspect>.md` and share `.claude/skills/review-local/protocol.md`; with another tool, use those files as the reviewers' prompts.

| Aspect | Looks for |
|--------|-----------|
| Requirements and scope | Unmet acceptance criteria, scope creep, call sites left unchanged |
| Design and API | Duplicated abstractions, a larger API than needed, inconsistency with neighboring functions |
| Correctness and logic | Wrong formulas, indices, branches and error paths |
| Edge cases and input validation | Empty, single, degenerate, non-finite, wrong-typed or wrong-shaped inputs |
| Test quality | Untested branches (patch coverage), weak assertions, missing properties, flaky tests |
| Documentation consistency | Docstrings, Examples, Sphinx pages and CLI help that disagree with the code |
| Project conventions | This guide's rules not enforced by tools, commit and branch names |
| Readability and maintainability | Naming, duplication, long functions, missing or stale comments |
| Numerical robustness | Cancellation, overflow, dtype handling, tolerances, conditioning |
| PyTorch and framework integration | Devices, autograd, in-place operations, Lightning, scikit-learn and MLflow conventions |
| Performance | Complexity, Python loops over tensors, large intermediates, device transfers |
| Backward compatibility | Changed public API, file formats or CLI without a breaking-change declaration |
| Security and safety | Unsafe deserialization, file paths, subprocess use, workflow changes |
| Reproducibility and side effects | Global RNG or configuration changes, mutated arguments, nondeterminism |
| Dependencies and packaging | Undeclared or unguarded imports, version bounds, `environment.yml` sync |

The first eight aspects run for every change (correctness and edge cases may be skipped when no code changes); the others run when the diff touches what they cover. `.claude/skills/review-local/SKILL.md` has the exact rules. Record the result in the PR description under **Local review**.

### Distribution checks

The `distribution` CI job builds a wheel from tracked working-tree files and a
second wheel through an sdist. It checks package contents, the type marker and
license, then installs each wheel into a temporary target and exercises imports,
fit/predict and CLI help outside the checkout. Run it locally with runtime
dependencies plus `setuptools` and `wheel` installed:

```bash
python tools/check_distribution.py
```

The check also adds temporary namespace probes to verify that future
`torch_bsf` subpackages are retained and unrelated packages are excluded.
Builds and installations leave the current Python environment unchanged.

### PR review and merge

PyTest and Conda run on every PR (including drafts and forks) when it is opened,
updated, reopened, or marked ready for review, and on pushes to `master`. Feature
branch pushes use the PR checks to avoid duplicate runs. Contributions from forks
may require maintainer approval before GitHub starts their workflows; resolve
merge conflicts before expecting PR checks to run.

The Conda workflow builds and smoke-tests Docker images on PRs without
publishing them. Images are published to GHCR only for published releases.
The Release workflow calls the Docker workflow
after release-please creates a release; manually published releases trigger it
directly. Stable semantic versions receive the `latest` image tag automatically;
prereleases do not.

Release Please uses the supported `googleapis/release-please-action` repository,
pinned to [v5.0.0](https://github.com/googleapis/release-please-action/releases/tag/v5.0.0)
(Node 24). Its [manifest configuration](https://github.com/googleapis/release-please-action/tree/45996ed1f6d02564a971a2fa1b5860e934307cf7#advanced-release-configuration)
is in `release-please-config.json`; the removed v3 Action inputs `package-name`
and `bump-minor-pre-major` belong in that file. Do not also pass the Action's
`release-type` input: it selects a different mode that bypasses the manifest.
The root package keeps the release identifier `torch_bsf`, distribution name
`pytorch-bsf` in `pyproject.toml`, and existing `v0.x.y` tags without a component
prefix. The manifest starts at the existing `v0.24.0` release; future release PRs
update `.release-please-manifest.json`, `pyproject.toml` and `CHANGELOG.md` together.
Do not clear the manifest or set a new bootstrap version during an Action update.

The root component's `release_created` and `tag_name` outputs connect to the
Docker and PyPI build jobs. Both build only when `release_created == 'true'`,
and both build the reported release tag. The separate PyPI publishing job also
requires a successful PyPI build. Keep the reusable Docker call: releases created
with `GITHUB_TOKEN` do not start another release-event workflow.

Validate release configuration without creating a release or publishing anything:

```bash
# Node 24 or later; these dependencies are only for the offline fixture
npm ci --prefix tests/tools/release-please-fixture --ignore-scripts --no-audit --no-fund
npm test --prefix tests/tools/release-please-fixture
```

The **Release configuration** PR check runs the same fixture. It uses the real
Release Please 17.6.0 engine bundled in Action v5.0.0 and checks version bumps
against golden results verified with 15.13.0 from the previous Action v3.7.13.
The legacy engine is no longer installed because it brings vulnerable dependencies.
From 0.24.0, breaking changes and features produce 0.25.0; fixes produce 0.24.1.
It validates the schemas, release
history boundary, Python/CHANGELOG/manifest updates and publication wiring with
an in-memory repository that rejects writes. When changing the Action pin,
update the fixture's engine dependency and lockfile to match the upstream
Action's lockfile, then rerun the check.

#### PyPI Trusted Publishing and provenance

PyPI publication uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
and the official PyPA publish Action, pinned to
[v1.14.3](https://github.com/pypa/gh-action-pypi-publish/releases/tag/v1.14.3).
The `pypi-build` job builds the release tag without OIDC or PyPI credentials and
uploads the wheel and sdist as a single artifact. The `pypi` job downloads that
artifact by its immutable ID from the same workflow run and fails on a digest
mismatch. It has no checkout or build steps; only this job has `id-token: write`.
There is no API-token fallback.

Complete setup steps 1–2 before merging the migration. Retire the old token in
step 3 after the first verified release:

1. In the repository's **Settings → Environments**, create `pypi`. Under
   **Deployment branches and tags**, choose **Selected branches and tags** and
   allow only the branch `master`. The workflow starts on a `master` push, so
   restricting this environment to release tags would block publication even
   though the build checks out a release tag. Required reviewers are optional;
   leave them unset to retain automatic releases.
2. On the existing PyPI project's
   [Publishing settings](https://pypi.org/manage/project/pytorch-bsf/settings/publishing/),
   add a GitHub Trusted Publisher with these exact values:

   | Field | Value |
   |-------|-------|
   | Owner | `opthub-org` |
   | Repository | `pytorch-bsf` |
   | Workflow filename | `release-please-action.yml` |
   | Environment | `pypi` |

   The workflow field is the filename, without `.github/workflows/`. Review the
   [public transparency-log notice](https://docs.pypi.org/attestations/producing-attestations/#prerequisites)
   before enabling attestations: signing records are public and permanent.
3. After the first successful Trusted Publishing release and provenance check,
   revoke the old project publishing token in PyPI and remove the GitHub Actions
   secret `PYPI_API_TOKEN`. Keeping the old token during migration does not make
   this workflow use it.

`attestations: true` explicitly enables
[PEP 740 PyPI Publish Attestations](https://docs.pypi.org/attestations/) for every
distribution file. No signing key or new runtime dependency is needed. Regular
PR checks, package installation and the Release Please merge/release process
remain the same.

For the first release, check **both** the wheel and sdist on PyPI's **Security**
tab. Each should report Trusted Publishing and a PyPI Publish Attestation from
`opthub-org/pytorch-bsf`, `release-please-action.yml`, and the expected workflow
commit. Compare that commit with the publishing workflow run's triggering SHA.
Separately confirm that the build checked out the tag reported by Release Please,
and that the published files' SHA-256 hashes match the distributions in that
run's retained artifact. The workflow commit can be newer than the release tag
when a later master push recovers a previously uncreated release. A Publish
Attestation identifies the publishing workflow; it does not prove the source
commit used by the build. For cryptographic verification, follow the
[consumption guide](https://docs.pypi.org/attestations/consuming-attestations/):

```bash
# Use a separate tooling environment; this is not a library dependency.
uvx --from pypi-attestations pypi-attestations verify pypi \
  --repository https://github.com/opthub-org/pytorch-bsf '<PyPI file download URL>'
```

Run this for each of the two file URLs from PyPI. These attestations establish
the publishing identity and file integrity; they do not certify code safety,
dependency safety or reproducible builds. Full SLSA build provenance is a
separate feature. Normal pip and uv installs do not yet verify these attestations
automatically.

For failures before any upload, correct the publisher/environment configuration
or wait for the relevant service to recover, then use **Re-run failed jobs** on
the same workflow run. The distribution artifact is retained for seven days,
and a build retry gets a new artifact name. Re-running the entire workflow can
skip publishing because Release Please has already created the release.

If an upload partially succeeded, a retry can fail on duplicate files. Compare
the already published files' hashes and attestations with the retained artifact
before planning recovery; do not blindly enable `skip-existing`. Do not rebuild
or replace an already published file under the same version. Changes to the
repository owner/name, workflow filename or environment require updating the
PyPI Trusted Publisher registration as well.

#### Submitting and merging a PR

1. Push your branch and open a Pull Request targeting `master`.
2. In the PR description, include what changed, the related issue (`Closes #XXX`), how to test it, and the result of the local review.
3. After opening the PR and after every push, check whether Codex is reviewing or has reviewed the current head. If neither applies and no request for that head is already pending, automatically request review with an `@codex review` PR comment and verify acknowledgement for that SHA. The `work-on-issue` skill authorizes these review-request comments on its own PR. Wait for all CI checks (GitHub Actions) and the Codex review of the current head commit. Repeat after every push; results for an older SHA do not cover the new head.
4. Handle every unresolved review thread, including outdated threads: verify the finding, fix and test it or decide why no change is appropriate, then reply in the thread with the fixing commit SHA or a concrete reason. Resolve only threads that are fixed or substantively answered, then check for new threads after each review round. The `work-on-issue` skill authorizes and requires these replies and resolutions on the PR it opened; its [PR review step](.claude/skills/work-on-issue/SKILL.md#10-wait-for-ci-and-handle-pr-review-threads) documents paginated GraphQL `reviewThreads`, REST replies (`gh api repos/opthub-org/pytorch-bsf/pulls/<N>/comments/<id>/replies -f body=…`), and GraphQL `resolveReviewThread` commands. The `review-local` skill's no-push/no-comment limit applies only to the local review itself.
5. Merge once you have an approval and explicit authorization to merge. Invoking `work-on-issue` does not authorize merging, closing issues, or commenting elsewhere on GitHub.
6. Delete the branch after merging.

### Codecov authentication and upload verification

The PyTest `build` job uploads `coverage.xml` only on the full-integration Python
version selected by `environment.yml`. Upload errors fail that job; a green
workflow from before this policy is not proof that Codecov accepted a report.

- Ordinary PRs and `master` pushes use the existing `CODECOV_TOKEN` Actions
  secret, configured under **Settings → Secrets and variables → Actions**.
- Same-repository PR runs triggered by `dependabot[bot]` use Codecov's OIDC
  authentication. The `build` job grants `id-token: write`; other jobs retain
  their existing permissions. OIDC requires no copied Dependabot secret or
  Codecov token-policy change. Dependabot cannot read ordinary Actions secrets, and manually rerunning
  its workflow retains the original restricted context.
- Public fork PRs retain Codecov's tokenless upload path. They receive no
  repository secret; the Codecov action identifies the fork and prefixes its
  upload branch. Keep `pull_request` as the trigger when testing contributor code.

For an Action upgrade, verify the actual upload as well as the CI result:

1. Record the PR head SHA and open its PyTest `build` job for the full-integration
   Python version. For Dependabot, verify that the workflow's original actor is
   `dependabot[bot]`; a maintainer's own PR does not exercise that context.
2. Expand **Upload coverage to Codecov**. Check that it finds `coverage.xml`,
   uploads for the recorded PR head SHA, and finishes without authentication,
   queueing or upload errors. A message that a report was queued alone does not
   prove that processing succeeded.
3. Open the matching commit at
   `https://app.codecov.io/gh/opthub-org/pytorch-bsf/commit/<head-sha>` and confirm
   that its processed report and upload correspond to that run. Check the
   applicable `codecov/patch` status too. A change outside measured source can
   have no patch status; still require the processed commit report.
4. Require CI and the Codex review for that same current head, plus zero
   unresolved review threads, before merging. After this authentication change
   reaches `master`, rebase the original Dependabot Action-upgrade PR to test
   both the new Action and the real Dependabot context.

If OIDC fails, check the job's `id-token: write` permission and Codecov's response
before rerunning. If using token authentication for Dependabot instead, a
repository administrator must copy the repository's Codecov upload token to
**Settings → Secrets and variables → Dependabot** as `CODECOV_TOKEN` and disable
OIDC for that path; the OIDC option takes precedence over the token input. Never
post token values in logs, issues or PRs. See the
[Codecov Action authentication options](https://github.com/codecov/codecov-action#using-oidc),
[Codecov token policy](https://docs.codecov.com/docs/codecov-tokens), and
[GitHub's Dependabot workflow restrictions](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/dependabot-on-actions).

---

## Commit Rules

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. `release-please` uses this format to automate CHANGELOG generation and version management.

### Format

```
<type>(<scope>): <subject>
```

- `<scope>` is optional.
- Write `<subject>` in English using the imperative present tense (e.g., "add feature" ✓, "added feature" ✗).
- Keep the first line under 72 characters.

### Commit types

| Type | Description | Version impact |
|------|-------------|---------------|
| `feat` | Add a new feature | Minor version bump |
| `fix` | Fix a bug | Patch version bump |
| `docs` | Documentation changes only | None |
| `style` | Changes that do not affect behavior (whitespace, formatting) | None |
| `refactor` | Code change that is neither a fix nor a feature | None |
| `test` | Add or update tests | None |
| `chore` | Changes to the build process or auxiliary tools | None |
| `ci` | Changes to CI configuration | None |
| `deps` | Add or update dependencies | None |
| `perf` | Performance improvement | None |

### Breaking changes

For changes that break backward compatibility, add `BREAKING CHANGE:` to the commit body or footer. Before 1.0 this triggers a minor version bump; after 1.0 it triggers a major version bump.

```
feat!: remove deprecated normalize parameter

BREAKING CHANGE: The `normalize` parameter has been removed.
Use `preprocessing` instead.
```

### Examples

```
feat(model_selection): add elastic net grid search
fix: resolve UnpicklingError in PyTorch 2.6+
docs: add CONTRIBUTING.md
deps: update starlette constraint for mlflow compatibility
test: add parametrized tests for BezierSimploid
ci: add Python 3.14 to test matrix
```

---

## Coding Conventions

### Linter and formatter

Conventions that a tool can check are enforced by the pre-commit hooks (see [Install the git hook](#install-the-git-hook)) rather than only written here; a violation is reported with the reason and the fix.

- [Ruff](https://docs.astral.sh/ruff/) lints, sorts imports and formats code (`ruff check --fix` and `ruff format`).
- [numpydoc](https://numpydoc.readthedocs.io/en/latest/validation.html) checks that public objects have docstrings, their parameters are documented, return values have descriptions, summaries use the infinitive, and NumPy-style sections and type entries are well formed. The checks are configured in `[tool.numpydoc_validation]` of `pyproject.toml`; private modules (a path component starting with `_`, except `__init__.py`), private objects and their descendants are skipped. Document the fields of a dataclass under `Attributes`, as numpydoc cannot see the generated `__init__`. Parameter type entries and runnable examples remain manual review items; extended summaries and See Also sections are optional when they add no useful context. To skip one check for one object, put `# numpydoc ignore=<CODE>` on its `def` or `class` line, with a comment explaining why.
- `tools/lint_conventions.py` checks the conventions of this file that Ruff cannot check (BSF001, BSF002 below).
- [typos](https://github.com/crate-ci/typos) finds misspellings; false positives are suppressed narrowly in `_typos.toml`. It runs only in the local git hook for now, and CI skips it, while the rate of false positives is observed. Add a false positive to `_typos.toml` (or report it) rather than rewording correct text.

CI runs the same hooks (except typos) on all files and fails if they report a problem or change a file. To run them manually:

```bash
# Run the hooks on all files, as CI does
SKIP=typos pre-commit run --all-files
# typos: local only while its false positives are observed
pre-commit run typos --all-files

# Or run Ruff directly
ruff check .           # lint and sort imports (add --fix to apply automatic fixes)
ruff format --check .  # check formatting (drop --check to rewrite files)
```

The Ruff version is pinned by `rev` in `.pre-commit-config.yaml`, which is the single source of truth: the git hook and CI both use it, and Dependabot opens a weekly PR to bump it. Run `pre-commit autoupdate` only if you need a newer Ruff before Dependabot does. The `ruff` in the `[develop]` extras is unpinned (for editors and direct use); if its results differ from the hook's, trust `pre-commit run --all-files`.

Configuration in `pyproject.toml` (`[tool.ruff]` section):

- `line-length = 127`
- `target-version = "py311"`
- `select = ["E", "F", "W", "I", "UP", "B", "SIM", "C90", "D", "PGH003", "PLC2401", "PLC2403", "TID251"]` (pycodestyle errors and warnings, Pyflakes, isort, pyupgrade, flake8-bugbear, flake8-simplify, McCabe complexity, pydocstyle, codes on `# type: ignore`, ASCII identifiers and imports, banned APIs)
- `max-complexity = 10` (`[tool.ruff.lint.mccabe]` section): split a function whose cyclomatic complexity exceeds 10 into smaller helpers
- pydocstyle (`D`) uses the `numpy` convention, applies to `torch_bsf/` only, and ignores `D105` (magic methods) and `D401` (imperative summaries)
- `TID251` bans `from __future__ import annotations` (`[tool.ruff.lint.flake8-tidy-imports.banned-api]`)

`B905` requires an explicit `strict=` on every `zip()`: pass `strict=True` when the iterables must have the same length, and `strict=False` only when truncation is intended.

### Ignoring bulk formatting commits in git blame

Commits that only reformat code (for example, the switch to Ruff in #364) are listed in `.git-blame-ignore-revs` at the repository root, so that `git blame` shows the commit that actually wrote each line. GitHub's blame view reads this file automatically. To use it locally, run this once in your clone:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

After a PR whose main purpose is a repository-wide reformat (for example, a formatter upgrade or a new rule that rewrites many files) is merged, open a follow-up PR that appends its commit to `.git-blame-ignore-revs`: a comment line with the commit subject, then the full 40-character SHA. Use the squash-merged SHA on `master`, not a SHA from the PR branch. Keep reformatting PRs free of unrelated changes, because every line they touch is hidden from blame.

### Type checking

Add type hints to all functions in `torch_bsf`, public or private. Use mypy for static analysis of both the package and the tests (as CI does):

```bash
mypy torch_bsf tests
```

Configuration in `pyproject.toml` (`[tool.mypy]` section):

- `python_version = 3.11` (type checking targets the minimum supported version)
- `warn_return_any`, `warn_unused_configs`, `warn_unused_ignores`, `warn_redundant_casts`
- `disallow_untyped_defs`, `disallow_incomplete_defs`: every function in `torch_bsf` must be fully annotated
- `check_untyped_defs`: the bodies of unannotated functions (e.g., tests) are type-checked too
- `disallow_any_generics`: write `dict[str, Any]`, not a bare `dict`, and `npt.NDArray[Any]` (`import numpy.typing as npt`), not a bare `np.ndarray`, which older NumPy versions reject
- `[[tool.mypy.overrides]]` for `tests.*` turns off `disallow_untyped_defs` and `disallow_incomplete_defs`, so test functions and fixtures need no annotations
- Optional dependencies without type information (`mlflow`, `scipy`, `sklearn`, `matplotlib`, `mpl_toolkits`) use `ignore_missing_imports`

Every `# type: ignore` must name its error code, e.g. `# type: ignore[arg-type]` for a test that passes an invalid value on purpose (Ruff `PGH003`).

### Style guidelines

- **Indentation**: 4 spaces
- **Maximum line length**: 127 characters (`line-length` in `[tool.ruff]`)
- **Strings**: Double quotes (`ruff format` default)
- **Docstrings**: NumPy style (Ruff `D` and numpydoc check the layout)
- **Type hints**: Python 3.11 built-in syntax without `from __future__ import annotations` (Ruff `TID251`); quote a forward reference, e.g. `-> "BezierSpace"` inside the class `BezierSpace`

### "Bezier" vs "Bézier" spelling

Use the spelling consistently depending on the context:

| Spelling | Where to use |
|----------|-------------|
| `Bezier` (no accent) | Python identifiers (`BezierSimploid`, `fit_simploidal_set`, `plot_bezier_simplex`, …) and any string that may appear in a console (CLI `description`/`help` strings, `print()` output, log/warning/error messages) |
| `Bézier` (with accent) | Everything else: docstrings, RST/Markdown documentation, code comments, `pyproject.toml` metadata, example script module docstrings |

The rationale for keeping `Bezier` in console-facing text is that this library is used on HPC clusters where the locale may be `LANG=C`, causing non-ASCII characters to be displayed incorrectly or to raise encoding errors.

Rule BSF001 of `tools/lint_conventions.py`, run by the pre-commit hook, enforces ASCII in console-facing strings (its docstring lists the calls it checks; a deliberate exception takes `# lint-conventions: ignore[BSF001]` and a reason).

---

## Writing Tests

Tests live in the `tests/` directory.

### Running tests

```bash
# Run all tests
pytest tests/

# Run a specific file
pytest tests/simploidal.py

# Run with coverage reports (coverage.xml for diff-cover, htmlcov/ for a browser)
pytest tests/ --cov=torch_bsf --cov-report=xml --cov-report=html

# What the `build` job of .github/workflows/python-package.yml runs
pytest --verbose --durations=0 --junit-xml=pytest.xml --cov=torch_bsf --cov-report=term-missing --cov-report=xml
```

`addopts` in `[tool.pytest.ini_options]` of `pyproject.toml` holds only `--assert=plain` and
`--doctest-modules`, so a plain `pytest` run writes no report files (`coverage.xml`, `htmlcov/`,
`pytest.xml`); pass the options above when you need them.

The tests run in random order with `pytest-randomly`, in CI as well as locally: it is in the
`develop` extra, and every CI job that runs the test suite installs it. A test that depends on the
side effects of another test therefore fails in both places. pytest prints the seed in its header
(`Using --randomly-seed=<seed>`); rerun with `pytest --randomly-seed=<seed>` to repeat
an order, or with `-p no:randomly` to run the files in their written order.

### Naming conventions

- Test file names mirror the module under test (e.g., `torch_bsf/splitting.py` → `tests/splitting.py`).
- Tests for development scripts live in `tests/tools/` and mirror the script name
  (e.g., `tools/update_python_support.py` → `tests/tools/update_python_support.py`).
  They run with the normal pytest suite. The standard-library-only updater also
  runs through `unittest` in the lint and dependency lifecycle jobs, without
  numerical dependencies.
- Test function names start with `test_` or `_test_`.

### Writing parametrized tests

Use `@pytest.mark.parametrize` to cover multiple input combinations:

```python
import pytest
import torch_bsf as tbsf


@pytest.mark.parametrize(
    "dim, n_values, degree",
    [(dim, n_values, degree) for dim in range(1, 4) for n_values in range(1, 3) for degree in range(3)],
)
def test_zeros(dim: int, n_values: int, degree: int) -> None:
    bs = tbsf.BezierSimploid.zeros((dim,), degree, n_values)
    assert bs.n_params == dim + 1
    assert bs.n_values == n_values
    assert bs.degree == (degree,)
```

### Property-based tests of numerical code

Hand-picked examples rarely include the inputs on which numerical code breaks. Code that computes with floating-point numbers (kernels, geometry, splitting, interpolation, …) must also be tested with [Hypothesis](https://hypothesis.readthedocs.io/) properties: invariants that hold for every valid input, checked on generated ones.

- **What to state**: invariances (translation and scaling of coordinates, scaling of coefficients or data), identities (a map and its inverse, a child model and its parent), bounds (range preservation, partition of unity, nonnegativity), continuity across shared faces, and that invalid input (NaN, infinity, empty, wrong type or shape) raises the documented exception.
- **Which inputs**: use the shared strategies of `tests/strategies.py`, which generate extreme values on purpose: translations up to `1e12`, scalings from `1e-6` to `1e6`, powers of two up to `2**±1000` (near the float64 limits), split ratios within `1e-12` of 0 and 1, and points on vertices and faces or with coordinates down to the smallest normal float64. Add a strategy there rather than in a test file, so that later properties can reuse it.
- **Tolerances**: never use the default tolerance of `torch.testing.assert_close` or an arbitrary `1e-6`. Derive the bound from the unit roundoff `U = 2**-53` (the number of roundings, the condition number of the problem, the magnitude of the data), and write the derivation next to the test, as in `_split_tolerance` of `tests/splitting.py`. An exact operation (a translation or a power-of-two scaling whose results are representable) should give an exactly equal result. Rule BSF002 of `tools/lint_conventions.py`, run by the pre-commit hook, requires every `assert_close` in `tests/` to pass both `rtol` and `atol` (`rtol=0, atol=0` for an exact comparison).
- **Counterexamples**: when a property fails, keep the shrunk input as a regression test (`@example` or a plain test) next to the property. Fix small bugs in the same PR. For a larger one, open an issue and mark the property `@pytest.mark.xfail(strict=True, reason="#<issue>")`.

The Hypothesis profile is chosen with the `HYPOTHESIS_PROFILE` environment variable (see `tests/conftest.py`):

| Profile | Use |
|---------|-----|
| `ci` (default) | Few examples, derandomized: every run tests the same inputs, so CI is reproducible and fast |
| `thorough` | 2000 random examples per property, for hunting bugs locally; failures are saved in `.hypothesis/` and replayed first |

```bash
HYPOTHESIS_PROFILE=thorough pytest tests/splitting.py tests/interpolation.py -p no:randomly
```

Run the `thorough` profile on the properties you add or whose code you change before opening a PR.

When revisiting older tests, replace repeated fixed inputs with a generated property when the assertion
expresses a general contract. Keep the old boundary cases as `@example` inputs when they need guaranteed
coverage. Keep explicit API examples, format/CLI expectations, dependency failures, known numerical
regressions and optimizer convergence benchmarks: their particular inputs or execution conditions matter.
Do not replace these with assertions that merely check output shape or finiteness.

Examples of migrated contracts:

| Tests | Generated contracts |
|-------|---------------------|
| `tests/sampling.py` | Grid cardinality and lattice coordinates, simplex bounds, seeded reproducibility, Sobol warnings, skeleton risk coefficients against exact rationals, the paper tables and the closed forms of the theory chapter, and skeleton sample allocation |
| `tests/preprocessing.py` | Fitted affine maps, inverse roundtrips, constant features and normalization statistics |
| `tests/simploidal.py` | Mixed-degree index counts, nonnegative partition of unity, face restrictions and degeneracies |
| `tests/io.py` | Exact coefficient, dtype, topology and prediction roundtrips |
| `tests/model_selection/kfold.py` | Disjoint and balanced folds, complete coverage, reproducibility and local RNG state |

An integration example can stay fixed when its mathematical identity is already covered by a property
at a lower level (for example, subdivision in `tests/unified_utilities.py` and `tests/splitting.py`).
For filesystem properties, create a temporary directory inside the property body: pytest's `tmp_path`
fixture is shared by all Hypothesis examples of a test invocation.

### Docstring tests

Write runnable examples in docstrings for all public APIs. They are verified automatically in CI:

```python
def fit(params, values, degree):
    """Fit a Bézier simplex to the given data.

    Parameters
    ----------
    params : torch.Tensor
        The parameter data on the simplex.
    values : torch.Tensor
        The label data.
    degree : int
        The degree of the Bézier simplex.

    Returns
    -------
    BezierSimploid
        A fitted Bézier simplex.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf
    >>> params = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> values = torch.tensor([[0.0], [0.25], [1.0]])
    >>> bs = torch_bsf.fit(params=params, values=values, degree=2)
    """
```

### Sphinx HTML and doctest

The `docs` job in the PyTest workflow builds HTML and executes Sphinx doctests
once on Python 3.14. Both commands treat warnings as errors and continue to
report other problems, so broken references, missing images and malformed
headings fail the PR before publication. Use the same interpreter for installation
and both builds. From the repository root, install Sphinx extensions and every
optional integration used by the examples, then restore the editable install:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e '.[all]'
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m sphinx -b doctest -W --keep-going docs docs/_build/doctest
```

In a fresh pip environment, the CI job first installs CPU PyTorch with
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
An existing environment can retain its suitable PyTorch installation.
The requirements file includes the local package (`.`); installing editable
afterward ensures that later edits are used. Run these commands through the
`pytorch-bsf` environment's Python when using Conda.

Only executable doctest and `testcode` blocks are tested; a plain `code-block`
is displayed without execution. Add assertions or expected output for examples
whose behavior should be checked.

The separate GitHub Pages workflow builds multiple tags and branches with
`sphinx-multiversion`. Historical tags keep their documentation sources and
release metadata, using the current build configuration. Their warnings do
not replace the strict checks of the current PR.
External link checking is not part of this PR gate.

---

## Repository Structure

```
pytorch-bsf/
├── .claude/                          # Claude Code harness (worktrees/ is local, not tracked)
│   ├── agents/                       # Reviewer agents (review-<aspect>.md) run by review-local
│   └── skills/                       # Project skills, invoked as slash commands
│       ├── create-issue/SKILL.md     # File a GitHub issue
│       ├── review-local/             # Local review before opening a PR
│       │   ├── SKILL.md              # Spawns the reviewer agents in parallel and verifies their findings
│       │   └── protocol.md           # How every reviewer agent reviews and reports
│       ├── setup-env/SKILL.md        # Set up or repair the development environment
│       └── work-on-issue/SKILL.md    # Implement an issue end to end, up to an open PR
├── .github/
│   ├── workflows/                    # GitHub Actions workflows
│   │   ├── python-package.yml        # pre-commit (Ruff, numpydoc-validation, lint-conventions), pytest, mypy, Sphinx; lowest and optional deps
│   │   ├── release-please-action.yml # Automated releases (CHANGELOG + PyPI publish)
│   │   ├── docker.yml                # Docker image published to ghcr.io on release
│   │   ├── sphinx-pages.yml          # Build and publish Sphinx docs to GitHub Pages
│   │   ├── codeql-analysis.yml       # Security scanning
│   │   └── python-package-conda.yml  # End-to-end tests in a Conda environment
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml            # Bug report form
│   │   ├── feature_request.yml       # Feature / improvement form
│   │   ├── documentation.yml         # Documentation form
│   │   └── config.yml                # Disables blank issues
│   ├── CODEOWNERS                    # Code ownership settings
│   └── dependabot.yml                # Automated dependency updates (pip and pre-commit hooks)
├── torch_bsf/                        # Main package
│   ├── __init__.py                   # Public API exports (BezierSpace, BezierModel, fit, minimize, …)
│   ├── __main__.py                   # CLI entry point (python -m torch_bsf)
│   ├── _cli_args.py                  # Shared CLI arguments and data loading
│   ├── _numeric.py                   # Numerical reductions shared by prediction and initialization
│   ├── _problem.py                   # Weighted, constrained least-squares problem shared by the solvers
│   ├── _skeleton.py                  # Faces and controls shared by skeleton sampling and fitting
│   ├── active_learning.py            # Acquisition functions (query_by_committee, farthest_points)
│   ├── core.py                       # Spaces, observations and models (BezierSpace, Points, BezierModel, …)
│   ├── ensemble.py                   # Ensemble of fitted models
│   ├── fitting.py                    # fit for every space and solver; fit_skeleton_regression
│   ├── geometry.py                   # Public topology builders (Simploid, SimploidalSet from _kernels/sets.py)
│   ├── interpolation.py              # Shape-preserving cubic interpolation on simplicial complexes
│   ├── io.py                         # Versioned model persistence (save_model, load_model, …)
│   ├── optimization.py               # Objective minimization (minimize, PreconditionedSGD, …)
│   ├── plotting.py                   # Visualization (plot_model)
│   ├── point_cloud.py                # Value-only fitting (fit_point_cloud, fit_skeleton)
│   ├── preprocessing.py              # Data scaling (MinMax, Std, Quantile, None)
│   ├── results.py                    # Fit results, diagnostics, ConvergenceError and ConvergenceWarning
│   ├── sampling.py                   # sample, grid; optimal skeleton sampling and its risk
│   ├── sklearn.py                    # scikit-learn regressor (BezierRegressor)
│   ├── solvers.py                    # Least-squares solver configurations (DenseLeastSquares, …)
│   ├── splitting.py                  # Exact local subdivision (subdivide, subdivide_space)
│   ├── validator.py                  # Parsers of CLI arguments
│   ├── py.typed                      # PEP 561 type hint marker
│   ├── _kernels/                     # Private Bézier simploid kernels behind the public API
│   │   ├── __init__.py               # Kernel exports (BezierSimploid, SimploidalSet, …)
│   │   ├── api.py                    # Kernel-level fit, fit_kfold, fit_sgd and splitting
│   │   ├── bezier.py                 # BezierSimploid model
│   │   ├── combinatorics.py          # Control point indices and Bernstein basis
│   │   ├── fitting.py                # Fitting Bézier simploids and simploidal Bézier spaces
│   │   ├── io.py                     # Saving and loading Bézier simploids
│   │   ├── sets.py                   # Simploid and simploidal set implementation
│   │   ├── sgd.py                    # Fitting to Pareto sets by stochastic gradient descent
│   │   ├── space.py                  # Simploidal Bézier spaces
│   │   ├── _active_learning.py       # Next-point suggestion
│   │   ├── _conforming.py            # Conforming subdivision of a simploidal Bézier space
│   │   ├── _lightning.py             # Lightning module of the "trainer" solver
│   │   ├── _plotting.py              # Plots of Bézier simplices, simploids and spaces
│   │   ├── _sampling.py              # Grid, random and Sobol sampling of simplices
│   │   └── _splitting.py             # De Casteljau splitting of Bézier simplices and simploids
│   ├── integrations/                 # Optional adapters; importing them loads no framework
│   │   ├── __init__.py               # Package marker; import each adapter module explicitly
│   │   ├── lightning.py              # Opt-in Lightning execution (regression_module)
│   │   └── mlflow.py                 # MLflow recording (log_result)
│   └── model_selection/              # Model selection sub-package
│       ├── __init__.py               # Public exports (KFold, cross_validate, select_degree, …)
│       ├── _validation.py            # Cross-validation and recorded degree search
│       └── elastic_net_grid.py       # Elastic net grid search CLI
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures and the Hypothesis profiles
│   ├── strategies.py                 # Hypothesis strategies shared by the property-based tests
│   ├── simploidal.py                 # Tests for the Bézier simploids of torch_bsf._kernels
│   ├── test_contributor_examples.py  # Runs the CONTRIBUTING.md CLI example and issue form examples
│   ├── …                             # One file per module (see Naming conventions), plus contract tests
│   ├── model_selection/              # Tests for torch_bsf.model_selection
│   ├── tools/                        # All tests for the scripts in tools/
│   │   ├── __init__.py
│   │   ├── lint_conventions.py        # AST convention checker regressions
│   │   └── update_python_support.py   # Lifecycle tests (pytest or standalone unittest)
│   └── data/                         # Test data files
├── docs/                             # Sphinx documentation
│   ├── conf.py                       # Sphinx configuration
│   ├── index.rst                     # Documentation home page
│   ├── requirements.txt              # Documentation build dependencies
│   ├── advanced/                     # Advanced topics (RST)
│   ├── applications/                 # Application examples (RST)
│   └── …                             # Other pages, images (_static/), benchmark results, templates, branding
├── examples/                         # Example scripts
│   ├── quickstart/
│   │   └── run.sh                    # Quickstart shell script
│   └── …                             # Benchmarks (*benchmark*.py), doc figures (generate_*.py), other scripts
├── tools/                            # Development scripts
│   ├── check_distribution.py         # Builds and checks the distributions (the distribution CI job)
│   ├── lint_conventions.py           # Checks the conventions that Ruff cannot check (a pre-commit hook)
│   └── update_python_support.py      # Synchronizes supported Python versions with the release lifecycle
├── .git-blame-ignore-revs            # Bulk formatting commits skipped by git blame
├── .gitattributes                    # Git LFS settings for binary files
├── .gitignore                        # Files ignored by Git
├── .pre-commit-config.yaml           # Git hooks (Ruff, numpydoc, tools/lint_conventions.py, typos) run on commit; CI runs all but typos
├── CHANGELOG.md                      # Version history (auto-generated by release-please)
├── CONTRIBUTING.md                   # This file
├── Dockerfile                        # Docker image definition
├── LICENSE                           # MIT License
├── MANIFEST.in                       # Extra files in the source distribution
├── MLproject                         # MLflow project definition
├── README.md                         # Project overview and usage
├── _typos.toml                       # Exceptions for the typos spell checker
├── environment.yml                   # Conda environment definition
├── params.csv                        # Sample parameters for the CLI
├── pyproject.toml                    # Package configuration, dependencies, tool settings
└── values.csv                        # Sample values for the CLI
```
