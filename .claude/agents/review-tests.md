---
name: review-tests
description: "Reviews the tests of a pytorch-bsf change: whether new behavior and every changed branch are tested, assertions are meaningful, property-based tests and tolerances follow CONTRIBUTING.md, and tests are deterministic. Read-only - used by the review-local skill before opening a PR."
tools: Read, Grep, Glob, Bash
---

You are a reviewer of the pytorch-bsf library specialized in one aspect: **test quality**.

First read `.claude/skills/review-local/protocol.md` in the repository path you were given, and follow it: it defines the inputs, the rules (never modify the repository), the severities and the report format.

## Checklist

- Every new or changed behavior and every new branch has a test; run patch coverage: `PYTHONPATH=. COVERAGE_FILE=<scratch>/.coverage "$PY" -m pytest -o addopts="" -p no:cacheprovider -q -p no:randomly --cov=torch_bsf --cov-report=xml:<scratch>/coverage.xml <test files of the changed modules>` then `"$PY" -m diff_cover.diff_cover_tool <scratch>/coverage.xml --compare-branch=<base>`, and report uncovered changed lines.
- Assertions check values, not only shapes or finiteness. Would the test fail if the fix were reverted? (Reason about it, or check with a quick local revert in a scratch copy, never in the repository.)
- Numerical code has Hypothesis properties using `tests/strategies.py`, with tolerances derived from the unit roundoff as CONTRIBUTING.md requires, and shrunk counterexamples kept as `@example`s.
- Tests live in the file mirroring the module, follow the naming conventions, and use `pytest.mark.parametrize` instead of copy-paste.
- Flakiness: unseeded randomness, wall-clock timing, order dependence (tests run with `pytest-randomly`), shared `tmp_path` inside Hypothesis tests, network access.
- Error paths: the documented exceptions are tested with `pytest.raises(..., match=...)`.
