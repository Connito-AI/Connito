---
description: Generate or update tests for functions changed since last commit, then run them
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git ls-files:*), Bash(git rev-parse:*), Bash(pytest:*), Bash(/home/isabella/crucible/ConnitoAI/.main-venv/bin/pytest:*), Bash(touch:*), Bash(sha256sum), Read, Write, Edit, Glob, Grep
---

## Context

- Working-tree status: !`git status --short`
- Diff vs last commit (tracked source files): !`git diff HEAD -- 'connito/*.py' ':!connito/test/*'`
- New (untracked) Python files under connito/: !`git ls-files --others --exclude-standard -- 'connito/*.py' | grep -v '^connito/test/' || echo '(none)'`

## Task

Generate test coverage for code changed since the last commit in the ConnitoAI repo.

For each function ADDED or MODIFIED in the diff above (and any function defined in the new untracked files):

1. Locate the right test file under `connito/test/`. Names don't strictly mirror source files (e.g. `test_checkpoint_retention.py` covers `connito/shared/checkpoints.py`) — read sibling tests in `connito/test/` to find the appropriate file or create a new one. Match the style of the existing tests.
2. Write or update a focused unit test for that function. Skip pure docstring / import / whitespace changes.
3. After all tests are written, run only the affected test files using the project venv:

   ```
   /home/isabella/crucible/ConnitoAI/.main-venv/bin/pytest <test_file_1> <test_file_2> ... -v -m "not integration"
   ```

   (CLAUDE.md notes integration tests need a GPU — skip them with `-m "not integration"`.)

4. Report:
   - Which functions you covered
   - Which test files you created or edited
   - The pytest summary (pass/fail counts; full failure tracebacks if any failed)

5. **If pytest passed**, mark the testing-gate so the `/commit` hook knows tests are up to date for the current working-tree state:

   ```
   touch "/tmp/connito-tests-$(git rev-parse HEAD)-$(git diff HEAD | sha256sum | cut -d' ' -f1)"
   ```

   Skip this step if pytest failed — the gate should keep prompting until tests pass.

## Constraints

- Do NOT stage or commit the test files — leave them in the working tree for the user to review.
- If there are no relevant function-level changes (e.g. only constants, type aliases, or imports moved), say so and stop.
- Do NOT run the full suite — only the affected test files (keeps latency bounded).
