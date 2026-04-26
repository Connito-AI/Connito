#!/usr/bin/env bash
# UserPromptSubmit hook: when the user runs /commit, check whether they've already
# run /generate-tests for the current working-tree state. If not, inject context
# telling Claude to confirm with the user before proceeding.

set -uo pipefail

REPO=/home/isabella/crucible/ConnitoAI

input=$(cat)
prompt=$(printf '%s' "$input" | jq -r '.prompt // empty')

# Match raw `/commit` invocation only (not "commit my code" or other text)
case "$prompt" in
  /commit|/commit\ *|$'/commit\n'*) ;;
  *) exit 0 ;;
esac

cd "$REPO" 2>/dev/null || exit 0
head_sha=$(git rev-parse HEAD 2>/dev/null) || exit 0
diff_hash=$(git diff HEAD 2>/dev/null | sha256sum | cut -d' ' -f1)
sentinel="/tmp/connito-tests-${head_sha}-${diff_hash}"

if [ -f "$sentinel" ]; then
  exit 0
fi

jq -n --arg msg "TESTING-GATE: The user invoked /commit but has not run /generate-tests since their last code change. BEFORE doing any commit work, ask the user verbatim: \"You haven't run /generate-tests for the current changes. Run it first to generate or update tests for the changed code? (y/N)\" — then wait for their answer. If they say yes, run /generate-tests first and then continue with /commit. If they say no, proceed with /commit normally." '{
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: $msg
  }
}'
