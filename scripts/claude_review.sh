#!/usr/bin/env bash
# Local Claude review of the current branch against REVIEW.md and the ownership check
# (.claude/classifiers/ownership/prompt.md), before a pull request is opened.
#
#   scripts/claude_review.sh            review HEAD against the merge base with $MYO_REVIEW_BASE
#   scripts/claude_review.sh --hook     same, as the pre-push hook: skips when `claude` is not installed
#
# Env: MYO_REVIEW_BASE (default: myohub/dev, else origin/dev), MYO_SIM_ROOT (sibling myo_sim checkout for
# cross-repo reuse search; default ../myo_sim), MYO_CLAUDE_REVIEW=0 to skip, =strict to block the push on a fail.
# Read-only: Claude may read files and run git diff/log/show; it edits nothing and pushes nothing.
set -euo pipefail

hook=0; [[ "${1:-}" == "--hook" ]] && hook=1
mode="${MYO_CLAUDE_REVIEW:-advisory}"
[[ "$mode" == "0" ]] && exit 0
if ! command -v claude >/dev/null 2>&1; then
  ((hook)) && exit 0
  echo "claude CLI not found (https://code.claude.com)" >&2; exit 1
fi

root="$(git rev-parse --show-toplevel)"; cd "$root"
base="${MYO_REVIEW_BASE:-}"
if [[ -z "$base" ]]; then
  for b in myohub/dev origin/dev; do git rev-parse -q --verify "$b" >/dev/null && { base=$b; break; }; done
fi
[[ -z "$base" ]] && { echo "set MYO_REVIEW_BASE" >&2; exit 1; }
mb="$(git merge-base "$base" HEAD)"
if git diff --quiet "$mb" HEAD; then exit 0; fi

sim="${MYO_SIM_ROOT:-$root/../myo_sim}"
extra=(); [[ -d "$sim" ]] && extra=(--add-dir "$sim")

prompt="Review the change \`git diff $mb HEAD\` in this MyoSuite repository (commits: \`git log $mb..HEAD\`).
Apply REVIEW.md, and answer the question in .claude/classifiers/ownership/prompt.md, using the default branch ($base)
and $( [[ -d "$sim" ]] && echo "the myo_sim checkout at $sim" || echo "no myo_sim checkout (say that myo_sim reuse is unverified)" )
to look for existing primitives. Treat the latest commit message body as the pull request description.
Report findings ranked by severity, each with file:line, the existing symbol or myo_sim API to use, and the fix.
End with exactly one line: 'VERDICT: pass' or 'VERDICT: fail'. Do not edit files."

out="$(claude -p "$prompt" ${extra[@]+"${extra[@]}"} \
  --allowedTools "Read" "Grep" "Glob" "Bash(git diff:*)" "Bash(git log:*)" "Bash(git show:*)" "Bash(git merge-base:*)")"
printf '%s\n' "$out"
if [[ "$mode" == "strict" ]] && grep -q '^VERDICT: fail' <<<"$out"; then
  echo "claude_review: ownership/review check failed (MYO_CLAUDE_REVIEW=strict)" >&2; exit 1
fi
exit 0
