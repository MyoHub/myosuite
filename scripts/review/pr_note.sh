#!/usr/bin/env bash
# Writes the "Local checks" block of a pull request description from the records that scripts/review/local_review.sh keeps
# per commit (in .git/myo-review/). The records are self-reported by the author's machine; they are not proof.
#
#   scripts/review/pr_note.sh print [sha]    print the block (default: HEAD)
#   scripts/review/pr_note.sh update [sha] [branch]   replace the block in the open PR of <branch> (default: the
#                                            current branch; needs `gh`)
#   gh pr create --body "$(scripts/review/pr_note.sh print)"   when opening a PR by hand
set -euo pipefail
cmd="${1:-print}"; sha="$(git rev-parse "${2:-HEAD}")"
rec="$(git rev-parse --git-common-dir)/myo-review"
start='<!-- myo-review:start -->'; end='<!-- myo-review:end -->'

field() { [[ -f "$1" ]] && sed -n "s/^$2: //p" "$1" | head -1; }
line() {  # line <label> <check>
  local f="$rec/$sha.$2"
  if [[ -f "$f" ]]; then
    echo "| $1 | $(field "$f" status) | $(field "$f" by) | $(field "$f" at) |"
  else
    echo "| $1 | **not run** on this commit | | |"
  fi
}
block() {
  echo "$start"
  echo "**Local checks** for \`${sha:0:10}\` (self-reported by \`scripts/review/local_review.sh\`; see REVIEW.md)"
  echo
  echo "| Check | Result | By | When (UTC) |"
  echo "|---|---|---|---|"
  line "pre-commit hooks (branch changes)" precommit
  line "Agent review (REVIEW.md, ownership)" review
  local f="$rec/$sha.review"
  if [[ -f "$f" ]] && grep -q '^---$' "$f"; then
    echo; echo "<details><summary>Review output</summary>"; echo
    n="$(sed '1,/^---$/d' "$f" | wc -l | tr -d ' ')"
    sed '1,/^---$/d' "$f" | head -n 300
    [[ "$n" -gt 300 ]] && printf '\n(truncated: %s more lines; run scripts/review/pr_note.sh print locally)\n' "$((n-300))"
    echo; echo "</details>"
  fi
  echo "$end"
}

case "$cmd" in
  print) block ;;
  update)
    command -v gh >/dev/null 2>&1 || exit 0
    branch="${3:-$(git rev-parse --abbrev-ref HEAD)}"; branch="${branch#refs/heads/}"
    num="$(gh pr view "$branch" --json number,state -q 'select(.state=="OPEN") | .number' 2>/dev/null || true)"
    if [[ -z "$num" ]]; then
      echo "pr_note: no open PR for $branch yet; open it with: gh pr create --body \"\$(scripts/review/pr_note.sh print)\"" >&2
      exit 0
    fi
    body="$(gh pr view "$num" --json body -q .body)"
    new="$(block)"
    if grep -qF "$start" <<<"$body"; then
      body="$(S="$start" E="$end" N="$new" awk '
        $0==ENVIRON["S"] {print ENVIRON["N"]; skip=1; next} $0==ENVIRON["E"] {skip=0; next} !skip {print}' <<<"$body")"
    else
      body="$body"$'\n\n'"$new"
    fi
    # REST, not `gh pr edit`: the latter fails on repos with classic Projects (GraphQL projectCards).
    gh api -X PATCH "repos/{owner}/{repo}/pulls/$num" -f body="$body" >/dev/null && echo "pr_note: updated the Local checks block of PR #$num" >&2 ;;
  *) echo "usage: pr_note.sh print [sha] | update [sha] [branch]" >&2; exit 2 ;;
esac
