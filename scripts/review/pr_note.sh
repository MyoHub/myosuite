#!/usr/bin/env bash
# Writes the "Local checks" block of a pull request description from the records that scripts/review/local_review.sh keeps
# per commit (in .git/myo-review/). The records are self-reported by the author's machine; they are not proof.
#
#   scripts/review/pr_note.sh print [sha]             print the block (default: HEAD)
#   scripts/review/pr_note.sh body [branch]           print the open PR description (without the Local checks block)
#   scripts/review/pr_note.sh update [sha] [branch]   replace the block in the open PR of <branch> (default: the
#                                                     current branch; needs `gh`)
#   gh pr create --body "$(scripts/review/pr_note.sh print)"   when opening a PR by hand
set -euo pipefail
cmd="${1:-print}"
rec="$(git rev-parse --git-common-dir)/myo-review"
start='<!-- myo-review:start -->'; end='<!-- myo-review:end -->'

slug_from_remote_url() {  # slug_from_remote_url <url> -> owner/repo
  local url="$1"
  url="${url%.git}"
  if [[ "$url" =~ github\.com[:/]([^/]+/[^/]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

urlencode_param() {  # urlencode_param <raw>
  local raw="$1" i ch out="" hex
  local LC_ALL=C
  for ((i = 0; i < ${#raw}; i++)); do
    ch="${raw:i:1}"
    case "$ch" in
      [a-zA-Z0-9._~-]) out+="$ch" ;;
      *) printf -v hex '%%%02X' "'$ch"; out+="$hex" ;;
    esac
  done
  printf '%s' "$out"
}

resolve_open_pr() {  # resolve_open_pr <branch> -> prints "<base_slug> <pr_number>" on match
  command -v gh >/dev/null 2>&1 || return 1
  local br="$1" push_remote head_slug="" base_slug="" head_owner num api_out head_q
  [[ -n "$br" && "$br" != "HEAD" ]] || return 1

  push_remote="${PRE_COMMIT_REMOTE_NAME:-$(git config "branch.${br}.pushRemote" 2>/dev/null || git config remote.pushDefault 2>/dev/null || git config "branch.${br}.remote" 2>/dev/null || echo origin)}"
  if git remote get-url --push "$push_remote" >/dev/null 2>&1; then
    head_slug="$(slug_from_remote_url "$(git remote get-url --push "$push_remote")" || true)"
  fi
  if [[ -z "$head_slug" ]] && api_out="$(gh api "repos/{owner}/{repo}" --jq .full_name 2>/dev/null)"; then
    head_slug="$api_out"
  fi
  [[ -n "$head_slug" && "$head_slug" == */* ]] || return 1
  head_owner="${head_slug%%/*}"

  # Candidate base repos: upstream/myohub remote if configured, parent repo when head_slug is a fork, and head_slug itself.
  local candidates=() r url parent_slug=""
  for r in myohub upstream origin; do
    if url="$(git remote get-url "$r" 2>/dev/null)"; then
      if base_slug="$(slug_from_remote_url "$url" || true)" && [[ -n "$base_slug" ]]; then
        candidates+=("$base_slug")
      fi
    fi
  done
  if api_out="$(gh api "repos/$head_slug" --jq '.parent.full_name // empty' 2>/dev/null)"; then
    parent_slug="$api_out"
  fi
  [[ -n "$parent_slug" && "$parent_slug" == */* ]] && candidates+=("$parent_slug")
  candidates+=("$head_slug")

  head_q="$(urlencode_param "${head_owner}:${br}")"
  local seen=" " cand
  for cand in "${candidates[@]}"; do
    [[ "$seen" == *" $cand "* ]] && continue
    seen+=" $cand "
    num=""
    if api_out="$(gh api --paginate "repos/$cand/pulls?state=open&head=${head_q}&per_page=100" \
      --jq "([.[] | select(.head.ref == \"$br\" and (.head.repo.full_name | ascii_downcase) == (\"$head_slug\" | ascii_downcase)) | .number])[0] // empty" 2>/dev/null)"; then
      num="$api_out"
    fi
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      printf '%s %s\n' "$cand" "$num"
      return 0
    fi
  done
  return 1
}

field() { [[ -f "$1" ]] && sed -n "s/^$2: //p" "$1" | head -1; }
line() {  # line <label> <check> <sha>
  local f="$rec/$3.$2"
  if [[ -f "$f" ]]; then
    echo "| $1 | $(field "$f" status) | $(field "$f" by) | $(field "$f" at) |"
  else
    echo "| $1 | **not run** on this commit | | |"
  fi
}
block() {  # block <sha>
  local sha="$1"
  echo "$start"
  echo "**Local checks** for \`${sha:0:10}\` (self-reported by \`scripts/review/local_review.sh\`; see REVIEW.md)"
  echo
  echo "| Check | Result | By | When (UTC) |"
  echo "|---|---|---|---|"
  line "pre-commit hooks (branch changes)" precommit "$sha"
  line "Agent review (REVIEW.md, ownership)" review "$sha"
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
  print)
    sha="$(git rev-parse "${2:-HEAD}")"
    block "$sha" ;;
  body)
    branch="${2:-${MYO_REVIEW_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}}"; branch="${branch#refs/heads/}"
    if resolved="$(resolve_open_pr "$branch")"; then
      base_slug="${resolved%% *}"; num="${resolved#* }"
      if raw_body="$(gh api "repos/$base_slug/pulls/$num" --jq '.body // ""' 2>/dev/null)"; then
        awk '/<!-- myo-review:start -->/{skip=1} !skip{print} /<!-- myo-review:end -->/{skip=0}' <<<"$raw_body" \
          | head -c 12000 || true
      fi
    fi ;;
  update)
    sha="$(git rev-parse "${2:-HEAD}")"
    command -v gh >/dev/null 2>&1 || exit 0
    branch="${3:-${MYO_REVIEW_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}}"; branch="${branch#refs/heads/}"
    if [[ -z "$branch" || "$branch" == "HEAD" ]]; then
      echo "pr_note: detached HEAD without branch name; pass branch as 3rd arg or set MYO_REVIEW_BRANCH" >&2
      exit 0
    fi
    if ! resolved="$(resolve_open_pr "$branch")"; then
      echo "pr_note: no open PR for branch ${branch} yet; open it with: gh pr create --body \"\$(scripts/review/pr_note.sh print)\"" >&2
      exit 0
    fi
    base_slug="${resolved%% *}"; num="${resolved#* }"
    body="$(gh api "repos/$base_slug/pulls/$num" --jq '.body // ""')"
    new="$(block "$sha")"
    if grep -qF "$start" <<<"$body"; then
      body="$(S="$start" E="$end" N="$new" awk '
        $0==ENVIRON["S"] {print ENVIRON["N"]; skip=1; next} $0==ENVIRON["E"] {skip=0; next} !skip {print}' <<<"$body")"
    else
      body="$body"$'\n\n'"$new"
    fi
    gh api -X PATCH "repos/$base_slug/pulls/$num" -f body="$body" >/dev/null \
      && echo "pr_note: updated the Local checks block of PR #$num" >&2 ;;
  *) echo "usage: pr_note.sh print [sha] | body [branch] | update [sha] [branch]" >&2; exit 2 ;;
esac
