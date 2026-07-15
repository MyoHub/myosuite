#!/usr/bin/env python3
"""Strip or reject commit messages that attribute authorship to AI assistants.

The CLA check requires every author/co-author to have signed the CLA. AI
tools cannot sign it, so Co-Authored-By (and similar) trailers for Cursor,
Claude, Codex, Gemini, etc. break CI. Soft guidance in CLAUDE.md is not
enough — some tools inject these trailers after the message is composed.

Modes:
  prepare-commit-msg (``--strip``): remove matching lines so injected
    trailers never land in history.
  commit-msg (default): fail if any matching lines remain.
"""

from __future__ import annotations

import re
import sys

# Match Co-Authored-By / Signed-off-by / Made-with lines that name known AI
# tools or their noreply addresses. Keep this list tool-agnostic: Cursor,
# Anthropic/Claude, OpenAI/Codex, Google/Gemini, Copilot, etc.
_AI_TRAILER = re.compile(
    r"(?im)^(?:"
    r"co-authored-by|signed-off-by|made-with|generated-by"
    r")\s*:.*(?:"
    r"\bcursor\b|"
    r"cursoragent@|"
    r"@cursor\.com\b|"
    r"\bclaude\b|"
    r"\banthropic\b|"
    r"noreply@anthropic\.com|"
    r"@anthropic\.com\b|"
    r"\bcodex\b|"
    r"\bopenai\b|"
    r"@openai\.com\b|"
    r"\bchatgpt\b|"
    r"\bgemini\b|"
    r"\bbard\b|"
    r"@google\.com\b|"
    r"\bcopilot\b|"
    r"noreply@github\.com.*copilot"
    r")"
)


def _hits(text: str) -> list[str]:
    return [line for line in text.splitlines() if _AI_TRAILER.search(line)]


def _strip(text: str) -> str:
    kept = [
        line for line in text.splitlines(keepends=True) if not _AI_TRAILER.search(line)
    ]
    # Collapse trailing blank lines left by removed trailers.
    while kept and kept[-1].strip() == "":
        kept.pop()
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    return "".join(kept)


def main(argv: list[str]) -> int:
    strip = False
    args = argv[1:]
    if args and args[0] == "--strip":
        strip = True
        args = args[1:]

    if len(args) != 1:
        print(
            "usage: reject_ai_coauthor.py [--strip] <commit-msg-file>",
            file=sys.stderr,
        )
        return 2

    path = args[0]
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        print(f"reject_ai_coauthor: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    hits = _hits(text)
    if not hits:
        return 0

    if strip:
        try:
            open(path, "w", encoding="utf-8").write(_strip(text))
        except OSError as exc:
            print(f"reject_ai_coauthor: cannot write {path}: {exc}", file=sys.stderr)
            return 2
        print(
            "reject_ai_coauthor: stripped AI attribution trailer(s):\n"
            + "\n".join(f"  {line}" for line in hits),
            file=sys.stderr,
        )
        return 0

    print(
        "ERROR: AI assistant co-author / attribution trailers are forbidden.\n"
        "This repo's CLA check requires every commit author/co-author to have\n"
        "signed the CLA; AI tools cannot sign it.\n"
        "\n"
        "Remove these line(s) from the commit message and retry:\n"
        + "\n".join(f"  {line}" for line in hits)
        + "\n\n"
        "Also turn off Cursor Settings → Agents → Attribution if it is on.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
