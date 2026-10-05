#!/usr/bin/env python3
"""Strip or reject commit messages that attribute authorship to AI assistants.

The CLA check requires every author/co-author to have signed the CLA. AI
tools cannot sign it, so Co-Authored-By (and similar) trailers for Cursor,
Claude, Codex, Gemini, etc. break CI. Soft guidance in CLAUDE.md is not
enough — some tools inject these trailers after the message is composed.

Only known AI identities are matched (the names and addresses the tools sign
with, see ``_AI_PRODUCT`` / ``_AI_NAME`` / ``_AI_EMAIL``), so people who merely
share a word with a tool ("Claude Dupont", "Anna Bard", any ``@google.com``
colleague) keep their trailers.

Modes:
  prepare-commit-msg (``--strip``): remove matching lines so injected
    trailers never land in history.
  commit-msg (default): fail if any matching lines remain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Optionally indented trailer with an ASCII or full-width colon. Person keys hold
# "Name <email>", tool keys the name of a tool.
_TRAILER = re.compile(
    r"\s*(?:(?P<person>co-authored-by|co-developed-by|signed-off-by)"
    r"|(?P<tool>made-with|generated-by|generated-with|assisted-by))"
    r"\s*[:\N{FULLWIDTH COLON}]\s*(?P<value>.*?)\s*",
    re.IGNORECASE,
)
# Product names nobody is called: AI wherever they appear in a name.
_AI_PRODUCT = re.compile(
    r"\b(?:copilot|cursor|cursoragent|codex|chatgpt|gpt-?\d[\w.-]*|windsurf|codeium"
    r"|aider|openhands)\b",
    re.IGNORECASE,
)
# Names AI tools sign with that people may share ("Claude", "Bard"): AI only as the
# whole name, optionally with model / version qualifiers and "[bot]"
# ("Claude Opus 4.5", "gemini-code-assist[bot]"), so "Claude Dupont" is a person.
_AI_NAME = re.compile(
    r"(?:(?:anthropic\s+)?claude|anthropic|openai|(?:google\s+)?(?:gemini|bard|jules)"
    r"|google-labs-jules|devin(?:\s+ai|-ai-integration)?|amazon[\s-]+q)"
    r"(?:[\s:-]+(?:\d[\w.-]*|\(.*\)|code|cli|agent|assist|assistant|chat|developer"
    r"|opus|sonnet|haiku|instant|pro|flash|ultra|ai|bot))*"
    r"(?:\[bot\])?",
    re.IGNORECASE,
)
# Addresses AI tools commit with: the vendors' no-reply / agent mailboxes and the
# GitHub accounts of the coding agents (``<id>+<login>@users.noreply.github.com``).
_AI_EMAIL = re.compile(
    r"(?:no-?reply|codex)@(?:anthropic\.com|cursor\.(?:com|sh)|openai\.com|aider\.chat"
    r"|codeium\.com|windsurf\.com|google\.com|amazon\.com)"
    r"|cursoragent@[\w.-]+|openhands@all-hands\.dev"
    r"|(?:\d+\+)?(?:copilot|(?:claude|copilot-swe-agent|cursor|devin-ai-integration"
    r"|google-labs-jules|gemini-code-assist|chatgpt-codex-connector"
    r"|amazon-q-developer|codeium|windsurf|coderabbitai|sweep-ai)\[bot\])"
    r"@users\.noreply\.github\.com",
    re.IGNORECASE,
)
# Tool keys never name a person, so a tool word anywhere in the value counts.
_AI_TOOL_WORD = re.compile(
    r"\b(?:claude|anthropic|cursor|codex|openai|chatgpt|gpt-?\d[\w.-]*|gemini|bard"
    r"|copilot|aider|devin|windsurf|codeium|jules|amazon\s+q|openhands)\b",
    re.IGNORECASE,
)
# Footer line such as "Generated with [Claude Code](https://claude.com/claude-code)"
# (not a "#" comment line, which git drops).
_FOOTER = re.compile(
    r"[^\w#]*generated\s+(?:with|by)\s+\[?(?P<tool>[^\[\]()]+?)\]?\s*(?:\(\S*\))?\W*",
    re.IGNORECASE,
)
# ``git commit -v`` appends the diff below this line; git drops it from the message.
_SCISSORS = re.compile(r"^\S{1,4} -{24} >8 -{24}\s*$", re.MULTILINE)


def _is_ai_name(name: str) -> bool:
    name = name.strip()
    return bool(_AI_PRODUCT.search(name) or _AI_NAME.fullmatch(name))


def _names_ai(value: str) -> bool:
    """Whether a person trailer's ``Name <email>`` value is a known AI identity."""
    emails = re.findall(r"<([^<>]*)>", value) or [value]
    return _is_ai_name(value.split("<", 1)[0]) or any(
        _AI_EMAIL.fullmatch(email.strip()) for email in emails
    )


def _is_ai_line(line: str) -> bool:
    """Whether *line* is an AI co-author / attribution trailer or footer."""
    line = line.rstrip("\r\n")
    trailer = _TRAILER.fullmatch(line)
    if trailer is not None:
        value = trailer["value"]
        if trailer["tool"] is not None and _AI_TOOL_WORD.search(value):
            return True
        return _names_ai(value)
    footer = _FOOTER.fullmatch(line)
    return footer is not None and _is_ai_name(footer["tool"])


def _split_message(text: str) -> tuple[str, str]:
    """``(message, verbose_diff)``: the text above git's scissors line, and the rest."""
    scissors = _SCISSORS.search(text)
    cut = len(text) if scissors is None else scissors.start()
    return text[:cut], text[cut:]


def _hits(text: str) -> list[str]:
    message, _ = _split_message(text)
    return [line for line in message.splitlines() if _is_ai_line(line)]


def _strip(text: str) -> str:
    message, diff = _split_message(text)
    kept = [line for line in message.splitlines(keepends=True) if not _is_ai_line(line)]
    # Collapse trailing blank lines left by removed trailers.
    while kept and kept[-1].strip() == "":
        kept.pop()
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    return "".join(kept) + diff


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

    path = Path(args[0])
    try:
        # Bytes in, bytes out: any encoding (i18n.commitEncoding) and line ending
        # survives a --strip unchanged apart from the removed lines.
        text = path.read_bytes().decode("utf-8", "surrogateescape")
    except OSError as exc:
        print(f"reject_ai_coauthor: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    hits = _hits(text)
    if not hits:
        return 0

    # Report undecodable bytes / emoji as escapes rather than crash the hook.
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    report = "\n".join(f"  {line}" for line in hits)
    if strip:
        try:
            path.write_bytes(_strip(text).encode("utf-8", "surrogateescape"))
        except OSError as exc:
            print(f"reject_ai_coauthor: cannot write {path}: {exc}", file=sys.stderr)
            return 2
        print(
            "reject_ai_coauthor: stripped AI attribution trailer(s):\n" + report,
            file=sys.stderr,
        )
        return 0

    print(
        "ERROR: AI assistant co-author / attribution trailers are forbidden.\n"
        "This repo's CLA check requires every commit author/co-author to have\n"
        "signed the CLA; AI tools cannot sign it.\n"
        "\n"
        "Remove these line(s) from the commit message and retry:\n" + report + "\n\n"
        "Also turn off Cursor Settings → Agents → Attribution if it is on.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
