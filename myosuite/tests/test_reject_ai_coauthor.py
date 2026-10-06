# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``scripts/reject_ai_coauthor.py`` commit-message (CLA) guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.reject_ai_coauthor import _hits, main

pytestmark = pytest.mark.tier1

AI_LINES = [
    "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>",
    "co-authored-by: claude <noreply@anthropic.com>",
    "  Co-authored-by: Claude <noreply@anthropic.com>",
    "Co-Authored-By\N{FULLWIDTH COLON} Claude <noreply@anthropic.com>",
    "Co-authored-by: Claude Sonnet 4.5 (1M context) <noreply@anthropic.com>",
    "Co-authored-by: claude[bot] <209825114+claude[bot]@users.noreply.github.com>",
    "Signed-off-by: Claude <noreply@anthropic.com>",
    "Co-authored-by: Jane Doe <noreply@anthropic.com>",
    "Co-authored-by: Claude",
    "Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
    "Co-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
    "Co-authored-by: GitHub Copilot <noreply@github.com>",
    "Co-authored-by: Cursor Agent <cursoragent@cursor.com>",
    "Co-authored-by: Cursor Bot <bot@cursor.sh>",
    "Made-with: Cursor",
    "Co-authored-by: Codex <codex@openai.com>",
    "Co-authored-by: ChatGPT <noreply@openai.com>",
    "Co-authored-by: chatgpt-codex-connector[bot] <1+chatgpt-codex-connector[bot]@users.noreply.github.com>",
    "Co-authored-by: Gemini <noreply@google.com>",
    "Co-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>",
    "Co-authored-by: Bard",
    "Co-authored-by: google-labs-jules[bot] <161369871+google-labs-jules[bot]@users.noreply.github.com>",
    "Co-authored-by: aider (gpt-4o) <noreply@aider.chat>",
    "Co-authored-by: devin-ai-integration[bot] <158243242+devin-ai-integration[bot]@users.noreply.github.com>",
    "Co-authored-by: Windsurf <noreply@codeium.com>",
    "Co-authored-by: Amazon Q <noreply@amazon.com>",
    "Co-authored-by: OpenHands <openhands@all-hands.dev>",
    "Co-authored-by: GPT-4o <someone@example.org>",
    "Generated-by: GitHub Copilot",
    "Assisted-by: Claude:claude-3-opus coccinelle",
    "\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
    "Generated with Cursor",
]

HUMAN_LINES = [
    "Co-authored-by: Claude Dupont <claude.dupont@epfl.ch>",
    "Signed-off-by: Jane Doe <jane.doe@google.com>",
    "Co-authored-by: Anna Bard <anna@example.org>",
    "Co-authored-by: Devin Smith <devin@example.org>",
    "Co-authored-by: Jules Verne <jules@example.org>",
    "Co-authored-by: Jane Doe <12345+janedoe@users.noreply.github.com>",
    "Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
    "Co-authored-by: Ana Gemini Rossi <ana@example.org>",
    "Co-authored-by: Jane Doe (Anthropic) <jane.doe@anthropic.com>",
    "Reviewed-by: Claude Dupont <claude.dupont@epfl.ch>",
    "Compare the Claude and Codex rollouts: co-authored-by humans only.",
    "# Co-authored-by: Claude <noreply@anthropic.com>",
    "Generated with Sphinx",
]


@pytest.mark.parametrize("line", AI_LINES)
def test_ai_trailers_are_caught(line: str) -> None:
    """Every known AI identity is caught, whatever the casing or indentation."""
    assert _hits(f"Fix the thing\n\nBody.\n\n{line}\n") == [line]


@pytest.mark.parametrize("line", HUMAN_LINES)
def test_human_trailers_are_kept(line: str) -> None:
    """People who share a word with a tool (or its domain) are not AI trailers."""
    assert _hits(f"Fix the thing\n\nBody.\n\n{line}\n") == []


def _message(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_bytes(data)
    return path


def test_strip_removes_only_the_ai_trailer_and_keeps_the_bytes(tmp_path: Path) -> None:
    """``--strip`` keeps human co-authors, CRLF line ends and non-UTF-8 bytes."""
    path = _message(
        tmp_path,
        b"Caf\xe9 fix\r\n\r\nBody\r\n\r\n"
        b"Co-authored-by: Claude Dupont <claude.dupont@epfl.ch>\r\n"
        b"Co-Authored-By: Claude <noreply@anthropic.com>\r\n",
    )

    assert main(["reject_ai_coauthor.py", "--strip", str(path)]) == 0
    assert path.read_bytes() == (
        b"Caf\xe9 fix\r\n\r\nBody\r\n\r\n"
        b"Co-authored-by: Claude Dupont <claude.dupont@epfl.ch>\r\n"
    )
    assert main(["reject_ai_coauthor.py", str(path)]) == 0


def test_commit_msg_mode_rejects_a_non_utf8_message_with_ai_trailer(
    tmp_path: Path,
) -> None:
    """A latin-1 message (``i18n.commitEncoding``) is checked instead of crashing."""
    clean = _message(tmp_path, b"Caf\xe9 fix\n")
    assert main(["reject_ai_coauthor.py", str(clean)]) == 0

    clean.write_bytes(
        "Caf\xe9 fix\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n".encode(
            "latin-1"
        )
    )
    assert main(["reject_ai_coauthor.py", str(clean)]) == 1


def test_verbose_diff_below_the_scissors_line_is_ignored(tmp_path: Path) -> None:
    """``git commit -v`` diff lines are not message text (git drops them)."""
    diff = (
        "# ------------------------ >8 ------------------------\n"
        "diff --git a/NOTES b/NOTES\n"
        " Co-authored-by: Claude <noreply@anthropic.com>\n"
    )
    path = _message(tmp_path, f"Fix the thing\n\n{diff}".encode())

    assert main(["reject_ai_coauthor.py", str(path)]) == 0
    path.write_bytes(
        f"Fix\n\nCo-authored-by: Claude <noreply@anthropic.com>\n{diff}".encode()
    )
    assert main(["reject_ai_coauthor.py", "--strip", str(path)]) == 0
    assert path.read_bytes() == f"Fix\n{diff}".encode()
