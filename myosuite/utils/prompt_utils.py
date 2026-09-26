# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Verbosity control for MyoSuite output.

Set the ``MYOSUITE_VERBOSITY`` environment variable to one of:
ALL / INFO / WARN (default) / ERROR / ONCE / ALWAYS / SILENT.
"""

from __future__ import annotations

import enum
import os

from termcolor import cprint


class Prompt(enum.IntEnum):
    """Verbosity levels for :func:`prompt`."""

    ALL = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    ONCE = 4
    ALWAYS = 5
    SILENT = 6


_PROMPT_CACHE: list[int] = []

_LEVEL_MAP = {
    "ALL": Prompt.ALL,
    "INFO": Prompt.INFO,
    "WARN": Prompt.WARN,
    "ERROR": Prompt.ERROR,
    "ONCE": Prompt.ONCE,
    "ALWAYS": Prompt.ALWAYS,
    "SILENT": Prompt.SILENT,
}

_raw = os.getenv("MYOSUITE_VERBOSITY", "WARN").upper()
if _raw not in _LEVEL_MAP:
    raise ValueError(
        f"Unknown MYOSUITE_VERBOSITY value: {_raw!r}. Choose from {list(_LEVEL_MAP)}"
    )
VERBOSE_MODE: Prompt = _LEVEL_MAP[_raw]


def prompt(
    data: object,
    color: str | None = None,
    on_color: str | None = None,
    flush: bool = False,
    end: str = "\n",
    type: Prompt = Prompt.INFO,
) -> None:
    """Print *data* if its verbosity *type* meets the current threshold.

    Args:
        data: Message to print (converted to str if needed).
        color: Foreground colour passed to :func:`termcolor.cprint`.
        on_color: Background colour passed to :func:`termcolor.cprint`.
        flush: Whether to flush stdout immediately.
        end: Line ending character.
        type: Verbosity level of this message.
    """
    global _PROMPT_CACHE

    if type == Prompt.ONCE:
        h = hash(str(data))
        if h in _PROMPT_CACHE:
            return
        _PROMPT_CACHE.append(h)
        type = Prompt.ALWAYS

    if on_color is None:
        if type == Prompt.WARN:
            color = "black"
            on_color = "on_yellow"
        elif type == Prompt.ERROR:
            on_color = "on_red"

    if VERBOSE_MODE == Prompt.SILENT:
        return
    if type >= VERBOSE_MODE:
        cprint(str(data), color=color, on_color=on_color, flush=flush, end=end)
