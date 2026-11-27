# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Migration stub directing SB3 callback imports to ``myosuite_ppo.sb3_callbacks``."""

from __future__ import annotations

from typing import Any

_MOVED_SYMBOLS = frozenset({"OnnxCheckpointCallback"})


def __getattr__(name: str) -> Any:
    if name in _MOVED_SYMBOLS:
        raise ImportError(
            f"`myosuite.utils.sb3_callbacks.{name}` moved to the standalone `myosuite-ppo` "
            "package (`training/ppo`) so runtime `myosuite` never imports `training/`. "
            "Install `pip install -e '.[rl]' && pip install --no-deps -e training/ppo` (from a MyoSuite source checkout) "
            "and import from `myosuite_ppo.sb3_callbacks`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
