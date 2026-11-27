# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility stub keeping ``solved_from_info`` and directing SB3 sweep imports to ``myosuite_ppo.sb3_sweep``."""

from __future__ import annotations

from typing import Any

from myosuite.benchmarks import solved_from_info

__all__ = ["solved_from_info"]

_MOVED_SYMBOLS = frozenset(
    {
        "WinnerPpoConfig",
        "budget_for",
        "improved",
        "improvement_threshold",
        "is_excluded_env_id",
        "parse_act_dim",
    }
)


def __getattr__(name: str) -> Any:
    if name in _MOVED_SYMBOLS:
        raise ImportError(
            f"`myosuite.utils.sb3_sweep.{name}` moved to the standalone `myosuite-ppo` "
            "package (`training/ppo`) so runtime `myosuite` never imports `training/`. "
            "Install `pip install -e '.[rl]' && pip install --no-deps -e training/ppo` (from a MyoSuite source checkout) "
            "and import from `myosuite_ppo.sb3_sweep`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
