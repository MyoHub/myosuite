# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_task_config`."""

import importlib as _importlib
import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_task_config import *  # noqa: F403

_sys.modules[__name__] = _importlib.import_module(
    "myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_task_config"
)
