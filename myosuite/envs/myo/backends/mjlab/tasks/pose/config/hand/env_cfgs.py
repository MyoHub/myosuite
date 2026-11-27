# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.envs.tasks.hand.pose.hand_mjlab_cfg`."""

import importlib as _importlib
import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from myosuite.envs.tasks.hand.pose.hand_mjlab_cfg import *  # noqa: F403

_sys.modules[__name__] = _importlib.import_module(
    "myosuite.envs.tasks.hand.pose.hand_mjlab_cfg"
)
