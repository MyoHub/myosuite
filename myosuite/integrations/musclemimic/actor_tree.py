# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.policies.actor_tree`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.policies import actor_tree as _target

if _TYPE_CHECKING:
    from myosuite.policies.actor_tree import *  # noqa: F403

_sys.modules[__name__] = _target
