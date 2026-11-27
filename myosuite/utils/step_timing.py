# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.core.step_timing`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.core import step_timing as _target

if _TYPE_CHECKING:
    from myosuite.core.step_timing import *  # noqa: F403

_sys.modules[__name__] = _target
