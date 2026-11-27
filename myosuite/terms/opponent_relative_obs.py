# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.core.terms.opponent_relative_obs`."""

import importlib as _importlib
import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from myosuite.core.terms.opponent_relative_obs import *  # noqa: F403

_sys.modules[__name__] = _importlib.import_module(
    "myosuite.core.terms.opponent_relative_obs"
)
