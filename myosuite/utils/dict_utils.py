# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.core.dict_utils`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING
import unittest as _unittest

from myosuite.core import dict_utils as _target

if _TYPE_CHECKING:
    from myosuite.core.dict_utils import *  # noqa: F403

if __name__ == "__main__":
    _unittest.main(module=_target)

_sys.modules[__name__] = _target
