# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.viz.examine_reference`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.viz import examine_reference as _target

if __name__ == "__main__":
    _target.examine_reference()

if _TYPE_CHECKING:
    from myosuite.viz.examine_reference import *  # noqa: F403

_sys.modules[__name__] = _target
