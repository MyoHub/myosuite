# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.benchmarks.examine_env`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.benchmarks import examine_env as _target

if __name__ == "__main__":
    _target._reexec_if_mjpython_module_on_macos()
    _target.main()

if _TYPE_CHECKING:
    from myosuite.benchmarks.examine_env import *  # noqa: F403

_sys.modules[__name__] = _target
