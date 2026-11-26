# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.benchmarks.term_catalog`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.benchmarks import term_catalog as _target

if _TYPE_CHECKING:
    from myosuite.benchmarks.term_catalog import *  # noqa: F403

if __name__ == "__main__":
    raise SystemExit(_target.main())

_sys.modules[__name__] = _target
