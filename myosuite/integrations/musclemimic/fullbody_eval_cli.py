# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.benchmarks.musclemimic_fullbody_eval_cli`."""

import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

from myosuite.benchmarks import musclemimic_fullbody_eval_cli as _target

if __name__ == "__main__":
    raise SystemExit(_target.main())

if _TYPE_CHECKING:
    from myosuite.benchmarks.musclemimic_fullbody_eval_cli import *  # noqa: F403

_sys.modules[__name__] = _target
