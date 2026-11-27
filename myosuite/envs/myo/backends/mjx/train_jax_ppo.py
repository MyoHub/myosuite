# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible alias for :mod:`myosuite.envs.backends._experimental_mjx.train_jax_ppo`."""

import importlib as _importlib
import sys as _sys

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from myosuite.envs.backends._experimental_mjx.train_jax_ppo import *  # noqa: F403

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module(
        "myosuite.envs.backends._experimental_mjx.train_jax_ppo",
        run_name="__main__",
        alter_sys=True,
    )
else:
    _sys.modules[__name__] = _importlib.import_module(
        "myosuite.envs.backends._experimental_mjx.train_jax_ppo"
    )
