# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade for :mod:`myosuite.envs.tasks.games.chasetag.chase_tag_vs`."""

import importlib as _importlib

_target = _importlib.import_module("myosuite.envs.tasks.games.chasetag.chase_tag_vs")
__all__ = list(
    getattr(_target, "__all__", [k for k in _target.__dict__ if not k.startswith("_")])
)
__getattr__ = _target.__getattribute__
del _importlib, _target
