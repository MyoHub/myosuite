# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade for :mod:`myosuite.envs.tasks.suites.challenge`."""

import importlib as _importlib

from . import (  # noqa: F401
    baoding,
    bimanual,
    challenge_common,
    chase_tag_fb_model,
    chase_tag_vs,
    chasetag,
    combat_model_meta,
    relocate,
    reorient,
    run_track,
    soccer,
    tabletennis,
    tabletennis_spec,
)

_target = _importlib.import_module("myosuite.envs.tasks.suites.challenge")
__all__ = list(
    getattr(_target, "__all__", [k for k in _target.__dict__ if not k.startswith("_")])
)
__getattr__ = _target.__getattribute__
del _importlib, _target
