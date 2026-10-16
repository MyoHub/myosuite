# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoHalloween: muscle-driven pumpkin dodge on MyoFullBody."""

from myosuite.envs.myo.tasks.halloween.halloween_model import (
    ASSET_DIR,
    PUMPKIN_BODY,
    PUMPKIN_JOINT,
    build_default_halloween_spec,
    build_halloween_spec,
    compile_halloween_model,
    throw_pumpkin,
)

__all__ = [
    "ASSET_DIR",
    "PUMPKIN_BODY",
    "PUMPKIN_JOINT",
    "build_default_halloween_spec",
    "build_halloween_spec",
    "compile_halloween_model",
    "throw_pumpkin",
]
