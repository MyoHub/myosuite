# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible scene management facade."""

from myosuite.scenes import library
from myosuite.scenes.library import (
    SCENES,
    FloorSpec,
    ObjectSpec,
    SceneSpec,
)

__all__ = ["FloorSpec", "ObjectSpec", "SCENES", "SceneSpec", "library"]
