# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility facade re-exporting the BVH / LAFAN1 capture adapter."""

from __future__ import annotations

import importlib
from typing import Any

_PUBLIC_NAMES = (
    "BVH_MAPS",
    "BVH_TO_MODEL",
    "DEFAULT_BVH_ALIASES",
    "LAFAN1_MYOFULLBODY",
    "BVHCalibration",
    "BVHClip",
    "BVHHinge",
    "BVHJoint",
    "BVHMap",
    "BVHOrientation",
    "BVHPoint",
    "BVHSource",
    "bvh_source",
    "calibrate_bvh",
    "knee_flexion",
    "landmark_bodies",
    "map_dict",
    "parse_bvh",
)

__all__ = list(_PUBLIC_NAMES)


def __getattr__(name: str) -> Any:
    mod = importlib.import_module("myosuite.integrations.capture.bvh")
    return getattr(mod, name)


def __dir__() -> list[str]:
    mod = importlib.import_module("myosuite.integrations.capture.bvh")
    return sorted(set(globals()) | set(_PUBLIC_NAMES) | set(dir(mod)))
