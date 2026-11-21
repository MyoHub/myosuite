# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Motion-capture file readers and BVH / LAFAN1 source adapters."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "BVHCalibration": "myosuite.integrations.capture.bvh",
    "BVHClip": "myosuite.integrations.capture.bvh",
    "BVHHinge": "myosuite.integrations.capture.bvh",
    "BVHJoint": "myosuite.integrations.capture.bvh",
    "BVHMap": "myosuite.integrations.capture.bvh",
    "BVHOrientation": "myosuite.integrations.capture.bvh",
    "BVHPoint": "myosuite.integrations.capture.bvh",
    "BVHSource": "myosuite.integrations.capture.bvh",
    "BVH_MAPS": "myosuite.integrations.capture.bvh",
    "BVH_TO_MODEL": "myosuite.integrations.capture.bvh",
    "DEFAULT_BVH_ALIASES": "myosuite.integrations.capture.bvh",
    "LAFAN1_MYOFULLBODY": "myosuite.integrations.capture.bvh",
    "bvh_source": "myosuite.integrations.capture.bvh",
    "calibrate_bvh": "myosuite.integrations.capture.bvh",
    "knee_flexion": "myosuite.integrations.capture.bvh",
    "landmark_bodies": "myosuite.integrations.capture.bvh",
    "map_dict": "myosuite.integrations.capture.bvh",
    "parse_bvh": "myosuite.integrations.capture.bvh",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
