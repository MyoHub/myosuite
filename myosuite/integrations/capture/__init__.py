# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Motion-capture and camera-calibration file adapters."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "load_pose2sim_cameras": "myosuite.integrations.capture.pose2sim",
    "save_pose2sim_cameras": "myosuite.integrations.capture.pose2sim",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
