# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Motion-capture file I/O (OpenSim TRC, C3D markers/analogs/force plates, CSV)."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "C3DTrial": "myosuite.integrations.capture.markers",
    "ForcePlateData": "myosuite.integrations.capture.markers",
    "MarkerClip": "myosuite.integrations.capture.markers",
    "fill_marker_gaps": "myosuite.integrations.capture.markers",
    "open_c3d": "myosuite.integrations.capture.markers",
    "read_c3d_trial": "myosuite.integrations.capture.markers",
    "read_markers": "myosuite.integrations.capture.markers",
    "write_trc": "myosuite.integrations.capture.markers",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
