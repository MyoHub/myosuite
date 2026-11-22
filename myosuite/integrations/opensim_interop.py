# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Legacy private names for the OpenSim scale-file helpers now in myo_sim."""

from __future__ import annotations

from typing import Any

from myosuite.core.body_scaling import _library

_NAMES = {
    "_MYOFULLBODY_SEGMENTS": "_COMPOSED_SEGMENTS",
    "_MYOLEG_SEGMENTS": "_STANDALONE_LEG_SEGMENTS",
    "_read_opensim_scales": "read_opensim_scales",
    "_segments_for": "_segments_for",
}


def __getattr__(name: str) -> Any:
    if name in _NAMES:
        return getattr(_library("opensim_scales"), _NAMES[name])
    raise AttributeError(name)
