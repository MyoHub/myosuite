# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared XML-path resolvers for bundled myo model assets.

Resolution order for each model family:
  1. Local task-assets directory (custom overrides / task-specific additions)
  2. myo_sim pip package — ``models/legacy/<family>/`` (feat/legacy-models-pip+)
  3. myo_sim pip package — ``models/<family>/`` (backward compat for older builds)
  4. Fall through to the local path (caller decides how to handle missing file)
"""

from __future__ import annotations

import pathlib

_ASSETS_ROOT = pathlib.Path(__file__).parent


def resolve_elbow_xml(filename: str = "myoelbow_1dof6muscles.xml") -> pathlib.Path:
    local = _ASSETS_ROOT / "elbow" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "elbow" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "elbow" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_finger_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "finger" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "finger" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_arm_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "arm" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "arm" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "arm" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_osl_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "leg" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "osl" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local
