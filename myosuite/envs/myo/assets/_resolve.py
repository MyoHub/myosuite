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
import warnings

_ASSETS_ROOT = pathlib.Path(__file__).parent

TORSO_PIP_CALIBRATION_WARNING = (
    "This env's torso uses myo_sim pip's muscle/tendon calibration, not the "
    "original MyoChallenge competition models. myo_sim pip mirrors right-side "
    "torso muscle parameters (gainprm/biasprm/lengthrange) onto the left side; "
    "the original competition models had independently-fit, slightly "
    "asymmetric left/right calibration. Results are not expected to match "
    "the original competition submissions exactly."
)


def warn_torso_pip_calibration_divergence() -> None:
    """Warn that this env's torso muscles diverge from the original competition models.

    See the resolve_torso_xml docstring above for the asymmetric-vs-mirrored
    left-muscle calibration divergence between the legacy/competition torso
    models and myo_sim pip's torso muscle/tendon files.
    """
    warnings.warn(TORSO_PIP_CALIBRATION_WARNING, UserWarning, stacklevel=3)


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


def resolve_torso_xml(filename: str) -> pathlib.Path:
    """Resolve torso host XML: local bundled override first, then pip myo_sim."""
    local = _ASSETS_ROOT / "torso" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "torso" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "torso" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_leg_xml(filename: str) -> pathlib.Path:
    """Resolve leg host XML: local bundled override first, then pip myo_sim.

    ``myolegs_with_torso.xml`` is bundled locally because the currently pinned
    myo_sim pip package's ``leg/myolegs.xml`` has no "torso" body (legs+pelvis
    only); the local file wraps the pip-resolved leg assets/tendon/muscle/chain
    in the bundled rigid-torso fallback so envs that need ``model.body("torso")``
    (e.g. ``LegWalkEnvV0._get_torso_angle``) keep working.
    """
    local = _ASSETS_ROOT / "leg" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "leg" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "leg" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local
