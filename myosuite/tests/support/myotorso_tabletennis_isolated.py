# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for the MyoTorso + table-tennis-style single-arm isolated MJCF.

The on-disk file is
:file:`myosuite/envs/myo/assets/arm/myotorso_tabletennis_isolated.xml`, which
includes ``myotorso_arm_chain_host.xml`` (same ``full_body`` composition as
``myoarm_tabletennis.xml``). That chain has **one** ``myoarm_body``
(conventionally the right arm in MyoSuite), plus pelvis and legs — **not** a
MuscleMimic dual-arm model.

For a true bimanual reference, use ``musclemimic_models`` via
:func:`myosuite.integrations.musclemimic.bimanual_model.build_mimic_bimanual_spec`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from myosuite.utils.simhive_path import resolve_model_xml_path

_ARM_HOST_XML = (
    Path(__file__).resolve().parents[2]
    / "envs"
    / "myo"
    / "assets"
    / "arm"
    / "myotorso_arm_chain_host.xml"
)
_ISOLATED_XML = (
    Path(__file__).resolve().parents[2]
    / "envs"
    / "myo"
    / "assets"
    / "arm"
    / "myotorso_tabletennis_isolated.xml"
)


def myotorso_arm_host_xml_path() -> Path:
    """Return the path to ``myotorso_arm_chain_host.xml``."""
    return _ARM_HOST_XML


def myotorso_tabletennis_isolated_xml_path() -> Path:
    """Return the path to ``myotorso_tabletennis_isolated.xml``."""
    return _ISOLATED_XML


def compile_myotorso_arm_host() -> mujoco.MjModel:
    """Compile torso + single-arm + legs host (table-tennis ``full_body``)."""
    resolved = resolve_model_xml_path(_ARM_HOST_XML)
    return mujoco.MjModel.from_xml_path(str(resolved))


def build_joint_value_map(
    ref: mujoco.MjModel, host: mujoco.MjModel, ref_qpos: mujoco.MjData
) -> dict[int, float]:
    """Map host joint qpos from bimanual reference qpos (right arm only).

    Args:
        ref: Compiled MuscleMimic bimanual model.
        host: Compiled ``myotorso_arm_chain_host`` model.
        ref_qpos: ``MjData`` for *ref* with ``qpos`` already set.

    Returns:
        Mapping ``host_joint_id -> scalar qpos value`` for hinge joints that
        exist in both models as ``{myoarm_name}`` on the host and
        ``{myoarm_name}_r`` on the bimanual reference.
    """
    out: dict[int, float] = {}
    ref_by_name = {ref.joint(i).name: i for i in range(ref.njnt) if ref.joint(i).name}
    for hj in range(host.njnt):
        hname = host.joint(hj).name
        if not hname:
            continue
        # Host joints may carry _r suffix (new arm) or bare names (old arm).
        # Try direct match first, then append _r for old-style host models.
        rid = ref_by_name.get(hname) or ref_by_name.get(f"{hname}_r")
        if rid is None:
            continue
        if int(ref.jnt_type[rid]) != int(host.jnt_type[hj]):
            continue
        if int(ref.jnt_type[rid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        rqadr = int(ref.jnt_qposadr[rid])
        out[hj] = float(ref_qpos.qpos[rqadr])
    return out


def body_relative_pos(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, frame_body: str
) -> tuple[float, float, float]:
    """Return *body_name* origin in *frame_body* local frame (position only)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body)
    if bid < 0 or fid < 0:
        raise ValueError(f"Missing body {body_name!r} or {frame_body!r}")
    dx = np.asarray(data.xpos[bid] - data.xpos[fid], dtype=np.float64)
    r = np.asarray(data.xmat[fid], dtype=np.float64).reshape(3, 3)
    local = r.T @ dx
    return float(local[0]), float(local[1]), float(local[2])
