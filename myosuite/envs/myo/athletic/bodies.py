# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Logical body / joint name maps for athletic plants.

myo_sim's revised myofullbody renames right-arm joints (``_r``) and inserts
neck dofs after the lumbar ones. Athletic terms must look joints and sites up
by name through these maps, never by hard-coded qpos indices.
"""

from __future__ import annotations

from typing import Any

from myosuite.envs.myo.athletic.specs import BodyMap

# OpenSim / Bolt sprinter and myofullbody_simple share these names.
SPRINTER_BODY_MAP = BodyMap()
FULLBODY_SIMPLE_BODY_MAP = BodyMap()
# Default myofullbody: same logical names; resolve through mj_name2id so a
# later myo_sim rename only needs a map edit.
FULLBODY_BODY_MAP = BodyMap()


def joint_qposadr(model: Any, joint_name: str) -> int:
    """qpos address of a named hinge/slide/ball joint."""
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"joint {joint_name!r} not in model")
    return int(model.jnt_qposadr[jid])


def body_id(model: Any, body_name: str) -> int:
    """MuJoCo body id for a named body."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"body {body_name!r} not in model")
    return int(bid)
