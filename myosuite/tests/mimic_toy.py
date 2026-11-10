# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A fixed-base ``myoMimic-v0`` test body with muscles and a motor: myo_sim's 2-DoF elbow
(``myoelbow2dof``) plus a torque motor on the shoulder and a first-order (filter) motor on
the elbow. Also used by ``training/supermimic``'s tests."""

from __future__ import annotations

import dataclasses

import mujoco

from myosuite.envs.myo.tasks.mimic.tracking.models import (
    MODELS,
    MimicModelCfg,
    register_model,
)

MIXED_MODEL = "myoelbow2dof_motors"


def _spec(self_collisions: bool = True) -> mujoco.MjSpec:
    spec = MODELS["myoelbow2dof"].build_spec(self_collisions)
    for name, joint, dyn, gear in (
        ("shoulder_motor", "r_shoulder_elev", mujoco.mjtDyn.mjDYN_NONE, 5.0),
        ("elbow_motor", "r_elbow_flex", mujoco.mjtDyn.mjDYN_FILTER, 2.0),
    ):
        act = spec.add_actuator(name=name, target=joint)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = dyn
        act.dynprm[0] = 0.03
        act.gear[0] = gear
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = (-1.0, 1.0)
    return spec


def register_mixed_model() -> MimicModelCfg:
    """Register (once) and return the mixed muscle / motor test body."""
    if MIXED_MODEL not in MODELS:
        register_model(
            dataclasses.replace(
                MODELS["myoelbow2dof"],
                name=MIXED_MODEL,
                build_spec=_spec,
                motor_expr=(".*_motor",),
            )
        )
    return MODELS[MIXED_MODEL]
