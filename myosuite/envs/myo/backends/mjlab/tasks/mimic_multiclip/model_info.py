# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Joint and muscle bookkeeping shared by the musculoskeletal robots, derived from a compiled entity model.

Every robot here is a tree of hinge and slide joints under one free joint (the root), part of them tied
to another joint by a joint ``<equality>``, and Hill-type muscles that each drive one spatial tendon.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/assets/robots/model_info.py``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings


class ModelInfo:
    """Joint and muscle bookkeeping derived from the compiled entity model.

    * root: the body carrying the model's free joint;
    * independent joints: every non-root joint that is not the driven side of a joint ``<equality>``
      (MS-Human-700: 37, legs 2x7, spine 9, arms 2x7), in model order -- the joint coordinates of the
      state, of the reference motions and of the world model;
    * coupled joints: the joints set by ``q_c = qpos0_c + poly(q_i - qpos0_i)`` from one independent joint
      (MS-Human-700: 42), or held at ``qpos0_c + polycoef[0]`` by an equality naming no second joint;
    * muscles: in actuator order (= activation order), with the tendon each one drives.
    """

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1, f"expected one free joint (the root), found {len(free)}"
        self.root_joint_name = m.joint(free[0]).name
        self.root_id = int(m.jnt_bodyid[free[0]])
        self.root_name = m.body(self.root_id).name
        coupled = {}
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not m.eq_active0[e]:
                continue
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            c = m.eq_data[e, :5].copy()
            if (
                j2 < 0
            ):  # joint1 held at qpos0 + polycoef[0]; MuJoCo ignores the other coefficients
                j2, c[1:] = None, 0.0
            coupled[j1] = (j2, c)
        # addresses in topological order (chained couplings, cycles rejected) for the projection
        self.couplings = JointCouplings.from_model(m)
        self.coupled = (
            coupled  # joint id -> (independent joint id or None, polycoef[5])
        )
        self.joint_ids = [
            j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_names = [m.joint(j).name for j in self.joint_ids]
        self.indep_ids = [j for j in self.joint_ids if j not in coupled]
        self.indep_names = [m.joint(j).name for j in self.indep_ids]
        self.indep_cols = [
            self.joint_ids.index(j) for j in self.indep_ids
        ]  # positions among the non-root joints
        self.qpos0 = m.qpos0.copy()
        assert m.jnt_limited[
            self.indep_ids
        ].all(), "every independent joint needs a range"
        self.q_lo, self.q_hi = m.jnt_range[self.indep_ids].T.copy()
        assert (m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON).all()
        assert (m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).all()
        self.num_muscles = m.nu
        self.muscle_names = [m.actuator(a).name for a in range(m.nu)]
        self.muscle_tendon_names = [
            m.tendon(int(t)).name for t in m.actuator_trnid[:, 0]
        ]


def root_offset(m: mujoco.MjModel) -> np.ndarray:
    """ROOT_OFFSET with R_canonical = R_root @ ROOT_OFFSET = identity in the model's default pose (the free
    joint's qpos0), which each robot config makes the upright pose facing +x."""
    adr = m.jnt_qposadr[m.body_jntadr[ModelInfo(m).root_id]]
    R_rest = np.zeros(9)
    mujoco.mju_quat2Mat(R_rest, m.qpos0[adr + 3 : adr + 7])
    return R_rest.reshape(3, 3).T.copy()
