# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reachable target positions of a reach task, shared by the CPU env and the mjlab twin."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

_N_SAMPLES = 100_000
_JOINT_TYPES = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
_CACHE: dict[tuple, np.ndarray] = {}


def reachable_target_points(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    low: np.ndarray,
    high: np.ndarray,
    n_samples: int = _N_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """World positions of the sites over the joint ranges, restricted to the target box.

    Joint configurations are drawn uniformly within the joint limits (other independent
    joints stay at ``qpos0``), then projected onto active joint equalities. A configuration
    is kept when it respects the joint limits and every site lies inside its box. This
    establishes kinematic reachability, not dynamic or collision-free reachability.
    The result is deterministic and cached, so both backends see the same table.

    Args:
        model: Compiled model.
        site_ids: Ids of the sites that must reach their targets.
        low: Box lower corners, shape ``(k, 3)``.
        high: Box upper corners, shape ``(k, 3)``.
        n_samples: Number of joint configurations to draw.
        seed: Seed of the sampling.

    Returns:
        Array ``(M, k, 3)``: for each kept configuration, the position of every site.

    Raises:
        ValueError: If no valid configuration puts the sites inside their boxes, or an
            active non-joint equality would require a different projection.
    """
    ids = tuple(int(i) for i in site_ids)
    low, high = (
        np.asarray(low, float).reshape(len(ids), 3),
        np.asarray(high, float).reshape(len(ids), 3),
    )
    if np.any((model.eq_active0 != 0) & (model.eq_type != mujoco.mjtEq.mjEQ_JOINT)):
        raise ValueError("workspace sampling supports only active joint equalities")
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    key = (
        hashlib.sha256(binary).digest(),
        ids,
        low.tobytes(),
        high.tobytes(),
        n_samples,
        seed,
    )
    if key not in _CACHE:
        rng = np.random.default_rng(seed)
        qpos = np.tile(model.qpos0, (n_samples, 1))
        for j in range(model.njnt):
            if model.jnt_limited[j] and int(model.jnt_type[j]) in _JOINT_TYPES:
                lo, hi = model.jnt_range[j]
                qpos[:, model.jnt_qposadr[j]] = rng.uniform(lo, hi, n_samples)
        project_joint_couplings(JointCouplings.from_model(model), qpos)
        valid = np.isfinite(qpos).all(axis=1)
        for j in range(model.njnt):
            if model.jnt_limited[j] and int(model.jnt_type[j]) in _JOINT_TYPES:
                lo, hi = model.jnt_range[j]
                valid &= (qpos[:, model.jnt_qposadr[j]] >= lo) & (
                    qpos[:, model.jnt_qposadr[j]] <= hi
                )
        qpos = qpos[valid]
        data = mujoco.MjData(model)
        positions = np.empty((len(qpos), len(ids), 3))
        for i in range(len(qpos)):
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(model, data)
            positions[i] = data.site_xpos[list(ids)]
        inside = np.all((positions >= low) & (positions <= high), axis=(1, 2))
        if not inside.any():
            raise ValueError("no joint configuration reaches the target box")
        _CACHE[key] = positions[inside]
    return _CACHE[key]
