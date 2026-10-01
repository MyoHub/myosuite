# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reachable target positions of a reach task, shared by the CPU env and the mjlab twin."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

_N_SAMPLES = 100_000
_JOINT_TYPES = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
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

    Joint configurations are drawn uniformly within the joint limits (other joints stay at
    ``qpos0``); a configuration is kept when every site lies inside its box. Sampling
    targets from the result guarantees that a policy can reach them. The result is
    deterministic and cached, so the CPU env and the mjlab twin see the same table.

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
        ValueError: If no configuration puts the sites inside their boxes.
    """
    ids = tuple(int(i) for i in site_ids)
    low, high = (
        np.asarray(low, float).reshape(len(ids), 3),
        np.asarray(high, float).reshape(len(ids), 3),
    )
    key = (
        model.jnt_range.tobytes(),
        model.qpos0.tobytes(),
        model.body_pos.tobytes(),
        model.body_quat.tobytes(),
        model.site_pos[list(ids)].tobytes(),
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
            if model.jnt_limited[j] and model.jnt_type[j] in _JOINT_TYPES:
                lo, hi = model.jnt_range[j]
                qpos[:, model.jnt_qposadr[j]] = rng.uniform(lo, hi, n_samples)
        data = mujoco.MjData(model)
        positions = np.empty((n_samples, len(ids), 3))
        for i in range(n_samples):
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(model, data)
            positions[i] = data.site_xpos[list(ids)]
        inside = np.all((positions >= low) & (positions <= high), axis=(1, 2))
        if not inside.any():
            raise ValueError("no joint configuration reaches the target box")
        _CACHE[key] = positions[inside]
    return _CACHE[key]
