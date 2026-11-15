# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MuJoCo's own residuals of the joint equalities, for coupling checks in tests."""

from __future__ import annotations

import mujoco
import numpy as np


def joint_equality_residuals(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity residuals (``efc_pos``, ``efc_vel``) of the active
    ``mjEQ_JOINT`` rows at the state *qpos*, *qvel* (zero velocity if ``None``).

    Args:
        model: Compiled model whose CPU layout *qpos* and *qvel* follow.
        qpos: Positions ``(nq,)``.
        qvel: Velocities ``(nv,)``.

    Returns:
        The two residual vectors, one entry per active joint equality.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    rows = np.flatnonzero(
        data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    )
    rows = rows[model.eq_type[data.efc_id[rows]] == mujoco.mjtEq.mjEQ_JOINT]
    return data.efc_pos[rows].copy(), data.efc_vel[rows].copy()
