# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Inverse kinematics on native MuJoCo models."""

from __future__ import annotations

import collections
import logging
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

_INVALID_JOINT_NAMES_TYPE = (
    "`joint_names` must be either None, a list, a tuple, or a numpy array; got {}."
)
_REQUIRE_TARGET_POS_OR_QUAT = (
    "At least one of `target_pos` or `target_quat` must be specified."
)

IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


def _resolve_model_and_data(
    physics: Any,
    data: mujoco.MjData | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, bool]:
    """Extract ``(MjModel, MjData, owns_fresh_data)`` from *physics*."""
    if isinstance(physics, mujoco.MjModel):
        if data is not None:
            return physics, data, False
        return physics, mujoco.MjData(physics), True
    if (
        isinstance(physics, tuple)
        and len(physics) == 2
        and isinstance(physics[0], mujoco.MjModel)
        and isinstance(physics[1], mujoco.MjData)
    ):
        return physics[0], physics[1], False
    raw_model = getattr(physics, "model", None)
    raw_data = data if data is not None else getattr(physics, "data", None)
    model = getattr(raw_model, "_model", getattr(raw_model, "ptr", raw_model))
    resolved_data = getattr(raw_data, "_data", getattr(raw_data, "ptr", raw_data))
    if isinstance(model, mujoco.MjModel) and isinstance(resolved_data, mujoco.MjData):
        return model, resolved_data, False
    raise TypeError(
        "Expected `MjModel`, `(MjModel, MjData)`, or an object with `.model`/`.data`; "
        f"got {type(physics)!r}."
    )


def qpos_from_site_pose(
    physics: Any,
    site_name: str,
    target_pos: np.ndarray | None = None,
    target_quat: np.ndarray | None = None,
    joint_names: list[str] | tuple[str, ...] | np.ndarray | None = None,
    tol: float = 1e-14,
    rot_weight: float = 1.0,
    regularization_threshold: float = 0.1,
    regularization_strength: float = 3e-2,
    max_update_norm: float = 2.0,
    progress_thresh: float = 20.0,
    max_steps: int = 500,
    inplace: bool = False,
    data: mujoco.MjData | None = None,
) -> IKResult:
    """Find joint positions that satisfy a target site position and/or rotation.

    Args:
        physics: An ``MjModel``, ``(MjModel, MjData)`` tuple, or an object
            exposing ``.model`` and ``.data`` attributes.
        site_name: Name of the target site.
        target_pos: Desired ``(3,)`` Cartesian site position, or ``None``.
        target_quat: Desired ``(4,)`` site quaternion ``(w, x, y, z)``, or ``None``.
        joint_names: Optional sequence of joint names allowed to move.
        tol: Stopping tolerance on ``err_norm``.
        rot_weight: Weight on rotational error relative to translational error.
        regularization_threshold: Error norm above which L2 damping is active.
        regularization_strength: Quadratic damping coefficient on joint updates.
        max_update_norm: Maximum L2 norm of per-iteration joint update.
        progress_thresh: Early-stop threshold on ``err_norm / update_norm``.
        max_steps: Maximum solver iterations.
        inplace: Whether to mutate ``data`` in place rather than operating on a copy.
        data: Optional ``MjData`` when ``physics`` is an ``MjModel``.

    Returns:
        ``IKResult(qpos, err_norm, steps, success)``.
    """
    if target_pos is None and target_quat is None:
        raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    model, orig_data, created_data = _resolve_model_and_data(physics, data=data)
    if inplace or created_data:
        work_data = orig_data
    else:
        work_data = mujoco.MjData(model)
        mujoco.mj_copyData(work_data, model, orig_data)

    dtype = np.float64
    if target_pos is not None and target_quat is not None:
        jac = np.zeros((6, model.nv), dtype=dtype)
        err = np.zeros(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    elif target_pos is not None:
        jac = np.zeros((3, model.nv), dtype=dtype)
        err = np.zeros(3, dtype=dtype)
        jac_pos, jac_rot = jac, None
        err_pos, err_rot = err, None
    else:
        jac = np.zeros((3, model.nv), dtype=dtype)
        err = np.zeros(3, dtype=dtype)
        jac_pos, jac_rot = None, jac
        err_pos, err_rot = None, err

    target_pos_arr = (
        np.asarray(target_pos, dtype=dtype) if target_pos is not None else None
    )
    target_quat_arr = (
        np.asarray(target_quat, dtype=dtype) if target_quat is not None else None
    )

    update_nv = np.zeros(model.nv, dtype=dtype)
    site_xquat = np.empty(4, dtype=dtype) if target_quat_arr is not None else None
    neg_site_xquat = np.empty(4, dtype=dtype) if target_quat_arr is not None else None
    err_rot_quat = np.empty(4, dtype=dtype) if target_quat_arr is not None else None

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name!r}")

    dof_indices: slice | np.ndarray
    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, tuple, np.ndarray)):
        dofs: list[int] = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(jname))
            if jid < 0:
                raise ValueError(f"Unknown joint: {jname!r}")
            dofs.extend(np.flatnonzero(model.dof_jntid == jid).tolist())
        dof_indices = np.asarray(dofs, dtype=np.intp)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    mujoco.mj_fwdPosition(model, work_data)
    site_xpos = work_data.site_xpos[site_id]
    site_xmat = work_data.site_xmat[site_id]

    steps = 0
    err_norm = 0.0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0
        if target_pos_arr is not None and err_pos is not None:
            err_pos[:] = target_pos_arr - site_xpos
            err_norm += float(np.linalg.norm(err_pos))
        if (
            target_quat_arr is not None
            and err_rot is not None
            and site_xquat is not None
            and neg_site_xquat is not None
            and err_rot_quat is not None
        ):
            mujoco.mju_mat2Quat(site_xquat, site_xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat_arr, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1.0)
            err_norm += float(np.linalg.norm(err_rot)) * rot_weight

        if err_norm < tol:
            logger.debug("Converged after %i steps: err_norm=%3g", steps, err_norm)
            success = True
            break

        mujoco.mj_jacSite(model, work_data, jac_pos, jac_rot, site_id)
        jac_joints = jac[:, dof_indices]
        reg_strength = (
            regularization_strength if err_norm > regularization_threshold else 0.0
        )
        update_joints = nullspace_method(
            jac_joints, err, regularization_strength=reg_strength
        )
        update_norm = float(np.linalg.norm(update_joints))
        if update_norm <= 1e-12:
            break

        progress_criterion = err_norm / update_norm
        if progress_criterion > progress_thresh:
            logger.debug(
                "Step %2i: err_norm / update_norm (%3g) > tolerance (%3g). Halting.",
                steps,
                progress_criterion,
                progress_thresh,
            )
            break

        if update_norm > max_update_norm:
            update_joints = update_joints * (max_update_norm / update_norm)

        update_nv.fill(0.0)
        update_nv[dof_indices] = update_joints
        mujoco.mj_integratePos(model, work_data.qpos, update_nv, 1.0)
        mujoco.mj_fwdPosition(model, work_data)

    if not success and max_steps > 0 and steps == max_steps - 1:
        logger.warning(
            "Failed to converge after %i steps: err_norm=%3g", steps, err_norm
        )

    qpos = work_data.qpos if (inplace and not created_data) else work_data.qpos.copy()
    return IKResult(qpos=qpos, err_norm=float(err_norm), steps=steps, success=success)


def nullspace_method(
    jac_joints: np.ndarray,
    delta: np.ndarray,
    regularization_strength: float = 0.0,
) -> np.ndarray:
    """Calculate joint velocities to achieve a specified end-effector delta."""
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        hess_approx = (
            hess_approx + np.eye(hess_approx.shape[0]) * regularization_strength
        )
        return np.linalg.solve(hess_approx, joint_delta)
    return np.linalg.lstsq(hess_approx, joint_delta, rcond=None)[0]
