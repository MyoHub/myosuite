# Copyright (c) MyoSuite Authors. All rights reserved.
# Portions adapted from dexsuite/dex-retargeting (Copyright (c) 2022 Yuzhe Qin, MIT License)
# and embodied-ai-nexus/Eden (Copyright (c) 2026 Embodied AI Nexus / Genesis Authors, Apache-2.0).
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Dexterous hand frame calibration, anatomical presets, and SLSQP fingertip retargeting.

Provides two building blocks in pure NumPy, SciPy, and CPU MuJoCo:

1. **MANO-to-model frame calibration** (:func:`align_up_and_front`,
   :func:`hand_frame_from_sites`, :func:`mano_to_wrist_rot`) and anatomical
   presets for ``myoHand`` and ``myoArm`` (:func:`preset_for_model`).
2. **Kinematic hand retargeting** (:class:`DexHandRetargeter`) supporting
   ``"position"``, ``"vector"``, and ``"dexpilot"`` objectives solved via
   ``scipy.optimize.minimize(..., method="SLSQP")`` with analytic ``mj_jacSite``
   gradients and exact :class:`~myosuite.physics.joint_coupling.JointCouplings`
   chain-rule projection.

Attribution:
    - The DexPilot pinch-projection and inter-finger vector objective follows
      Handa et al., *"DexPilot: Vision-Based Teleoperation of Dexterous Robotic
      Hand-Arm System"*, ICRA 2020 (``arXiv:1910.03135``).
    - The ``position`` / ``vector`` / ``dexpilot`` Smooth-L1 cost structure and
      hysteresis projection state machine are adapted from Qin et al.,
      ``dexsuite/dex-retargeting`` (MIT License) and ``embodied-ai-nexus/Eden``
      (``eden/utils/dex_retargeter``, Apache-2.0).
    - MuJoCo ``mj_jacSite`` / ``mj_jacBody`` analytic Jacobians, polynomial
      ``JointCouplings`` chain-rule projection, and the ``myoHand`` / ``myoArm``
      anatomical presets (:data:`MYOHAND_PRESET`, :data:`MYOARM_HAND_PRESET`) are
      MyoSuite-specific implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

_DEXPILOT_SECONDARY_PROJECT_DIST = 0.03

_MANO_UP_FRONT: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "right": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
}

TIP_MANO_INDICES: tuple[int, int, int, int, int] = (4, 8, 12, 16, 20)
"""Indices of ``(thumb, index, middle, ring, pinky)`` tips in the 21-keypoint MANO layout."""


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-10:
        raise ValueError(f"Expected non-zero 3-vector, got {v}.")
    return arr / n


def align_up_and_front(
    up: tuple[float, float, float] | np.ndarray,
    front: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return ``(3, 3)`` rotation mapping asset ``front -> +X`` and ``up -> +Z``."""
    z_asset = _unit(np.asarray(up, dtype=np.float64))
    x_raw = _unit(np.asarray(front, dtype=np.float64))
    x_asset = _unit(x_raw - np.dot(x_raw, z_asset) * z_asset)
    y_asset = np.cross(z_asset, x_asset)
    return np.stack([x_asset, y_asset, z_asset], axis=0)


def mano_to_wrist_rot(
    wrist_up: tuple[float, float, float] | np.ndarray,
    wrist_front: tuple[float, float, float] | np.ndarray,
    *,
    side: str = "right",
) -> np.ndarray:
    """Return ``(3, 3)`` rotation mapping MANO wrist-local vectors to robot wrist-local frame.

    Args:
        wrist_up: Finger extension axis in the robot wrist body's local frame.
        wrist_front: Palm-normal axis in the robot wrist body's local frame.
        side: ``"right"`` or ``"left"``.

    Returns:
        ``R_mano_to_wrist`` in ``SO(3)``.
    """
    if side not in _MANO_UP_FRONT:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    mano_up, mano_front = _MANO_UP_FRONT[side]
    r_wrist_to_world = align_up_and_front(wrist_up, wrist_front)
    r_mano_to_world = align_up_and_front(mano_up, mano_front)
    return r_wrist_to_world.T @ r_mano_to_world


def hand_frame_from_sites(
    model: mujoco.MjModel,
    wrist_body: str,
    middle_tip_site: str,
    thumb_tip_site: str,
    pinky_tip_site: str,
    *,
    side: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``(wrist_up, wrist_front)`` in ``wrist_body``'s frame from ``qpos0``."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    project_joint_couplings(JointCouplings.from_model(model), data.qpos)
    mujoco.mj_kinematics(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
    if bid < 0:
        raise ValueError(f"Body {wrist_body!r} not found in model.")
    r_w = data.xmat[bid].reshape(3, 3)
    p_w = data.xpos[bid]

    def _local_site(name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site {name!r} not found in model.")
        return r_w.T @ (data.site_xpos[sid] - p_w)

    up = _unit(_local_site(middle_tip_site))
    radial_raw = _local_site(thumb_tip_site) - _local_site(pinky_tip_site)
    radial = _unit(radial_raw - np.dot(radial_raw, up) * up)
    front = np.cross(radial, up) if side == "right" else np.cross(up, radial)
    return up, _unit(front)


@dataclass(frozen=True)
class HandPreset:
    """Site, body, and joint naming convention for a musculoskeletal hand model."""

    name: str
    side: str
    wrist_body: str
    palm_site: str
    tip_sites: tuple[str, str, str, str, str]
    finger_joints: tuple[str, ...]
    wrist_joints: tuple[str, ...] = ()
    wrist_up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    wrist_front: tuple[float, float, float] = (0.0, 0.0, -1.0)


def _finger_joint_names(
    suffix: str, thumb_cmc_order: tuple[str, str] = ("cmc_abduction", "cmc_flexion")
) -> tuple[str, ...]:
    base = [
        thumb_cmc_order[0],
        thumb_cmc_order[1],
        "mp_flexion",
        "ip_flexion",
    ]
    for d in (2, 3, 4, 5):
        base.extend(
            [f"mcp{d}_flexion", f"mcp{d}_abduction", f"pm{d}_flexion", f"md{d}_flexion"]
        )
    return tuple(f"{j}{suffix}" for j in base)


MYOHAND_PRESET = HandPreset(
    name="myohand",
    side="right",
    wrist_body="lunate_r",
    palm_site="S_grasp_r",
    tip_sites=("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
    finger_joints=_finger_joint_names("_r", ("cmc_flexion", "cmc_abduction")),
    wrist_joints=("pro_sup_r", "deviation_r", "flexion_r"),
    wrist_up=(0.0, -1.0, 0.0),
    wrist_front=(0.0, 0.0, -1.0),
)

MYOARM_HAND_PRESET = HandPreset(
    name="myoarm",
    side="right",
    wrist_body="lunate",
    palm_site="S_grasp",
    tip_sites=("THtip", "IFtip", "MFtip", "RFtip", "LFtip"),
    finger_joints=_finger_joint_names("", ("cmc_abduction", "cmc_flexion")),
    wrist_joints=("pro_sup", "deviation", "flexion"),
    wrist_up=(0.0, -1.0, 0.0),
    wrist_front=(0.0, 0.0, -1.0),
)


def preset_for_model(model: mujoco.MjModel, side: str = "right") -> HandPreset:
    """Resolve a :class:`HandPreset` matching the sites and joints present in ``model``."""
    suffix_candidates = ("_r", "") if side == "right" else ("_l", "")
    for sfx in suffix_candidates:
        tips = tuple(f"{t}{sfx}" for t in ("THtip", "IFtip", "MFtip", "RFtip", "LFtip"))
        if all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) >= 0 for s in tips
        ):
            wrist_body = f"lunate{sfx}"
            palm_site = f"S_grasp{sfx}"
            all_jnts = {model.joint(i).name for i in range(model.njnt)}
            cmc_order = (
                ("cmc_flexion", "cmc_abduction")
                if f"cmc_flexion{sfx}" in all_jnts
                and model.joint(f"cmc_flexion{sfx}").id
                < model.joint(f"cmc_abduction{sfx}").id
                else ("cmc_abduction", "cmc_flexion")
            )
            f_jnts = tuple(
                j for j in _finger_joint_names(sfx, cmc_order) if j in all_jnts
            )
            w_jnts = tuple(
                f"{j}{sfx}"
                for j in ("pro_sup", "deviation", "flexion")
                if f"{j}{sfx}" in all_jnts
            )
            front = (0.0, 0.0, -1.0) if side == "right" else (0.0, 0.0, 1.0)
            return HandPreset(
                name=f"auto{sfx or '_nosfx'}",
                side=side,
                wrist_body=wrist_body,
                palm_site=palm_site,
                tip_sites=tips,  # type: ignore[arg-type]
                finger_joints=f_jnts,
                wrist_joints=w_jnts,
                wrist_up=(0.0, -1.0, 0.0),
                wrist_front=front,
            )
    raise ValueError(
        f"Could not auto-detect fingertip sites (THtip..LFtip) for side={side!r} in model."
    )


def _huber_val_grad(r: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise SmoothL1 / Huber value and derivative w.r.t. ``r``."""
    abs_r = np.abs(r)
    quad = abs_r < beta
    val = np.where(quad, 0.5 * (r * r) / beta, abs_r - 0.5 * beta)
    grad = np.where(quad, r / beta, np.sign(r))
    return val, grad


def _generate_dexpilot_pairs(num_fingers: int) -> tuple[list[int], list[int]]:
    origin_idx: list[int] = []
    task_idx: list[int] = []
    for i in range(1, num_fingers):
        for j in range(i + 1, num_fingers + 1):
            origin_idx.append(j)
            task_idx.append(i)
    for i in range(1, num_fingers + 1):
        origin_idx.append(0)
        task_idx.append(i)
    return origin_idx, task_idx


class DexHandRetargeter:
    """SLSQP fingertip retargeter for MuJoCo hand models (``position``, ``vector``, ``dexpilot``).

    Implements the DexPilot pinch-projection objective of Handa et al. (ICRA 2020)
    as formulated in ``dexsuite/dex-retargeting`` (MIT) and ``embodied-ai-nexus/Eden``
    (Apache-2.0), using native MuJoCo ``mj_jacSite`` Jacobians and polynomial
    :class:`~myosuite.physics.joint_coupling.JointCouplings` chain-rule gradients.

    Args:
        model: Compiled ``mujoco.MjModel``.
        preset: Hand naming preset (auto-detected via :func:`preset_for_model` when ``None``).
        mode: ``"dexpilot"`` (default), ``"vector"``, or ``"position"``.
        include_wrist: Optimize wrist DOFs alongside finger DOFs when ``mode="position"``.
        scaling: Multiplier applied to human wrist-to-finger vectors.
        huber_delta: Smooth-L1 transition threshold in metres.
        norm_delta: Temporal smoothness weight ``||q - q_prev||^2``.
        project_dist: DexPilot thumb-to-finger pinch snap distance (m).
        escape_dist: DexPilot pinch release hysteresis distance (m).
        eta1: Projected thumb-to-finger pinch distance (m).
        eta2: Projected secondary finger-to-finger distance (m).
        maxiter: Maximum SLSQP iterations per frame.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        preset: HandPreset | None = None,
        *,
        mode: str = "dexpilot",
        include_wrist: bool = False,
        scaling: float = 1.0,
        huber_delta: float = 0.03,
        norm_delta: float = 4e-3,
        project_dist: float = 0.03,
        escape_dist: float = 0.05,
        eta1: float = 1e-4,
        eta2: float = 3e-2,
        maxiter: int = 60,
    ) -> None:
        if mode not in ("dexpilot", "vector", "position"):
            raise ValueError(
                f"mode must be 'dexpilot', 'vector', or 'position'; got {mode!r}."
            )
        self.model = model
        self.data = mujoco.MjData(model)
        self.preset = preset or preset_for_model(model)
        self.mode = mode
        self.scaling = float(scaling)
        self.huber_delta = float(huber_delta)
        self.norm_delta = float(norm_delta)
        self.project_dist = float(project_dist)
        self.escape_dist = float(escape_dist)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.maxiter = int(maxiter)

        self.couplings = JointCouplings.from_model(model)
        coupled_dep = set(self.couplings.dep_qpos.tolist())

        joint_names = list(self.preset.finger_joints)
        if include_wrist:
            joint_names = list(self.preset.wrist_joints) + joint_names

        qpos_adrs: list[int] = []
        dof_adrs: list[int] = []
        bounds: list[tuple[float, float]] = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint {jname!r} not found in model.")
            qadr = int(model.jnt_qposadr[jid])
            if qadr in coupled_dep:
                continue
            qpos_adrs.append(qadr)
            dof_adrs.append(int(model.jnt_dofadr[jid]))
            if model.jnt_limited[jid]:
                lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            else:
                lo, hi = -np.pi, np.pi
            bounds.append((lo, hi))

        self.target_joint_names = tuple(joint_names)
        self.qpos_adrs = np.asarray(qpos_adrs, dtype=np.int64)
        self.dof_adrs = np.asarray(dof_adrs, dtype=np.int64)
        self.bounds = bounds
        self.opt_dof = len(self.qpos_adrs)

        self.wrist_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.preset.wrist_body
        )
        if self.wrist_body_id < 0:
            raise ValueError(
                f"Wrist body {self.preset.wrist_body!r} not found in model."
            )
        self.tip_site_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                for s in self.preset.tip_sites
            ],
            dtype=np.int64,
        )
        if np.any(self.tip_site_ids < 0):
            raise ValueError(
                f"Missing tip sites in model for preset {self.preset.name!r}."
            )

        self.R_mano_to_wrist = mano_to_wrist_rot(
            self.preset.wrist_up, self.preset.wrist_front, side=self.preset.side
        )

        num_fingers = len(self.tip_site_ids)
        if mode == "dexpilot":
            orig_i, task_i = _generate_dexpilot_pairs(num_fingers)
            self._origin_idx = np.asarray(orig_i, dtype=np.int64)
            self._task_idx = np.asarray(task_i, dtype=np.int64)
            self.human_pair_indices = (
                np.stack([self._origin_idx, self._task_idx], axis=0) * 4
            )
            n_pairs = num_fingers * (num_fingers - 1) // 2
            self._projected = np.zeros(n_pairs, dtype=bool)
            s2_orig, s2_task = [], []
            for i in range(num_fingers - 2):
                for j in range(i + 1, num_fingers - 1):
                    s2_orig.append(j)
                    s2_task.append(i)
            self._s2_orig = np.asarray(s2_orig, dtype=np.int64)
            self._s2_task = np.asarray(s2_task, dtype=np.int64)
            self._proj_dist = np.array(
                [eta1] * (num_fingers - 1)
                + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2),
                dtype=np.float64,
            )
        elif mode == "vector":
            self._origin_idx = np.zeros(num_fingers, dtype=np.int64)
            self._task_idx = np.arange(1, num_fingers + 1, dtype=np.int64)
            self.human_pair_indices = np.vstack(
                [np.zeros(num_fingers, dtype=np.int64), np.asarray(TIP_MANO_INDICES)]
            )
        else:
            self._origin_idx = np.empty(0, dtype=np.int64)
            self._task_idx = np.empty(0, dtype=np.int64)
            self.human_pair_indices = np.asarray(TIP_MANO_INDICES, dtype=np.int64)

        self._last_x = model.qpos0[self.qpos_adrs].astype(np.float64).copy()
        self._jac_buf = np.zeros((3, model.nv), dtype=np.float64)

    def reset(self, qpos0: np.ndarray | None = None) -> None:
        """Reset warm-start joint angles and DexPilot pinch-projection state."""
        base = (
            self.model.qpos0 if qpos0 is None else np.asarray(qpos0, dtype=np.float64)
        )
        self._last_x = base[self.qpos_adrs].astype(np.float64).copy()
        if self.mode == "dexpilot":
            self._projected[:] = False

    def _anchors_and_jacs(
        self, x: np.ndarray, base_qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(6, 3)`` positions ``[wrist, 5 tips]`` and ``(6, 3, opt_dof)`` Jacobians."""
        q = base_qpos.copy()
        q[self.qpos_adrs] = x
        if len(self.couplings.dep_qpos) > 0:
            project_joint_couplings(self.couplings, q)
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        pos = np.empty((1 + len(self.tip_site_ids), 3), dtype=np.float64)
        jac = np.empty((1 + len(self.tip_site_ids), 3, self.model.nv), dtype=np.float64)

        pos[0] = self.data.xpos[self.wrist_body_id]
        mujoco.mj_jacBody(
            self.model, self.data, self._jac_buf, None, self.wrist_body_id
        )
        jac[0] = self._jac_buf

        for i, sid in enumerate(self.tip_site_ids, start=1):
            pos[i] = self.data.site_xpos[sid]
            mujoco.mj_jacSite(self.model, self.data, self._jac_buf, None, int(sid))
            jac[i] = self._jac_buf

        if len(self.couplings.dep_qpos) > 0:
            c = self.couplings
            ind_val = q[c.ind_qpos]
            p = c.poly
            slope = p[:, 1] + ind_val * (
                2.0 * p[:, 2] + ind_val * (3.0 * p[:, 3] + 4.0 * p[:, 4] * ind_val)
            )
            dep_dofs = self.model.jnt_dofadr[
                [
                    next(
                        j
                        for j in range(self.model.njnt)
                        if self.model.jnt_qposadr[j] == dq
                    )
                    for dq in c.dep_qpos
                ]
            ]
            ind_dofs = self.model.jnt_dofadr[
                [
                    next(
                        j
                        for j in range(self.model.njnt)
                        if self.model.jnt_qposadr[j] == iq
                    )
                    for iq in c.ind_qpos
                ]
            ]
            for dd, idf, s in zip(dep_dofs, ind_dofs, slope, strict=True):
                jac[:, :, idf] += jac[:, :, dd] * s

        return pos, jac[:, :, self.dof_adrs]

    def keypoints_to_targets(
        self,
        keypoints_21: np.ndarray,
        wrist_rot: np.ndarray | None = None,
        base_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map ``(21, 3)`` MANO keypoints into the robot's current wrist world frame."""
        kp = np.asarray(keypoints_21, dtype=np.float64)
        if kp.shape != (21, 3):
            raise ValueError(f"Expected (21, 3) MANO keypoints, got {kp.shape}.")
        q = (
            self.model.qpos0.copy()
            if base_qpos is None
            else np.asarray(base_qpos, np.float64).copy()
        )
        q[self.qpos_adrs] = self._last_x
        if len(self.couplings.dep_qpos) > 0:
            project_joint_couplings(self.couplings, q)
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        r_robot_wrist = self.data.xmat[self.wrist_body_id].reshape(3, 3)
        p_robot_wrist = self.data.xpos[self.wrist_body_id]

        rel_world = kp - kp[0:1]
        if wrist_rot is not None:
            r_mano_w = np.asarray(wrist_rot, dtype=np.float64).reshape(3, 3)
            rel_mano_local = rel_world @ r_mano_w
        else:
            rel_mano_local = rel_world
        rel_robot_local = rel_mano_local @ self.R_mano_to_wrist.T
        mapped_kp = p_robot_wrist[None, :] + rel_robot_local @ r_robot_wrist.T

        if self.mode == "position":
            return p_robot_wrist[None, :] + self.scaling * (
                mapped_kp[np.asarray(TIP_MANO_INDICES)] - p_robot_wrist[None, :]
            )
        orig_kp = mapped_kp[self.human_pair_indices[0]]
        task_kp = mapped_kp[self.human_pair_indices[1]]
        return task_kp - orig_kp

    def retarget_step(
        self,
        ref_value: np.ndarray,
        base_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve one frame and return a full ``(nq,)`` ``qpos`` vector."""
        from scipy.optimize import minimize

        q_base = (
            self.model.qpos0.astype(np.float64).copy()
            if base_qpos is None
            else np.asarray(base_qpos, dtype=np.float64).copy()
        )
        last_x = self._last_x.copy()
        ref = np.asarray(ref_value, dtype=np.float64)

        if self.mode == "position":
            if ref.shape != (len(self.tip_site_ids), 3):
                raise ValueError(
                    f"Position mode expects {(len(self.tip_site_ids), 3)}, got {ref.shape}."
                )

            def obj_pos(x: np.ndarray) -> tuple[float, np.ndarray]:
                pos, jac = self._anchors_and_jacs(x, q_base)
                tips, j_tips = pos[1:], jac[1:]
                diff = tips - ref
                h_val, h_grad = _huber_val_grad(diff, self.huber_delta)
                loss = float(h_val.mean() + self.norm_delta * np.sum((x - last_x) ** 2))
                grad_pos = h_grad / diff.size
                grad_x = np.einsum(
                    "kc,kcd->d", grad_pos, j_tips
                ) + 2.0 * self.norm_delta * (x - last_x)
                return loss, grad_x

            res = minimize(
                obj_pos,
                last_x,
                method="SLSQP",
                jac=True,
                bounds=self.bounds,
                options={"maxiter": self.maxiter, "ftol": 1e-6},
            )
        else:
            n_vec = len(self._origin_idx)
            if ref.shape != (n_vec, 3):
                raise ValueError(
                    f"{self.mode} mode expects {(n_vec, 3)}, got {ref.shape}."
                )

            if self.mode == "dexpilot":
                len_proj = len(self._projected)
                len_s2 = len(self._s2_task)
                len_s1 = len_proj - len_s2
                dist = np.linalg.norm(ref[:len_proj], axis=1)
                self._projected[:len_s1][dist[:len_s1] < self.project_dist] = True
                self._projected[:len_s1][dist[:len_s1] > self.escape_dist] = False
                if len_s2 > 0:
                    both = (
                        self._projected[:len_s1][self._s2_orig]
                        & self._projected[:len_s1][self._s2_task]
                        & (dist[len_s1:len_proj] <= _DEXPILOT_SECONDARY_PROJECT_DIST)
                    )
                    self._projected[len_s1:len_proj] = both
                high_w = np.array([200.0] * len_s1 + [400.0] * len_s2, dtype=np.float64)
                pair_w = np.where(self._projected, high_w, 1.0)
                num_f = len(self.tip_site_ids)
                weights = np.concatenate(
                    [pair_w, np.full(num_f, float(len_proj + num_f), dtype=np.float64)]
                )
                scaled = ref * self.scaling
                unit_dir = ref[:len_proj] / (dist[:, None] + 1e-6)
                proj_vec = unit_dir * self._proj_dist[:, None]
                target_vec = np.concatenate(
                    [
                        np.where(self._projected[:, None], proj_vec, scaled[:len_proj]),
                        scaled[len_proj:],
                    ],
                    axis=0,
                )
            else:
                weights = np.ones(n_vec, dtype=np.float64)
                target_vec = ref * self.scaling

            def obj_vec(x: np.ndarray) -> tuple[float, np.ndarray]:
                pos, jac = self._anchors_and_jacs(x, q_base)
                r_vec = pos[self._task_idx] - pos[self._origin_idx]
                err = r_vec - target_vec
                norms = np.linalg.norm(err, axis=1)
                h_val, h_grad = _huber_val_grad(norms, self.huber_delta)
                loss = float(
                    np.sum(h_val * weights) / n_vec
                    + self.norm_delta * np.sum((x - last_x) ** 2)
                )
                scale = (h_grad * weights / n_vec) / np.maximum(norms, 1e-8)
                grad_vec = err * scale[:, None]
                j_diff = jac[self._task_idx] - jac[self._origin_idx]
                grad_x = np.einsum(
                    "kc,kcd->d", grad_vec, j_diff
                ) + 2.0 * self.norm_delta * (x - last_x)
                return loss, grad_x

            res = minimize(
                obj_vec,
                last_x,
                method="SLSQP",
                jac=True,
                bounds=self.bounds,
                options={"maxiter": self.maxiter, "ftol": 1e-6},
            )

        x_opt = np.clip(
            np.asarray(res.x, dtype=np.float64),
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
        )
        self._last_x = x_opt
        q_out = q_base.copy()
        q_out[self.qpos_adrs] = x_opt
        if len(self.couplings.dep_qpos) > 0:
            project_joint_couplings(self.couplings, q_out)
        return q_out

    def retarget_keypoints(
        self,
        keypoints: np.ndarray,
        wrist_rot: np.ndarray | None = None,
        base_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget ``(21, 3)`` or ``(T, 21, 3)`` MANO keypoints to ``qpos``."""
        kp = np.asarray(keypoints, dtype=np.float64)
        single = kp.ndim == 2
        if single:
            kp = kp[None, ...]
        n_frames = len(kp)
        w_rot = None if wrist_rot is None else np.asarray(wrist_rot, dtype=np.float64)
        if w_rot is not None and w_rot.ndim == 2:
            w_rot = w_rot[None, ...]
        out = np.empty((n_frames, self.model.nq), dtype=np.float64)
        for t in range(n_frames):
            q_frame = (
                None
                if base_qpos is None
                else (base_qpos if base_qpos.ndim == 1 else base_qpos[t])
            )
            ref = self.keypoints_to_targets(
                kp[t], None if w_rot is None else w_rot[t], q_frame
            )
            out[t] = self.retarget_step(ref, q_frame)
        return out[0] if single else out


__all__ = [
    "MYOARM_HAND_PRESET",
    "MYOHAND_PRESET",
    "TIP_MANO_INDICES",
    "DexHandRetargeter",
    "HandPreset",
    "align_up_and_front",
    "hand_frame_from_sites",
    "mano_to_wrist_rot",
    "preset_for_model",
]
