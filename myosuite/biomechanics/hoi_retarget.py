# Copyright (c) MyoSuite Authors. All rights reserved.
# Portions adapted from embodied-ai-nexus/Eden (Copyright (c) 2026 Embodied AI Nexus /
# Genesis Authors, Apache-2.0).
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Hand-object interaction (HOI) trajectory utilities and contact-aware MuJoCo IK.

Provides:

1. :class:`HOIClip` container and trajectory transforms (:func:`translate_hoi_clip`,
   :func:`rotate_hoi_clip`, :func:`mirror_hoi_clip_lr`, :func:`trim_hoi_clip`,
   :func:`resample_hoi_clip`) built on :func:`myosuite.physics.quat.get_quat_ops`
   and :class:`scipy.spatial.transform.Slerp`.
2. :func:`retarget_hoi_mujoco`, which solves per-frame fingertip targets via
   :func:`myosuite.biomechanics.inverse_kinematics.solve_ik` (``mink`` differential
   QP IK) and optionally runs MuJoCo contact-settle substeps so fingers resolve
   penetration against manipulated objects.

Attribution:
    - The :class:`HOIClip` trajectory modifier conventions and contact-settle
      workflow are adapted from ``embodied-ai-nexus/Eden`` (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

from myosuite.biomechanics.hand_retarget import HandPreset, preset_for_model
from myosuite.biomechanics.inverse_kinematics import solve_ik
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.quat import get_quat_ops


@dataclass
class HOIClip:
    """Normalized hand-object interaction trajectory clip.

    Attributes:
        fps: Frame rate in Hz.
        keypoints: ``(T, 21, 3)`` canonical MANO keypoints in world frame.
        wrist_pos: ``(T, 3)`` wrist position in world frame.
        wrist_rot: ``(T, 3, 3)`` wrist rotation matrix in world frame.
        side: ``"right"`` or ``"left"``.
        object_pos: Optional ``(T, 3)`` manipulated object position in world frame.
        object_quat: Optional ``(T, 4)`` manipulated object quaternion ``(w, x, y, z)``.
    """

    fps: float
    keypoints: np.ndarray
    wrist_pos: np.ndarray
    wrist_rot: np.ndarray
    side: str = "right"
    object_pos: np.ndarray | None = None
    object_quat: np.ndarray | None = None


def translate_hoi_clip(
    clip: HOIClip, offset: tuple[float, float, float] | np.ndarray
) -> HOIClip:
    """Shift all world-frame positions of ``clip`` by ``offset``."""
    off = np.asarray(offset, dtype=np.float64).reshape(1, 3)
    return replace(
        clip,
        keypoints=clip.keypoints + off[:, None, :],
        wrist_pos=clip.wrist_pos + off,
        object_pos=None if clip.object_pos is None else clip.object_pos + off,
    )


def rotate_hoi_clip(
    clip: HOIClip,
    axis: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
    angle_deg: float = 0.0,
) -> HOIClip:
    """Rotate all world-frame positions and orientations about ``axis`` through origin."""
    ax_raw = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(ax_raw))
    if norm < 1e-10:
        raise ValueError(f"Expected non-zero rotation axis, got {axis}.")
    if angle_deg == 0.0:
        return clip

    qm = get_quat_ops(np)
    q_rot = qm.axis_angle2quat(ax_raw / norm, np.radians(angle_deg))
    r_mat = qm.quat2mat(q_rot)
    obj_q = (
        None
        if clip.object_quat is None
        else qm.mul_quat(
            np.broadcast_to(q_rot, clip.object_quat.shape), clip.object_quat
        )
    )
    return replace(
        clip,
        keypoints=clip.keypoints @ r_mat.T,
        wrist_pos=clip.wrist_pos @ r_mat.T,
        wrist_rot=np.einsum("ab,tbc->tac", r_mat, clip.wrist_rot),
        object_pos=None if clip.object_pos is None else clip.object_pos @ r_mat.T,
        object_quat=obj_q,
    )


def mirror_hoi_clip_lr(clip: HOIClip, axis: int = 1) -> HOIClip:
    """Mirror an HOI clip across world-frame axis ``axis`` (default ``1`` = Y) and swap side."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
    pos_sign = np.ones(3, dtype=np.float64)
    pos_sign[axis] = -1.0
    quat_sign = np.ones(4, dtype=np.float64)
    quat_sign[1:] = -pos_sign

    m_world = np.diag(pos_sign)
    m_local = np.diag([-1.0, 1.0, 1.0])
    wrist_rot = np.einsum("ab,tbc,cd->tad", m_world, clip.wrist_rot, m_local)

    return replace(
        clip,
        side="left" if clip.side == "right" else "right",
        keypoints=clip.keypoints * pos_sign[None, None, :],
        wrist_pos=clip.wrist_pos * pos_sign[None, :],
        wrist_rot=wrist_rot,
        object_pos=(
            None if clip.object_pos is None else clip.object_pos * pos_sign[None, :]
        ),
        object_quat=(
            None if clip.object_quat is None else clip.object_quat * quat_sign[None, :]
        ),
    )


def trim_hoi_clip(clip: HOIClip, start: int = 0, end: int | None = None) -> HOIClip:
    """Slice ``clip`` to frames ``[start:end]``."""
    sl = slice(start, end)
    return replace(
        clip,
        keypoints=clip.keypoints[sl].copy(),
        wrist_pos=clip.wrist_pos[sl].copy(),
        wrist_rot=clip.wrist_rot[sl].copy(),
        object_pos=None if clip.object_pos is None else clip.object_pos[sl].copy(),
        object_quat=None if clip.object_quat is None else clip.object_quat[sl].copy(),
    )


def resample_hoi_clip(
    clip: HOIClip, speed_scale: float = 1.0, out_fps: float | None = None
) -> HOIClip:
    """Resample ``clip`` for playback speed scaling and/or frame-rate conversion."""
    if speed_scale <= 0.0:
        raise ValueError(f"speed_scale must be positive, got {speed_scale}.")
    target_fps = clip.fps if out_fps is None else float(out_fps)
    n_in = len(clip.keypoints)
    if n_in <= 1:
        return replace(clip, fps=target_fps)

    duration_out = (n_in - 1) / (clip.fps * speed_scale)
    n_out = max(2, round(duration_out * target_fps) + 1)
    idx = np.linspace(0.0, float(n_in - 1), n_out)
    low = np.clip(np.floor(idx).astype(np.int64), 0, n_in - 2)
    frac = idx - low

    def _lerp(arr: np.ndarray) -> np.ndarray:
        f = frac.reshape((-1,) + (1,) * (arr.ndim - 1))
        return (1.0 - f) * arr[low] + f * arr[low + 1]

    times_in = np.arange(n_in, dtype=np.float64)
    w_rot_out = Slerp(times_in, Rotation.from_matrix(clip.wrist_rot))(idx).as_matrix()

    obj_q: np.ndarray | None = None
    if clip.object_quat is not None:
        obj_xyzw = Slerp(
            times_in, Rotation.from_quat(clip.object_quat[:, [1, 2, 3, 0]])
        )(idx).as_quat()
        obj_q = obj_xyzw[:, [3, 0, 1, 2]]
        for i in range(1, len(obj_q)):
            if float(np.dot(obj_q[i - 1], obj_q[i])) < 0.0:
                obj_q[i] = -obj_q[i]

    return HOIClip(
        fps=target_fps,
        keypoints=_lerp(clip.keypoints),
        wrist_pos=_lerp(clip.wrist_pos),
        wrist_rot=w_rot_out,
        side=clip.side,
        object_pos=None if clip.object_pos is None else _lerp(clip.object_pos),
        object_quat=obj_q,
    )


@dataclass(frozen=True)
class HoiPhysicsIKConfig:
    """Solver hyperparameters for :func:`retarget_hoi_mujoco`."""

    ik_tol: float = 1e-4
    ik_max_iters: int = 80
    sim_dt: float = 0.002
    settle_steps: int = 3
    smooth_window: int = 3
    object_solref: tuple[float, float] = (0.004, 1.0)
    servo_kp: float = 40.0


def retarget_hoi_mujoco(
    spec: mujoco.MjSpec,
    tip_targets: np.ndarray,
    *,
    preset: HandPreset | None = None,
    object_body: str | None = None,
    object_poses: np.ndarray | None = None,
    cfg: HoiPhysicsIKConfig = HoiPhysicsIKConfig(),
) -> dict[str, Any]:
    """Contact-aware MuJoCo HOI IK combining :func:`solve_ik` and contact settle.

    Solves per-frame fingertip site targets using :func:`solve_ik` (``mink`` QP IK
    on the model's joint limits and polynomial couplings), optionally steps a
    lightweight servo-held MuJoCo simulation for ``cfg.settle_steps`` substeps so
    contacts push fingers out of object penetration, and applies a zero-lag
    centered moving-average filter when ``cfg.smooth_window > 1``.

    Args:
        spec: ``MjSpec`` containing the hand/arm model (and optional free object).
        tip_targets: ``(T, 5, 3)`` world-frame fingertip target positions ordered
            ``(thumb, index, middle, ring, pinky)``.
        preset: Hand naming preset (auto-detected on compiled ``spec`` if ``None``).
        object_body: Name of a free-joint object body in ``spec``, if present.
        object_poses: ``(T, 7)`` object ``[x, y, z, qw, qx, qy, qz]`` poses.
        cfg: Solver configuration.

    Returns:
        Dict with ``qpos`` ``(T, nq)``, ``tip_error_mean_m``, and
        ``per_frame_tip_error_m``.
    """
    tips = np.asarray(tip_targets, dtype=np.float64)
    if tips.ndim != 3 or tips.shape[1:] != (5, 3):
        raise ValueError(f"Expected (T, 5, 3) tip_targets, got {tips.shape}.")
    n_frames = len(tips)

    work = spec.copy()
    probe_model = work.compile()
    hand = preset or preset_for_model(probe_model)
    probe_couplings = JointCouplings.from_model(probe_model)
    coupled_dep = set(probe_couplings.dep_qpos.tolist())

    for act in list(work.actuators):
        work.delete(act)

    candidate_joints = list(dict.fromkeys((*hand.wrist_joints, *hand.finger_joints)))
    ik_joints = [
        j
        for j in candidate_joints
        if mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0
    ]

    if cfg.settle_steps > 0:
        for jname in ik_joints:
            jid = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if int(probe_model.jnt_qposadr[jid]) in coupled_dep:
                continue
            act = work.add_actuator(
                name=f"ik_hold_{jname}",
                target=jname,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = cfg.servo_kp
            act.biasprm[1] = -cfg.servo_kp

        if object_body is not None and object_poses is not None:
            work.worldbody.add_body(name="_ik_target_obj", mocap=True)
            eq = work.add_equality(
                name="_ik_eq_obj",
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=object_body,
                name2="_ik_target_obj",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
            )
            eq.active = False
            eq.solref = list(cfg.object_solref)

    m = work.compile()
    m.opt.timestep = cfg.sim_dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    couplings = JointCouplings.from_model(m)
    if len(couplings.dep_qpos) > 0:
        project_joint_couplings(couplings, d.qpos)

    tip_site_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in hand.tip_sites
    ]
    act_qadr = (
        m.jnt_qposadr[m.actuator_trnid[:, 0]]
        if m.nu > 0
        else np.empty(0, dtype=np.int64)
    )

    obj_poses_arr: np.ndarray | None = None
    obj_mocap_id = -1
    obj_eq_id = -1
    obj_qpos_adr = -1
    if object_body is not None and object_poses is not None:
        obj_poses_arr = np.asarray(object_poses, dtype=np.float64)
        if obj_poses_arr.shape != (n_frames, 7):
            raise ValueError(
                f"Expected ({n_frames}, 7) object_poses, got {obj_poses_arr.shape}."
            )
        obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if obj_bid < 0:
            raise ValueError(f"Object body {object_body!r} not found in spec.")
        if m.body_jntnum[obj_bid] > 0:
            obj_qpos_adr = int(m.jnt_qposadr[m.body_jntadr[obj_bid]])
            d.qpos[obj_qpos_adr : obj_qpos_adr + 7] = obj_poses_arr[0]
        if cfg.settle_steps > 0:
            obj_mocap_id = int(
                m.body_mocapid[
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "_ik_target_obj")
                ]
            )
            obj_eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "_ik_eq_obj")

    prev_qpos = d.qpos.copy()
    qpos_traj = np.empty((n_frames, m.nq), dtype=np.float64)
    errs = np.empty(n_frames, dtype=np.float64)

    for t in range(n_frames):
        seed = prev_qpos.copy()
        if obj_qpos_adr >= 0 and obj_poses_arr is not None:
            seed[obj_qpos_adr : obj_qpos_adr + 7] = obj_poses_arr[t]

        targets = {site_name: tips[t, i] for i, site_name in enumerate(hand.tip_sites)}
        q_ik = solve_ik(
            m,
            targets,
            joints=ik_joints,
            qpos=seed,
            strict=False,
            retry=(t == 0),
            tol=cfg.ik_tol,
            max_iters=cfg.ik_max_iters,
        )
        if len(couplings.dep_qpos) > 0:
            project_joint_couplings(couplings, q_ik)

        d.qpos[:] = q_ik
        d.qvel[:] = 0.0
        if cfg.settle_steps > 0:
            if m.nu > 0:
                d.ctrl[:] = q_ik[act_qadr]
            if obj_mocap_id >= 0 and obj_poses_arr is not None:
                d.mocap_pos[obj_mocap_id] = obj_poses_arr[t, :3]
                d.mocap_quat[obj_mocap_id] = obj_poses_arr[t, 3:7]
                d.eq_active[obj_eq_id] = 1
            for _ in range(cfg.settle_steps):
                mujoco.mj_step(m, d)
            if obj_eq_id >= 0:
                d.eq_active[obj_eq_id] = 0
            if len(couplings.dep_qpos) > 0:
                project_joint_couplings(couplings, d.qpos)

        mujoco.mj_forward(m, d)
        errs[t] = float(
            np.mean(
                [
                    np.linalg.norm(d.site_xpos[sid] - tips[t, i])
                    for i, sid in enumerate(tip_site_ids)
                ]
            )
        )
        qpos_traj[t] = d.qpos
        prev_qpos = d.qpos.copy()

    if cfg.smooth_window > 1 and n_frames > cfg.smooth_window:
        smoothed = uniform_filter1d(
            qpos_traj, size=cfg.smooth_window, axis=0, mode="nearest"
        )
        if obj_qpos_adr >= 0:
            q_slice = slice(obj_qpos_adr + 3, obj_qpos_adr + 7)
            smoothed[:, q_slice] /= np.linalg.norm(
                smoothed[:, q_slice], axis=-1, keepdims=True
            )
        if len(couplings.dep_qpos) > 0:
            for t in range(n_frames):
                project_joint_couplings(couplings, smoothed[t])
        qpos_traj = smoothed

    return {
        "qpos": qpos_traj.astype(np.float32),
        "tip_error_mean_m": float(errs.mean()),
        "per_frame_tip_error_m": errs.astype(np.float32),
    }


__all__ = [
    "HOIClip",
    "HoiPhysicsIKConfig",
    "mirror_hoi_clip_lr",
    "resample_hoi_clip",
    "retarget_hoi_mujoco",
    "rotate_hoi_clip",
    "translate_hoi_clip",
    "trim_hoi_clip",
]
