# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Boxing demo renderer: right hand → body capsule, then → head sphere.

By default a **single** BOB is used: one opponent in front of the actor in a
custom scripted scene.

Pass ``--dual-opponent`` to add a second BOB on the left (render demo only) and
run the original three-phase script: left jab → left BOB head, right hook →
right BOB body, right jab → right BOB head.

  Phase A — RIGHT HOOK → BODY capsule:
      Right arm hook (shoulder_elv=0.60, rot=0.25, flex=0.70) drives the right
      fist into the torso capsule at Z≈0.95.

  Phase B — RIGHT JAB → HEAD sphere:
      Right jab (shoulder_elv=1.10, rot=0.30, flex=0.80) hits the head sphere.

  (``--dual-opponent`` only) Phase L — LEFT JAB → left BOB head at ``_LEFT_OPP_POS``.

Usage::

    uv run python scripts/boxing_render.py [--output boxing_demo.mp4] [--fps 30]
    uv run python scripts/boxing_render.py --dual-opponent   # two BOBs

Two-agent Gymnasium env (random muscle actions, live MuJoCo viewer)::

    uv run python scripts/boxing_render.py --boxing-vs
    uv run python scripts/boxing_render.py --boxing-vs --vs-episodes 5 --vs-seed 42

``myoChallengeBoxingVs-v0`` with **agent_1** driven by ``mannequin_exact_clone.npz``
(same scene as true 1v1; you control only **agent_0**).  Episodes end if **either**
fighter falls or is KO'd — random muscle noise on agent_0 usually makes **you**
(red) collapse first; use ``--vs-clone-player half`` for a standing partner so the
clone (blue) is visible longer::

    uv run python scripts/boxing_render.py --boxing-vs-clone --vs-clone-player half
    uv run python scripts/boxing_render.py --record-boxing-vs-clone --output boxing_vs_clone.mp4 --steps 900

Optional custom policy: ``--clone-policy /path/to/policy.npz``.

On macOS, if the viewer fails to open, use ``mjpython`` instead of ``python``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.utils.video_io import write_video

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Joint qadr indices (compile_mimic_fullbody_mjmodel)
# --------------------------------------------------------------------------- #
# Right arm
_R_ELV = 36  # shoulder_elv_r
_R_ROT = 38  # shoulder_rot_r
_R_FLEX = 39  # elbow_flex_r

# Left arm
_L_ELV = 54  # shoulder_elv_l
_L_ROT = 56  # shoulder_rot_l
_L_FLEX = 57  # elbow_flex_l

# Neutral (qpos0 values = 0 for all driven joints)
_NEUTRAL_R = {_R_ELV: 0.0, _R_ROT: 0.0, _R_FLEX: 0.0}
_NEUTRAL_L = {_L_ELV: 0.0, _L_ROT: 0.0, _L_FLEX: 0.0}

# Calibrated punch poses (verified to place fist within reach of target zone)
_POSE_L_HEAD = {_L_ELV: 0.95, _L_ROT: 0.20, _L_FLEX: 1.10}  # left jab → head sphere
_POSE_R_BODY = {_R_ELV: 0.60, _R_ROT: 0.25, _R_FLEX: 0.70}  # right hook → body capsule
_POSE_R_HEAD = {_R_ELV: 1.10, _R_ROT: 0.30, _R_FLEX: 0.80}  # right jab → head sphere

# Left-side mirrored opponent (render demo only — not part of the env task)
_LEFT_OPP_POS = [+0.40, -0.15, 0.0]

HIT_THRESHOLD = 0.15  # metres — default boxing target proximity threshold

_PUNCHING_BAG_OBJ = (
    Path(__file__).resolve().parent.parent
    / "myosuite"
    / "envs"
    / "myo"
    / "assets"
    / "PunchingBag.obj"
)
_MESH_FACE_ACTOR_QUAT = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
_MESH_FACE_ACTOR_MIRROR_QUAT = [-0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
_OPPONENT_POS = [-0.035, -0.33, 0.0]
_OPPONENT_MASS = 5000.0
_HEAD_COL_Z = 1.25
_HEAD_COL_R = 0.13
_BODY_COL_Z = 0.95
_BODY_COL_R = 0.135
_BODY_COL_H = 0.30


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _add_boxing_fullbody_scene(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Attach a single BOB opponent to the fullbody spec."""
    fist_body = spec.body("3proxph_r")
    fist_body.add_site(name="fist_r", pos=[0.0, 0.0, 0.0], size=[0.015, 0.0, 0.0])
    left_fist_body = spec.body("3proxph_l")
    left_fist_body.add_site(name="fist_l", pos=[0.0, 0.0, 0.0], size=[0.015, 0.0, 0.0])

    mesh = spec.add_mesh()
    mesh.name = "punching_bag_mesh"
    mesh.file = str(_PUNCHING_BAG_OBJ)
    mesh.scale = [1.0, 1.0, 1.0]
    mesh.refpos = [0.0, 0.0, 0.0]

    mat = spec.add_material()
    mat.name = "punching_bag_mat"
    mat.rgba = [0.65, 0.18, 0.08, 1.0]
    mat.shininess = 0.3
    mat.specular = 0.2

    opponent = spec.worldbody.add_body(name="opponent", pos=_OPPONENT_POS)
    opponent.gravcomp = 1.0
    opponent.add_joint(name="opponent_free", type=mujoco.mjtJoint.mjJNT_FREE)
    opponent.add_geom(
        name="opponent_inertia",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.05],
        size=[0.05, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
        mass=_OPPONENT_MASS,
        contype=0,
        conaffinity=0,
    )
    visual_geom = opponent.add_geom(
        name="opponent_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        contype=0,
        conaffinity=0,
    )
    visual_geom.meshname = "punching_bag_mesh"
    visual_geom.material = "punching_bag_mat"
    visual_geom.quat = _MESH_FACE_ACTOR_QUAT
    head_col = opponent.add_geom(
        name="opponent_head_col",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, _HEAD_COL_Z],
        size=[_HEAD_COL_R, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
        mass=0.1,
        contype=1,
        conaffinity=1,
    )
    head_col.quat = _MESH_FACE_ACTOR_QUAT
    head_col.condim = 4
    body_col = opponent.add_geom(
        name="opponent_body_col",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=[0.0, 0.0, _BODY_COL_Z],
        size=[_BODY_COL_R, _BODY_COL_H, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
        mass=0.1,
        contype=1,
        conaffinity=1,
    )
    body_col.quat = _MESH_FACE_ACTOR_QUAT
    body_col.condim = 4
    head_site = opponent.add_site(
        name="bag_head_zone", pos=[0.0, 0.0, _HEAD_COL_Z], size=[0.02, 0.0, 0.0]
    )
    head_site.quat = _MESH_FACE_ACTOR_QUAT
    body_site = opponent.add_site(
        name="bag_body_zone", pos=[0.0, 0.0, _BODY_COL_Z], size=[0.02, 0.0, 0.0]
    )
    body_site.quat = _MESH_FACE_ACTOR_QUAT
    return spec


def _interp_pose(
    qpos: np.ndarray,
    start: dict[int, float],
    end: dict[int, float],
    t: float,
) -> None:
    """Cosine ease-in-out interpolation of arm joint angles (in-place)."""
    alpha = 0.5 * (1.0 - np.cos(np.pi * t))
    for idx in set(start) | set(end):
        s = start.get(idx, float(qpos[idx]))
        e = end.get(idx, float(qpos[idx]))
        qpos[idx] = s + alpha * (e - s)


def _check_contacts(
    data: mujoco.MjData,
    hand_body_ids: set[int],
    opponent_body_id: int,
) -> bool:
    """Return True if any contact links a hand geom to the opponent body."""
    for k in range(data.ncon):
        c = data.contact[k]
        b1 = int(data.model.geom_bodyid[int(c.geom1)])
        b2 = int(data.model.geom_bodyid[int(c.geom2)])
        if (b1 == opponent_body_id and b2 in hand_body_ids) or (
            b2 == opponent_body_id and b1 in hand_body_ids
        ):
            return True
    return False


def _resolve_hand_bodies(model: mujoco.MjModel, suffixes: list[str]) -> set[int]:
    """Collect body IDs for all hand bodies matching the given name suffixes."""
    names = [
        "radius",
        "lunate",
        "scaphoid",
        "pisiform",
        "triquetrum",
        "capitate",
        "trapezium",
        "trapezoid",
        "firstmc",
        "secondmc",
        "2proxph",
        "2distph",
        "3proxph",
        "3midph",
        "3distph",
        "4proxph",
        "5proxph",
    ]
    ids: set[int] = set()
    for base in names:
        for sfx in suffixes:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{base}{sfx}")
            if bid != -1:
                ids.add(bid)
    return ids


def _add_left_opponent(spec: mujoco.MjSpec) -> None:
    """Inject a mirrored left-side BOB opponent into *spec* (render demo only).

    The mesh asset ``punching_bag_mesh`` and material ``punching_bag_mat`` are
    already registered by :func:`_add_boxing_fullbody_scene`; this function
    re-uses them and only adds the body + geoms.

    Zones:
        ``left_opp_head_zone`` — head sphere at Z=1.25 (same height as right)
    """
    left_opp = spec.worldbody.add_body(name="left_opponent", pos=_LEFT_OPP_POS)
    left_opp.gravcomp = 1.0
    left_opp.add_joint(name="left_opponent_free", type=mujoco.mjtJoint.mjJNT_FREE)

    # Inertia anchor (same huge mass — immovable)
    left_opp.add_geom(
        name="left_opp_inertia",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.05],
        size=[0.05, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
        mass=_OPPONENT_MASS,
        contype=0,
        conaffinity=0,
    )

    # Visual mesh (re-uses existing punching_bag_mesh + punching_bag_mat)
    vg = left_opp.add_geom(
        name="left_opp_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        contype=0,
        conaffinity=0,
    )
    vg.meshname = "punching_bag_mesh"
    vg.quat = _MESH_FACE_ACTOR_MIRROR_QUAT
    vg.material = "punching_bag_mat"

    # Head collision sphere
    hc = left_opp.add_geom(
        name="left_opp_head_col",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, _HEAD_COL_Z],
        size=[_HEAD_COL_R, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
        mass=0.1,
        contype=1,
        conaffinity=1,
    )
    hc.quat = _MESH_FACE_ACTOR_MIRROR_QUAT
    hc.condim = 4

    lz = left_opp.add_site(
        name="left_opp_head_zone",
        pos=[0.0, 0.0, _HEAD_COL_Z],
        size=[0.02, 0.0, 0.0],
    )
    lz.quat = _MESH_FACE_ACTOR_MIRROR_QUAT


def _add_overlay(
    frame: np.ndarray,
    hits: dict[str, int],
    contacts: dict[str, bool],
    dists: dict[str, float],
) -> np.ndarray:
    """Draw hit counters and proximity distances onto *frame*.  No-op if cv2 absent."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return frame

    frame = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (8, 8), (380, 130), (20, 20, 20), -1)

    rows = [
        ("L HEAD  (sphere )", "l_head", (100, 200, 255)),
        ("R BODY  (capsule)", "r_body", (100, 255, 160)),
        ("R HEAD  (sphere )", "r_head", (255, 200, 80)),
    ]
    for i, (label, key, color) in enumerate(rows):
        active = hits[key] > 0 or contacts.get(key, False)
        col = color if active else (120, 120, 120)
        d = dists.get(key, 0.0)
        text = f"{label}: {hits[key]:3d}  d={d:.3f}m  {'CONTACT' if contacts.get(key) else ''}"
        cv2.putText(
            frame,
            text,
            (14, 32 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            col,
            1,
            cv2.LINE_AA,
        )

    if any(v > 0 for v in hits.values()):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 200, 200), 4)
    return frame


# --------------------------------------------------------------------------- #
# Main demo
# --------------------------------------------------------------------------- #


def run_demo(
    output: str = "boxing_demo.mp4",
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    *,
    dual_opponent: bool = False,
    cam_azimuth: float | None = None,
    cam_elevation: float | None = None,
    cam_distance: float | None = None,
    cam_lookat: tuple[float, float, float] | None = None,
) -> Path:
    """Render boxing demo and save as MP4.

    Args:
        output: Output file path.
        fps: Frames per second.
        width: Render width in pixels.
        height: Render height in pixels.
        dual_opponent: If True, add a second BOB for the left-arm head phase.

    Returns:
        Path to the saved MP4 file.
    """
    from myosuite.envs.modular_env import _build_model_from_task_model

    base_model, base_spec = _build_model_from_task_model("musclemimic_fullbody")
    spec = _add_boxing_fullbody_scene(base_spec)
    if dual_opponent:
        _add_left_opponent(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    # ---- Resolve site IDs ----
    fist_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fist_r")
    fist_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fist_l")
    r_head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bag_head_zone")
    r_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bag_body_zone")
    l_head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_opp_head_zone")

    # ---- Body/geom IDs for contact detection ----
    r_opp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "opponent")
    l_opp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_opponent")

    # ---- Material IDs for hit flash ----
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "punching_bag_mat")
    mat_rgba_orig = model.mat_rgba[mat_id].copy()
    mat_rgba_head = np.array([0.9, 0.15, 0.10, 1.0], dtype=np.float32)
    mat_rgba_body = np.array([0.2, 0.80, 0.25, 1.0], dtype=np.float32)
    mat_rgba_contact = np.array([1.0, 0.40, 0.00, 1.0], dtype=np.float32)

    # ---- freejoint qadr for each opponent (save/restore during kinematics) ----
    def _opp_qadr(name: str) -> tuple[int, int]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    r_qadr, r_dadr = _opp_qadr("opponent_free")
    if dual_opponent:
        l_qadr, l_dadr = _opp_qadr("left_opponent_free")
        n_body_qpos = min(r_qadr, l_qadr)  # qpos before first opponent freejoint
    else:
        l_qadr = l_dadr = -1
        n_body_qpos = r_qadr

    right_hand_ids = _resolve_hand_bodies(model, ["_r"])
    left_hand_ids = _resolve_hand_bodies(model, ["_l"])

    logger.info(
        "Model: nq=%d nv=%d nu=%d  r_qadr=%d dual=%s",
        model.nq,
        model.nv,
        model.nu,
        r_qadr,
        dual_opponent,
    )

    # ---- Camera ----
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if cam_lookat is not None:
        cam.lookat[:] = list(cam_lookat)
    elif dual_opponent:
        cam.lookat[:] = [0.0, -0.10, 1.1]
        cam.azimuth = 180.0  # face-on: actor between the two BOBs
    else:
        # Single BOB sits at negative X — bias lookat so bag + torso stay in frame
        cam.lookat[:] = [-0.35, -0.10, 1.05]
        cam.azimuth = 168.0
    cam.distance = 2.8 if cam_distance is None else float(cam_distance)
    cam.elevation = -8.0 if cam_elevation is None else float(cam_elevation)
    if cam_azimuth is not None:
        cam.azimuth = float(cam_azimuth)

    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ---- Animation phases ----
    # (label, n_frames, right_arm_start, right_arm_end, left_arm_start, left_arm_end, dynamics)
    _phases_dual = [
        ("neutral_1", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
        (
            "l_windup",
            20,
            _NEUTRAL_R,
            _NEUTRAL_R,
            _NEUTRAL_L,
            {**_NEUTRAL_L, _L_ELV: 0.3},
            True,
        ),
        (
            "l_jab_extend",
            35,
            _NEUTRAL_R,
            _NEUTRAL_R,
            {**_NEUTRAL_L, _L_ELV: 0.3},
            _POSE_L_HEAD,
            True,
        ),
        (
            "l_hold_contact",
            30,
            _NEUTRAL_R,
            _NEUTRAL_R,
            _POSE_L_HEAD,
            _POSE_L_HEAD,
            True,
        ),
        ("l_retract", 25, _NEUTRAL_R, _NEUTRAL_R, _POSE_L_HEAD, _NEUTRAL_L, True),
        ("neutral_2", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
        (
            "r_hook_windup",
            20,
            _NEUTRAL_R,
            {**_NEUTRAL_R, _R_ELV: 0.2},
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        (
            "r_hook_extend",
            35,
            {**_NEUTRAL_R, _R_ELV: 0.2},
            _POSE_R_BODY,
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        ("r_hold_body", 30, _POSE_R_BODY, _POSE_R_BODY, _NEUTRAL_L, _NEUTRAL_L, True),
        ("r_retract_body", 25, _POSE_R_BODY, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, True),
        ("neutral_3", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
        (
            "r_jab_windup",
            20,
            _NEUTRAL_R,
            {**_NEUTRAL_R, _R_ELV: 0.3},
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        (
            "r_jab_extend",
            35,
            {**_NEUTRAL_R, _R_ELV: 0.3},
            _POSE_R_HEAD,
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        ("r_hold_head", 30, _POSE_R_HEAD, _POSE_R_HEAD, _NEUTRAL_L, _NEUTRAL_L, True),
        ("r_retract_head", 25, _POSE_R_HEAD, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, True),
        ("neutral_4", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
    ]
    _phases_single = [
        ("neutral_1", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
        (
            "r_hook_windup",
            20,
            _NEUTRAL_R,
            {**_NEUTRAL_R, _R_ELV: 0.2},
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        (
            "r_hook_extend",
            35,
            {**_NEUTRAL_R, _R_ELV: 0.2},
            _POSE_R_BODY,
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        ("r_hold_body", 30, _POSE_R_BODY, _POSE_R_BODY, _NEUTRAL_L, _NEUTRAL_L, True),
        ("r_retract_body", 25, _POSE_R_BODY, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, True),
        ("neutral_2", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
        (
            "r_jab_windup",
            20,
            _NEUTRAL_R,
            {**_NEUTRAL_R, _R_ELV: 0.3},
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        (
            "r_jab_extend",
            35,
            {**_NEUTRAL_R, _R_ELV: 0.3},
            _POSE_R_HEAD,
            _NEUTRAL_L,
            _NEUTRAL_L,
            True,
        ),
        ("r_hold_head", 30, _POSE_R_HEAD, _POSE_R_HEAD, _NEUTRAL_L, _NEUTRAL_L, True),
        ("r_retract_head", 25, _POSE_R_HEAD, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, True),
        ("neutral_3", 15, _NEUTRAL_R, _NEUTRAL_R, _NEUTRAL_L, _NEUTRAL_L, False),
    ]
    PHASES = _phases_dual if dual_opponent else _phases_single

    hits = {"l_head": 0, "r_body": 0, "r_head": 0}
    frames: list[np.ndarray] = []

    dict(_NEUTRAL_R)
    dict(_NEUTRAL_L)

    for phase_name, n_frames, rs, re, ls, le, use_dynamics in PHASES:
        logger.info("Phase: %s (%d frames, dyn=%s)", phase_name, n_frames, use_dynamics)
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)

            target_r_full = np.zeros(model.nq)
            target_l_full = np.zeros(model.nq)
            for k, v in rs.items():
                e = re.get(k, float(data.qpos[k]))
                s = rs.get(k, float(data.qpos[k]))
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                target_r_full[k] = s + alpha * (e - s)
            for k, v in ls.items():
                e = le.get(k, float(data.qpos[k]))
                s = ls.get(k, float(data.qpos[k]))
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                target_l_full[k] = s + alpha * (e - s)

            # Save opponent states
            r_qsave = data.qpos[r_qadr : r_qadr + 7].copy()
            r_vsave = data.qvel[r_dadr : r_dadr + 6].copy()
            if dual_opponent:
                l_qsave = data.qpos[l_qadr : l_qadr + 7].copy()
                l_vsave = data.qvel[l_dadr : l_dadr + 6].copy()

            if use_dynamics:
                dt = float(model.opt.timestep) * 5
                for k in rs:
                    dofadr = int(
                        model.jnt_dofadr[
                            mujoco.mj_name2id(
                                model, mujoco.mjtObj.mjOBJ_JOINT, _QADR_TO_JOINT_R[k]
                            )
                        ]
                    )
                    data.qvel[dofadr] = (target_r_full[k] - float(data.qpos[k])) / dt
                for k in ls:
                    dofadr = int(
                        model.jnt_dofadr[
                            mujoco.mj_name2id(
                                model, mujoco.mjtObj.mjOBJ_JOINT, _QADR_TO_JOINT_L[k]
                            )
                        ]
                    )
                    data.qvel[dofadr] = (target_l_full[k] - float(data.qpos[k])) / dt
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data, 5)

            # Pin body to qpos0, set arm targets, restore opponents
            data.qpos[:n_body_qpos] = model.qpos0[:n_body_qpos]
            for k in _QADR_TO_JOINT_R:
                data.qpos[k] = (
                    target_r_full[k]
                    if target_r_full[k] != 0 or k in re
                    else model.qpos0[k]
                )
            for k in _QADR_TO_JOINT_L:
                data.qpos[k] = (
                    target_l_full[k]
                    if target_l_full[k] != 0 or k in le
                    else model.qpos0[k]
                )
            data.qpos[r_qadr : r_qadr + 7] = r_qsave
            data.qvel[r_dadr : r_dadr + 6] = r_vsave
            if dual_opponent:
                data.qpos[l_qadr : l_qadr + 7] = l_qsave
                data.qvel[l_dadr : l_dadr + 6] = l_vsave
            mujoco.mj_forward(model, data)

            # ---- Hit detection ----
            fp_r = data.site_xpos[fist_r_id]
            fp_l = data.site_xpos[fist_l_id]

            d_r_body = float(np.linalg.norm(data.site_xpos[r_body_id] - fp_r))
            d_r_head = float(np.linalg.norm(data.site_xpos[r_head_id] - fp_r))
            if dual_opponent and l_head_id >= 0 and l_opp_id >= 0:
                d_l_head = float(np.linalg.norm(data.site_xpos[l_head_id] - fp_l))
                prox_l_head = d_l_head < HIT_THRESHOLD
                con_l_head = _check_contacts(data, left_hand_ids, l_opp_id)
            else:
                d_l_head = 1e6
                prox_l_head = False
                con_l_head = False

            prox_r_body = d_r_body < HIT_THRESHOLD
            prox_r_head = d_r_head < HIT_THRESHOLD

            con_r_body = _check_contacts(data, right_hand_ids, r_opp_id)
            con_r_head = _check_contacts(data, right_hand_ids, r_opp_id)

            if prox_l_head:
                hits["l_head"] += 1
            if prox_r_body:
                hits["r_body"] += 1
            if prox_r_head:
                hits["r_head"] += 1

            # ---- Material flash (shared punching_bag_mat on all BOB meshes) ----
            if con_r_body or con_r_head or con_l_head:
                model.mat_rgba[mat_id] = mat_rgba_contact
            elif prox_r_body:
                model.mat_rgba[mat_id] = mat_rgba_body
            elif prox_l_head or prox_r_head:
                model.mat_rgba[mat_id] = mat_rgba_head
            else:
                model.mat_rgba[mat_id] = mat_rgba_orig

            # ---- Render ----
            renderer.update_scene(data, camera=cam)
            frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
            frame = _add_overlay(
                frame,
                hits,
                {"l_head": con_l_head, "r_body": con_r_body, "r_head": con_r_head},
                {"l_head": d_l_head, "r_body": d_r_body, "r_head": d_r_head},
            )
            frames.append(frame)

    renderer.close()

    out_path = Path(output)
    write_video(
        str(out_path),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": str(fps), "-vcodec": "libx264", "-pix_fmt": "yuv420p"},
    )

    logger.info(
        "Saved %d frames → %s  l_head=%d r_body=%d r_head=%d",
        len(frames),
        out_path,
        hits["l_head"],
        hits["r_body"],
        hits["r_head"],
    )
    return out_path


# --------------------------------------------------------------------------- #
# Gymnasium: myoChallengeBoxingVs-v0 (two MuscleMimic agents)
# --------------------------------------------------------------------------- #

BOXING_VS_DEFAULT_ID = "myoChallengeBoxingVs-v0"
BOXING_VS_CLONE_DEFAULT_ID = "myoChallengeBoxingVsClone-v0"


def _sample_boxing_vs_actions(
    action_space: Any,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample a valid dict action for ``BoxingVsEnv`` (one Box per agent)."""
    actions: dict[str, np.ndarray] = {}
    for agent_id, space in action_space.items():
        actions[agent_id] = rng.uniform(
            np.asarray(space.low, dtype=np.float32),
            np.asarray(space.high, dtype=np.float32),
            size=space.shape,
        ).astype(np.float32)
    return actions


def play_boxing_vs(
    *,
    env_id: str = BOXING_VS_DEFAULT_ID,
    seed: int | None = None,
    max_episodes: int = 20,
    render_mode: str = "human",
    policy_path: str | Path | None = None,
    vs_clone_player: str = "half",
) -> None:
    """Run a registered boxing env with random actions and on-screen rendering.

    This is for visually inspecting a registered boxing scene. Supports both:

    * multi-agent dict spaces (e.g. ``myoChallengeBoxingVs-v0``)
    * single-agent box spaces (e.g. ``myoChallengeBoxingP0-v0``,
      ``myoChallengeBoxingVsClone-v0``)

    Actions are independent uniform samples per muscle (same idea as
    ``examine_env``).

    Args:
        env_id: Registered Gymnasium id (default two-agent boxing vs).
        seed: If set, passed to the first :meth:`~gymnasium.Env.reset` and used
            for action sampling for all episodes.
        max_episodes: Number of full episodes before returning (use Ctrl+C to stop).
        render_mode: ``human`` for interactive viewer, ``rgb_array`` for off-screen
            frames (no interactive window).
        policy_path: If set, passed to ``gym.make`` (e.g. ``mannequin_exact_clone.npz``
            for ``myoChallengeBoxingVsClone-v0``).
        vs_clone_player: For ``myoChallengeBoxingVsClone-v0`` only: ``half`` (stable
            mid-muscle) or ``random`` for the controlled agent's actions.
    """
    from myosuite import register_all_envs

    register_all_envs()
    import gymnasium as gym

    make_kw: dict[str, Any] = {"render_mode": render_mode, "disable_env_checker": True}
    if policy_path is not None:
        make_kw["policy_path"] = Path(policy_path)
    env = gym.make(env_id, **make_kw)
    rng = np.random.default_rng(seed)
    try:
        for ep in range(max_episodes):
            _obs, _info = env.reset(seed=seed if ep == 0 else None)
            done = False
            n_steps = 0
            while not done:
                if env_id == BOXING_VS_CLONE_DEFAULT_ID:
                    actions = _player_action_for_vs_clone_record(
                        env.action_space, rng, vs_clone_player
                    )
                else:
                    actions = _sample_action_for_space(env.action_space, rng)
                _obs, rewards, terminated, truncated, _info = env.step(actions)
                if isinstance(terminated, dict):
                    done = any(terminated.values()) or any(truncated.values())
                else:
                    done = bool(terminated or truncated)
                n_steps += 1
            if isinstance(rewards, dict):
                reward_log: Any = {k: float(v) for k, v in rewards.items()}
            else:
                reward_log = float(rewards)
            logger.info(
                "Episode %d finished in %d steps | rewards=%s",
                ep,
                n_steps,
                reward_log,
            )
    finally:
        env.close()


def _sample_action_for_space(
    action_space: Any, rng: np.random.Generator
) -> np.ndarray | dict[str, np.ndarray]:
    """Sample random actions for Box or dict-style action spaces."""
    if isinstance(action_space, dict):
        return _sample_boxing_vs_actions(action_space, rng)
    if hasattr(action_space, "spaces"):
        spaces = getattr(action_space, "spaces")
        if isinstance(spaces, dict):
            return _sample_boxing_vs_actions(spaces, rng)
    return rng.uniform(
        np.asarray(action_space.low, dtype=np.float32),
        np.asarray(action_space.high, dtype=np.float32),
        size=action_space.shape,
    ).astype(np.float32)


def _player_action_for_vs_clone_record(
    action_space: Any,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray | dict[str, np.ndarray]:
    """Player-side action for single-agent BoxingVsClone (Box space only).

    Args:
        action_space: Gymnasium space for the controlled agent.
        rng: RNG (used when ``mode`` is ``\"random\"``).
        mode: ``\"random\"`` — uniform in the action box; ``\"half\"`` — constant
            mid-range activation (stable stance for demos / MP4).

    Returns:
        Action array or dict (dict path falls back to random sampling).
    """
    if mode not in ("random", "half"):
        raise ValueError(f"mode must be 'random' or 'half', got {mode!r}")
    if isinstance(action_space, dict) or (
        hasattr(action_space, "spaces")
        and isinstance(getattr(action_space, "spaces", None), dict)
    ):
        return _sample_action_for_space(action_space, rng)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if mode == "half":
        return (0.5 * (low + high)).astype(np.float32)
    return _sample_action_for_space(action_space, rng)


def render_registered_env(
    *,
    env_id: str,
    output: str,
    fps: int,
    width: int,
    height: int,
    steps: int,
    seed: int | None,
    cam_azimuth: float = 165.0,
    cam_elevation: float = -14.0,
    cam_distance: float | None = None,
    cam_lookat: tuple[float, float, float] | None = None,
    policy_path: str | Path | None = None,
    vs_clone_player: str = "half",
) -> Path:
    """Roll out a registered env with random (or fixed) actions and save MP4."""
    from myosuite import register_all_envs

    register_all_envs()
    import gymnasium as gym

    make_kw: dict[str, Any] = {"disable_env_checker": True}
    if policy_path is not None:
        make_kw["policy_path"] = Path(policy_path)
    env = gym.make(env_id, **make_kw)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    uw = env.unwrapped
    model = uw.model
    data = uw.data

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if cam_lookat is None:
        cam.lookat[:] = np.asarray(model.stat.center, dtype=np.float64)
    else:
        cam.lookat[:] = list(cam_lookat)
    cam.distance = (
        float(model.stat.extent) * 0.7 if cam_distance is None else float(cam_distance)
    )
    cam.azimuth = float(cam_azimuth)
    cam.elevation = float(cam_elevation)

    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    n_done = 0
    try:
        for _ in range(steps):
            if env_id == BOXING_VS_CLONE_DEFAULT_ID:
                action = _player_action_for_vs_clone_record(
                    env.action_space, rng, vs_clone_player
                )
            else:
                action = _sample_action_for_space(env.action_space, rng)
            _obs, _rwd, terminated, truncated, _info = env.step(action)

            if isinstance(terminated, dict):
                step_done = any(bool(v) for v in terminated.values()) or any(
                    bool(v) for v in truncated.values()
                )
            else:
                step_done = bool(terminated) or bool(truncated)
            if step_done:
                n_done += 1
                env.reset()

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
    finally:
        renderer.close()
        env.close()

    out_path = Path(output)
    write_video(
        str(out_path),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": str(fps), "-vcodec": "libx264", "-pix_fmt": "yuv420p"},
    )

    logger.info(
        "Saved %d frames → %s  env_id=%s resets=%d",
        len(frames),
        out_path,
        env_id,
        n_done,
    )
    return out_path


# Joint name maps for dofadr lookup
_QADR_TO_JOINT_R: dict[int, str] = {
    _R_ELV: "shoulder_elv_r",
    _R_ROT: "shoulder_rot_r",
    _R_FLEX: "elbow_flex_r",
}
_QADR_TO_JOINT_L: dict[int, str] = {
    _L_ELV: "shoulder_elv_l",
    _L_ROT: "shoulder_rot_l",
    _L_FLEX: "elbow_flex_l",
}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--boxing-vs",
        action="store_true",
        help=(
            "Play the registered two-agent env (see --vs-env-id) with random "
            "actions instead of exporting the scripted BOB MP4 demo."
        ),
    )
    p.add_argument(
        "--boxing-vs-clone",
        action="store_true",
        help=(
            "Play myoChallengeBoxingVsClone-v0: same 1v1 arena as BoxingVs, but "
            "agent_1 is driven by mannequin_exact_clone.npz (see --clone-policy)."
        ),
    )
    p.add_argument(
        "--record-boxing-vs-clone",
        action="store_true",
        help=(
            "Write an MP4 of myoChallengeBoxingVsClone-v0 (clone on agent_1). "
            "Uses --vs-clone-player for agent_0 (default half = stable stance). "
            "Also --output, --steps, --seed, camera flags."
        ),
    )
    p.add_argument(
        "--vs-clone-player",
        choices=("random", "half"),
        default="half",
        help=(
            "Controlled agent (agent_0) in VsClone: 'half' = mid muscle activation "
            "(keeps you upright for demos); 'random' = uniform [0,1] (often collapses "
            "before the clone)."
        ),
    )
    p.add_argument(
        "--clone-policy",
        type=str,
        default=None,
        help=(
            "Path to standalone BC policy .npz for VsClone (default: repo "
            "tutorials/mc26/baselines/boxing/mannequin_exact_clone.npz)."
        ),
    )
    p.add_argument(
        "--vs-env-id",
        default=BOXING_VS_DEFAULT_ID,
        help="Gymnasium env id when --boxing-vs is set.",
    )
    p.add_argument(
        "--vs-seed",
        type=int,
        default=None,
        help="RNG seed for reset + random actions when --boxing-vs is set.",
    )
    p.add_argument(
        "--vs-episodes",
        type=int,
        default=20,
        help="Max full episodes when --boxing-vs is set (Ctrl+C to exit early).",
    )
    p.add_argument(
        "--vs-render",
        choices=("human", "rgb_array"),
        default="human",
        help="Render mode when --boxing-vs is set.",
    )
    p.add_argument(
        "--env-id",
        default=None,
        help=(
            "Render a registered Gymnasium env with random actions and write MP4 "
            "(e.g., myoChallengeBoxingMannequin-v0)."
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of random env steps when --env-id is set.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed when --env-id is set.",
    )
    p.add_argument("--output", default="boxing_demo.mp4", help="Output MP4 path")
    p.add_argument("--fps", type=int, default=30, help="Frames per second")
    p.add_argument("--width", type=int, default=640, help="Render width")
    p.add_argument("--height", type=int, default=480, help="Render height")
    p.add_argument("--cam-azimuth", type=float, default=None)
    p.add_argument("--cam-elevation", type=float, default=None)
    p.add_argument("--cam-distance", type=float, default=None)
    p.add_argument("--cam-lookat-x", type=float, default=None)
    p.add_argument("--cam-lookat-y", type=float, default=None)
    p.add_argument("--cam-lookat-z", type=float, default=None)
    p.add_argument(
        "--dual-opponent",
        action="store_true",
        help="Add a second BOB for the left-arm head phase (two meshes in scene).",
    )
    args = p.parse_args()
    cam_lookat = None
    if (
        args.cam_lookat_x is not None
        and args.cam_lookat_y is not None
        and args.cam_lookat_z is not None
    ):
        cam_lookat = (
            float(args.cam_lookat_x),
            float(args.cam_lookat_y),
            float(args.cam_lookat_z),
        )
    if args.boxing_vs and args.boxing_vs_clone:
        p.error("Use only one of --boxing-vs and --boxing-vs-clone.")
    if args.record_boxing_vs_clone and args.boxing_vs:
        p.error("Use only one of --boxing-vs and --record-boxing-vs-clone.")
    if args.record_boxing_vs_clone and args.boxing_vs_clone:
        p.error("Use only one of --boxing-vs-clone and --record-boxing-vs-clone.")
    if args.record_boxing_vs_clone and args.env_id:
        p.error("Do not combine --record-boxing-vs-clone with --env-id.")
    clone_policy = Path(args.clone_policy) if args.clone_policy else None

    if args.record_boxing_vs_clone:
        out = render_registered_env(
            env_id=BOXING_VS_CLONE_DEFAULT_ID,
            output=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            cam_azimuth=165.0 if args.cam_azimuth is None else args.cam_azimuth,
            cam_elevation=-14.0 if args.cam_elevation is None else args.cam_elevation,
            cam_distance=args.cam_distance,
            cam_lookat=cam_lookat,
            policy_path=clone_policy,
            vs_clone_player=args.vs_clone_player,
        )
        print(f"Done (boxing vs clone MP4): {out}")
        return
    if args.boxing_vs:
        play_boxing_vs(
            env_id=args.vs_env_id,
            seed=args.vs_seed,
            max_episodes=args.vs_episodes,
            render_mode=args.vs_render,
            vs_clone_player=args.vs_clone_player,
        )
        print("Done (boxing vs).")
        return
    if args.boxing_vs_clone:
        play_boxing_vs(
            env_id=BOXING_VS_CLONE_DEFAULT_ID,
            seed=args.vs_seed,
            max_episodes=args.vs_episodes,
            render_mode=args.vs_render,
            policy_path=clone_policy,
            vs_clone_player=args.vs_clone_player,
        )
        print("Done (boxing vs clone).")
        return
    if args.env_id:
        out = render_registered_env(
            env_id=args.env_id,
            output=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            cam_azimuth=165.0 if args.cam_azimuth is None else args.cam_azimuth,
            cam_elevation=-14.0 if args.cam_elevation is None else args.cam_elevation,
            cam_distance=args.cam_distance,
            cam_lookat=cam_lookat,
            policy_path=clone_policy
            if args.env_id == BOXING_VS_CLONE_DEFAULT_ID
            else None,
        )
        print(f"Done: {out}")
        return
    out = run_demo(
        output=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        dual_opponent=args.dual_opponent,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_distance=args.cam_distance,
        cam_lookat=cam_lookat,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
