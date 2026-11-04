# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Retarget FMPose3D monocular 3D human poses onto MyoLeg and MyoFullBody.

`FMPose3D <https://github.com/AdaptiveMotorControlLab/FMPose3D>`_ (Wang, Yu
and Mathis, CVPR 2026) lifts video frames to the 17 joints of Human3.6M.
:func:`retarget_fmpose3d` turns its output into a
:class:`~myosuite.core.trajectory_io.MotionClip` of a MyoSuite model.

FMPose3D is not a dependency: it needs ``torch<2.11`` and Python < 3.13, while
``myosuite[mjlab]`` needs ``torch>=2.13``. Run it in its own environment and
pass its ``Pose3DResult``, its ``poses_3d`` array or an ``.npz`` saved from it.

Each frame is rebuilt with the model's segment lengths along FMPose3D's segment
directions before :func:`~myosuite.core.retargeting.retarget` fits it. One
camera fixes directions far better than lengths, and Human3.6M's hip joints sit
about 0.13 m either side of the pelvis, wider than the models' hip centres.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from myosuite.core.body_scaling import _assembled_qpos, scale_model
from myosuite.core.retargeting import retarget
from myosuite.core.trajectory_io import MotionClip

#: FMPose3D's 17 Human3.6M joints in output order. Human3.6M names the ankles
#: "foot" and the nose "neck/nose".
H36M_JOINTS = (
    "hip",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "spine",
    "thorax",
    "nose",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
)
_INDEX = {name: i for i, name in enumerate(H36M_JOINTS)}

# Joint -> (body carrying its site, bodies whose mean origin locates the site).
# Spine, nose and head are left out: FMPose3D synthesises them from the hips,
# shoulders and face in its 2D stage, and the models have no matching point.
_CENTRES: dict[str, tuple[str, tuple[str, ...]]] = {
    "hip": ("pelvis", ("femur_r", "femur_l")),
    "r_hip": ("pelvis", ("femur_r",)),
    "l_hip": ("pelvis", ("femur_l",)),
    "r_knee": ("tibia_r", ("tibia_r",)),
    "l_knee": ("tibia_l", ("tibia_l",)),
    "r_ankle": ("talus_r", ("talus_r",)),
    "l_ankle": ("talus_l", ("talus_l",)),
    "thorax": ("thorax", ("thorax",)),
    "r_shoulder": ("humerus_r", ("humerus_r",)),
    "l_shoulder": ("humerus_l", ("humerus_l",)),
    "r_elbow": ("ulna_r", ("ulna_r",)),
    "l_elbow": ("ulna_l", ("ulna_l",)),
    "r_wrist": ("lunate_r", ("lunate_r",)),
    "l_wrist": ("lunate_l", ("lunate_l",)),
}
_LEGS = ("hip", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle")
# MyoLeg's trunk is one body with its pelvis, without a thorax. Its thorax site
# goes where MyoFullBody's thorax origin is at qpos0: 0.56 m above the hip
# centres and 0.028 m behind them. Only its direction from the hips is fitted,
# which sets the pelvis tilt the hips alone leave open.
_TRUNK_HEIGHT, _TRUNK_BACK = 0.56, 0.028
# Joints held at the seed, with the joints coupled to them, although they move
# a site. MyoFullBody's lumbar flexion: the hip and thorax joints cannot tell it
# from pelvis tilt, and left free it drifts against pelvis tilt and hip flexion
# from frame to frame. Forearm rotation: it moves the wrist point by 14 mm per
# radian in any pose, so 1 cm of noise turns it by 40 degrees. Joints that move
# no site (ankle, subtalar, toes, wrist) are held as well; see _free_joints.
_HELD = ("flex_extension", "pro_sup_r", "pro_sup_l")
# Segments rebuilt from the root outwards, as (parent joint, child joint).
_SEGMENTS = (
    ("hip", "r_hip"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("hip", "l_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("hip", "thorax"),
    ("thorax", "r_shoulder"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("thorax", "l_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
)
# FMPose3D's camera frame (x right, y down, z forward) to the z-up world of a
# level camera.
_LEVEL_CAMERA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_SITE_PREFIX = "h36m_"


def retarget_fmpose3d(
    spec: mujoco.MjSpec,
    poses: Any,
    *,
    fps: float | None = None,
    frame: Literal["camera", "world"] | None = None,
    upright: bool = True,
    ground_each_frame: bool = True,
    scales: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    total_mass: float | None = None,
) -> tuple[mujoco.MjModel, MotionClip]:
    """Fit MyoLeg or MyoFullBody to FMPose3D human poses.

    Example::

        # In an environment with fmpose3d, on a folder of video frames:
        result = FMPose3DInference().predict("frames/")
        np.savez("walk.npz", poses_3d=result.poses_3d,
                 valid_frames_mask=result.valid_frames_mask, fps=30)
        # In MyoSuite:
        model, clip = retarget_fmpose3d(spec, "walk.npz")

    The poses are turned into a z-up world, rebuilt with the model's segment
    lengths along their own segment directions, and fitted by
    :func:`~myosuite.core.retargeting.retarget` to joint-centre sites added to
    a copy of *spec* (``h36m_<joint>``): the hip centres and their midpoint,
    the knees, the ankles, a trunk point and, on MyoFullBody, the shoulders,
    elbows and wrists.

    * With *upright* the median hip-to-thorax direction is turned to the
      model's trunk direction at its seed pose, which assumes the trunk is
      upright on average, as in walking. Otherwise the vertical is that of a
      level camera, or FMPose3D's own world frame.
    * The first valid frame is turned to face the way the model faces.
    * FMPose3D poses are root-relative, so the clip has no horizontal travel.
      With *ground_each_frame* every frame's lowest sole point is put on the
      ground, which assumes a foot is down in every frame; otherwise the
      pelvis keeps its height and the clip is lifted as a whole.
    * Joints that move none of the sites keep the seed pose: Human3.6M has no
      foot or hand points, so the ankle, subtalar and toe joints and the
      wrist stay put. So does forearm rotation, which barely moves the wrist
      point, and MyoFullBody's lumbar flexion: the joints cannot tell it from
      pelvis tilt, so a forward lean of the trunk shows as pelvis tilt, as on
      MyoLeg. The axial rotation of a straight limb is also left open.

    Args:
        spec: MyoLeg or MuscleMimic MyoFullBody spec; it is not changed.
        poses: FMPose3D's ``Pose3DResult``, an array ``(T, 17, 3)`` or
            ``(17, 3)`` in metres, or an ``.npz`` holding ``poses_3d``,
            ``poses_3d_world`` or ``pose3d`` (optionally with
            ``valid_frames_mask`` and ``fps``).
        fps: Frame rate in Hz, required for several frames unless the
            ``.npz`` holds it.
        frame: ``"camera"`` for ``poses_3d`` (x right, y down, z forward) or
            ``"world"`` for ``poses_3d_world`` and the demo's ``pose3d``.
            Taken from a result or ``.npz``; arrays default to ``"camera"``.
        upright: Take the vertical from the median hip-to-thorax direction.
        ground_each_frame: Put every frame's lowest sole point on the ground.
        scales: Subject scales for
            :func:`~myosuite.core.body_scaling.scale_model`, applied first.
        total_mass: Subject mass in kg, for ``scale_model``.

    Returns:
        The model with the ``h36m_*`` sites (scaled if asked; its ``qpos``
        layout is that of *spec*) and the clip. Its ``metadata`` holds
        retarget's entries and ``fmpose3d``: the rotation from the input
        frame, the frame, the fitted Human3.6M joints, the model joints held
        at the seed pose (``held``) and the valid frames.

    Raises:
        ValueError: For poses that are not the 17 human joints, a spec without
            MyoLeg's leg bodies, or *upright* without hip and thorax joints.
    """
    positions, valid, rate, inferred = _load_poses(poses)
    frame = frame or inferred
    if frame not in ("camera", "world"):
        raise ValueError(f'frame must be "camera" or "world", got {frame!r}.')
    joints = _joints_in(spec)
    model = _with_sites(spec, joints, scales, total_mass)
    seed = _assembled_qpos(model)
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    mujoco.mj_kinematics(model, data)
    centres = {
        joint: data.site_xpos[model.site(_SITE_PREFIX + joint).id].copy()
        for joint in joints
    }

    positions = np.where(valid[:, None, None], positions, np.nan)
    rotation = _LEVEL_CAMERA if frame == "camera" else np.eye(3)
    world = positions @ rotation.T
    if upright:
        tilt = _z_to(_median_trunk(world)).T
        world, rotation = world @ tilt.T, tilt @ rotation
    yaw = _heading(world, centres)
    world, rotation = world @ yaw.T, yaw @ rotation
    if upright and "thorax" in centres:  # lean as the model's trunk does at its seed
        lean = _z_to(centres["thorax"] - centres["hip"])
        world, rotation = world @ lean.T, lean @ rotation
    targets = _rebuild(world, joints, centres)
    shift = centres["hip"] - targets["hip"]  # every frame over the seed's pelvis
    free, held = _free_joints(model)
    clip = retarget(
        model,
        {_SITE_PREFIX + joint: path + shift for joint, path in targets.items()},
        fps=fps if fps is not None else rate,
        joints=free,
        qpos=seed,
        ground_each_frame=ground_each_frame,
    )
    clip.metadata["fmpose3d"] = {
        "rotation": rotation,
        "frame": frame,
        "joints": list(targets),
        "held": held,
        "valid_frames": valid,
    }
    return model, clip


def _load_poses(poses: Any) -> tuple[np.ndarray, np.ndarray, float | None, str]:
    """Positions ``(T, 17, 3)``, valid frames, frame rate and frame of *poses*."""
    mask, rate, frame = None, None, "camera"
    if isinstance(poses, str | os.PathLike):
        with np.load(Path(poses)) as archive:
            for key, kind in (
                ("poses_3d", "camera"),
                ("poses_3d_world", "world"),
                ("pose3d", "world"),
            ):
                if key in archive:
                    array, frame = archive[key], kind
                    break
            else:
                raise ValueError(
                    f"{poses} holds no poses_3d, poses_3d_world or pose3d array."
                )
            if "valid_frames_mask" in archive:
                mask = archive["valid_frames_mask"]
            if "fps" in archive:
                rate = float(archive["fps"])
    elif hasattr(poses, "poses_3d"):  # FMPose3D's Pose3DResult
        array, mask = poses.poses_3d, getattr(poses, "valid_frames_mask", None)
    else:
        array = poses
    positions = np.array(array, dtype=float)
    if positions.ndim == 2:
        positions = positions[None]
    if positions.ndim != 3 or positions.shape[1:] != (len(H36M_JOINTS), 3):
        raise ValueError(
            f"poses must hold the {len(H36M_JOINTS)} Human3.6M joints, shape "
            f"(T, {len(H36M_JOINTS)}, 3); got {positions.shape}. FMPose3D's "
            "26-joint animal poses have no MyoSuite model."
        )
    valid = np.ones(len(positions), dtype=bool)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool).reshape(-1)
        if valid.shape != (len(positions),):
            raise ValueError("valid_frames_mask must have one entry per frame.")
    return positions, valid, rate, frame


def _joints_in(spec: mujoco.MjSpec) -> list[str]:
    """The joints whose bodies *spec* has; the legs are required."""
    bodies = {body.name for body in spec.bodies}
    needed = {
        name for joint in _LEGS for name in (_CENTRES[joint][0], *_CENTRES[joint][1])
    }
    if missing := sorted(needed - bodies):
        raise ValueError(
            "retarget_fmpose3d needs MyoLeg or the MuscleMimic MyoFullBody; the spec "
            f"has no {', '.join(missing)} body."
        )
    joints = [
        joint
        for joint, (carrier, located_by) in _CENTRES.items()
        if {carrier, *located_by} <= bodies
    ]
    if "thorax" not in joints and "torso" in bodies:  # MyoLeg
        joints.append("thorax")
    return joints


def _with_sites(
    spec: mujoco.MjSpec,
    joints: list[str],
    scales: Mapping[str, Any] | str | os.PathLike[str] | None,
    total_mass: float | None,
) -> mujoco.MjModel:
    """Compile a copy of *spec* with a site at each joint centre, scaled if asked."""
    nominal = spec.copy().compile()
    data = mujoco.MjData(nominal)
    data.qpos[:] = _assembled_qpos(nominal)
    mujoco.mj_kinematics(nominal, data)
    names = {nominal.body(i).name for i in range(nominal.nbody)}
    edited = spec.copy()
    for joint in joints:
        carrier, located_by = _CENTRES[joint]
        if carrier not in names:  # MyoLeg's thorax, on its trunk
            right, left = (data.xpos[nominal.body(f"femur_{s}").id] for s in "rl")
            forward = np.cross(left - right, [0.0, 0.0, 1.0])
            forward /= np.linalg.norm(forward)
            carrier, up = "torso", np.array([0.0, 0.0, _TRUNK_HEIGHT])
            point = 0.5 * (right + left) + up - _TRUNK_BACK * forward
        else:
            point = np.mean(
                [data.xpos[nominal.body(name).id] for name in located_by], axis=0
            )
        body = nominal.body(carrier).id
        local = data.xmat[body].reshape(3, 3).T @ (point - data.xpos[body])
        edited.body(carrier).add_site(name=_SITE_PREFIX + joint, pos=local)
    if scales is None and total_mass is None:
        return edited.compile()
    return scale_model(edited, {} if scales is None else scales, total_mass=total_mass)


def _free_joints(model: mujoco.MjModel) -> tuple[list[str] | None, list[str]]:
    """Joints the fit may move (``None``: all) and the joints held at the seed.

    Held are the joints of ``_HELD``, every joint that moves no ``h36m_*``
    site (its body is no strict ancestor of a site's body: the ankle and wrist
    joints, whose sites sit at their own centres, and everything distal to
    them), and the joints coupled to a held joint. A joint coupled to a free
    joint moves with it.
    """
    couplings = [
        (int(model.eq_obj1id[e]), int(model.eq_obj2id[e]))
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[e]
    ]
    followers = {follower for follower, leader in couplings if leader >= 0}
    moving = set()  # strict ancestors of the bodies carrying the sites
    for site in range(model.nsite):
        if not model.site(site).name.startswith(_SITE_PREFIX):
            continue
        body = int(model.body_parentid[model.site_bodyid[site]])
        while body > 0 and body not in moving:
            moving.add(body)
            body = int(model.body_parentid[body])
    held = {
        joint
        for joint in range(model.njnt)
        if joint not in followers
        and (model.joint(joint).name in _HELD or model.jnt_bodyid[joint] not in moving)
    }
    grown = True
    while grown:  # a coupled joint (obj1) follows its leader (obj2)
        grown = False
        for follower, leader in couplings:
            if leader in held and follower not in held:
                held.add(follower)
                grown = True
    if not held:
        return None, []
    names = [model.joint(j).name for j in range(model.njnt)]
    free = [names[j] for j in range(model.njnt) if j not in held]
    return free, [names[j] for j in sorted(held)]


def _median_trunk(world: np.ndarray) -> np.ndarray:
    """Median direction from the hip to the thorax joint."""
    trunk = world[:, _INDEX["thorax"]] - world[:, _INDEX["hip"]]
    norms = np.linalg.norm(trunk, axis=1)
    usable = np.isfinite(norms) & (norms > 0)
    if not usable.any():
        raise ValueError(
            "upright=True needs finite hip and thorax joints; pass upright=False."
        )
    return np.median(trunk[usable] / norms[usable, None], axis=0)


def _z_to(direction: np.ndarray) -> np.ndarray:
    """Smallest rotation turning +z to *direction*."""
    quat, turn = np.zeros(4), np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, direction / np.linalg.norm(direction))
    mujoco.mju_quat2Mat(turn, quat)
    return turn.reshape(3, 3)


def _heading(world: np.ndarray, centres: Mapping[str, np.ndarray]) -> np.ndarray:
    """Rotation about z turning the first valid frame's hip line to the model's."""
    lateral = world[:, _INDEX["l_hip"]] - world[:, _INDEX["r_hip"]]
    finite = np.flatnonzero(np.all(np.isfinite(lateral), axis=1))
    if not len(finite):
        raise ValueError("No frame has finite hip joints.")
    model_lateral = centres["l_hip"] - centres["r_hip"]
    first = lateral[finite[0]]
    angle = np.arctan2(model_lateral[1], model_lateral[0]) - np.arctan2(
        first[1], first[0]
    )
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rebuild(
    world: np.ndarray, joints: list[str], centres: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Joint paths with the model's segment lengths along the poses' directions."""
    targets = {"hip": world[:, _INDEX["hip"]].copy()}
    for parent, child in _SEGMENTS:
        if parent not in targets or child not in joints:
            continue
        direction = world[:, _INDEX[child]] - world[:, _INDEX[parent]]
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = np.where(norm > 0, direction / norm, np.nan)
        length = np.linalg.norm(centres[child] - centres[parent])
        targets[child] = targets[parent] + length * unit
    return targets
