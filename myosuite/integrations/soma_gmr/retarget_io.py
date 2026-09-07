# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Convert GMR-family (amathislab/gmr_plus) retargeted motion pickles into
MyoSuite MotionClip-compatible NPZ files for the ``myofullbody`` skeleton.

GMR retargets against the *raw* ``musclemimic_models`` ``myofullbody`` MJCF,
which includes finger joints and has no ``*_mimic`` sites. MyoSuite's
MuscleMimic full-body envs compile that same MJCF with
``disable_fingers=True`` (dropping finger DoFs) and add the ``*_mimic``
sites (see ``myosuite.integrations.musclemimic.fullbody_model``). So a
GMR ``myofullbody`` pickle is *wider* than the model MyoSuite actually
simulates on -- this module drops the finger coordinates by joint name
(reusing the same ``FINGER_JOINT_TOKENS`` list MyoSuite's model builder
removes) before writing qpos/qvel, and forward-kinematics the mimic-site
positions on MyoSuite's own compiled model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np

from myosuite.core.trajectory_io import (
    _joint_qpos_size,
    _joint_qvel_size,
    _resolve_model_indices,
)
from myosuite.integrations.musclemimic.bimanual_model import FINGER_JOINT_TOKENS
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)

# Site order myosuite.envs.myo.tasks.mimic.clip_env.MuscleMimicClipEnvV0
# expects in a clip's `site_xpos` array (T, 17, 3). Mirrors that module's
# private `_CLIP_SITE_ORDER` exactly; kept as a literal here (rather than
# importing the private symbol) with a test asserting the two stay in sync.
GMR_CLIP_SITE_ORDER: tuple[str, ...] = (
    "upper_body_mimic",
    "head_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "pelvis_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
)


def load_gmr_pkl(path: str | Path) -> dict[str, np.ndarray]:
    """Load a GMR ``bvh_to_robot.py`` / ``smplh_to_robot.py`` output pickle.

    Args:
        path: Path to the GMR ``.pkl`` file.

    Returns:
        Dict with ``root_pos`` (F, 3), ``root_rot`` (F, 4, xyzw), ``dof_pos``
        (F, raw_nq - 7), and ``fps`` (float). ``dof_pos`` spans the *raw*
        (finger-included) myofullbody model GMR retargets against.

    Raises:
        KeyError: If any required key is missing from the pickle.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = ("root_pos", "root_rot", "dof_pos", "fps")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"GMR motion pickle {path} is missing keys: {missing}")
    return data


def _raw_myofullbody_model() -> mujoco.MjModel:
    """The unmodified ``musclemimic_models`` myofullbody model GMR uses.

    Includes finger joints (which MyoSuite's MuscleMimic full-body envs
    remove by default) and lacks the ``*_mimic`` sites MyoSuite adds, so
    this is only used to interpret the raw GMR pickle -- never to simulate.
    """
    from musclemimic_models import get_xml_path

    return mujoco.MjModel.from_xml_path(str(get_xml_path("myofullbody")))


def _gmr_pkl_to_qpos(
    gmr_data: dict[str, np.ndarray], raw_model: mujoco.MjModel
) -> np.ndarray:
    """Concatenate GMR's (root_pos, root_rot, dof_pos) into raw-model qpos.

    GMR stores ``root_rot`` scalar-last (xyzw); MuJoCo qpos free-joint
    quaternions are scalar-first (wxyz).
    """
    root_pos = np.asarray(gmr_data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(gmr_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(gmr_data["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (F, 3), got {root_pos.shape}")
    if root_rot_xyzw.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot must have shape (F, 4), got {root_rot_xyzw.shape}")
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
    if qpos.shape[1] != raw_model.nq:
        raise ValueError(
            f"root_pos+root_rot+dof_pos width {qpos.shape[1]} does not match "
            f"myofullbody raw model nq={raw_model.nq}. Was this pickle really "
            "retargeted to 'myofullbody'?"
        )
    return qpos


def _per_coordinate_joint_names(model: mujoco.MjModel, *, use_qvel: bool) -> list[str]:
    """Per-coordinate joint name list (length nv or nq), in model order."""
    width = model.nv if use_qvel else model.nq
    names: list[str] = [""] * width
    for joint_id in range(model.njnt):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            or f"joint_{joint_id}"
        )
        joint_type = int(model.jnt_type[joint_id])
        if use_qvel:
            start = int(model.jnt_dofadr[joint_id])
            size = _joint_qvel_size(joint_type)
        else:
            start = int(model.jnt_qposadr[joint_id])
            size = _joint_qpos_size(joint_type)
        for k in range(size):
            names[start + k] = name
    return names


def _drop_finger_coordinates(
    values: np.ndarray,
    coordinate_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Drop columns whose joint name is a MyoSuite-removed finger joint."""
    keep = [
        i for i, name in enumerate(coordinate_names) if name not in FINGER_JOINT_TOKENS
    ]
    reduced_names = [coordinate_names[i] for i in keep]
    return values[:, keep], reduced_names


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference qvel from consecutive qpos frames.

    Uses ``mj_differentiatePos`` rather than naive subtraction so the
    free-joint quaternion difference becomes a proper angular velocity. The
    last frame repeats the second-to-last velocity.
    """
    n_frames = qpos.shape[0]
    qvel = np.zeros((n_frames, model.nv), dtype=np.float64)
    for t in range(n_frames - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t], qpos[t + 1])
    if n_frames > 1:
        qvel[-1] = qvel[-2]
    return qvel


def _permute_to_model_order(
    model: mujoco.MjModel,
    values: np.ndarray,
    coordinate_names: list[str],
    *,
    use_qvel: bool,
) -> np.ndarray:
    """Scatter ``values`` (named columns) into ``model``'s native coordinate order.

    Requires ``coordinate_names`` to exactly cover every coordinate of
    ``model`` (bijective) -- raises via :func:`_resolve_model_indices` if a
    referenced joint doesn't exist on ``model``.
    """
    expected_width = model.nv if use_qvel else model.nq
    model_indices = _resolve_model_indices(
        clip_width=len(coordinate_names),
        expected_width=expected_width,
        joint_names=coordinate_names,
        mj_model=model,
        field_name="qvel" if use_qvel else "qpos",
        use_qvel=use_qvel,
    )
    out = np.zeros((values.shape[0], expected_width), dtype=values.dtype)
    out[:, model_indices] = values
    return out


def _compute_site_xpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    site_names: tuple[str, ...],
) -> np.ndarray:
    """Forward-kinematics site positions for ``site_names`` at every frame."""
    site_ids = []
    for name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Model has no site named {name!r}")
        site_ids.append(site_id)
    data = mujoco.MjData(model)
    n_frames = qpos.shape[0]
    site_xpos = np.zeros((n_frames, len(site_ids), 3), dtype=np.float64)
    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        site_xpos[t] = data.site_xpos[site_ids]
    return site_xpos


def gmr_pkl_to_motion_clip_npz(
    pkl_path: str | Path,
    out_path: str | Path,
    *,
    model: mujoco.MjModel | None = None,
) -> Path:
    """Convert a GMR ``myofullbody`` retargeting pickle to a MotionClip NPZ.

    The output matches what both
    ``myosuite.core.trajectory_io.load_motion_clip`` and
    ``myosuite.envs.myo.tasks.mimic.clip_env.MuscleMimicClipEnvV0`` expect:
    full-width ``qpos``/``qvel`` (on ``model``, finger DoFs already dropped)
    plus ``site_xpos`` for the 17 mimic sites in :data:`GMR_CLIP_SITE_ORDER`.

    Args:
        pkl_path: GMR ``.pkl`` output (e.g. from gmr_plus's
            ``scripts/bvh_to_robot.py --robot myofullbody`` or
            ``scripts/smplh_to_robot.py --robot myofullbody``).
        out_path: Destination ``.npz`` path.
        model: Target ``myofullbody`` MjModel to write the clip against.
            Defaults to the same (finger-removed, mimic-site-bearing) model
            MyoSuite's MuscleMimic full-body envs use
            (:func:`compile_mimic_fullbody_mjmodel`).

    Returns:
        The resolved ``out_path``.
    """
    gmr_data = load_gmr_pkl(pkl_path)
    raw_model = _raw_myofullbody_model()
    if model is None:
        model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())

    qpos_raw = _gmr_pkl_to_qpos(gmr_data, raw_model)
    fps = float(np.asarray(gmr_data["fps"]).reshape(()))
    dt = 1.0 / fps
    qvel_raw = _compute_qvel(raw_model, qpos_raw, dt)

    qpos_names_raw = _per_coordinate_joint_names(raw_model, use_qvel=False)
    qvel_names_raw = _per_coordinate_joint_names(raw_model, use_qvel=True)
    qpos_reduced, qpos_names = _drop_finger_coordinates(qpos_raw, qpos_names_raw)
    qvel_reduced, qvel_names = _drop_finger_coordinates(qvel_raw, qvel_names_raw)

    if len(qpos_names) != model.nq or len(qvel_names) != model.nv:
        raise ValueError(
            "After dropping finger coordinates, GMR's myofullbody pickle has "
            f"nq={len(qpos_names)}, nv={len(qvel_names)} but the target model "
            f"has nq={model.nq}, nv={model.nv}. The raw musclemimic_models "
            "myofullbody MJCF and MyoSuite's compiled mimic model have "
            "diverged beyond the known finger-joint removal -- update "
            "FINGER_JOINT_TOKENS or investigate the model mismatch before "
            "trusting this conversion."
        )

    qpos = _permute_to_model_order(model, qpos_reduced, qpos_names, use_qvel=False)
    qvel = _permute_to_model_order(model, qvel_reduced, qvel_names, use_qvel=True)
    site_xpos = _compute_site_xpos(model, qpos, GMR_CLIP_SITE_ORDER)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        site_names=np.array(GMR_CLIP_SITE_ORDER),
        frequency=np.asarray(fps, dtype=np.float64),
    )
    return out_path
