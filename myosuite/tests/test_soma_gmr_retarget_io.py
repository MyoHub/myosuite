# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GMR (amathislab/gmr_plus) -> MyoSuite MotionClip converter.

Covers three layers:
  * Fast, dependency-free unit tests of the pkl -> NPZ conversion against a
    synthetic (but shape-correct) GMR pickle, including the finger-joint
    coordinate drop between GMR's raw myofullbody model and MyoSuite's
    compiled (finger-removed) mimic model.
  * An optional smoke test against the real, shipped
    ``bvh_soma_to_myofullbody.json`` IK config (finite-output only -- see
    ``test_bvh_soma_config_smoke``'s docstring for why it doesn't assert
    tracking quality).
  * An optional end-to-end tracking-quality test that runs the real
    ``general_motion_retargeting`` (gmr_plus) IK retargeter against the real
    ``myofullbody`` model with a synthetic but kinematically-consistent,
    position-only marker sequence, then feeds the converted clip into
    MuscleMimicClipEnvV0, asserting meaningful and left/right-symmetric range
    of motion. All three are skipped unless gmr_plus is importable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_io import load_motion_clip
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.soma_gmr.retarget_io import (
    GMR_CLIP_SITE_ORDER,
    _raw_myofullbody_model,
    gmr_pkl_to_motion_clip_npz,
    load_gmr_pkl,
)

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module")
def myofullbody_model():
    """MyoSuite's compiled mimic model: fingers removed, has *_mimic sites."""
    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    return model


@pytest.fixture(scope="module")
def raw_model():
    """GMR's raw musclemimic_models myofullbody model: fingers included."""
    return _raw_myofullbody_model()


def test_clip_site_order_matches_clip_env():
    """GMR_CLIP_SITE_ORDER must stay in sync with clip_env's private order.

    MuscleMimicClipEnvV0 re-indexes a clip's on-disk site_xpos using its own
    `_CLIP_SITE_ORDER` -> model-site-order table. If the two orders drift
    apart, sites get silently swapped instead of raising an error.
    """
    from myosuite.envs.myo.tasks.mimic.clip_env import _CLIP_SITE_ORDER

    assert GMR_CLIP_SITE_ORDER == _CLIP_SITE_ORDER


def _make_synthetic_gmr_pkl(raw_model, n_frames: int = 5, fps: float = 30.0) -> dict:
    """Build a shape-correct fake GMR output pickle for `myofullbody`.

    Width matches GMR's *raw* (finger-included) model, matching what
    gmr_plus's `scripts/{bvh,smplh,smplx}_to_robot.py --robot myofullbody`
    actually write. Root translates along +x at constant velocity and yaws
    slowly; dof_pos is a small sinusoidal sweep away from the keyframe pose.
    Not a real retargeting result -- just enough to exercise the converter's
    math.
    """
    rng = np.random.default_rng(0)
    nq_local = raw_model.nq - 7
    t = np.arange(n_frames) / fps

    root_pos = np.stack([0.1 * t, np.zeros_like(t), 0.9 * np.ones_like(t)], axis=1)
    yaw = 0.05 * t
    root_rot_xyzw = np.stack(
        [
            np.zeros_like(t),
            np.zeros_like(t),
            np.sin(yaw / 2.0),
            np.cos(yaw / 2.0),
        ],
        axis=1,
    )

    base_dof = (
        raw_model.key_qpos[0][7:]
        if raw_model.nkey > 0
        else np.zeros(nq_local, dtype=np.float64)
    )
    wiggle = (
        0.01 * np.sin(2.0 * np.pi * t)[:, None] * rng.standard_normal((1, nq_local))
    )
    dof_pos = base_dof[None, :] + wiggle

    return {
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "fps": fps,
    }


def test_load_gmr_pkl_missing_key_raises(tmp_path: Path):
    bad_path = tmp_path / "bad.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"root_pos": np.zeros((2, 3))}, f)
    with pytest.raises(KeyError):
        load_gmr_pkl(bad_path)


def test_gmr_pkl_to_motion_clip_npz_shapes_and_finiteness(
    tmp_path: Path, raw_model, myofullbody_model
):
    gmr_data = _make_synthetic_gmr_pkl(raw_model, n_frames=6, fps=30.0)
    pkl_path = tmp_path / "synthetic_myofullbody.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)

    out_path = gmr_pkl_to_motion_clip_npz(
        pkl_path, tmp_path / "synthetic_myofullbody.npz", model=myofullbody_model
    )
    assert out_path.is_file()

    npz = np.load(out_path, allow_pickle=True)
    # Output is on the *target* (finger-removed) model, narrower than the raw
    # GMR pickle -- that's the point of the finger-coordinate drop.
    assert npz["qpos"].shape == (6, myofullbody_model.nq)
    assert npz["qvel"].shape == (6, myofullbody_model.nv)
    assert myofullbody_model.nq < raw_model.nq, "fixture should actually drop fingers"
    assert npz["site_xpos"].shape == (6, len(GMR_CLIP_SITE_ORDER), 3)
    assert tuple(str(n) for n in npz["site_names"]) == GMR_CLIP_SITE_ORDER

    assert np.all(np.isfinite(npz["qpos"]))
    assert np.all(np.isfinite(npz["qvel"]))
    assert np.all(np.isfinite(npz["site_xpos"]))

    # Root xy velocity should recover the ~0.1 m/s (in x) we synthesized,
    # not something wildly off (which would indicate a quaternion-order bug
    # feeding garbage into mj_differentiatePos, or a root-joint index bug in
    # the finger-coordinate permutation).
    vx = npz["qvel"][:-1, 0]
    assert np.all(np.abs(vx - 0.1) < 0.05)


def test_gmr_pkl_to_motion_clip_npz_loads_via_trajectory_io(
    tmp_path: Path, raw_model, myofullbody_model
):
    gmr_data = _make_synthetic_gmr_pkl(raw_model, n_frames=4, fps=30.0)
    pkl_path = tmp_path / "clip.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)
    out_path = gmr_pkl_to_motion_clip_npz(
        pkl_path, tmp_path / "clip.npz", model=myofullbody_model
    )

    clip = load_motion_clip(
        out_path, expected_nq=myofullbody_model.nq, expected_nv=myofullbody_model.nv
    )
    assert clip.qpos.shape == (4, myofullbody_model.nq)
    assert clip.qvel is not None and clip.qvel.shape == (4, myofullbody_model.nv)


def test_gmr_pkl_to_motion_clip_npz_wrong_width_raises(
    tmp_path: Path, raw_model, myofullbody_model
):
    gmr_data = _make_synthetic_gmr_pkl(raw_model, n_frames=3, fps=30.0)
    gmr_data["dof_pos"] = gmr_data["dof_pos"][:, :-1]  # drop one DoF -> width mismatch
    pkl_path = tmp_path / "bad_width.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)
    with pytest.raises(ValueError, match="does not match myofullbody raw model"):
        gmr_pkl_to_motion_clip_npz(
            pkl_path, tmp_path / "bad_width.npz", model=myofullbody_model
        )


def test_converted_clip_drives_mimic_clip_env(
    tmp_path: Path, raw_model, myofullbody_model
):
    """The converted NPZ must satisfy MuscleMimicClipEnvV0's own load contract."""
    from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

    gmr_data = _make_synthetic_gmr_pkl(raw_model, n_frames=10, fps=30.0)
    pkl_path = tmp_path / "clip_for_env.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)
    out_path = gmr_pkl_to_motion_clip_npz(
        pkl_path, tmp_path / "clip_for_env.npz", model=myofullbody_model
    )

    env = MuscleMimicClipEnvV0(clip_path=out_path, seed=0, use_obs_normalizer=False)
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs))
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert np.all(np.isfinite(obs2))
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


# ---------------------------------------------------------------------------
# End-to-end: real gmr_plus IK retargeting -> converter -> MyoSuite mimic env.
# ---------------------------------------------------------------------------

gmr = pytest.importorskip(
    "general_motion_retargeting",
    reason="amathislab/gmr_plus not installed; skipping SOMA/BVH->myofullbody e2e test",
)


def _fix_initial_configuration(retargeter) -> None:
    """Nudge any out-of-range limited hinge/slide DoF to its range midpoint.

    The shipped musclemimic_models myofullbody.xml has a handful of coupled
    OpenSim-derived DoFs (e.g. knee translation, driven by an mjEQ_JOINT
    polynomial off the primary knee angle) whose declared `range` doesn't
    actually include the model's own qpos0/keyframe-0 default (0.0) -- a
    pre-existing MJCF/keyframe inconsistency, not something this test or the
    SOMA/GMR conversion introduces. mink's ConfigurationLimit checks the raw
    starting qpos before the IK solver's EqualityConstraintTask has a chance
    to pull dependent DoFs back into range, so it trips immediately on frame
    0 without this.
    """
    model = retargeter.model
    qpos0 = retargeter.configuration.data.qpos.copy()
    hinge_or_slide = (
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    )
    for jid in range(model.njnt):
        if not model.jnt_limited[jid] or int(model.jnt_type[jid]) not in hinge_or_slide:
            continue
        qadr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        if not (lo - 1e-6 <= qpos0[qadr] <= hi + 1e-6):
            qpos0[qadr] = 0.5 * (lo + hi)
    retargeter.configuration.update(qpos0)


def test_bvh_soma_config_smoke(tmp_path: Path):
    """Smoke-test the real, shipped bvh_soma_to_myofullbody.json IK config.

    This only asserts the real GMR asset loads and produces a finite,
    converged result for a single static frame. It does NOT assert
    tracking quality: that config's rotation offsets are calibrated to a
    specific BVH rig's bone-local convention that isn't documented, and an
    earlier version of this test fed it made-up identity quaternions plus
    position targets on femur_l/tibia_l -- both of which that config
    assigns *zero* position weight, so those targets had no effect at all
    while the wrong orientation targets fought the (correctly-weighted)
    pelvis/torso position targets, producing a visually implausible,
    left/right-asymmetric retargeting result that nonetheless passed the
    old (too-weak) finite-output-only assertions. See
    test_end_to_end_soma_style_retarget_to_muscle_activation_env below for
    the real tracking-quality checks, done against a config this test file
    fully controls and validates instead.
    """
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT

    ik_config_path = IK_CONFIG_DICT["bvh_soma"]["myofullbody"]
    assert Path(ik_config_path).is_file()

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_soma",
        tgt_robot="myofullbody",
        ik_config_path=str(ik_config_path),
        verbose=False,
    )
    _fix_initial_configuration(retargeter)

    # retarget() requires every body referenced by the config's IK tasks to
    # be present in human_data (missing keys raise KeyError), so supply all
    # of them at a plausible rest position -- values don't need to be
    # meaningful for a finite-output-only smoke check.
    cfg = json.loads(Path(ik_config_path).read_text())
    all_bodies = {
        entry[0]
        for table in (cfg["ik_match_table1"], cfg["ik_match_table2"])
        for entry in table.values()
    }
    human_data = {
        body: (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        for body in all_bodies
    }
    qpos, err = retargeter.retarget(human_data)
    assert np.all(np.isfinite(qpos))
    assert np.all(np.isfinite(np.asarray(err)))


# ---------------------------------------------------------------------------
# A minimal, self-authored, position-only IK config for the tracking-quality
# end-to-end test below. Real marker-based capture systems (like SOMA's
# ChArUco suit) provide 3D point positions, not calibrated bone
# orientations, so orientation cost is 0 everywhere -- this also sidesteps
# needing to reverse-engineer bvh_soma_to_myofullbody.json's undocumented
# rig-specific rotation offsets. femur_l/r (hip) is deliberately excluded:
# the hip joint center is rigidly fixed relative to the pelvis by anatomy,
# so giving it its own independently-invented target fights the pelvis
# target through hip_flexion and pins that joint at its limit (this was the
# actual root cause of the previous version's bad-looking squat, together
# with using the wrong world-frame "forward" sign -- see FORWARD_SIGN below).
# ---------------------------------------------------------------------------

_MARKER_TO_BODY: dict[str, str] = {
    "pelvis": "Hips",
    "lumbar1": "Chest",
    "head": "Head",
    "humerus_l": "LeftShoulder",
    "ulna_l": "LeftElbow",
    "lunate_l": "LeftWrist",
    "humerus_r": "RightShoulder",
    "ulna_r": "RightElbow",
    "lunate_r": "RightWrist",
    "tibia_l": "LeftKnee",
    "calcn_l": "LeftAnkle",
    "toes_l": "LeftToe",
    "tibia_r": "RightKnee",
    "calcn_r": "RightAnkle",
    "toes_r": "RightToe",
}
_IDENTITY_WXYZ = [1.0, 0.0, 0.0, 0.0]

# Empirically verified against the compiled myofullbody keyframe (see
# scratch verification: increasing `hip_flexion_r` moves the knee toward
# -y world): this model's anatomical "anterior/forward" is -y, not +y.
FORWARD_SIGN = -1.0

THIGH, SHANK = 0.42, 0.43
UPPER_ARM, FOREARM = 0.28, 0.25
PELVIS_Z0, CHEST_DZ, HEAD_DZ = 0.90, 0.25, 0.60
HIP_X, SHOULDER_X = 0.10, 0.20


def _build_position_only_ik_config() -> dict:
    table = {
        body: [marker, 60, 0, [0.0, 0.0, 0.0], _IDENTITY_WXYZ]
        for body, marker in _MARKER_TO_BODY.items()
    }
    table["pelvis"][1] = 150  # extra root weight for stability
    return {
        "robot_root_name": "pelvis",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        "human_height_assumption": 1.7,
        "human_scale_table": {m: 1.0 for m in _MARKER_TO_BODY.values()},
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "ik_match_table1": table,
        "ik_match_table2": {k: list(v) for k, v in table.items()},
    }


def _two_link_forward(
    hip_yz: np.ndarray, ankle_yz: np.ndarray, l1: float, l2: float
) -> np.ndarray:
    """Constant-segment-length knee position from hip/ankle (sagittal plane).

    Bends the knee toward FORWARD_SIGN so the resulting motion is a real
    (not just plausible-looking) 2-link squat: |hip-knee|=l1 and
    |knee-ankle|=l2 exactly, for any hip height the ankle can still reach.
    """
    d_vec = ankle_yz - hip_yz
    d = np.clip(np.linalg.norm(d_vec), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    cos_a = np.clip((l1**2 + d**2 - l2**2) / (2 * l1 * d), -1.0, 1.0)
    a = np.arccos(cos_a)
    dir_hip_ankle = d_vec / d
    perp = np.array([-dir_hip_ankle[1], dir_hip_ankle[0]])
    if np.sign(perp[0]) != np.sign(FORWARD_SIGN):
        perp = -perp
    return hip_yz + l1 * (np.cos(a) * dir_hip_ankle + np.sin(a) * perp)


def _synthetic_squat_armraise_motion(n_frames: int, fps: float):
    """Bilateral squat (0->deep->0) with a bilateral lateral arm raise.

    Every target position is derived from constant-length-segment geometry
    (2-link squat IK for the legs, straight-arm rotation for the arms), so
    the motion is kinematically self-consistent, not just approximately
    plausible.
    """
    t = np.arange(n_frames) / fps
    period = t[-1] if t[-1] > 0 else 1.0
    squat = 0.5 * (1 - np.cos(2 * np.pi * t / period))  # 0 -> 1 -> 0
    raise_ = 0.5 * (1 - np.cos(np.pi * t / period))  # 0 -> 1

    frames = []
    for i in range(n_frames):
        s, a = float(squat[i]), float(raise_[i])
        pelvis_z = PELVIS_Z0 - 0.28 * s
        pelvis_y = -FORWARD_SIGN * 0.03 * s  # slight posterior lean while squatting
        marker: dict[str, np.ndarray] = {
            "Hips": np.array([0.0, pelvis_y, pelvis_z]),
            "Chest": np.array([0.0, pelvis_y, pelvis_z + CHEST_DZ]),
            "Head": np.array([0.0, pelvis_y, pelvis_z + HEAD_DZ]),
        }

        theta = np.radians(-80 + 160 * a)  # -80deg (arm down) -> +80deg (overhead)
        for side, sign in (("Left", 1.0), ("Right", -1.0)):
            shoulder = np.array(
                [sign * SHOULDER_X, pelvis_y, pelvis_z + CHEST_DZ - 0.05]
            )
            dir_xz = np.array([sign * np.cos(theta), np.sin(theta)])
            elbow_xz = np.array([shoulder[0], shoulder[2]]) + UPPER_ARM * dir_xz
            wrist_xz = elbow_xz + FOREARM * dir_xz
            marker[f"{side}Shoulder"] = shoulder
            marker[f"{side}Elbow"] = np.array([elbow_xz[0], shoulder[1], elbow_xz[1]])
            marker[f"{side}Wrist"] = np.array([wrist_xz[0], shoulder[1], wrist_xz[1]])

            ankle = np.array([sign * 0.10, FORWARD_SIGN * 0.05, 0.08])
            toe = np.array([sign * 0.10, FORWARD_SIGN * 0.20, 0.03])
            hip = np.array([sign * HIP_X, pelvis_y, pelvis_z])
            knee_yz = _two_link_forward(
                np.array([hip[1], hip[2]]), np.array([ankle[1], ankle[2]]), THIGH, SHANK
            )
            marker[f"{side}Knee"] = np.array([hip[0], knee_yz[0], knee_yz[1]])
            marker[f"{side}Ankle"] = ankle
            marker[f"{side}Toe"] = toe

        frame = {
            name: (pos.astype(np.float64), np.array(_IDENTITY_WXYZ, dtype=np.float64))
            for name, pos in marker.items()
        }
        frames.append(frame)
    return frames


def _joint_range_deg(model, qpos_arr: np.ndarray, name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jid]
    series = qpos_arr[:, qadr]
    lo, hi = model.jnt_range[jid]
    within = bool(np.all(series >= lo - 1e-6) and np.all(series <= hi + 1e-6))
    return np.degrees(series), within


def test_end_to_end_soma_style_retarget_to_muscle_activation_env(
    tmp_path: Path, myofullbody_model
):
    """Real gmr_plus IK (mink + myofullbody MJCF) -> converter -> mimic env.

    Exercises the full chain proposed for interfacing SOMA-style kinematic
    reconstruction with MyoSuite: a synthetic but kinematically-consistent
    marker sequence (position-only, like a real marker-suit capture) is
    retargeted with the *real* GeneralMotionRetargeting IK solver against
    the *real* raw myofullbody model, converted to a MotionClip NPZ (finger
    coordinates dropped, mimic sites forward-kinematically computed), and
    used to drive MuscleMimicClipEnvV0 (whose action space is the
    Hill-type muscle activation MyoSuite forks off the same reconstructed
    kinematics).

    Unlike a smoke test, this asserts the retargeting actually *worked*:
    meaningful (not near-zero) bilateral knee/shoulder range of motion,
    left/right symmetry for a bilaterally-symmetric input motion, and every
    joint staying within the model's own physiological limits. These are
    exactly the properties that were silently violated by the previous
    version of this test (see test_bvh_soma_config_smoke's docstring).
    """
    from general_motion_retargeting import GeneralMotionRetargeting

    cfg = _build_position_only_ik_config()
    ik_config_path = tmp_path / "position_only_myofullbody.json"
    ik_config_path.write_text(json.dumps(cfg))

    retargeter = GeneralMotionRetargeting(
        src_human="custom_position_only",
        tgt_robot="myofullbody",
        ik_config_path=str(ik_config_path),
        verbose=False,
    )
    _fix_initial_configuration(retargeter)

    n_frames, fps = 20, 20.0
    human_frames = _synthetic_squat_armraise_motion(n_frames, fps)
    qpos_frames = []
    for human_data in human_frames:
        qpos, _err = retargeter.retarget(dict(human_data))
        qpos_frames.append(np.asarray(qpos, dtype=np.float64).copy())
    qpos_frames = np.stack(qpos_frames, axis=0)
    assert qpos_frames.shape[1] == retargeter.model.nq  # GMR's raw model width
    assert np.all(np.isfinite(qpos_frames))

    gmr_data = {
        "root_pos": qpos_frames[:, :3],
        # mujoco qpos quat is wxyz; GMR pkl convention is xyzw -- convert back
        # so gmr_pkl_to_motion_clip_npz's wxyz<-xyzw handling round-trips.
        "root_rot": qpos_frames[:, [4, 5, 6, 3]],
        "dof_pos": qpos_frames[:, 7:],
        "fps": fps,
    }
    pkl_path = tmp_path / "soma_squat_myofullbody.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)

    clip_path = gmr_pkl_to_motion_clip_npz(
        pkl_path, tmp_path / "soma_squat_myofullbody.npz", model=myofullbody_model
    )
    npz = np.load(clip_path, allow_pickle=True)
    clip_qpos = npz["qpos"]

    # -- Tracking-quality checks (what actually caught the original bug) --
    for r_name, l_name, min_rom_deg, max_asym_deg in (
        ("knee_angle_r", "knee_angle_l", 40.0, 25.0),
        ("shoulder_elv_r", "shoulder_elv_l", 40.0, 45.0),
    ):
        r_deg, r_ok = _joint_range_deg(myofullbody_model, clip_qpos, r_name)
        l_deg, l_ok = _joint_range_deg(myofullbody_model, clip_qpos, l_name)
        assert r_ok, f"{r_name} exceeded the model's own physiological range"
        assert l_ok, f"{l_name} exceeded the model's own physiological range"
        r_rom, l_rom = r_deg.max() - r_deg.min(), l_deg.max() - l_deg.min()
        assert r_rom >= min_rom_deg, (
            f"{r_name} barely moved (ROM={r_rom:.1f} deg) -- the squat/arm-raise "
            "target for this joint likely has no effect (e.g. zero IK weight)"
        )
        assert l_rom >= min_rom_deg, f"{l_name} barely moved (ROM={l_rom:.1f} deg)"
        asym = float(np.max(np.abs(r_deg - l_deg)))
        assert asym <= max_asym_deg, (
            f"{r_name}/{l_name} diverged by {asym:.1f} deg for a bilaterally "
            "symmetric input motion -- one side likely isn't tracking correctly"
        )

    # -- Fork B: the same clip drives real Hill-type muscle-activation dynamics --
    from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

    env = MuscleMimicClipEnvV0(clip_path=clip_path, seed=0, use_obs_normalizer=False)
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs))

    rewards = []
    for _ in range(5):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        rewards.append(reward)
        if terminated or truncated:
            break
    assert len(rewards) > 0
