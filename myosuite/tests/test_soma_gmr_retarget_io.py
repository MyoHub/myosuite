# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GMR (amathislab/gmr_plus) -> MyoSuite MotionClip converter.

Covers two layers:
  * Fast, dependency-free unit tests of the pkl -> NPZ conversion against a
    synthetic (but shape-correct) GMR pickle, including the finger-joint
    coordinate drop between GMR's raw myofullbody model and MyoSuite's
    compiled (finger-removed) mimic model.
  * An optional end-to-end test that runs the real ``general_motion_retargeting``
    (gmr_plus) IK retargeter against the real ``myofullbody`` model with a
    synthetic BVH-style human pose sequence, then feeds the converted clip into
    MuscleMimicClipEnvV0. Skipped unless gmr_plus is importable.
"""

from __future__ import annotations

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

# Standard BVH body names used by gmr_plus's bvh_soma_to_myofullbody.json IK
# config (also the naming SOMA's own Blender/BVH tooling uses).
_BVH_BODIES: tuple[str, ...] = (
    "Hips",
    "Chest",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)

# Rough neutral offsets (metres) for a T-pose-ish rig, root at the pelvis.
# Not a real skeleton -- just plausible enough for the IK solver to converge
# to *some* pose per frame, which is all this test needs to exercise the
# real mink/mujoco retargeting code path end to end.
_BVH_REST_POS: dict[str, tuple[float, float, float]] = {
    "Hips": (0.0, 0.0, 0.9),
    "Chest": (0.0, 0.0, 1.15),
    "Head": (0.0, 0.0, 1.5),
    "LeftArm": (0.2, 0.0, 1.35),
    "LeftForeArm": (0.45, 0.0, 1.35),
    "LeftHand": (0.7, 0.0, 1.35),
    "RightArm": (-0.2, 0.0, 1.35),
    "RightForeArm": (-0.45, 0.0, 1.35),
    "RightHand": (-0.7, 0.0, 1.35),
    "LeftLeg": (0.1, 0.0, 0.9),
    "LeftShin": (0.1, 0.0, 0.5),
    "LeftFoot": (0.1, 0.05, 0.08),
    "LeftToeBase": (0.1, 0.15, 0.04),
    "RightLeg": (-0.1, 0.0, 0.9),
    "RightShin": (-0.1, 0.0, 0.5),
    "RightFoot": (-0.1, 0.05, 0.08),
    "RightToeBase": (-0.1, 0.15, 0.04),
}

_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _synthetic_bvh_human_data(n_frames: int = 8):
    """Synthetic per-frame human_data dicts in gmr_plus's expected format.

    Each frame maps body_name -> (pos: (3,), quat_wxyz: (4,)). Animates a
    small left-arm raise over the sequence, holding everything else at rest,
    since real SOMA/BVH mocap data isn't available in this environment.
    """
    frames = []
    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)
        raise_z = 0.35 * s  # left arm rises over the sequence
        frame = {}
        for body in _BVH_BODIES:
            x, y, z = _BVH_REST_POS[body]
            if body in ("LeftForeArm", "LeftHand"):
                z += raise_z
            frame[body] = (
                np.array([x, y, z], dtype=np.float64),
                np.array(_IDENTITY_QUAT_WXYZ, dtype=np.float64),
            )
        frames.append(frame)
    return frames


def test_end_to_end_soma_style_bvh_retarget_to_muscle_activation_env(
    tmp_path: Path, myofullbody_model
):
    """Real gmr_plus IK (mink + myofullbody MJCF) -> converter -> mimic env.

    Exercises the full chain proposed for interfacing SOMA-style kinematic
    reconstruction with MyoSuite: a human pose sequence in the same
    body-name schema gmr_plus's `bvh_soma_to_myofullbody.json` config
    expects is retargeted with the *real* GeneralMotionRetargeting IK solver
    against the *real* raw myofullbody model, converted to a MotionClip NPZ
    (finger coordinates dropped, mimic sites forward-kinematically
    computed), and used to drive MuscleMimicClipEnvV0 (whose action space is
    the Hill-type muscle activation MyoSuite forks off the same
    reconstructed kinematics).

    The input pose sequence is synthetic (no licensed SOMA/AMASS capture is
    available in this sandbox) but the retargeting and simulation code paths
    are the real ones, unmocked.
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

    # The shipped musclemimic_models myofullbody.xml has a handful of
    # coupled OpenSim-derived DoFs (e.g. knee translation, driven by an
    # mjEQ_JOINT polynomial off the primary knee angle) whose declared
    # `range` doesn't actually include the model's own qpos0/keyframe-0
    # default (0.0) -- a pre-existing MJCF/keyframe inconsistency, not
    # something this test or the SOMA/GMR conversion introduces. mink's
    # ConfigurationLimit checks the *raw* starting qpos before the IK
    # solver's EqualityConstraintTask has a chance to pull dependent DoFs
    # back into range, so it trips immediately on frame 0. Nudge any
    # out-of-range limited hinge/slide joint to its range midpoint once,
    # up front, so the solver starts from a legal configuration.
    model = retargeter.model
    qpos0 = retargeter.configuration.data.qpos.copy()
    hinge_or_slide = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    for jid in range(model.njnt):
        if not model.jnt_limited[jid] or int(model.jnt_type[jid]) not in (
            int(t) for t in hinge_or_slide
        ):
            continue
        qadr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        if not (lo - 1e-6 <= qpos0[qadr] <= hi + 1e-6):
            qpos0[qadr] = 0.5 * (lo + hi)
    retargeter.configuration.update(qpos0)

    human_frames = _synthetic_bvh_human_data(n_frames=8)
    qpos_frames = []
    for human_data in human_frames:
        qpos, _err = retargeter.retarget(dict(human_data))
        qpos_frames.append(np.asarray(qpos, dtype=np.float64).copy())
    qpos_frames = np.stack(qpos_frames, axis=0)
    assert qpos_frames.shape[1] == retargeter.model.nq  # GMR's raw model width
    assert np.all(np.isfinite(qpos_frames))

    fps = 30.0
    gmr_data = {
        "root_pos": qpos_frames[:, :3],
        # mujoco qpos quat is wxyz; GMR pkl convention is xyzw -- convert back
        # so gmr_pkl_to_motion_clip_npz's wxyz<-xyzw handling round-trips.
        "root_rot": qpos_frames[:, [4, 5, 6, 3]],
        "dof_pos": qpos_frames[:, 7:],
        "fps": fps,
    }
    pkl_path = tmp_path / "soma_bvh_myofullbody.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(gmr_data, f)

    clip_path = gmr_pkl_to_motion_clip_npz(
        pkl_path, tmp_path / "soma_bvh_myofullbody.npz", model=myofullbody_model
    )

    from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

    env = MuscleMimicClipEnvV0(clip_path=clip_path, seed=0, use_obs_normalizer=False)
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs))

    rewards = []
    for _ in range(5):
        action = np.zeros(
            env.action_space.shape, dtype=np.float32
        )  # zero muscle activation
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        rewards.append(reward)
        if terminated or truncated:
            break
    assert len(rewards) > 0
