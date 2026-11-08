# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Episode start and end of ``MuscleMimicClipEnvV0`` on synthetic clips."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.tasks.mimic.clip_env import (
    _CLIP_IDX_FOR_MODEL,
    _MODEL_SITE_ORDER,
    MuscleMimicClipEnvV0,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.tests.support.joint_couplings import joint_equality_residuals

pytestmark = pytest.mark.tier1


def _write_standing_clip(path: Path, n_frames: int, frame0_offset: float) -> Path:
    """Full-width clip that holds the keyframe pose; frame 0's sites are shifted up."""
    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    site_ids = [model.site(name).id for name in _MODEL_SITE_ORDER]
    sites = np.zeros((n_frames, len(_MODEL_SITE_ORDER), 3))
    sites[:, _CLIP_IDX_FOR_MODEL, :] = data.site_xpos[site_ids]
    sites[0, :, 2] += frame0_offset
    np.savez(
        path,
        qpos=np.tile(data.qpos, (n_frames, 1)),
        qvel=np.zeros((n_frames, model.nv)),
        site_xpos=sites,
        frequency=np.array(100.0),
    )
    return path


def _write_105_left_knee_clip(path: Path, n_frames: int) -> Path:
    """Knee sweep whose coupled knee joints follow ``musclemimic_models`` 1.0.5.

    Like the retargeted MuscleMimic clips: the left knee's ``rotation3`` is on the
    1.0.5 polynomial, up to 0.5 rad off 1.0.6. The sites are the forward
    kinematics of the stored state (both releases have the same kinematic tree).
    """
    cfg = default_mimic_fullbody_config()
    cfg.model_version = "1.0.5"
    model, _, _ = compile_mimic_fullbody_mjmodel(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    qpos = np.tile(data.qpos, (n_frames, 1))
    qvel = np.zeros((n_frames, model.nv))
    for name in ("knee_angle_l", "knee_angle_r"):
        qpos[:, model.jnt_qposadr[model.joint(name).id]] = np.linspace(0, 1.2, n_frames)
        qvel[:, model.jnt_dofadr[model.joint(name).id]] = 1.2 / (n_frames - 1) * 100
    project_joint_couplings(JointCouplings.from_model(model), qpos, qvel)
    site_ids = [model.site(name).id for name in _MODEL_SITE_ORDER]
    sites = np.zeros((n_frames, len(_MODEL_SITE_ORDER), 3))
    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        sites[t, _CLIP_IDX_FOR_MODEL] = data.site_xpos[site_ids]
    np.savez(path, qpos=qpos, qvel=qvel, site_xpos=sites, frequency=np.array(100.0))
    return path


def _env(clip_path: Path, **kwargs: object) -> MuscleMimicClipEnvV0:
    return MuscleMimicClipEnvV0(
        clip_path=clip_path,
        use_obs_normalizer=False,
        lookahead_k=1,
        lookahead_stride=1,
        **kwargs,
    )


def test_random_start_false_starts_every_episode_at_frame_zero(tmp_path: Path) -> None:
    env = _env(_write_standing_clip(tmp_path / "c.npz", 50, 0.0), random_start=False)
    for seed in range(3):
        env.reset(seed=seed)
        assert env._current_frame() == 0
    env.close()


def test_random_start_draws_from_the_env_rng(tmp_path: Path) -> None:
    env = _env(_write_standing_clip(tmp_path / "c.npz", 50, 0.0), seed=7)
    starts = []
    for seed in range(4):
        env.reset(seed=seed)
        first = env._current_frame()
        env.reset(seed=seed)
        assert env._current_frame() == first
        starts.append(first)
    assert len(set(starts)) > 1
    env.close()


def test_reset_puts_the_coupled_joints_on_their_polynomials(tmp_path: Path) -> None:
    """The clip holds the keyframe, whose patella is 5 cm off its knee coupling."""
    clip_path = _write_standing_clip(tmp_path / "c.npz", 20, 0.0)
    env = _env(clip_path, seed=0)
    stored = np.load(clip_path)["qpos"][0]
    assert np.abs(joint_equality_residuals(env.model, stored)[0]).max() > 0.05
    env.reset(seed=0)
    pos, vel = joint_equality_residuals(env.model, env.data.qpos, env.data.qvel)
    assert np.abs(pos).max() < 1e-12
    assert np.abs(vel).max() < 1e-12
    env.close()


def test_clip_references_are_on_the_couplings_of_the_model(tmp_path: Path) -> None:
    """A clip with the 1.0.5 left knee is served on the 1.0.6 couplings.

    Joint references, velocities and site targets are one reachable state per
    frame: the sites below the knees move with the projected joints.
    """
    clip_path = _write_105_left_knee_clip(tmp_path / "c.npz", 30)
    env = _env(clip_path, seed=0)
    stored = np.load(clip_path)
    worst = max(
        np.abs(joint_equality_residuals(env.model, q)[0]).max() for q in stored["qpos"]
    )
    assert worst > 0.4  # knee_angle_rotation3_l, as in the retargeted clips
    data = mujoco.MjData(env.model)
    site_ids = [env.model.site(name).id for name in _MODEL_SITE_ORDER]
    for t in range(len(env._clip_qpos)):
        pos, vel = joint_equality_residuals(
            env.model, env._clip_qpos[t], env._clip_qvel[t]
        )
        assert max(np.abs(pos).max(), np.abs(vel).max()) < 1e-12
        data.qpos[:] = env._clip_qpos[t]
        mujoco.mj_kinematics(env.model, data)
        np.testing.assert_allclose(
            env._clip_site_xpos[t], data.site_xpos[site_ids], rtol=0, atol=1e-12
        )
    np.testing.assert_array_equal(env._clip_joint_qpos, env._clip_qpos)
    np.testing.assert_array_equal(env._clip_joint_qvel, env._clip_qvel)
    moved = np.linalg.norm(
        env._clip_site_xpos - stored["site_xpos"][:, _CLIP_IDX_FOR_MODEL], axis=-1
    )
    assert moved[:, _MODEL_SITE_ORDER.index("left_toes_mimic")].max() > 0.02
    assert moved[:, _MODEL_SITE_ORDER.index("pelvis_mimic")].max() == 0.0
    env.close()


def test_stored_clip_references_keep_the_clip_and_project_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``MYOSUITE_MIMIC_CLIP_REFERENCES=stored`` serves the clip as retargeted."""
    monkeypatch.setenv("MYOSUITE_MIMIC_CLIP_REFERENCES", "stored")
    clip_path = _write_105_left_knee_clip(tmp_path / "c.npz", 10)
    env = _env(clip_path, seed=0)
    stored = np.load(clip_path)
    np.testing.assert_array_equal(env._clip_qpos, stored["qpos"])
    np.testing.assert_array_equal(env._clip_qvel, stored["qvel"])
    np.testing.assert_array_equal(
        env._clip_site_xpos, stored["site_xpos"][:, _CLIP_IDX_FOR_MODEL]
    )
    env.reset(seed=0)
    pos, vel = joint_equality_residuals(env.model, env.data.qpos, env.data.qvel)
    assert max(np.abs(pos).max(), np.abs(vel).max()) < 1e-12
    env.close()


def test_unknown_keyword_arguments_are_rejected(tmp_path: Path) -> None:
    clip_path = _write_standing_clip(tmp_path / "c.npz", 5, 0.0)
    with pytest.raises(TypeError, match="max_episode_steps"):
        _env(clip_path, max_episode_steps=10)


def test_clip_end_step_is_scored_against_the_last_frame(tmp_path: Path) -> None:
    """The truncating step compares the state with frame T-1, not wrapped frame 0.

    Two clips differ only in frame 0 (shifted 1 m up); from frame 0 both envs run
    the same physics, so the clip-end reward must not depend on frame 0.
    """
    n_frames = 4
    rewards, frames = [], []
    for name, shift in (("plain", 0.0), ("shifted", 1.0)):
        env = _env(
            _write_standing_clip(tmp_path / f"{name}.npz", n_frames, shift),
            random_start=False,
        )
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(1, n_frames + 1):
            _, reward, terminated, truncated, info = env.step(action)
            assert truncated == (step == n_frames)
            assert not terminated
        rewards.append(reward)
        frames.append(info["frame"])
        env.close()
    assert frames == [n_frames - 1, n_frames - 1]
    assert rewards[1] == pytest.approx(rewards[0], abs=1e-12)
