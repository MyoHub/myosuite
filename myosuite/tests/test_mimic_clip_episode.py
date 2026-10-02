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
