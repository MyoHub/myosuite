# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for partial-joint MuscleMimic clips."""

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


def _select_scalar_joints(
    model: mujoco.MjModel,
    count: int = 4,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    joint_names: list[str] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type in (
            int(mujoco.mjtJoint.mjJNT_FREE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        ):
            continue
        joint_names.append(joint_name)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        if len(joint_names) == count:
            break
    if len(joint_names) < count:
        raise AssertionError("Expected enough 1-DoF joints for the partial clip test.")
    return (
        tuple(joint_names),
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
    )


def _write_partial_clip(path: Path) -> tuple[tuple[str, ...], int]:
    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    joint_names, qpos_indices, qvel_indices = _select_scalar_joints(model)
    frames = 3
    clip_qpos = np.repeat(data.qpos[qpos_indices][None, :], frames, axis=0)
    clip_qvel = np.repeat(data.qvel[qvel_indices][None, :], frames, axis=0)

    site_ids = np.asarray(
        [model.site(name).id for name in _MODEL_SITE_ORDER], dtype=np.int32
    )
    model_sites = data.site_xpos[site_ids].copy()
    clip_sites = np.zeros((frames, len(_MODEL_SITE_ORDER), 3), dtype=np.float64)
    clip_sites[:, _CLIP_IDX_FOR_MODEL, :] = model_sites[None, :, :]

    np.savez(
        path,
        qpos=clip_qpos,
        qvel=clip_qvel,
        site_xpos=clip_sites,
        joint_names=np.asarray(joint_names),
    )
    return joint_names, frames


def test_partial_joint_clip_tracks_named_subset(tmp_path: Path) -> None:
    clip_path = tmp_path / "partial_clip.npz"
    joint_names, _ = _write_partial_clip(clip_path)

    env = MuscleMimicClipEnvV0(
        clip_path=clip_path,
        seed=0,
        use_obs_normalizer=False,
        lookahead_k=2,
        lookahead_stride=1,
    )
    try:
        obs, _ = env.reset()
        reward, info = env._compute_reward()

        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert tuple(env._qpos_selection.joint_names) == joint_names
        assert tuple(env._qvel_selection.joint_names) == joint_names
        assert info["joint_pos"] == pytest.approx(1.0)
        assert info["joint_vel"] == pytest.approx(1.0)
        assert info["root_pos"] is None
        assert info["root_vel"] is None
        assert info["root_orient"] is None
        assert env._check_termination() is False

        expected_lookahead = env._lookahead_k * (len(_MODEL_SITE_ORDER) * 3 + 1)
        assert env.observation_space.shape == (
            env.model.nq + env.model.nv + env.model.na + expected_lookahead,
        )
    finally:
        env.close()
