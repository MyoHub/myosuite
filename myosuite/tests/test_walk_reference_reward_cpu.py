# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only validation of the walk_env_reward reference reward.

This test checks that the backend-agnostic reference implementation in
``myosuite.terms.base_reward.walk_env_reward`` matches ``LegWalkEnvV0``
(``myoLegWalk-v0``) when ``task_state`` is built from the same ``obs_dict`` and
``qpos`` the env passes into the term.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("steps", [5])
def test_walk_reference_reward_matches_cpu(steps: int) -> None:
    """Compare LegWalkEnvV0 reward vs walk_env_reward on CPU."""
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0
    from myosuite.terms.base_reward import walk_env_reward
    from myosuite.envs.gymnasium_env import CpuEnvAccessor

    seed = 0
    rng = np.random.default_rng(seed)

    env = gym.make("myoLegWalk-v0", reset_type="init")
    obs, info = env.reset(seed=seed)
    del obs, info

    impl: LegWalkEnvV0 = env.unwrapped  # type: ignore[assignment]
    model = impl.model
    data = impl.data

    def _qadr(name: str) -> int:
        jnt_id = model.joint(name).id
        return int(model.jnt_qposadr[jnt_id])

    hip_angle_indices = (
        _qadr("hip_adduction_l"),
        _qadr("hip_adduction_r"),
        _qadr("hip_rotation_l"),
        _qadr("hip_rotation_r"),
    )

    hip_flex_indices = (
        _qadr("hip_flexion_l"),
        _qadr("hip_flexion_r"),
    )

    # Same default as LegWalkEnvV0.get_reward_dict (keyframe 0 root orientation).
    target_rot = (
        impl.target_rot.copy()
        if impl.target_rot is not None
        else model.key_qpos[0][3:7].copy()
    )

    act_dim = env.action_space.shape[0]

    for _ in range(steps):
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)
        obs, rwd, terminated, truncated, info = env.step(action)
        del obs, rwd, terminated, truncated

        rwd_cpu = info["rwd_dict"]
        obs_dict = info["obs_dict"]

        accessor = CpuEnvAccessor(model, data, ctrl_dt=impl.dt)
        task_state = {
            "qpos": data.qpos.copy(),
            "height": float(np.asarray(obs_dict["height"][0]).item()),
            "com_vel": np.asarray(obs_dict["com_vel"][0]),
            "phase_var": obs_dict["phase_var"],
        }

        ref = walk_env_reward(
            accessor,
            task_state,
            hip_flex_indices=hip_flex_indices,
            hip_angle_indices=hip_angle_indices,
            target_rot=target_rot,
            target_vel=(impl.target_x_vel, impl.target_y_vel),
            min_height=impl.min_height,
            max_rot=impl.max_rot,
        )

        atol_per_key = {
            "vel_reward": 1e-7,
            "done": 1e-7,
            "ref_rot": 1e-7,
            "joint_angle_rew": 1e-7,
            "cyclic_hip": 1e-7,
            "dense": 1e-7,
        }

        for key in ("vel_reward", "done", "ref_rot", "joint_angle_rew", "cyclic_hip"):
            cpu_val = float(rwd_cpu[key])
            ref_val = float(ref[key])
            assert np.allclose(
                ref_val,
                cpu_val,
                atol=atol_per_key[key],
            ), f"{key} mismatch: cpu={cpu_val}, ref={ref_val}"

        dense_cpu = float(rwd_cpu["dense"])
        dense_ref = float(ref["dense"])
        assert np.allclose(
            dense_ref,
            dense_cpu,
            atol=atol_per_key["dense"],
        ), f"dense mismatch: cpu={dense_cpu}, ref={dense_ref}"

    env.close()
