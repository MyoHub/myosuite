# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import jax
from jax import numpy as jp
from flax import struct

import io
import imageio

from myosuite.envs.myo.backends.mjx.brax_env import State, wrap_for_brax_training


def _unbatch_first(arr: jp.ndarray, unbatched_ndim: int) -> jp.ndarray:
    return arr[0] if arr.ndim > unbatched_ndim else arr


def make_minimal_state(full_state):
    """Create a minimal single-env State suitable for rendering only."""

    @struct.dataclass
    class MinimalData:
        qpos: jp.ndarray
        qvel: jp.ndarray
        mocap_pos: jp.ndarray
        mocap_quat: jp.ndarray
        xfrc_applied: jp.ndarray

    minimal_data = MinimalData(
        qpos=_unbatch_first(full_state.data.qpos, 1),
        qvel=_unbatch_first(full_state.data.qvel, 1),
        mocap_pos=_unbatch_first(full_state.data.mocap_pos, 2),
        mocap_quat=_unbatch_first(full_state.data.mocap_quat, 2),
        xfrc_applied=_unbatch_first(full_state.data.xfrc_applied, 2),
    )

    return State(
        data=minimal_data,  # only the arrays used for rendering
        obs={},  # empty dummy
        reward=jp.array(0.0),  # dummy
        done=jp.array(False),  # dummy
        metrics={},  # empty
        info={},  # empty
    )


def make_policy_params_fn(env):
    eval_env = wrap_for_brax_training(
        env, episode_length=env._config.max_episode_steps, action_repeat=1
    )
    jit_env_reset = jax.jit(eval_env.reset)
    jit_env_step = jax.jit(eval_env.step)

    def make_jit_policy(make_policy, params, deterministic):
        return jax.jit(make_policy(params, deterministic))

    def policy_params_fn(num_steps, make_policy, params):
        policy = make_jit_policy(make_policy, params, deterministic=True)

        key = jax.random.PRNGKey(seed=num_steps)
        key, subkey = jax.random.split(key)
        state = jit_env_reset(rng=subkey[None, :])
        if hasattr(eval_env, "_target_sids"):
            targets = state.info["targets"]
            if targets.ndim >= 2 and targets.shape[0] == 1:
                targets = targets[0]
            eval_env._mj_model.site_pos[eval_env._target_sids] = targets

        rollout = [make_minimal_state(state)]
        for _ in range(eval_env._config.max_episode_steps):
            key, subkey = jax.random.split(key)
            action, _ = policy(state.obs, subkey)
            state = jit_env_step(state, action)
            rollout.append(make_minimal_state(state))

        frames = eval_env.unwrapped.render(rollout, height=240, width=320)

        video_bytes = io.BytesIO()
        imageio.mimwrite(video_bytes, frames, format="mp4")
        video_bytes.seek(0)

        try:
            import wandb  # noqa: PLC0415
        except ImportError:
            wandb = None
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {"rollout_video": wandb.Video(video_bytes, format="mp4")},
                step=num_steps,
            )

    return policy_params_fn
