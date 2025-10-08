import jax
from jax import numpy as jp
from flax import struct
from mujoco_playground._src.mjx_env import State

import io
import imageio
import wandb

from mujoco_playground import wrapper

def make_minimal_state(full_state):
    """Create a minimal State suitable for rendering only."""

    @struct.dataclass
    class MinimalData:
        qpos: jp.ndarray
        qvel: jp.ndarray
        mocap_pos: jp.ndarray
        mocap_quat: jp.ndarray
        xfrc_applied: jp.ndarray

    minimal_data = MinimalData(
        qpos=full_state.data.qpos,
        qvel=full_state.data.qvel,
        mocap_pos=full_state.data.mocap_pos,
        mocap_quat=full_state.data.mocap_quat,
        xfrc_applied=full_state.data.xfrc_applied
    )

    return State(
        data=minimal_data,      # only the arrays used for rendering
        obs={},                 # empty dummy
        reward=jp.array(0.0),   # dummy
        done=jp.array(False),   # dummy
        metrics={},             # empty
        info={}                 # empty
    )

def make_policy_params_fn(env):

    eval_env = wrapper.wrap_for_brax_training(env, episode_length=env._config.max_episode_steps, action_repeat=1)
    jit_env_reset = jax.jit(eval_env.reset)
    jit_env_step = jax.jit(eval_env.step)
    def make_jit_policy(make_policy, params, deterministic):
        return jax.jit(make_policy(params, deterministic))

    def policy_params_fn(num_steps, make_policy, params):

        policy = make_jit_policy(make_policy, params, deterministic=True)

        key = jax.random.PRNGKey(seed=num_steps)
        key, subkey = jax.random.split(key)
        state = jit_env_reset(rng=subkey[None,:])
        if hasattr(eval_env, "_target_sids"):
            eval_env._mj_model.site_pos[eval_env._target_sids] = state.info['targets']

        rollout = [make_minimal_state(state)]
        for i in range(eval_env._config.max_episode_steps):
            key, subkey = jax.random.split(key)
            action, _ = policy(state.obs, subkey[None, :])
            state = jit_env_step(state, action)
            rollout.append(make_minimal_state(state))

        frames = eval_env.render(rollout, height=240, width=320)

        video_bytes = io.BytesIO()
        imageio.mimwrite(video_bytes, frames, format='mp4')
        video_bytes.seek(0)

        wandb.log({"rollout_video": wandb.Video(video_bytes, format="mp4")}, step=num_steps)

    return policy_params_fn
