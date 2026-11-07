# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Env state, physics step, renderer and brax training wrappers of the MJX backend.

Adapted by the MyoSuite Authors from ``mujoco_playground`` 0.2.0, whose 1.0
release (a PyTorch / MuJoCo Warp rewrite) dropped them:

- ``State``, ``step`` and ``render_array`` from ``_src/mjx_env.py``;
- ``AutoResetWrapper`` (``BraxAutoResetWrapper``) and ``wrap_for_brax_training``
  from ``_src/wrapper.py``.

Changes: type hints; no ``State.tree_replace``; ``render_array`` renders lists
only, without a progress bar or heightfield upload; the auto-reset has no
``full_reset`` mode and ``wrap_for_brax_training`` no domain randomization.
Rollouts match 0.2.0's bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Wrapper
from brax.envs.wrappers import training as brax_training
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    """Environment state for training and inference.

    Attributes:
        data: MJX simulation data.
        obs: Observation dict (``{"state": ...}``, plus wrapper keys).
        reward: Reward of the last step.
        done: 1.0 when the episode terminated, else 0.0.
        metrics: Named scalars that brax logs per episode.
        info: Task state, step counter, rng and wrapper bookkeeping.
    """

    data: mjx.Data
    obs: Mapping[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, Any]


def step(
    model: mjx.Model, data: mjx.Data, action: jax.Array, n_substeps: int = 1
) -> mjx.Data:
    """Hold *action* as ``ctrl`` for *n_substeps* MJX steps."""

    def single_step(data: mjx.Data, _: Any) -> tuple[mjx.Data, None]:
        return mjx.step(model, data.replace(ctrl=action)), None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]


def render_array(
    mj_model: mujoco.MjModel,
    trajectory: Sequence[State],
    height: int = 480,
    width: int = 640,
    camera: str | None = None,
    scene_option: mujoco.MjvOption | None = None,
    modify_scene_fns: Sequence[Callable[[mujoco.MjvScene], None]] | None = None,
) -> list[np.ndarray]:
    """Render one frame per state of a single-env trajectory.

    Args:
        mj_model: Host model of the env.
        trajectory: States to render.
        height: Image height in pixels.
        width: Image width in pixels.
        camera: Camera name; ``None`` uses the free camera.
        scene_option: Visualisation options.
        modify_scene_fns: Per-frame callbacks that edit the scene before rendering.

    Returns:
        One ``(height, width, 3)`` uint8 image per state.
    """
    data = mujoco.MjData(mj_model)
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        for i, state in enumerate(trajectory):
            data.qpos, data.qvel = state.data.qpos, state.data.qvel
            data.mocap_pos, data.mocap_quat = (
                state.data.mocap_pos,
                state.data.mocap_quat,
            )
            data.xfrc_applied = state.data.xfrc_applied
            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(
                data,
                camera=camera if camera is not None else -1,
                scene_option=scene_option,
            )
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            frames.append(renderer.render())
    return frames


_KEY = "AutoResetWrapper"


class AutoResetWrapper(Wrapper):
    """Reset done envs to the data and observation of their first reset.

    ``info`` carries on: the env re-samples its own task when an episode ends.
    Brax's own ``AutoResetWrapper`` keys on ``State.pipeline_state``.
    """

    def reset(self, rng: jax.Array) -> State:
        """Reset all envs and cache the first state.

        Args:
            rng: Batched random keys, shape ``(num_envs, 2)``.

        Returns:
            Batched initial state.
        """
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[f"{_KEY}_first_data"] = state.data
        state.info[f"{_KEY}_first_obs"] = state.obs
        state.info[f"{_KEY}_rng"] = rng
        state.info[f"{_KEY}_done_count"] = jp.zeros(key.shape[:-1], dtype=int)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Step all envs; those that are done restart from the cached first state.

        Args:
            state: Batched state.
            action: Batched actions.

        Returns:
            Batched next state.
        """
        reset_rng = jax.vmap(jax.random.split)(state.info[f"{_KEY}_rng"])[..., 0]
        reset_data = state.info[f"{_KEY}_first_data"]
        reset_obs = state.info[f"{_KEY}_first_obs"]
        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(steps=jp.where(state.done, jp.zeros_like(steps), steps))
        state = self.env.step(state.replace(done=jp.zeros_like(state.done)), action)

        def where_done(x: jax.Array, y: jax.Array) -> jax.Array:
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)
        state.info[f"{_KEY}_done_count"] += state.done.astype(int)
        state.info[f"{_KEY}_rng"] = reset_rng
        return state.replace(data=data, obs=obs)


def wrap_for_brax_training(
    env: Any,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[..., Any] | None = None,
) -> AutoResetWrapper:
    """Vmap, episode and auto-reset wrappers, the ``wrap_env_fn`` of brax's trainers.

    Args:
        env: A MyoSuite MJX env, or a wrapper of one such as ``FatigueWrapper``.
        episode_length: Steps before truncation.
        action_repeat: Env steps per action.
        randomization_fn: Not supported; brax passes ``None`` unless asked.

    Returns:
        The wrapped env, batched over its first axis.

    Raises:
        NotImplementedError: If *randomization_fn* is given.
    """
    if randomization_fn is not None:
        raise NotImplementedError(
            "MyoSuite's MJX envs do not support domain randomization."
        )
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return AutoResetWrapper(env)
