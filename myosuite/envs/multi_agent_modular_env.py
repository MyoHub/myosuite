# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ModularMultiAgentTaskEnv — data-driven two-agent competitive environment.

The entire episode loop (reset, step, render, close) is implemented here.
Task-specific logic (model, obs, damage, reward) lives entirely in
:class:`~myosuite.core.multi_agent_config.MultiAgentTaskConfig` hook methods.

Example::

    env = ModularMultiAgentTaskEnv(BoxingVsTaskConfig())
    obs, info = env.reset(seed=0)
    # obs = {"agent_0": np.ndarray, "agent_1": np.ndarray}

    actions = {"agent_0": ..., "agent_1": ...}
    obs, rewards, terminated, truncated, info = env.step(actions)
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.core.multi_agent_config import MultiAgentTaskConfig

logger = logging.getLogger(__name__)


class ModularMultiAgentTaskEnv(gym.Env):
    """Two-agent simultaneous-step environment driven by :class:`MultiAgentTaskConfig`.

    Both agents act each step; the episode ends when any agent falls, is KO'd,
    or the step limit is reached.

    Interface::

        env = ModularMultiAgentTaskEnv(task_config)
        obs, info = env.reset(seed=0)
        # obs["agent_0"], obs["agent_1"] — per-agent observation arrays

        actions = {"agent_0": a0, "agent_1": a1}
        obs, rewards, terminated, truncated, info = env.step(actions)
        # All return dicts keyed by agent id, except info (flat dict).

    Args:
        task_config: Task configuration supplying model, obs, and reward hooks.
        render_mode: ``"human"`` (interactive) or ``"rgb_array"`` (off-screen).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_config: MultiAgentTaskConfig,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = task_config
        self._agents = task_config.agents
        self.render_mode = render_mode

        self.model, self.data, self._meta = task_config.build_model()
        self._spec: mujoco.MjSpec | None = task_config.build_spec()
        self._n_substeps: int = task_config.n_substeps

        # Episode state — initialised properly in reset().
        self._health: dict[str, float] = {a: 0.0 for a in self._agents}
        self._step_count: int = 0

        {a: 0.0 for a in self._agents}
        obs_dims = {
            a: task_config.obs_dim(self.model, self.data, self._meta, a)
            for a in self._agents
        }
        act_sizes = {
            a: len(task_config.act_indices(self._meta, a)) for a in self._agents
        }

        self.observation_space = gym.spaces.Dict(
            {
                a: gym.spaces.Box(
                    -np.inf, np.inf, shape=(obs_dims[a],), dtype=np.float32
                )
                for a in self._agents
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                a: gym.spaces.Box(
                    low=np.zeros(act_sizes[a], dtype=np.float32),
                    high=np.ones(act_sizes[a], dtype=np.float32),
                    dtype=np.float32,
                )
                for a in self._agents
            }
        )

        self._viewer: Any = None
        self._renderer: Any = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the episode.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Ignored (reserved for future use).

        Returns:
            Tuple of (observations dict, info dict).
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        # See ModularTaskEnv.reset() in modular_env.py: mj_resetData alone
        # zeros ALL qpos, which is not a valid standing pose for any
        # humanoid/leg host model and collapses instantly under gravity.
        # Two-agent combined models don't carry a compiled-in keyframe
        # (MjSpec.attach drops them -- see two_agent_standing_qpos), so the
        # per-agent standing pose is reconstructed once at model-build time
        # and cached on meta; prefer that, falling back to a raw keyframe 0
        # if the model happens to have one, before agent on_reset() runs.
        standing_qpos = getattr(self._meta, "standing_qpos", None)
        if standing_qpos is not None:
            self.data.qpos[:] = standing_qpos
        elif self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._config.on_reset(self.model, self.data, self._meta)

        self._health = {a: 0.0 for a in self._agents}
        self._step_count = 0

        obs = self._get_obs()
        info = self._config.get_info(
            self.data,
            self._meta,
            self._health,
            {a: 0.0 for a in self._agents},
            {a: False for a in self._agents},
            {a: False for a in self._agents},
            self._step_count,
        )
        return obs, info

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Advance physics by one control step with all agents acting simultaneously.

        Args:
            actions: ``{agent_id: action_array}`` for every agent.  Each action
                is clipped to ``[0, 1]`` and written to the corresponding
                actuator indices.

        Returns:
            5-tuple ``(obs, rewards, terminated, truncated, info)`` — all dicts
            keyed by agent id except *info* which is a flat dict.
        """
        # Apply all actions simultaneously before any physics.
        for agent_id in self._agents:
            act = np.clip(actions[agent_id], 0.0, 1.0).astype(np.float64)
            idx = self._config.act_indices(self._meta, agent_id)
            self.data.ctrl[idx] = act

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Damage: each agent deals damage to its opponent.
        damage_delivered = self._config.compute_damage(
            self.model, self.data, self._meta
        )

        scale = self._config.health_update_scale
        ko_thresh = self._config.ko_threshold
        for agent_id in self._agents:
            opp = self._opponent(agent_id)
            self._health[agent_id] = min(
                self._health[agent_id] + damage_delivered[opp] * scale,
                ko_thresh,
            )

        fell = {
            a: self._config.check_fell(self.data, self._meta, a) for a in self._agents
        }
        ko = {a: self._health[a] >= ko_thresh for a in self._agents}
        episode_over = any(fell.values()) or any(ko.values())

        terminated = {a: episode_over for a in self._agents}
        truncated = {
            a: (not episode_over)
            and (self._step_count >= self._config.max_episode_steps)
            for a in self._agents
        }

        rewards = {
            a: self._config.compute_reward(
                self.data,
                self._meta,
                a,
                damage_delivered,
                self._health,
                actions[a],
                fell,
                ko,
            )
            for a in self._agents
        }

        obs = self._get_obs()
        info = self._config.get_info(
            self.data,
            self._meta,
            self._health,
            damage_delivered,
            fell,
            ko,
            self._step_count,
        )

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the scene.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` in ``"rgb_array"`` mode,
            ``None`` in ``"human"`` mode.
        """
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                try:
                    import mujoco.viewer as mj_viewer

                    self._viewer = mj_viewer.launch_passive(self.model, self.data)
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Could not launch passive MuJoCo viewer", exc_info=True
                    )
            if self._viewer is not None:
                try:
                    self._viewer.sync()
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning(
                        "MuJoCo viewer sync failed; closing viewer", exc_info=True
                    )
                    self._viewer = None
        return None

    def close(self) -> None:
        """Release renderer and viewer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                import logging

                logging.getLogger(__name__).debug(
                    "MuJoCo viewer close raised; ignoring", exc_info=True
                )
            self._viewer = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            a: self._config.get_obs(self.model, self.data, self._meta, a, self._health)
            for a in self._agents
        }

    def _opponent(self, agent_id: str) -> str:
        """Return the opponent agent id for a 2-agent game.

        Args:
            agent_id: One of ``self._agents``.

        Returns:
            The other agent's id.

        Raises:
            ValueError: If the task has more or fewer than 2 agents.
        """
        others = [a for a in self._agents if a != agent_id]
        if len(others) != 1:
            raise ValueError(
                f"_opponent() requires exactly 2 agents; {agent_id!r} has {len(others)} opponents"
            )
        return others[0]


class ModularMultiAgentSingleAgentWrapper(gym.Wrapper):
    """Wraps ``ModularMultiAgentTaskEnv`` as a single-agent env.

    Presents one agent's observation/action space as the primary interface.
    The opponent's action is generated by a fixed policy callable.

    This replaces both ``BoxingVsSingleAgentWrapper`` and
    ``SaberVsSingleAgentWrapper``.

    Args:
        env: The underlying ``ModularMultiAgentTaskEnv``.
        agent_id: The agent id to expose as the primary agent.
        opponent_policy: Callable ``(obs: np.ndarray) -> np.ndarray`` that
            returns the opponent's action given its current observation.
            Defaults to sampling from the opponent's action space.

    Example::

        base = ModularMultiAgentTaskEnv(BoxingVsTaskConfig())
        env = ModularMultiAgentSingleAgentWrapper(
            base,
            agent_id="agent_0",
            opponent_policy=lambda obs: base.action_space["agent_1"].sample(),
        )
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(my_policy(obs))
    """

    def __init__(
        self,
        env: ModularMultiAgentTaskEnv,
        agent_id: str = "agent_0",
        opponent_policy: Any | None = None,
    ) -> None:
        super().__init__(env)
        if agent_id not in env.unwrapped._agents:
            raise ValueError(
                f"agent_id {agent_id!r} not in agents {env.unwrapped._agents!r}"
            )
        self._agent_id = agent_id
        self._opp_id = env.unwrapped._opponent(agent_id)

        if opponent_policy is None:
            self._opp_policy = lambda obs: env.action_space[self._opp_id].sample()
        else:
            self._opp_policy = opponent_policy

        self.observation_space = env.observation_space[agent_id]
        self.action_space = env.action_space[agent_id]

        self._last_opp_obs: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the wrapped environment.

        Args:
            seed: Forwarded to the base env.
            options: Forwarded to the base env.

        Returns:
            Tuple of ``(agent_obs, info)``.
        """
        obs_dict, info = self.env.reset(seed=seed, options=options)
        self._last_opp_obs = obs_dict[self._opp_id]
        return obs_dict[self._agent_id], info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the wrapped environment.

        Args:
            action: Action for the primary agent.

        Returns:
            5-tuple ``(obs, reward, terminated, truncated, info)`` for the
            primary agent.
        """
        opp_action = self._opp_policy(self._last_opp_obs)
        actions = {self._agent_id: action, self._opp_id: opp_action}
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        self._last_opp_obs = obs_dict[self._opp_id]
        return (
            obs_dict[self._agent_id],
            rewards[self._agent_id],
            terminated[self._agent_id],
            truncated[self._agent_id],
            info,
        )
