# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Two-agent competitive boxing environment: ``BoxingVsEnv``.

Two MuscleMimic full-body agents compete in a 5-second boxing round.
Each agent controls its own muscles independently; they step simultaneously.

Interface (parallel multi-agent, PettingZoo-compatible without the dependency)::

    env = BoxingVsEnv()
    obs, info = env.reset(seed=0)
    # obs = {"agent_0": np.ndarray, "agent_1": np.ndarray}

    actions = {
        "agent_0": policy_0(obs["agent_0"]),
        "agent_1": policy_1(obs["agent_1"]),
    }
    obs, rewards, terminated, truncated, info = env.step(actions)
    # All return values are dicts keyed by agent id.

Gymnasium spaces::

    env.observation_space["agent_0"]  # Box(-inf, inf, shape=(obs_dim_0,))
    env.action_space["agent_0"]       # Box(0, 1, shape=(n_act_0,))
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from myosuite.envs.multi_agent_modular_env import ModularMultiAgentTaskEnv
from myosuite.envs.multi_agent_modular_env import ModularMultiAgentSingleAgentWrapper
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import AGENTS
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_task_config import (
    BoxingVsTaskConfig,
)

logger = logging.getLogger(__name__)


class BoxingVsEnv(gym.Env):
    """Two-agent competitive boxing — parallel simultaneous-step interface.

    Both agents control their own ~300-actuator MuscleMimic full-body model.
    Each call to :meth:`step` advances physics by one control timestep
    (``ctrl_dt = 0.01 s``, composed of 5 MuJoCo substeps at ``sim_dt = 0.002 s``).

    Termination conditions (checked each step, any agent triggering ends the episode):
    - An agent's cumulative health reaches ``ko_health_threshold`` (KO).
    - An agent's pelvis drops below ``fall_pelvis_z_threshold`` m (fall).
    - ``max_episode_steps`` reached → truncation.

    Args:
        config: Task configuration.  Pass ``BoxingVsConfig(...)`` to override
            any default.
        render_mode: ``"human"`` (interactive viewer) or ``"rgb_array"``
            (off-screen).  ``None`` disables rendering.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: BoxingVsConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config or BoxingVsConfig()
        self.render_mode = render_mode
        self._agents = AGENTS

        cfg = BoxingVsTaskConfig(**vars(self._config))
        self._base_env = ModularMultiAgentTaskEnv(cfg, render_mode=render_mode)
        self.model = self._base_env.model
        self.data = self._base_env.data
        self._meta = self._base_env._meta  # pylint: disable=protected-access
        self.observation_space = self._base_env.observation_space
        self.action_space = self._base_env.action_space

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Ignored (reserved for future use).

        Returns:
            Tuple of (observations dict, info dict).
        """
        return self._base_env.reset(seed=seed, options=options)

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
        """Advance physics by one control step with both agents acting simultaneously.

        Args:
            actions: Dict ``{"agent_0": action_0, "agent_1": action_1}`` where
                each action is a float32 array in ``[0, 1]`` with shape
                ``(n_act_i,)``.

        Returns:
            5-tuple of (obs, rewards, terminated, truncated, info) — each a
            dict keyed by agent id, except info which is a flat dict.
        """
        return self._base_env.step(actions)

    def render(self) -> np.ndarray | None:
        """Render the scene.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` in ``"rgb_array"`` mode,
            or ``None`` in ``"human"`` mode.
        """
        return self._base_env.render()

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Legacy Gym API seed method for compatibility with examine_env.py.

        Args:
            seed: Optional RNG seed.

        Returns:
            List containing the seed value.
        """
        return [seed]

    def close(self) -> None:
        """Release renderer and viewer resources."""
        self._base_env.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Compute observations for both agents.

        Returns:
            Dict mapping agent id → observation vector.
        """
        return self._base_env._get_obs()  # pylint: disable=protected-access

    def _get_info(
        self,
        damage_delivered: dict[str, float] | None = None,
        fell: dict[str, bool] | None = None,
        ko: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Build the info dict for the current step.

        Args:
            damage_delivered: Per-agent damage dealt this step.
            fell: Per-agent fall flags.
            ko: Per-agent KO flags.

        Returns:
            Flat info dict with diagnostic scalars.
        """
        del damage_delivered, fell, ko
        return {}

    # ------------------------------------------------------------------
    # Per-agent helpers (split / merge)
    # ------------------------------------------------------------------

    def _opponent(self, agent_id: str) -> str:
        """Return opponent id for a two-agent setup."""
        return "agent_1" if agent_id == "agent_0" else "agent_0"

    @staticmethod
    def split_obs(
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unpack the obs dict into ``(obs_agent_0, obs_agent_1)``.

        Args:
            obs: Dict returned by :meth:`reset` or :meth:`step`.

        Returns:
            Tuple ``(obs_0, obs_1)``.
        """
        return obs["agent_0"], obs["agent_1"]

    @staticmethod
    def merge_actions(
        action_0: np.ndarray,
        action_1: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Pack two per-agent action arrays into the dict expected by :meth:`step`.

        Args:
            action_0: Action for ``agent_0``, shape ``(n_act_0,)``.
            action_1: Action for ``agent_1``, shape ``(n_act_1,)``.

        Returns:
            Dict ``{"agent_0": action_0, "agent_1": action_1}``.
        """
        return {"agent_0": action_0, "agent_1": action_1}

    @staticmethod
    def split_rewards(
        rewards: dict[str, float],
    ) -> tuple[float, float]:
        """Unpack the rewards dict into ``(reward_agent_0, reward_agent_1)``.

        Args:
            rewards: Rewards dict returned by :meth:`step`.

        Returns:
            Tuple ``(r0, r1)``.
        """
        return rewards["agent_0"], rewards["agent_1"]

    # ------------------------------------------------------------------
    # Convenience properties for training scripts
    # ------------------------------------------------------------------

    @property
    def agents(self) -> tuple[str, ...]:
        """Agent ids: ``("agent_0", "agent_1")``."""
        return AGENTS

    @property
    def health(self) -> dict[str, float]:
        """Current health for both agents (read-only copy)."""
        return dict(self._base_env._health)  # pylint: disable=protected-access

    @property
    def n_substeps(self) -> int:
        """Number of MuJoCo substeps per ``step()`` call."""
        return int(self._base_env._n_substeps)  # pylint: disable=protected-access

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._config.ctrl_dt

    @property
    def visual_keys(self) -> list[str]:
        """No visual observation keys (required by examine_env.py)."""
        return []

    # ------------------------------------------------------------------
    # Legacy examine_env.py compatibility
    # ------------------------------------------------------------------

    def evaluate_success(self, paths: list[dict[str, Any]]) -> float:
        """Return success rate over rollout paths.

        Boxing has no binary success criterion, so this always returns 0.0.

        Args:
            paths: Episode summary dicts returned by :meth:`examine_policy`.

        Returns:
            0.0 (no defined success criterion).
        """
        return 0.0

    def examine_policy(
        self,
        policy: Any,
        horizon: int | None = None,
        num_episodes: int = 1,
        mode: str = "exploration",
        render: str | None = None,
        camera_name: str | None = None,
        frame_size: tuple[int, int] = (640, 480),
        output_dir: str = "/tmp/",
        filename: str = "newvid",
        device_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Roll out a policy for ``num_episodes`` and return episode summaries.

        Compatibility shim for ``myosuite.utils.examine_env``.

        Args:
            policy: Policy with ``get_action(obs)`` returning
                ``(action, {... "evaluation": action})``.  For multi-agent
                boxing, ``action`` must be a ``dict`` keyed by agent id.
            horizon: Max steps per episode. Defaults to
                ``config.max_episode_steps``.
            num_episodes: Number of episodes to run.
            mode: ``"exploration"`` uses ``get_action(obs)[0]``;
                ``"evaluation"`` uses ``get_action(obs)[1]["evaluation"]``.
            render: ``"onscreen"`` for live viewer, ``"offscreen"`` for silent
                rendering, ``None`` / ``"none"`` to skip.
            camera_name: Unused (kept for API compatibility).
            frame_size: Unused (kept for API compatibility).
            output_dir: Unused (kept for API compatibility).
            filename: Unused (kept for API compatibility).
            device_id: Unused (kept for API compatibility).

        Returns:
            List of episode summary dicts with keys ``episode``, ``steps``,
            ``rewards``, and ``info``.
        """
        max_steps = horizon or self._config.max_episode_steps
        paths: list[dict[str, Any]] = []

        if render == "onscreen":
            self.render_mode = "human"

        for ep in range(num_episodes):
            obs, _ = self.reset()
            done = False
            t = 0
            ep_rwd: dict[str, float] = {a: 0.0 for a in AGENTS}
            ep_info: list[dict[str, Any]] = []

            while t < max_steps and not done:
                action_result = (
                    policy.get_action(obs)[0]
                    if mode == "exploration"
                    else policy.get_action(obs)[1]["evaluation"]
                )
                obs, rewards, terminated, truncated, info = self.step(action_result)

                for a in AGENTS:
                    ep_rwd[a] += float(rewards[a])
                ep_info.append(info)

                done = any(terminated.values()) or any(truncated.values())
                t += 1

                if render == "onscreen":
                    self.render()

            logger.info(
                "Episode %d: %d steps | rewards %s",
                ep,
                t,
                {a: f"{r:.3f}" for a, r in ep_rwd.items()},
            )
            paths.append(
                {"episode": ep, "steps": t, "rewards": ep_rwd, "info": ep_info}
            )

        return paths


# ---------------------------------------------------------------------------
# Single-agent wrapper
# ---------------------------------------------------------------------------


class BoxingVsSingleAgentWrapper(ModularMultiAgentSingleAgentWrapper):
    """Present one agent's view of ``BoxingVsEnv`` as a standard ``gym.Env``.

    Wraps the two-agent env so that a single-agent policy (e.g. one trained on
    ``myoChallengeBoxingP0-v0``) can be trained or evaluated using ordinary
    Gymnasium tooling.  The opponent's action is computed by a fixed callable
    supplied at construction time.

    The wrapped env's ``observation_space`` and ``action_space`` are the
    single-agent Box spaces; ``reset`` / ``step`` follow the single-agent API.

    Example::

        from myosuite.envs.myo.tasks.challenge.boxing_vs import BoxingVsEnv
        from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_env import BoxingVsSingleAgentWrapper

        base = BoxingVsEnv()
        opponent_policy = lambda obs: base.action_space["agent_1"].sample()
        env = BoxingVsSingleAgentWrapper(base, agent_id="agent_0",
                                         opponent_policy=opponent_policy)

        obs, info = env.reset(seed=0)          # obs.shape == (603,)
        action = my_policy(obs)                # shape (354,)
        obs, reward, terminated, truncated, info = env.step(action)

    Args:
        env: An unwrapped ``BoxingVsEnv`` instance (not a ``gym.make`` wrapper).
        agent_id: ``"agent_0"`` or ``"agent_1"`` — the agent this view controls.
        opponent_policy: Callable ``(obs: np.ndarray) -> np.ndarray`` that
            returns an action for the opponent given its observation.  Defaults
            to uniform random sampling from the opponent's action space.
    """

    def __init__(
        self,
        env: BoxingVsEnv,
        agent_id: str = "agent_0",
        opponent_policy: Any | None = None,
    ) -> None:
        """Backward-compatible alias around the modular single-agent wrapper."""
        super().__init__(env=env, agent_id=agent_id, opponent_policy=opponent_policy)
