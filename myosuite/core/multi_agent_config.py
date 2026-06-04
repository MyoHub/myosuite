# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Abstract base class for two-agent competitive task configurations.

``MultiAgentTaskConfig`` is the single-source-of-truth for a two-agent task:
it builds the MuJoCo model, supplies per-agent observation / action dimensions,
and implements the damage→health→KO termination contract used by both boxing
and lightsaber challenges.

``ModularMultiAgentTaskEnv`` calls these hooks to orchestrate the simultaneous
step loop so that concrete subclasses contain zero boilerplate.

Example::

    @dataclass
    class BoxingVsTaskConfig(MultiAgentTaskConfig):
        ko_threshold: float = 100.0
        ctrl_dt: float = 0.01

        def build_model(self):
            model, data, meta = build_boxing_vs_model(...)
            return model, data, meta

        def compute_damage(self, model, data, meta):
            return {a: compute_total_damage(...) for a in self.agents}

        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mujoco
import numpy as np


class MultiAgentTaskConfig(ABC):
    """Abstract base for two-agent competitive task configurations.

    Subclass this (typically as a ``@dataclass``) and implement all abstract
    methods to define a new competitive task.  ``ModularMultiAgentTaskEnv``
    drives the episode loop entirely through these hooks.

    Attributes:
        agents: Ordered tuple of agent identifiers.  Fixed at
            ``("agent_0", "agent_1")`` for all current challenges.
        max_episode_steps: Episode length limit (triggers truncation).
    """

    agents: tuple[str, ...] = ("agent_0", "agent_1")
    max_episode_steps: int = 1000

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
        """Build and return the combined MuJoCo model.

        Returns:
            Tuple of ``(model, data, meta)`` where *meta* is a task-specific
            pre-computed index cache (e.g. ``BoxingVsModelMeta``).
        """

    def build_spec(self) -> mujoco.MjSpec | None:
        """Build and return the combined MjSpec without compiling.

        Used by the export pipeline to obtain an XML-serialisable spec for
        programmatically-built models.  Override in task configs that have a
        spec builder; the default returns ``None``.

        Returns:
            :class:`mujoco.MjSpec` if supported, else ``None``.
        """
        return None

    @property
    @abstractmethod
    def n_substeps(self) -> int:
        """Number of MuJoCo substeps executed per :meth:`step` call."""

    # ------------------------------------------------------------------
    # Space helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def obs_dim(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
        agent_id: str,
    ) -> int:
        """Return the flat observation dimension for *agent_id*.

        Args:
            model: Compiled combined MjModel.
            data: Corresponding MjData (in reset state).
            meta: Task-specific index cache.
            agent_id: One of ``self.agents``.

        Returns:
            Integer observation size.
        """

    @abstractmethod
    def act_indices(self, meta: Any, agent_id: str) -> list[int]:
        """Return the actuator indices for *agent_id* in the combined model.

        Args:
            meta: Task-specific index cache.
            agent_id: One of ``self.agents``.

        Returns:
            List of integer indices into ``data.ctrl``.
        """

    # ------------------------------------------------------------------
    # Episode logic
    # ------------------------------------------------------------------

    def on_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
    ) -> None:
        """Called after ``mujoco.mj_resetData`` and ``mj_forward``.

        Override for domain randomisation, initial pose perturbation, etc.
        Default implementation does nothing.

        Args:
            model: Compiled combined MjModel.
            data: Freshly reset MjData.
            meta: Task-specific index cache.
        """

    @abstractmethod
    def get_obs(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
        agent_id: str,
        health: dict[str, float],
    ) -> np.ndarray:
        """Compute the observation vector for *agent_id*.

        Args:
            model: Compiled combined MjModel.
            data: Current MjData after physics step.
            meta: Task-specific index cache.
            agent_id: One of ``self.agents``.
            health: Current cumulative health for all agents.

        Returns:
            Float32 observation array.
        """

    @abstractmethod
    def compute_damage(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
    ) -> dict[str, float]:
        """Compute damage *delivered by* each agent this control step.

        The env accumulates ``damage_delivered[attacker]`` onto the defender's
        health each step.  Return zeros if damage is not applicable.

        Args:
            model: Compiled combined MjModel.
            data: Current MjData.
            meta: Task-specific index cache.

        Returns:
            Dict ``{agent_id: damage_delivered_this_step}`` for all agents.
        """

    def check_fell(
        self,
        data: mujoco.MjData,
        meta: Any,
        agent_id: str,
    ) -> bool:
        """Return ``True`` if *agent_id* has fallen (e.g. pelvis below floor).

        Default returns ``False`` — override for tasks with free-root joints
        (boxing) but leave as default for anchored models (saber).

        Args:
            data: Current MjData.
            meta: Task-specific index cache.
            agent_id: One of ``self.agents``.

        Returns:
            ``True`` if the agent has fallen.
        """
        return False

    def opponent_id(self, agent_id: str) -> str:
        """Return the opponent id for a 2-agent task.

        Args:
            agent_id: Agent id to resolve.

        Returns:
            The single opposing agent id.

        Raises:
            ValueError: If this is not a strict 2-agent config.
        """
        others = [agent for agent in self.agents if agent != agent_id]
        if len(others) != 1:
            raise ValueError(
                f"opponent_id() requires exactly 2 agents; got {len(self.agents)}"
            )
        return others[0]

    def default_step_info(
        self,
        health: dict[str, float],
        damage_delivered: dict[str, float],
        fell: dict[str, bool],
        ko: dict[str, bool],
        step_count: int,
    ) -> dict[str, Any]:
        """Build the standard per-step diagnostics info dictionary.

        Args:
            health: Per-agent cumulative health.
            damage_delivered: Per-agent damage delivered this step.
            fell: Per-agent fall flags.
            ko: Per-agent KO flags.
            step_count: Current step index.

        Returns:
            Standardized info dict keyed by ``<metric>/<agent_id>``.
        """
        info: dict[str, Any] = {"step": step_count}
        for agent in self.agents:
            info[f"health/{agent}"] = health[agent]
            info[f"damage_delivered/{agent}"] = damage_delivered[agent]
            info[f"fell/{agent}"] = fell[agent]
            info[f"ko/{agent}"] = ko[agent]
        return info

    @property
    def ko_threshold(self) -> float:
        """Cumulative health at which an agent is KO'd.  Default: 100.0."""
        return 100.0

    @property
    def health_update_scale(self) -> float:
        """Multiplier applied to damage when updating health (usually ctrl_dt).

        Default: 1.0.  Override to use ``ctrl_dt`` or another time scale.
        """
        return 1.0

    @abstractmethod
    def compute_reward(
        self,
        data: mujoco.MjData,
        meta: Any,
        agent_id: str,
        damage_delivered: dict[str, float],
        health: dict[str, float],
        actions: np.ndarray,
        fell: dict[str, bool],
        ko: dict[str, bool],
    ) -> float:
        """Compute the scalar reward for *agent_id* after a step.

        Args:
            data: Current MjData.
            meta: Task-specific index cache.
            agent_id: One of ``self.agents``.
            damage_delivered: Damage delivered by each agent this step.
            health: Cumulative health after the step.
            actions: Raw action array applied by *agent_id* this step.
            fell: Whether each agent fell this step.
            ko: Whether each agent is KO'd.

        Returns:
            Scalar reward.
        """

    @abstractmethod
    def get_info(
        self,
        data: mujoco.MjData,
        meta: Any,
        health: dict[str, float],
        damage_delivered: dict[str, float],
        fell: dict[str, bool],
        ko: dict[str, bool],
        step_count: int,
    ) -> dict[str, Any]:
        """Build the info dict returned by :meth:`step`.

        Args:
            data: Current MjData.
            meta: Task-specific index cache.
            health: Cumulative health after the step.
            damage_delivered: Damage delivered by each agent this step.
            fell: Whether each agent fell this step.
            ko: Whether each agent is KO'd.
            step_count: Current step count.

        Returns:
            Flat info dict (string keys).
        """
