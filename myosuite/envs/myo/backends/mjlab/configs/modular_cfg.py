# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Modular task configuration helpers for the mjlab (MuJoCo Warp) backend.

Translates a :class:`~myosuite.core.config.TaskConfig` into an mjlab-compatible
configuration dataclass.  This is the mjlab counterpart of
:class:`~myosuite.envs.modular_env.ModularTaskEnv` (CPU) and
:class:`~myosuite.envs.myo.backends.mjx.mjx_modular_env.MjxModularTaskEnv` (MJX).

Usage::

    from myosuite.core.config import TaskConfig, ActuatorGroupSpec
    from myosuite.envs.myo.backends.mjlab.configs.modular_cfg import make_modular_mjlab_cfg

    cfg = make_modular_mjlab_cfg(TaskConfig())
    # cfg.action_terms describes how to build MuscleActionTermCfg entries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from myosuite.core.config import TaskConfig


@dataclass
class ModularActionTermEntry:
    """One action term entry derived from an ActuatorGroupSpec.

    Args:
        name: Logical group name.
        actuator_type: ``"muscle"`` or ``"motor"``.
        condition: Physiological condition (``"normal"``, ``"fatigue"``,
            ``"sarcopenia"``).
        normalize_actions: Whether to apply sigmoid normalization.
        noise: Gaussian noise std added to actions.
    """

    name: str
    actuator_type: str = "muscle"
    condition: str = "normal"
    normalize_actions: bool = True
    noise: float = 0.0


@dataclass
class ModularMjlabCfg:
    """mjlab-backend configuration derived from a :class:`~myosuite.core.config.TaskConfig`.

    This is a plain-Python translation of ``TaskConfig`` fields into the
    mjlab task config convention.  Pass it to the mjlab ``ManagerBasedRlEnv``
    builder once mjlab/MuJoCo Warp becomes available.

    Args:
        model_recipe: Named model recipe (see ``myosuite.core.model_recipes``).
        scene: Scene name (``str``), list of scene names for randomization, or
            a ``Callable[[MjSpec], MjSpec]`` for spec-level transforms.
        sim_dt: MuJoCo simulation timestep in seconds.
        ctrl_dt: Control timestep in seconds.
        max_episode_steps: Episode truncation length.
        num_envs: Parallel environment batch size on GPU.
        obs_keys: Ordered observation term keys.
        reward_terms: Ordered reward term names.
        reward_weights: Per-term scalar weights.
        goal_type: Semantic goal type (``"joint_angles"`` or ``"site_positions"``).
        goal_randomize: Whether to randomize goals at each episode reset.
        goal_range: Joint/site name → ``(lo, hi)`` sampling range.
        action_terms: One entry per actuator group.
    """

    model_recipe: str = "elbow_standard"
    scene: Any = "flat_floor"
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 200
    num_envs: int = 4096
    obs_keys: list[str] = field(
        default_factory=lambda: ["joint_pos", "joint_vel", "muscle_act"]
    )
    reward_terms: list[str] = field(default_factory=lambda: ["pose"])
    reward_weights: dict[str, float] = field(default_factory=dict)
    goal_type: str = "joint_angles"
    goal_randomize: bool = True
    goal_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    action_terms: list[ModularActionTermEntry] = field(
        default_factory=lambda: [ModularActionTermEntry(name="muscles")]
    )


def make_modular_mjlab_cfg(task_config: TaskConfig) -> ModularMjlabCfg:
    """Translate a TaskConfig into a ModularMjlabCfg for the mjlab backend.

    Args:
        task_config: High-level task specification.

    Returns:
        mjlab-compatible configuration dataclass.

    Example::

        from myosuite.core.config import TaskConfig, ActuatorGroupSpec
        cfg = make_modular_mjlab_cfg(TaskConfig())
        assert cfg.model_recipe == "elbow_standard"
    """
    action_terms = [
        ModularActionTermEntry(
            name=g.name,
            actuator_type=g.actuator_type,
            condition=g.condition,
            normalize_actions=g.normalize_actions,
            noise=g.noise,
        )
        for g in task_config.actuators
    ]

    obs_keys = [
        k if isinstance(k, str) else getattr(k, "__name__", repr(k))
        for k in task_config.obs.keys
    ]
    reward_terms = [
        t if isinstance(t, str) else getattr(t, "__name__", repr(t))
        for t in task_config.reward.terms
    ]

    return ModularMjlabCfg(
        model_recipe=task_config.model,
        scene=task_config.scene,
        sim_dt=task_config.backend.sim_dt,
        ctrl_dt=task_config.backend.ctrl_dt,
        max_episode_steps=task_config.max_episode_steps,
        num_envs=int(task_config.backend.extra.get("num_envs", 4096)),
        obs_keys=obs_keys,
        reward_terms=reward_terms,
        reward_weights=dict(task_config.reward.weights),
        goal_type=task_config.goal.target_type,
        goal_randomize=task_config.goal.randomize,
        goal_range=dict(task_config.goal.range),
        action_terms=action_terms,
    )
