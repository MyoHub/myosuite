# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Environment and task configuration dataclasses.

This module provides two levels of configuration:

**Low-level** (Phases 1–4):
    :class:`BackendConfig` and :class:`EnvConfig` — thin wrappers used by
    the registry and existing env registrations.

**High-level** (Phase 5 — Modular Task System):
    :class:`ObsSpec`, :class:`GoalSpec`, :class:`RewardSpec`,
    :class:`ActuatorGroupSpec`, and :class:`TaskConfig` — data-driven task
    definitions that drive :class:`~myosuite.envs.modular_env.ModularTaskEnv`
    without subclassing.

Example::

    from myosuite.core.config import TaskConfig, GoalSpec, ObsSpec, RewardSpec

    @dataclass
    class ElbowPoseTask(TaskConfig):
        model: str = "elbow_standard"
        goal: GoalSpec = field(default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=True,
            range={"r_elbow_flex": (0.0, 2.27)},
        ))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ---------------------------------------------------------------------------
# Variant specification (Phase 2 — declarative muscle condition variants)
# ---------------------------------------------------------------------------


@dataclass
class VariantSpec:
    """Declares a configuration variant of a base task.

    Used in ``TaskConfig.variants`` to auto-register muscle-condition
    variants (sarcopenia, fatigue, reafferentation) without string-manipulation
    hacks.  :func:`~myosuite.core.registry.register_task` expands each
    ``VariantSpec`` into a separate Gymnasium environment registration.

    Args:
        suffix: Short identifier prepended after the ``"myo"`` prefix in the
            env ID (e.g. ``"Sarc"`` turns ``"myoElbowPose-v0"`` into
            ``"myoSarcElbowPose-v0"``).
        config_delta: Dict of ``TaskConfig`` field overrides to apply on top of
            the base config (e.g. ``{"actuators": [ActuatorGroupSpec(condition="sarcopenia")]}``).

    Example::

        @dataclass
        class ElbowPoseTask(TaskConfig):
            variants: ClassVar[list[VariantSpec]] = [
                VariantSpec("Sarc", {"actuators": [ActuatorGroupSpec(condition="sarcopenia")]}),
                VariantSpec("Fati", {"actuators": [ActuatorGroupSpec(condition="fatigue")]}),
            ]
    """

    suffix: str
    config_delta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level config (Phases 1–4)
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Physics-backend-specific settings.

    Args:
        n_substeps: Number of MuJoCo simulation steps per control step.
        ctrl_dt: Control timestep in seconds.
        sim_dt: Simulation timestep in seconds.
        extra: Additional backend-specific key-value pairs.
    """

    n_substeps: int = 10
    ctrl_dt: float = 0.01
    sim_dt: float = 0.001
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Top-level environment configuration.

    Args:
        env_id: Registered environment identifier (e.g. "myoElbowPose1D6MRandom-v0").
        model: Named model recipe from myosuite.core.model_recipes.
        scene: Named scene spec from myosuite.scenes.library.
        max_episode_steps: Episode length limit before truncation.
        backend: Backend-specific physics configuration.
    """

    env_id: str = ""
    model: str = "elbow_standard"
    scene: str = "flat_floor"
    max_episode_steps: int = 200
    backend: BackendConfig = field(default_factory=BackendConfig)


# ---------------------------------------------------------------------------
# High-level task specs (Phase 5 — Modular Task System)
# ---------------------------------------------------------------------------


@dataclass
class ObsSpec:
    """Declares which observation channels a task exposes.

    Each entry in ``keys`` maps to a term function in
    ``myosuite/terms/myo_obs_terms.py`` (e.g. ``"joint_pos"`` → calls
    ``joint_pos_obs(accessor)``).

    Args:
        keys: Ordered list of observation term names to concatenate into
            the observation vector.
        extra: Additional keyword arguments forwarded to each term function
            at call time (e.g. ``{"site_ids": [...]}`` for tip-position obs).
    """

    keys: list[str | Callable] = field(
        default_factory=lambda: ["joint_pos", "joint_vel", "muscle_act"]
    )
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalSpec:
    """Describes how episode goals are sampled and represented.

    Args:
        target_type: Semantic type of the goal.  One of:

            - ``"joint_angles"``   — target ``qpos`` for a set of joints.
            - ``"site_positions"`` — target 3-D Cartesian positions for sites.
            - ``"trajectory"``     — reference motion clip (MuscleMimic-style).

        randomize: If ``True``, sample a new target at each episode reset.
            If ``False``, use the fixed values in ``range``.
        range: Mapping from joint/site name to ``(lo, hi)`` sampling bounds.
            For ``"joint_angles"`` the values are in radians; for
            ``"site_positions"`` in metres.
        extra: Additional goal-specific parameters (e.g. ``{"motion_clip": Path(...)}``.
    """

    target_type: str = "joint_angles"
    randomize: bool = True
    range: dict[str, tuple[float, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardSpec:
    """Declares the reward terms and their weights for a task.

    Each entry in ``terms`` maps to a function in
    ``myosuite/terms/myo_reward_terms.py`` (e.g. ``"pose"`` → ``pose_reward``).

    Args:
        terms: Ordered list of reward term names to evaluate each step.
        weights: Per-term scalar multipliers applied to each term's ``"dense"``
            output before summing.  Defaults to ``1.0`` for unlisted terms.
        extra: Additional keyword arguments forwarded to every term function.
    """

    terms: list[str | Callable] = field(default_factory=lambda: ["pose"])
    weights: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def weight_for(self, term: str) -> float:
        """Return the scalar weight for *term*, defaulting to 1.0.

        Args:
            term: Reward term name.

        Returns:
            Weight value.
        """
        return self.weights.get(term, 1.0)


@dataclass
class ActuatorGroupSpec:
    """Describes a group of actuators (muscles or motors) and their condition.

    Args:
        name: Logical group name (e.g. ``"elbow_muscles"``).
        actuator_type: ``"muscle"`` for Hill-type muscles or ``"motor"`` for
            direct torque/position actuators.
        condition: Physiological condition applied at reset.  One of
            ``"normal"``, ``"fatigue"``, or ``"sarcopenia"``.
        normalize_actions: If ``True``, actions are passed through a sigmoid
            ``σ(5(a − 0.5))`` to map ``ℝ → (0, 1)`` before being written to
            ``ctrl``.  Set to ``False`` when actions are already in ``[0, 1]``.
    """

    name: str = "muscles"
    actuator_type: str = "muscle"
    condition: str = "normal"
    normalize_actions: bool = True
    noise: float = 0.0  # Gaussian noise std added to actions


@dataclass
class TaskConfig:
    """Data-driven task definition for :class:`~myosuite.envs.modular_env.ModularTaskEnv`.

    A :class:`TaskConfig` is the single source of truth for a task: model,
    scene, goal distribution, reward function, and action interface.
    Subclass and override individual fields to create task variants without
    code duplication::

        @dataclass
        class ElbowPoseSarcopeniaTask(ElbowPoseTask):
            actuators: list[ActuatorGroupSpec] = field(
                default_factory=lambda: [
                    ActuatorGroupSpec(name="elbow_muscles", condition="sarcopenia")
                ]
            )

    Args:
        model: Named model recipe from ``myosuite.core.model_recipes``
            (e.g. ``"elbow_standard"``).
        scene: Named scene spec from ``myosuite.scenes.library``
            (e.g. ``"flat_floor"``).
        max_episode_steps: Episode length limit before truncation.
        muscle_fatigue: If ``True``, apply cumulative 3-compartment fatigue
            dynamics to muscle excitations each control step.
        backend: Backend-specific physics settings.
        obs: Observation channel specification.
        goal: Goal sampling and representation specification.
        reward: Reward term and weight specification.
        actuators: List of actuator group specs (one per muscle/motor group).
        fragment_versions: Maps fragment name → expected version integer.
            CI fails if the installed fragment version does not match.
    """

    model: str = "elbow_standard"
    scene: str | list[str] | Callable = "flat_floor"
    max_episode_steps: int = 200
    muscle_fatigue: bool = False
    persist_muscle_fatigue: bool = False
    backend: BackendConfig = field(default_factory=BackendConfig)
    obs: ObsSpec = field(default_factory=ObsSpec)
    goal: GoalSpec = field(default_factory=GoalSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec()]
    )

    # Subclasses declare fragment version constraints here so CI can detect
    # stale task configs when a fragment XML is updated.
    fragment_versions: ClassVar[dict[str, int]] = {}

    # Subclasses declare muscle-condition variants here.  Each VariantSpec
    # causes register_task() to auto-register an additional environment with
    # the config_delta merged on top of the base config.
    variants: ClassVar[list[VariantSpec]] = []

    def to_env_config(self, env_id: str = "") -> EnvConfig:
        """Convert to a legacy :class:`EnvConfig` for registry compatibility.

        Args:
            env_id: Gymnasium environment ID to embed.

        Returns:
            An :class:`EnvConfig` with model, scene, and backend fields
            copied from this task config.  When ``scene`` is not a plain
            ``str`` (i.e. a list or callable), the legacy field defaults to
            ``"flat_floor"``.
        """
        return EnvConfig(
            env_id=env_id,
            model=self.model,
            scene=self.scene if isinstance(self.scene, str) else "flat_floor",
            max_episode_steps=self.max_episode_steps,
            backend=self.backend,
        )
