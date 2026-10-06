# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Environment and task configuration dataclasses.

This module provides two levels of configuration:

**Low-level** (Phases 1–4):
    :class:`BackendConfig` (timing) and :class:`EnvConfig` (the env instance
    to build, read by :func:`~myosuite.core.registry.make_env`).

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

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Variant specification (Phase 2 — declarative muscle condition variants)
# ---------------------------------------------------------------------------


@dataclass
class VariantSpec:
    """Declares a configuration variant of a base task.

    Used in ``TaskConfig.variants`` to auto-register variants (for example the
    muscle conditions sarcopenia, fatigue and reafferentation) without
    string-manipulation hacks.  :func:`~myosuite.core.registry.register_task`
    expands each ``VariantSpec`` into a separate Gymnasium environment registration.

    Args:
        suffix: Short identifier prepended after the ``"myo"`` prefix in the
            env ID (e.g. ``"Sarc"`` turns ``"myoElbowPose-v0"`` into
            ``"myoSarcElbowPose-v0"``).
        config_delta: Dict of ``TaskConfig`` field overrides to apply on top of
            the base config (e.g. ``{"max_episode_steps": 400}``).
        features: Muscle-command wrappers the variant registers (the same
            :class:`~gymnasium.envs.registration.WrapperSpec` as
            ``EnvConfig.features``), e.g. ``condition_wrapper_specs("fatigue")``.
            CPU registrations only.

    Example::

        @dataclass
        class ElbowPoseTask(TaskConfig):
            variants: ClassVar[list[VariantSpec]] = [
                VariantSpec("Sarc", features=condition_wrapper_specs("sarcopenia")),
                VariantSpec("Fati", features=condition_wrapper_specs("fatigue")),
            ]
    """

    suffix: str
    config_delta: dict[str, Any] = field(default_factory=dict)
    features: tuple[Any, ...] = ()


# ---------------------------------------------------------------------------
# Low-level config (Phases 1–4)
# ---------------------------------------------------------------------------


def check_control_step(n_substeps: int, sim_dt: float, ctrl_dt: float) -> None:
    """Check that one control step of ``ctrl_dt`` is ``n_substeps`` steps of ``sim_dt``.

    Args:
        n_substeps: Physics steps per control step.
        sim_dt: Physics timestep in seconds.
        ctrl_dt: Control timestep in seconds.

    Raises:
        ValueError: If ``n_substeps < 1`` or ``n_substeps * sim_dt != ctrl_dt``.
    """
    simulated = n_substeps * sim_dt
    if n_substeps < 1 or not math.isclose(
        simulated, ctrl_dt, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"Control step mismatch: ctrl_dt={ctrl_dt} s but n_substeps * sim_dt = "
            f"{n_substeps} * {sim_dt} = {simulated} s. Every backend simulates "
            "n_substeps steps of sim_dt per control step, so ctrl_dt must equal "
            "their product."
        )


@dataclass
class BackendConfig:
    """Physics-backend-specific settings.

    One rule on every backend: a control step is ``n_substeps`` physics steps of
    ``sim_dt`` (the CPU and MJX envs set the model timestep to ``sim_dt``, mjlab
    uses ``sim_dt`` and ``ctrl_dt``), so ``ctrl_dt`` must equal
    ``n_substeps * sim_dt``.  Observations and rewards scaled by the control
    step (e.g. ``joint_vel``) use ``ctrl_dt``.

    Args:
        n_substeps: Number of MuJoCo simulation steps per control step.
        ctrl_dt: Control timestep in seconds.
        sim_dt: Simulation timestep in seconds.
        extra: Additional backend-specific key-value pairs.

    Raises:
        ValueError: If ``ctrl_dt != n_substeps * sim_dt``.
    """

    n_substeps: int = 10
    ctrl_dt: float = 0.01
    sim_dt: float = 0.001
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_control_step(self.n_substeps, self.sim_dt, self.ctrl_dt)


@dataclass
class EnvConfig:
    """Backend-agnostic description of the env instance to build.

    Pass it to :func:`~myosuite.core.registry.make_env`, which resolves it against
    the registration of ``env_id`` (the defaults) and builds it on ``backend``.
    ``None`` means "keep the registered default".

    Args:
        env_id: Registered environment identifier (e.g. "myoElbowPose1D6MRandom-v0").
        backend: ``"cpu"``, ``"mjlab"`` (or the experimental ``"mjx"``).
        num_envs: Parallel envs; ``None`` is one on the CPU, the registered count on mjlab.
        max_episode_steps: Episode length limit before truncation.
        ctrl_dt: Control timestep in seconds (the single timing knob: physics
            substeps per control step, ``frame_skip`` or decimation, follow from
            it and the model timestep). Must be a whole multiple of the model
            timestep. The ``TaskConfig`` envs (``ModularTaskEnv``) take their
            timing from ``task_config.backend`` and raise a ``ValueError`` on
            every backend.
        features: Muscle-command features to activate (noise, fatigue, reafferentation,
            sarcopenia, custom excitation stages). Each entry is a wrapper class
            (``FatigueWrapper``), a ``(class, kwargs)`` pair
            (``(MotorNoiseWrapper, {"motor_noise": {...}})``) or a
            :class:`~gymnasium.envs.registration.WrapperSpec` (see
            :func:`myosuite.envs.wrappers.wrapper_spec`); stored as specs. Every backend
            runs them identically. All are off unless listed here or in the registration.
        task_kwargs: Task-specific constructor overrides (CPU env kwargs).
        backend_options: Options of one backend only (``device``, ``render_mode``, ...);
            nothing portable belongs here.
    """

    env_id: str = ""
    backend: str = "cpu"
    num_envs: int | None = None
    max_episode_steps: int | None = None
    ctrl_dt: float | None = None
    features: tuple[Any, ...] = ()
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from myosuite.envs.wrappers import normalize_features  # noqa: PLC0415

        self.features = normalize_features(self.features)


# ---------------------------------------------------------------------------
# High-level task specs (experimental ModularTaskEnv / TaskConfig route)
# ---------------------------------------------------------------------------


@dataclass
class ObsSpec:
    """Declares which observation channels a task exposes.

    Each entry in ``keys`` maps to a term function in
    ``myosuite/terms/base_obs.py`` (e.g. ``"joint_pos"`` → calls
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
            For ``"joint_angles"`` the values are scalars in radians. For
            ``"site_positions"`` each bound is an ``(x, y, z)`` position in metres (a
            scalar is broadcast to all three axes); the order of the mapping is the
            order of the sites in the sampled ``target_pos``. With an empty ``range``
            a ``"site_positions"`` goal is just ``extra`` (fixed, not sampled).
        extra: Additional goal-specific parameters (e.g. ``{"motion_clip": Path(...)}``.
    """

    target_type: str = "joint_angles"
    randomize: bool = True
    range: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def site_bounds(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Site names and ``(k, 3)`` lower/upper position bounds of a site goal.

        Returns:
            ``(names, lo, hi)`` in the order of :attr:`range`.

        Raises:
            ValueError: If :attr:`range` is empty.
        """
        if not self.range:
            raise ValueError("GoalSpec.range is empty: no site position bounds.")
        names = list(self.range)
        lo = np.stack(
            [np.broadcast_to(np.asarray(self.range[n][0], float), (3,)) for n in names]
        )
        hi = np.stack(
            [np.broadcast_to(np.asarray(self.range[n][1], float), (3,)) for n in names]
        )
        return names, lo, hi


@dataclass
class RewardSpec:
    """Declares the reward terms and their weights for a task.

    Each entry in ``terms`` maps to a function in
    ``myosuite/terms/base_reward.py`` (e.g. ``"pose"`` → ``pose_reward``).

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
    """Describes a group of actuators (muscles or motors).

    Physiological conditions and motor noise are not part of the task: they are
    features of the env instance (``EnvConfig.features``, see
    :mod:`myosuite.envs.wrappers`).

    Args:
        name: Logical group name (e.g. ``"elbow_muscles"``).
        actuator_type: ``"muscle"`` for Hill-type muscles or ``"motor"`` for
            direct torque/position actuators.
        normalize_actions: If ``True``, actions are passed through a sigmoid
            ``σ(5(a − 0.5))`` to map ``ℝ → (0, 1)`` before being written to
            ``ctrl``.  Set to ``False`` when actions are already in ``[0, 1]``.
    """

    name: str = "muscles"
    actuator_type: str = "muscle"
    normalize_actions: bool = True


@dataclass
class TaskConfig:
    """Data-driven task definition for :class:`~myosuite.envs.modular_env.ModularTaskEnv`.

    A :class:`TaskConfig` is the single source of truth for a task: model,
    scene, goal distribution, reward function, and action interface.
    Subclass and override individual fields to create task variants without
    code duplication::

        @dataclass
        class ElbowPoseRandomTask(ElbowPoseTask):
            goal: GoalSpec = field(
                default_factory=lambda: GoalSpec(randomize=True, range={...})
            )

    A task config holds the task only. Which muscle-command features are active
    (noise, fatigue, sarcopenia, reafferentation, custom stages) is a property of
    the env instance: ``EnvConfig.features`` or the registered ``features`` of a
    :class:`VariantSpec`.

    Args:
        model: Named model recipe from ``myosuite.core.model_recipes``
            (e.g. ``"elbow_standard"``).
        scene: Named scene spec from ``myosuite.scenes.library``
            (e.g. ``"flat_floor"``).
        max_episode_steps: Episode length limit before truncation.
        backend: Backend-specific physics settings.
        obs: Observation channel specification.
        goal: Goal sampling and representation specification.
        reward: Reward term and weight specification.
        actuators: List of actuator group specs (one per muscle/motor group).
        fragment_versions: Maps fragment name → expected version integer.
            ``scripts/check_fragment_compat.py`` fails if an installed
            fragment is newer than the declared version.
    """

    model: str = "elbow_standard"
    scene: str | list[str] | Callable = "flat_floor"
    max_episode_steps: int = 200
    backend: BackendConfig = field(default_factory=BackendConfig)
    obs: ObsSpec = field(default_factory=ObsSpec)
    goal: GoalSpec = field(default_factory=GoalSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec()]
    )

    # Subclasses declare fragment version constraints here so that
    # scripts/check_fragment_compat.py can detect stale task configs when a
    # fragment XML is updated.
    fragment_versions: ClassVar[dict[str, int]] = {}

    # Subclasses declare muscle-condition variants here.  Each VariantSpec
    # causes register_task() to auto-register an additional environment with
    # the config_delta merged on top of the base config.
    variants: ClassVar[list[VariantSpec]] = []
