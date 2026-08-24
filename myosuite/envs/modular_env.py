# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ModularTaskEnv — data-driven CPU environment driven by TaskConfig.

A :class:`ModularTaskEnv` needs no subclassing: observation channels,
reward terms, and goal sampling are fully specified by a
:class:`~myosuite.core.config.TaskConfig` dataclass.  This is the reference
CPU implementation for the Phase 5 Modular Task Configuration System.

Example::

    from dataclasses import dataclass, field
    from myosuite.core.config import TaskConfig, GoalSpec
    from myosuite.envs.modular_env import ModularTaskEnv

    @dataclass
    class ElbowTask(TaskConfig):
        model: str = "elbow_standard"
        goal: GoalSpec = field(default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=True,
            range={"r_elbow_flex": (0.0, 2.27)},
        ))

    env = ModularTaskEnv(ElbowTask())
    obs, info = env.reset()
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.core.config import GoalSpec, TaskConfig
from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.core.model_recipes import _MUSCLEMIMIC_NAMES, _musclemimic_build
from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv

logger = logging.getLogger(__name__)

# Maps ObsSpec key → function name suffix in myo_obs_terms (e.g. "joint_pos" → joint_pos_obs)
_OBS_TERM_SUFFIX = "_obs"

# Maps RewardSpec term → function name in myo_reward_terms (e.g. "pose" → pose_reward)
_REWARD_TERM_SUFFIX = "_reward"


def _load_obs_fn(key: str) -> Any:
    """Resolve an observation term key to its callable.

    Convention: ``key`` maps to ``<key>_obs`` in
    ``myosuite.terms.base_obs``.

    Args:
        key: Observation term name (e.g. ``"joint_pos"``).

    Returns:
        Callable term function.

    Raises:
        AttributeError: If no matching function exists in myo_obs_terms.
    """
    mod = importlib.import_module("myosuite.terms.base_obs")
    fn_name = f"{key}{_OBS_TERM_SUFFIX}"
    if not hasattr(mod, fn_name):
        raise AttributeError(
            f"Observation term {key!r} not found: expected function "
            f"{fn_name!r} in myosuite.terms.base_obs. "
            f"Available: {[n for n in dir(mod) if n.endswith(_OBS_TERM_SUFFIX)]}"
        )
    return getattr(mod, fn_name)


def _load_reward_fn(term: str) -> Any:
    """Resolve a reward term name to its callable.

    Convention: ``term`` maps to ``<term>_reward`` in
    ``myosuite.terms.base_reward``, with a fallback to an exact
    match for terms like ``"act_reg"`` and ``"joint_penalty"`` that do
    not follow the ``_reward`` suffix pattern.

    Args:
        term: Reward term name (e.g. ``"pose"``, ``"act_reg"``).

    Returns:
        Callable term function.

    Raises:
        AttributeError: If no matching function can be found.
    """
    mod = importlib.import_module("myosuite.terms.base_reward")
    # Try <term>_reward first, then exact match
    for fn_name in (f"{term}{_REWARD_TERM_SUFFIX}", term):
        if hasattr(mod, fn_name):
            return getattr(mod, fn_name)
    available = [n for n in dir(mod) if not n.startswith("_")]
    raise AttributeError(
        f"Reward term {term!r} not found in myosuite.terms.base_reward. "
        f"Available: {available}"
    )


def _resolve_obs_term(key: str | Any) -> tuple[str, Any]:
    """Resolve an obs term key (string or callable) to a (label, fn) pair.

    Args:
        key: Observation term name string or a callable directly.

    Returns:
        Tuple of (label, callable).
    """
    if callable(key):
        return getattr(key, "__name__", repr(key)), key
    return key, _load_obs_fn(key)


def _resolve_reward_term(term: str | Any) -> tuple[str, Any]:
    """Resolve a reward term (string or callable) to a (label, fn) pair.

    Args:
        term: Reward term name string or a callable directly.

    Returns:
        Tuple of (label, callable).
    """
    if callable(term):
        return getattr(term, "__name__", repr(term)), term
    return term, _load_reward_fn(term)


def _joint_qpos_width(model: mujoco.MjModel, j: int) -> int:
    """Return the number of generalized positions occupied by joint *j*."""
    jt = int(model.jnt_type[j])
    if jt == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if jt == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _sample_goal(
    goal_spec: GoalSpec,
    model: mujoco.MjModel,
    qpos_init: np.ndarray,
    np_random: np.random.Generator,
) -> dict[str, Any]:
    """Sample a goal dict from *goal_spec*.

    Args:
        goal_spec: Goal specification dataclass.
        model: Compiled MuJoCo model (joint addresses come from here).
        qpos_init: Baseline ``qpos`` vector (length ``model.nq``); coordinates
            not listed in ``goal_spec.range`` keep these values so
            ``target_angles`` matches :meth:`~CpuEnvAccessor.joint_pos` layout.
        np_random: NumPy random generator.

    Returns:
        Dict with ``"target_angles"`` for ``joint_angles`` targets,
        ``"target_pos"`` for ``site_positions`` targets.

    Raises:
        ValueError: If the target type is not supported.
    """
    match goal_spec.target_type:
        case "joint_angles":
            target = np.array(qpos_init, dtype=np.float64, copy=True)
            if target.shape != (model.nq,):
                raise ValueError(
                    f"qpos_init must have shape ({model.nq},); got {target.shape}"
                )
            for j in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if name not in goal_spec.range:
                    continue
                lo, hi = goal_spec.range[name]
                qadr = int(model.jnt_qposadr[j])
                span = _joint_qpos_width(model, j)
                if goal_spec.randomize:
                    target[qadr : qadr + span] = np_random.uniform(lo, hi, size=(span,))
                else:
                    target[qadr : qadr + span] = np.full((span,), float(lo))
            return {"target_angles": target}
        case "site_positions":
            # Minimal: return whatever extra the caller provided
            return dict(goal_spec.extra)
        case _:
            raise ValueError(
                f"Unsupported GoalSpec.target_type: {goal_spec.target_type!r}. "
                "Supported: 'joint_angles', 'site_positions'."
            )


def _build_model_from_task_model(
    model_name: str,
) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Build model/spec from a TaskConfig.model identifier.

    Supports:
    - Named ModelBuilder recipes (default path)
    - Direct XML path strings
    - Special MuscleMimic shortcuts:
      ``musclemimic_fullbody``, ``musclemimic_bimanual``,
      ``musclemimic_bimanual_fingers``,
      ``musclemimic_myotorso_bimanual``,
      ``musclemimic_myotorso_bimanual_fingers``
    """
    if model_name in _MUSCLEMIMIC_NAMES:
        return _musclemimic_build(model_name)

    xml_candidate = Path(model_name)
    if xml_candidate.suffix == ".xml" and xml_candidate.exists():
        model, spec = ModelBuilder.from_xml_file(xml_candidate).build()
        return model, spec

    return build_from_recipe(model_name)


class ModularTaskEnv(MyoGymnasiumEnv):
    """CPU environment fully driven by a :class:`~myosuite.core.config.TaskConfig`.

    No subclassing is needed: observation channels, reward terms, and goal
    sampling are all configured through the ``task_config`` argument.

    The observation vector is assembled by calling each named function in
    ``task_config.obs.keys`` (resolved from
    ``myosuite.terms.base_obs``) and concatenating the results.

    The reward scalar is the weighted sum of each term in
    ``task_config.reward.terms`` (resolved from
    ``myosuite.terms.base_reward``).  The ``"done"`` flag is ``True``
    if any reward term returns ``"done": True``.

    Args:
        task_config: Data-driven task specification.
        render_mode: Gymnasium render mode (``"human"`` or ``"rgb_array"``).
        **kwargs: Forwarded to :class:`MyoGymnasiumEnv`.

    Example:
        >>> from myosuite.core.config import TaskConfig
        >>> env = ModularTaskEnv(TaskConfig())
        >>> obs, info = env.reset(seed=0)
        >>> obs.shape
        (nobs,)
    """

    def __init__(
        self,
        task_config: TaskConfig,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_skip=task_config.backend.n_substeps,
            render_mode=render_mode,
            **kwargs,
        )
        self._task_config = task_config
        self._ctrl_dt = task_config.backend.ctrl_dt

        # Resolve obs/reward callables once at construction time
        self._obs_fns: list[tuple[str, Any]] = [
            _resolve_obs_term(key) for key in task_config.obs.keys
        ]
        self._reward_fns: list[tuple[str, Any]] = [
            _resolve_reward_term(term) for term in task_config.reward.terms
        ]

        # Build the MuJoCo model based on scene type
        scene = task_config.scene
        if callable(scene):
            base_model, base_spec = _build_model_from_task_model(task_config.model)
            base_spec = scene(base_spec)
            self.model = base_spec.compile()
            self._scene_models: dict[str, tuple] | None = None
        elif isinstance(scene, list):
            self._scene_models = {}
            for s in scene:
                m, _ = _build_model_from_task_model(task_config.model)
                self._scene_models[s] = (m, mujoco.MjData(m))
            first = scene[0]
            self.model, _ = self._scene_models[first]
        else:
            base_model, base_spec = _build_model_from_task_model(task_config.model)
            if any(g.condition == "sarcopenia" for g in task_config.actuators):
                base_spec = apply_sarcopenia_to_spec(base_spec)
                self.model = base_spec.compile()
            else:
                self.model = base_model
            self._scene_models = None

        self.data = mujoco.MjData(self.model)
        self._fatigue_model: Any = None
        self._fatigue_mask: Any = None
        if task_config.muscle_fatigue:
            from myosuite.core.muscle_conditions import CumulativeFatigue  # noqa: PLC0415

            self._fatigue_model = CumulativeFatigue(
                self.model, frame_skip=task_config.backend.n_substeps
            )
            self._fatigue_mask = (
                self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
            )
        # Infer observation and action spaces via a dummy forward pass
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        dummy_obs_dict = self._get_obs_dict(self._accessor)
        obs_dim = sum(np.atleast_1d(v).size for v in dummy_obs_dict.values())

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        n_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=np.zeros(n_actuators, dtype=np.float32),
            high=np.ones(n_actuators, dtype=np.float32),
            dtype=np.float32,
        )
        self.mj_instability_termination: bool = True
        self._validate_backend_timing()

    def _check_mj_instability_termination(self) -> bool:
        """Return True if the simulation is numerically unstable (NaN/Inf in qpos/qvel)."""
        return bool(
            np.any(~np.isfinite(self.data.qpos)) or np.any(~np.isfinite(self.data.qvel))
        )

    def _validate_backend_timing(self) -> None:
        """Validate control-step timing when strict alignment is requested."""
        if not self._task_config.backend.extra.get(
            "enforce_step_timing_alignment", False
        ):
            return
        expected = float(self._task_config.backend.ctrl_dt)
        effective = float(self.frame_skip) * float(self.model.opt.timestep)
        if abs(expected - effective) > 1e-10:
            raise ValueError(
                "Backend timing mismatch: ctrl_dt="
                f"{expected:.6f}, frame_skip*timestep={effective:.6f}. "
                "Update BackendConfig or model timestep so one control step "
                "matches the configured control interval."
            )

    # ------------------------------------------------------------------
    # MyoGymnasiumEnv interface
    # ------------------------------------------------------------------

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Assemble observation dict by calling each configured obs term.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Dict mapping term key to numpy array.
        """
        extra = {**self._task_config.obs.extra, **self._task_state}
        return {key: np.atleast_1d(fn(accessor, **extra)) for key, fn in self._obs_fns}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute reward by evaluating each configured reward term.

        Args:
            obs_dict: Output of ``_get_obs_dict`` (not used directly; terms
                read physics state via the accessor stored in
                ``self._task_state``).

        Returns:
            Dict with ``"dense"`` (float) and ``"done"`` (bool), plus
            per-term component entries for logging.
        """
        accessor = self._accessor
        task_state = self._task_state
        extra = dict(self._task_config.reward.extra)

        combined: dict[str, Any] = {}
        dense_total = 0.0
        done = False
        solved = False

        for term, fn in self._reward_fns:
            result = fn(accessor, task_state, **extra)
            weight = self._task_config.reward.weight_for(term)
            dense_total += weight * float(result.get("dense", 0.0))
            done = done or bool(result.get("done", False))
            solved = solved or bool(result.get("solved", False))
            # Merge component keys (prefix with term name to avoid collisions)
            for k, v in result.items():
                if k not in ("dense", "done"):
                    combined[f"{term}/{k}"] = v

        combined["dense"] = dense_total
        combined["done"] = done
        combined["solved"] = solved
        return combined

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Sample a new goal for the episode.

        Swaps the active MuJoCo model when scene is a list (randomizes over
        available scenes).

        Args:
            np_random: NumPy random generator provided by Gymnasium.

        Returns:
            Task state dict containing the sampled goal.
        """
        task_state = _sample_goal(
            self._task_config.goal,
            self.model,
            self.data.qpos.copy(),
            np_random,
        )
        if self._scene_models:
            scene = np_random.choice(list(self._scene_models.keys()))
            self.model, self.data = self._scene_models[scene]
            self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
            task_state["scene"] = scene
        self._maybe_sample_heading(task_state, np_random)
        return task_state

    def _maybe_sample_heading(
        self, task_state: dict[str, Any], np_random: np.random.Generator
    ) -> None:
        """Sample a per-episode commanded heading when the task requests it.

        Controlled by ``reward.extra["randomize_heading"]``. Writes the sampled
        unit ``heading_dir`` into ``task_state`` so both the ``heading_cmd``
        observation (merged with ``obs.extra``) and ``heading_reward`` (which
        reads ``task_state``) follow the same command. This is what lets a policy
        learn the command->direction mapping instead of a single fixed heading.

        A ``reward.extra["heading_choices"]`` list of ``(dx, dy)`` directions, if
        present, is sampled discretely; otherwise the heading is drawn uniformly
        from the full unit circle.
        """
        ex = self._task_config.reward.extra
        if not ex.get("randomize_heading"):
            return
        choices = ex.get("heading_choices")
        if choices:
            dx, dy = choices[int(np_random.integers(0, len(choices)))]
        else:
            angle = float(np_random.uniform(0.0, 2.0 * np.pi))
            dx, dy = np.cos(angle), np.sin(angle)
        task_state["heading_dir"] = (float(dx), float(dy))
        return

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset simulation and sample a new episode task state."""
        super(MyoGymnasiumEnv, self).reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            # mj_resetData alone zeros ALL qpos (including free-joint
            # orientation quaternions), which is not a valid pose for any
            # standing/walking host model -- it collapses instantly under
            # gravity. Load keyframe 0, which every bundled host XML ships
            # as a real standing/crouched pose (see e.g.
            # myosuite/envs/myo/assets/leg/myolegs_with_torso.xml's
            # <keyframe> block), so reset_task()'s goal/heading sampling
            # below starts from a physically valid pose instead of the
            # all-zero default.
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        if self._fatigue_model is not None:
            self._fatigue_model.reset()
        self._task_state = self.reset_task(self.np_random)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment, optionally applying action noise.

        Args:
            action: Control action array.
            **kwargs: Ignored compatibility kwargs (e.g. update_exteroception).

        Returns:
            5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        noise_std = (
            max(g.noise for g in self._task_config.actuators)
            if self._task_config.actuators
            else 0.0
        )
        if noise_std > 0.0:
            action = action + self.np_random.normal(
                0.0, noise_std, size=action.shape
            ).astype(action.dtype)
        _ = kwargs
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._task_state["last_action"] = action.astype(np.float32, copy=True)

        if self._fatigue_model is not None:
            action = action.copy()
            action[self._fatigue_mask], _, _ = self._fatigue_model.compute_act(
                action[self._fatigue_mask]
            )
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        if self.mujoco_render_frames:
            self.mj_render()

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)

        obs_dict = self._get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        reward = float(rwd_dict.get("dense", 0.0))
        terminated = bool(rwd_dict.get("done", False))
        info = {k: v for k, v in rwd_dict.items() if k not in ("dense", "done")}
        info["obs_dict"] = obs_dict
        info["rwd_dict"] = rwd_dict
        return obs, reward, terminated, False, info
