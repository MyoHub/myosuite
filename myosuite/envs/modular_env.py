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
from myosuite.physics.quat_math import mul_quat
from myosuite.envs.myo.tasks.challenge.saber_task_config import saber_term_params
from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
    SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS,
    SaberTargetPoolConfig,
    SaberTargetPoolRuntime,
    beat_saber_mocap_translation_delta,
    build_pool_indices,
    place_graveyard_positions,
    pool_post_physics,
    pool_pre_physics,
    pool_reset,
    resolve_recycle_contact_geom_ids,
)

logger = logging.getLogger(__name__)

# Right-hand table-tennis grasp prior; left uses same stems with ``_l``.
_TABLETENNIS_GRASP_SEED: dict[str, float] = {
    "elbow_flex": 0.805495,
    "pro_sup": -0.10284,
    "deviation": -0.08288,
    "flexion": -0.730422,
    "cmc_abduction": 0.0632,
    "cmc_flexion": 0.7,
    "mp_flexion": 0.07503,
    "ip_flexion": -0.296726,
    "mcp2_flexion": 0.72266,
    "mcp2_abduction": -0.136136,
    "pm2_flexion": 0.26707,
    "md2_flexion": 0.353475,
    "mcp3_flexion": 0.65982,
    "mcp3_abduction": -0.151844,
    "pm3_flexion": 0.42417,
    "md3_flexion": 0.51843,
    "mcp4_flexion": 0.919035,
    "mcp4_abduction": -0.20944,
    "pm4_flexion": 0.259215,
    "md4_flexion": 0.510575,
    "mcp5_flexion": 0.793355,
    "mcp5_abduction": -0.204204,
    "pm5_flexion": 0.227795,
}

# Site-local offsets tuned for bilateral palm/thumb contact stability.
_LEFT_SABER_GRASP_OFFSET = np.array(
    [-0.0244778, 0.03560197, -0.02699442], dtype=np.float64
)
_RIGHT_SABER_GRASP_OFFSET = np.array(
    [-0.03221418, 0.02980037, 0.03681982], dtype=np.float64
)


def _saber_upright_posture_from_xquat(quat: np.ndarray) -> float:
    """Return the world-z projection of the torso's local y-axis, clamped to [0, 1]."""
    quat32 = np.asarray(quat, dtype=np.float32)
    qw, qx, qy, qz = quat32
    upright = np.float32(2.0) * (qy * qz + qw * qx)
    return float(np.clip(upright, np.float32(0.0), np.float32(1.0)))


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
        self._saber_mocap_advance: dict[str, Any] | None = None
        self._saber_target_pool_enabled: bool = False
        self._target_pool_cfg: SaberTargetPoolConfig | None = None
        self._target_pool_runtime: SaberTargetPoolRuntime | None = None
        self._target_pool_mocap_ids: np.ndarray | None = None
        self._target_pool_site_ids: np.ndarray | None = None
        self._target_pool_geom_ids: np.ndarray | None = None
        self._target_pool_marker_geom_ids: np.ndarray | None = None
        self._target_pool_recycle_contact_geom_ids: set[int] | None = None
        self._saber_keyframe_qpos_adrs: np.ndarray | None = None
        self._saber_keyframe_ref_pos: np.ndarray | None = None
        self._configure_saber_mocap_driver()
        self._configure_saber_target_pool()

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
        _saber_cfg = getattr(self._task_config, "saber_cfg", None)
        extra = {
            **self._task_config.reward.extra,
            **(saber_term_params(_saber_cfg) if _saber_cfg is not None else {}),
        }

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
        return task_state

    def _configure_saber_mocap_driver(self) -> None:
        """If the task requests it, cache mocap ids for Beat Saber-style target motion."""
        self._saber_mocap_advance = None
        ex = self._task_config.reward.extra
        if ex.get("enable_saber_target_pool"):
            return
        if not ex.get("drive_saber_targets_kinematics"):
            return
        body_a = str(ex.get("saber_target_a_body", "saber_target_a"))
        body_b = str(ex.get("saber_target_b_body", "saber_target_b"))
        id_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_a)
        id_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_b)
        if id_a < 0 or id_b < 0:
            logger.warning(
                "drive_saber_targets_kinematics is set but bodies %r / %r were not found",
                body_a,
                body_b,
            )
            return
        mocap_a = int(self.model.body_mocapid[id_a])
        mocap_b = int(self.model.body_mocapid[id_b])
        if mocap_a < 0 or mocap_b < 0:
            logger.warning(
                "Saber targets %r / %r must be mocap bodies for kinematic drive",
                body_a,
                body_b,
            )
            return
        direction = np.asarray(
            ex.get("saber_target_advance_direction", [0.0, -1.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        speed = float(ex.get("saber_target_speed_mps", 0.55))
        self._saber_mocap_advance = {
            "mocap_ids": (mocap_a, mocap_b),
            "direction": direction,
            "speed": speed,
        }

    def _apply_saber_mocap_advance_pre_physics(self) -> None:
        """Translate strike-block mocap poses (one control step) before ``mj_step``."""
        cfg = self._saber_mocap_advance
        if cfg is None:
            return
        delta = beat_saber_mocap_translation_delta(
            cfg["direction"], cfg["speed"], self._ctrl_dt
        )
        for mid in cfg["mocap_ids"]:
            self.data.mocap_pos[mid] += delta

    def _configure_saber_target_pool(self) -> None:
        """If the task provides a SaberP0Cfg, resolve indices and reset the mocap pool."""
        saber_cfg = getattr(self._task_config, "saber_cfg", None)
        if saber_cfg is None or not saber_cfg.enable_saber_target_pool:
            return
        self._target_pool_cfg = SaberTargetPoolConfig(
            pool_size=int(saber_cfg.pool_size),
            graveyard_z=float(saber_cfg.graveyard_z),
            spawn_interval_steps=int(saber_cfg.spawn_interval_steps),
            sphere_radius=float(saber_cfg.sphere_radius),
            hit_threshold=None,
            kill_plane_y=float(saber_cfg.kill_plane_y),
            miss_zone_y=float(saber_cfg.miss_zone_y),
            ideal_hit_y_range=tuple(float(v) for v in saber_cfg.ideal_hit_y_range),
            max_target_misses=saber_cfg.max_target_misses,
            spawn_portal_x=tuple(float(v) for v in saber_cfg.spawn_portal_x),
            spawn_portal_y=tuple(float(v) for v in saber_cfg.spawn_portal_y),
            spawn_portal_z=tuple(float(v) for v in saber_cfg.spawn_portal_z),
            advance_direction=tuple(float(v) for v in saber_cfg.advance_direction),
            speed_mps_range=tuple(
                float(v) for v in saber_cfg.saber_target_speed_mps_range
            ),
            recycle_on_body_or_helmet_contact=bool(
                saber_cfg.recycle_on_body_or_helmet_contact
            ),
            recycle_contact_geom_substrings=saber_cfg.recycle_contact_geom_substrings
            or SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS,
            enforce_preferred_hand=bool(saber_cfg.enforce_preferred_hand),
            enforce_preferred_hit_area=bool(saber_cfg.enforce_preferred_hit_area),
            obstacle_fraction=float(saber_cfg.obstacle_fraction),
            slicing_accuracy_epsilon=float(saber_cfg.slicing_accuracy_epsilon),
        )
        self._target_pool_runtime = SaberTargetPoolRuntime()
        pool_reset(self._target_pool_runtime, self._target_pool_cfg.pool_size)
        (
            self._target_pool_mocap_ids,
            self._target_pool_site_ids,
            self._target_pool_geom_ids,
            self._target_pool_marker_geom_ids,
        ) = build_pool_indices(self.model, self._target_pool_cfg)
        self._target_pool_recycle_contact_geom_ids = resolve_recycle_contact_geom_ids(
            self.model, self._target_pool_cfg, self._target_pool_geom_ids
        )
        keyframe_qpos_adrs: list[int] = []
        for joint_name in ("left_saber_freejoint", "right_saber_freejoint"):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                self._saber_keyframe_qpos_adrs = None
                break
            qadr = int(self.model.jnt_qposadr[joint_id])
            keyframe_qpos_adrs.extend([qadr, qadr + 1, qadr + 2])
        else:
            self._saber_keyframe_qpos_adrs = np.asarray(
                keyframe_qpos_adrs, dtype=np.int32
            )
        self._saber_target_pool_enabled = True
        assert self._target_pool_cfg is not None
        assert self._target_pool_mocap_ids is not None
        place_graveyard_positions(
            self.data,
            self._target_pool_mocap_ids,
            self._target_pool_cfg.pool_size,
            self._target_pool_cfg.graveyard_z,
        )
        self._update_saber_target_pool_visuals()
        self._publish_target_pool_task_state()

    def _capture_saber_keyframe_pose_reference(self) -> None:
        """Capture the current saber translation state as the debug pose reference."""
        if self._saber_keyframe_qpos_adrs is None:
            self._saber_keyframe_ref_pos = None
            return
        self._saber_keyframe_ref_pos = (
            self.data.qpos[self._saber_keyframe_qpos_adrs].reshape(2, 3).copy()
        )

    def _compute_saber_keyframe_pose_error(self) -> float:
        """Return mean free-joint translation error from the reset saber pose."""
        if (
            self._saber_keyframe_qpos_adrs is None
            or self._saber_keyframe_ref_pos is None
            or len(self._saber_keyframe_qpos_adrs) == 0
        ):
            return 1.0
        current_sites = self.data.qpos[self._saber_keyframe_qpos_adrs].reshape(2, 3)
        return float(
            np.mean(
                np.linalg.norm(current_sites - self._saber_keyframe_ref_pos, axis=1)
            )
        )

    def _update_saber_target_pool_visuals(self) -> None:
        """Color active cubes and highlight the required hit side per target."""
        if (
            not self._saber_target_pool_enabled
            or self._target_pool_runtime is None
            or self._target_pool_geom_ids is None
            or self._target_pool_marker_geom_ids is None
        ):
            return
        # Keep cubes clearly visible in all camera angles/compression settings.
        cube_inactive = np.array([0.55, 0.55, 0.55, 0.85], dtype=np.float32)
        cube_active = np.array([0.80, 0.80, 0.80, 1.0], dtype=np.float32)
        obstacle_active = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        left_marker = np.array([0.05, 0.25, 1.0, 1.0], dtype=np.float32)
        right_marker = np.array([1.0, 0.10, 0.10, 1.0], dtype=np.float32)
        clear = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        active = self._target_pool_runtime.active
        preferred = self._target_pool_runtime.preferred_hand
        hit_face = self._target_pool_runtime.preferred_hit_area
        is_obstacle = self._target_pool_runtime.is_obstacle
        for i, gid in enumerate(self._target_pool_geom_ids):
            marker_ids = self._target_pool_marker_geom_ids[i]
            if active[i]:
                self.model.geom_rgba[int(gid)] = (
                    obstacle_active if bool(is_obstacle[i]) else cube_active
                )
                marker_color = left_marker if int(preferred[i]) == 0 else right_marker
                active_face_idx = int(hit_face[i])
                for face_idx, marker_gid in enumerate(marker_ids):
                    self.model.geom_rgba[int(marker_gid)] = (
                        marker_color
                        if (not bool(is_obstacle[i])) and face_idx == active_face_idx
                        else clear
                    )
            else:
                self.model.geom_rgba[int(gid)] = cube_inactive
                for marker_gid in marker_ids:
                    self.model.geom_rgba[int(marker_gid)] = clear

    def _publish_target_pool_task_state(self) -> None:
        """Copy pool runtime into ``self._task_state`` for obs/reward terms."""
        if not self._saber_target_pool_enabled or self._target_pool_runtime is None:
            return
        _saber_cfg = getattr(self._task_config, "saber_cfg", None)
        upright_body_name = (
            str(_saber_cfg.upright_body_name) if _saber_cfg is not None else "torso"
        )
        upright_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, upright_body_name
        )
        upright_posture = 0.0
        if upright_bid >= 0:
            upright_posture = _saber_upright_posture_from_xquat(
                self.data.xquat[int(upright_bid)]
            )
        r = self._target_pool_runtime
        self._task_state["target_pool_active"] = r.active.copy()
        self._task_state["target_pool_vel"] = r.vel.copy()
        self._task_state["target_pool_is_obstacle"] = r.is_obstacle.copy()
        self._task_state["target_pool_hit_area"] = r.preferred_hit_area.copy()
        self._task_state["target_pool_preferred_hand"] = r.preferred_hand.copy()
        self._task_state["target_pool_hit_this_step"] = bool(r.hit_this_step)
        self._task_state["target_pool_offset_at_hit"] = float(r.offset_at_hit)
        self._task_state["target_pool_hit_y_at_hit"] = float(r.hit_y_at_hit)
        self._task_state["target_pool_hit_y_deviation_at_hit"] = float(
            r.hit_y_deviation_at_hit
        )
        self._task_state["target_pool_face_surface_distance_at_hit"] = float(
            r.face_surface_distance_at_hit
        )
        self._task_state["target_pool_slicing_accuracy_this_step"] = float(
            r.slicing_accuracy_this_step
        )
        self._task_state["target_pool_timing_accuracy_this_step"] = float(
            r.timing_accuracy_this_step
        )
        dcap = float(r.d_min_closest) if r.d_min_closest < 1e5 else 1e6
        self._task_state["target_pool_d_min"] = dcap
        obstacle_dcap = float(r.d_min_obstacle) if r.d_min_obstacle < 1e5 else 1e6
        self._task_state["target_pool_obstacle_d_min"] = obstacle_dcap
        self._task_state["target_pool_obstacle_contact_this_step"] = bool(
            r.obstacle_contact_this_step
        )
        self._task_state["target_pool_obstacle_contacts"] = int(r.obstacle_contacts)
        self._task_state["target_pool_miss_zone"] = bool(r.miss_zone_this_step)
        self._task_state["target_pool_hits"] = int(r.hits)
        self._task_state["target_pool_misses"] = int(r.misses)
        self._task_state["target_pool_correct_hand_this_step"] = bool(
            r.correct_hand_this_step
        )
        self._task_state["target_pool_correct_face_this_step"] = bool(
            r.correct_face_this_step
        )
        self._task_state["target_pool_wrong_hand_this_step"] = bool(
            r.wrong_hand_this_step
        )
        self._task_state["target_pool_wrong_face_this_step"] = bool(
            r.wrong_face_this_step
        )
        self._task_state["target_pool_correct_hand_hits"] = int(r.correct_hand_hits)
        self._task_state["target_pool_correct_face_hits"] = int(r.correct_face_hits)
        self._task_state["target_pool_wrong_hand_hits"] = int(r.wrong_hand_hits)
        self._task_state["target_pool_wrong_face_hits"] = int(r.wrong_face_hits)
        self._task_state["target_pool_total_spawned"] = int(r.total_spawned)
        self._task_state["health_status"] = float(r.health_status)
        self._task_state["upright_posture"] = upright_posture
        self._task_state["saber_keyframe_pose_error"] = (
            self._compute_saber_keyframe_pose_error()
        )
        self._task_state["target_pool_health_depleted_this_step"] = bool(
            r.health_depleted_this_step
        )

    def _reset_saber_target_pool(self) -> None:
        """Episode reset: all targets to graveyard and pool queues cleared."""
        if not self._saber_target_pool_enabled or self._target_pool_cfg is None:
            return
        assert self._target_pool_runtime is not None
        assert self._target_pool_mocap_ids is not None
        pool_reset(self._target_pool_runtime, self._target_pool_cfg.pool_size)
        place_graveyard_positions(
            self.data,
            self._target_pool_mocap_ids,
            self._target_pool_cfg.pool_size,
            self._target_pool_cfg.graveyard_z,
        )
        self._update_saber_target_pool_visuals()
        mujoco.mj_forward(self.model, self.data)
        self._publish_target_pool_task_state()

    def _snap_free_sabers_to_grasp_sites(self) -> None:
        """Place freejoint sabers at hand grasp sites with matching orientation."""
        mount_quat = np.array(
            [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float64
        )
        mappings = (
            (
                "left_saber_freejoint",
                "S_grasp_left",
                "lunate_l",
                _LEFT_SABER_GRASP_OFFSET,
            ),
            (
                "right_saber_freejoint",
                "S_grasp",
                "lunate_r",
                _RIGHT_SABER_GRASP_OFFSET,
            ),
        )
        changed = False
        for joint_name, site_name, hand_body_name, site_local_offset in mappings:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name
            )
            if jid < 0 or sid < 0 or bid < 0:
                continue
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            qadr = int(self.model.jnt_qposadr[jid])
            dadr = int(self.model.jnt_dofadr[jid])
            hand_quat = self.data.xquat[bid].copy()
            quat = mul_quat(hand_quat, mount_quat)
            quat /= max(np.linalg.norm(quat), 1e-12)
            site_rot = self.data.site_xmat[sid].reshape(3, 3)
            self.data.qpos[qadr : qadr + 3] = (
                self.data.site_xpos[sid] + site_rot @ site_local_offset
            )
            self.data.qpos[qadr + 3 : qadr + 7] = quat
            self.data.qvel[dadr : dadr + 6] = 0.0
            changed = True
        if changed:
            mujoco.mj_forward(self.model, self.data)

    def _apply_tabletennis_grasp_seed(self) -> None:
        """Apply table-tennis-like arm/hand grasp posture at reset."""
        # Only apply for saber-freejoint scenes.
        lj = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_saber_freejoint"
        )
        rj = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_saber_freejoint"
        )
        if lj < 0 or rj < 0:
            return

        changed = False
        for stem, value in _TABLETENNIS_GRASP_SEED.items():
            for suffix in ("_r", "_l"):
                jname = f"{stem}{suffix}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                self.data.qpos[qadr] = float(value)
                changed = True
        if changed:
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _apply_saber_target_pool_pre_physics(self) -> None:
        """Advance mocap pool and spawn before ``mj_step``."""
        if not self._saber_target_pool_enabled or self._target_pool_cfg is None:
            return
        assert self._target_pool_runtime is not None
        assert self._target_pool_mocap_ids is not None
        pool_pre_physics(
            self.data,
            self._target_pool_cfg,
            self._target_pool_runtime,
            self._target_pool_mocap_ids,
            self.np_random,
            self._ctrl_dt,
        )
        self._update_saber_target_pool_visuals()

    def _post_step_saber_target_pool(self) -> None:
        """Hits, misses, and recycling after the control step."""
        if not self._saber_target_pool_enabled or self._target_pool_cfg is None:
            return
        assert self._target_pool_runtime is not None
        assert self._target_pool_mocap_ids is not None
        assert self._target_pool_site_ids is not None
        assert self._target_pool_geom_ids is not None
        pool_post_physics(
            self.model,
            self.data,
            self._target_pool_cfg,
            self._target_pool_runtime,
            self._target_pool_mocap_ids,
            self._target_pool_site_ids,
            self._target_pool_geom_ids,
            self._target_pool_recycle_contact_geom_ids,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset simulation and, when configured, the saber target pool."""
        super(MyoGymnasiumEnv, self).reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self._fatigue_model is not None:
            self._fatigue_model.reset()
        self._task_state = self.reset_task(self.np_random)
        self._reset_saber_target_pool()
        self._apply_tabletennis_grasp_seed()
        self._snap_free_sabers_to_grasp_sites()
        self._capture_saber_keyframe_pose_reference()
        self._publish_target_pool_task_state()
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment, optionally applying action noise and pool kinematics.

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

        self._apply_saber_target_pool_pre_physics()
        self._apply_saber_mocap_advance_pre_physics()
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
        self._post_step_saber_target_pool()
        self._publish_target_pool_task_state()

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
