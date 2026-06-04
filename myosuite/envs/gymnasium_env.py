# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
MyoGymnasiumEnv — clean Gymnasium base class for MyoSuite CPU environments.

Replaces MujocoEnv / env_base.py as the root of the CPU class hierarchy.
Not yet wired to registrations (Phase 1); full migration happens in Phase 2.

Key design decisions:
- Pure Gymnasium interface: reset() → (obs, info); step() → (obs, rwd, terminated, truncated, info)
- No @implement_for shims — gymnasium only
- Gymnasium TimeLimit wrapper handles truncation (no manual step counter here)
- Subclasses implement get_obs_dict(), get_reward_dict(), reset_task()
"""

from __future__ import annotations

import types
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import gymnasium as gym
from myosuite.utils.path_utils import evaluate_success as _evaluate_success
from myosuite.utils.policy_utils import examine_policy as _examine_policy

if TYPE_CHECKING:  # pragma: no cover
    from myosuite.viz.mj_renderer import MJRenderer

_REQUIRED_RWD_KEYS = frozenset({"dense", "done"})


def _validate_reward_dict(rwd_dict: dict) -> None:
    """Raise ``KeyError`` when a required reward key is missing.

    Called by :meth:`MyoGymnasiumEnv.step` after every ``get_reward_dict``
    call so that custom task authors catch schema violations immediately
    rather than producing silent NaNs downstream.

    Args:
        rwd_dict: Dict returned by ``get_reward_dict``.

    Raises:
        KeyError: If ``"dense"`` or ``"done"`` are absent.
    """
    missing = _REQUIRED_RWD_KEYS - rwd_dict.keys()
    if missing:
        raise KeyError(
            f"get_reward_dict() must return keys {_REQUIRED_RWD_KEYS}; "
            f"missing: {missing}"
        )


class CpuEnvAccessor:
    """EnvAccessor implementation for the CPU (MuJoCo CPU) backend.

    Wraps a mujoco.MjData / MjModel pair and returns numpy.ndarray from
    all accessor methods.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo simulation data.
        ctrl_dt: Control timestep in seconds.
    """

    def __init__(self, model: Any, data: Any, ctrl_dt: float = 0.01) -> None:
        self._model = model
        self._data = data
        self._ctrl_dt = ctrl_dt

    @property
    def physics_path(self) -> Any:
        from myosuite.core.protocols import PhysicsPath

        return PhysicsPath.CPU

    @property
    def model(self) -> Any:
        """The MuJoCo MjModel."""
        return self._model

    @property
    def data(self) -> Any:
        """The MuJoCo MjData."""
        return self._data

    def joint_pos(self) -> np.ndarray:
        return self._data.qpos.copy()

    def joint_vel(self) -> np.ndarray:
        return self._data.qvel.copy()

    def muscle_act(self) -> np.ndarray:
        return self._data.act.copy() if self._data.act is not None else np.zeros(0)

    def site_xpos(self, site_ids: Any) -> np.ndarray:
        return self._data.site_xpos[site_ids].copy()

    def site_id(self, name: str) -> int:
        import mujoco

        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            raise ValueError(f"Site {name!r} not found in model")
        return int(sid)

    def time(self) -> float:
        return float(self._data.time)

    def ctrl_range(self) -> np.ndarray:
        return self._model.actuator_ctrlrange.copy()

    def dt(self) -> float:
        return self._ctrl_dt

    def array_module(self) -> types.ModuleType:
        return np

    # ------------------------------------------------------------------
    # Muscle kinematics (Biomechanist / Neuroscientist API)
    # ------------------------------------------------------------------

    def muscle_length(self) -> np.ndarray:
        """Normalised muscle-tendon unit length for all actuators.

        Returns ``data.actuator_length`` — the total MTU length in metres
        as computed by MuJoCo's muscle model.  Divide by the optimal fibre
        length (``model.actuator_user[:, 0]`` in MyoSuite MJCF convention)
        to obtain the dimensionless fibre-length ratio used in Hill-type
        force-length curves.

        Returns:
            Array of shape ``(n_actuators,)`` in metres.
        """
        return self._data.actuator_length.copy()

    def muscle_velocity(self) -> np.ndarray:
        """Muscle-tendon unit lengthening velocity for all actuators.

        Returns ``data.actuator_velocity`` in m/s (positive = lengthening).
        Divide by ``Vmax * optimal_fibre_length`` to obtain the normalised
        contraction velocity used in Hill-type force-velocity curves.

        Returns:
            Array of shape ``(n_actuators,)`` in m/s.
        """
        return self._data.actuator_velocity.copy()

    def muscle_force(self) -> np.ndarray:
        """Applied actuator force for all actuators.

        Returns ``data.actuator_force`` in Newtons (positive = shortening
        force).  For Hill-type muscles this is the total force including
        active and passive components after pennation projection.

        Returns:
            Array of shape ``(n_actuators,)`` in N.
        """
        return self._data.actuator_force.copy()

    def moment_arm(self, joint_id: int) -> np.ndarray:
        """Moment arm of every actuator about a given joint.

        Computes the column of the actuator moment-arm matrix (``efc_J``)
        corresponding to *joint_id* using MuJoCo's ``mj_jacActDot``
        approach via ``mj_actuatorMoment``.

        Args:
            joint_id: Zero-based joint index (``model.joint(name).id``).

        Returns:
            Array of shape ``(n_actuators,)`` in metres.
        """
        import mujoco

        n_act = self._model.nu
        moment = np.zeros((n_act, self._model.nv), dtype=np.float64)
        mujoco.mj_actuatorMoment(self._model, self._data, moment)
        jadr = int(self._model.jnt_dofadr[joint_id])
        return moment[:, jadr].copy()


# Design note: MyoGymnasiumEnv intentionally does NOT inherit from
# gymnasium.envs.mujoco.MujocoEnv.  Three contracts are incompatible:
#   1. reset_model() must return an obs vector, but MyoSuite computes obs via
#      pluggable term functions and returns a task-state dict.
#   2. _initialize_simulation() calls mujoco.MjModel.from_xml_path() — no MjSpec
#      access — so muscle conditions (sarcopenia, fatigue) cannot be applied.
#   3. MujocoEnv.__init__() requires observation_space upfront, but we infer obs
#      size from an initial forward pass after model load.
# Instead we selectively adopt three useful utilities: render(), set_state(), and
# get_body_com() — matching MujocoEnv's public API without the inheritance burden.


class MyoGymnasiumEnv(gym.Env):
    """Base class for all MyoSuite CPU (Gymnasium) environments.

    Subclasses must implement:
    - ``get_obs_dict(accessor)``  → dict[str, np.ndarray]
    - ``get_reward_dict(obs_dict)`` → dict[str, float | bool]
    - ``reset_task(np_random)`` → dict  (task state for the episode)
    - ``setup_model()`` → (MjModel, MjData)  (or set self.model / self.data)

    The default step/reset logic handles action clipping, physics stepping,
    obs assembly, and reward extraction.

    Args:
        frame_skip: Number of MuJoCo substeps per gym step.
        render_mode: Rendering mode passed to gymnasium.Env.

    Example:
        >>> env = gym.make("myoElbowPose1D6MRandom-v0")
        >>> obs, info = env.reset(seed=42)
        >>> obs, rwd, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        frame_skip: int = 10,
        render_mode: str | None = None,
        mj_instability_termination: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.mj_instability_termination = mj_instability_termination

        # Subclass must set these (or call setup_model())
        self.model: Any = None
        self.data: Any = None
        self._ctrl_dt: float = 0.01

        # Task state for the current episode
        self._task_state: dict[str, Any] = {}
        self._accessor: CpuEnvAccessor | None = None
        self._mj_renderer_compat: MJRenderer | None = None
        # Legacy realtime-render toggle consumed by policy_utils.examine_policy*.
        self.mujoco_render_frames: bool = False
        # Backward-compatible visual key container expected by examine_env.py.
        self.visual_keys: list[str] = []

    # ------------------------------------------------------------------
    # Subclass interface (must override)
    # ------------------------------------------------------------------

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute observation dictionary from current physics state.

        Override in subclasses to implement observation logic.

        Args:
            accessor: CPU environment accessor wrapping model/data.

        Returns:
            Dict mapping obs group names to numpy arrays.
        """
        raise NotImplementedError

    def get_obs_dict(
        self, accessor_or_model=None, mj_data=None
    ) -> dict[str, np.ndarray]:
        """Compute observation dictionary, accepting both new and legacy call styles.

        Preferred: ``get_obs_dict(accessor)`` with a :class:`CpuEnvAccessor`.
        Legacy: ``get_obs_dict(mj_model, mj_data)`` — retained for test_envs and
        callers that pass (model, data); may be removed when all callers use
        accessor-only.

        Args:
            accessor_or_model: Either a :class:`CpuEnvAccessor` (new API) or a
                ``mujoco.MjModel`` instance (legacy API).
            mj_data: ``mujoco.MjData`` instance (legacy API only; ignored otherwise).

        Returns:
            Dict mapping obs group names to numpy arrays.
        """
        import mujoco

        accessor: CpuEnvAccessor | None
        if isinstance(accessor_or_model, mujoco.MjModel):
            accessor = CpuEnvAccessor(accessor_or_model, mj_data, self._ctrl_dt)
        elif accessor_or_model is None:
            accessor = self._accessor
            if accessor is None:
                raise RuntimeError(
                    "CpuEnvAccessor is not initialized; call reset() or pass a "
                    "CpuEnvAccessor instance to get_obs_dict()."
                )
        else:
            accessor = accessor_or_model
        assert accessor is not None
        return self._get_obs_dict(accessor)

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute reward dictionary from the observation dict.

        **Reward dict contract** (standard keys):

        - **Required:** ``"dense"`` (float) — scalar reward for this step;
          ``"done"`` (bool) — task-level episode end.
        - **Optional:** ``"sparse"`` (float), ``"solved"`` (bool); task-specific
          keys (e.g. ``vel_reward``, ``cyclic_hip``) may be included for logging.

        :meth:`step` returns the 5-tuple ``(obs, reward, terminated, truncated, info)``
        and sets ``info["rwd_dict"]`` to this full dict so benchmarks and callers
        can rely solely on 5-tuple step and ``info["rwd_dict"]``.

        Args:
            obs_dict: Output of get_obs_dict().

        Returns:
            Dict with at least "dense" and "done"; may include "sparse", "solved",
            and task-specific component keys.
        """
        raise NotImplementedError

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Sample a new task configuration for the episode.

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Task state dict stored as self._task_state.
        """
        return {}

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one control step.

        Args:
            action: Control command, clipped to action_space bounds.
            **kwargs: Ignored compatibility kwargs (e.g. update_exteroception).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        import mujoco

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        if self.mujoco_render_frames:
            self.mj_render()

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)
        _validate_reward_dict(rwd_dict)

        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        reward = float(rwd_dict.get("dense", 0.0))
        terminated = bool(rwd_dict.get("done", False))
        # Physics instability (bad acceleration warning) → truncated, not terminated,
        # consistent with the MjInstabilityTerminationWrapper and the MJX backend.
        truncated = (
            self.mj_instability_termination and self._check_mj_instability_termination()
        )
        info = {k: v for k, v in rwd_dict.items() if k not in ("dense", "done")}
        info["obs_dict"] = obs_dict
        info["rwd_dict"] = rwd_dict

        return obs, reward, terminated, truncated, info

    def forward(
        self, update_exteroception: bool = False
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Legacy forward() compatibility without stepping dynamics.

        Args:
            update_exteroception: Unused compatibility flag.

        Returns:
            Gymnasium-style 5-tuple for current state.
        """
        del update_exteroception
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
        if self.mujoco_render_frames:
            self.mj_render()
        return obs, reward, terminated, False, info

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for episode reproducibility.
            options: Optional reset options (currently unused).

        Returns:
            Tuple of (obs, info).
        """
        import mujoco

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._task_state = self.reset_task(self.np_random)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten observation dict to a single 1-D array.

        Args:
            obs_dict: Dict of observation arrays.

        Returns:
            Concatenated 1-D numpy array.
        """
        return np.concatenate([np.atleast_1d(v).ravel() for v in obs_dict.values()])

    def _ensure_obs_gymnasium_compliant(self, obs: np.ndarray) -> np.ndarray:
        """Cast obs to float32 and clip to observation_space so passive_env_checker is satisfied.

        Args:
            obs: Raw observation vector (may be float64 or outside declared bounds).

        Returns:
            float32 array within observation_space.low/high.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if getattr(self, "observation_space", None) is not None:
            obs = np.clip(
                obs,
                self.observation_space.low,
                self.observation_space.high,
            )
        return obs

    def render(self):
        """Render the environment using a lazy MujocoRenderer.

        Mirrors ``gymnasium.envs.mujoco.MujocoEnv.render()`` without
        inheriting from it.

        Returns:
            Rendered frame (ndarray for ``"rgb_array"``) or ``None``.
        """
        if self.render_mode is None:
            return None
        if not hasattr(self, "_mujoco_renderer") or self._mujoco_renderer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self._mujoco_renderer = MujocoRenderer(self.model, self.data)
        return self._mujoco_renderer.render(self.render_mode)

    def close(self) -> None:
        """Release rendering resources.

        Mirrors ``gymnasium.envs.mujoco.MujocoEnv.close()``.
        """
        if (
            hasattr(self, "_mj_renderer_compat")
            and self._mj_renderer_compat is not None
        ):
            try:
                self._mj_renderer_compat.close()
            except Exception:
                pass
            self._mj_renderer_compat = None
        if hasattr(self, "_mujoco_renderer") and self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None

    # ------------------------------------------------------------------
    # MujocoEnv-compatible utility helpers
    # ------------------------------------------------------------------

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Set joint positions and velocities, then forward the simulation.

        Mirrors ``gymnasium.envs.mujoco.MujocoEnv.set_state()``.

        Args:
            qpos: Joint position vector (shape ``(nq,)``).
            qvel: Joint velocity vector (shape ``(nv,)``).
        """
        import mujoco

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)

    def get_body_com(self, body_name: str) -> np.ndarray:
        """Return the centre-of-mass position of a subtree rooted at *body_name*.

        Mirrors ``gymnasium.envs.mujoco.MujocoEnv.get_body_com()``.

        Args:
            body_name: Name of the MuJoCo body.

        Returns:
            3-D centre-of-mass position as a copied numpy array.
        """
        return self.data.body(body_name).subtree_com.copy()

    # ------------------------------------------------------------------
    # Backward-compatibility shims (match old env_base.MujocoEnv / BaseV0 API)
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> list[int]:
        """Re-seed the environment's random number generator.

        Backward-compatible shim for the old ``BaseV0.seed()`` API.
        Gymnasium replaced ``seed()`` with the ``seed`` argument to ``reset()``;
        this method provides compatibility for code that calls ``env.seed()``.

        Args:
            seed: Integer seed or ``None`` to pick a random seed.

        Returns:
            Single-element list containing the effective seed.
        """
        import gymnasium.utils.seeding as _seeding

        self._input_seed = seed
        self.np_random, effective_seed = _seeding.np_random(seed)
        return [effective_seed]

    def get_input_seed(self) -> int | None:
        """Return the seed most recently passed to ``__init__`` or ``seed()``.

        Compatibility shim for ``BaseV0.get_input_seed()``; used by test_envs and
        legacy code. May be deprecated when callers are updated.

        Returns:
            The stored input seed, or ``None`` if never set.
        """
        return getattr(self, "_input_seed", None)

    def get_proprioception(self) -> tuple:
        """Return proprioceptive observations.

        Compatibility shim for ``BaseV0.get_proprioception()``; used by test_envs.
        ``MyoGymnasiumEnv`` does not configure proprioception keys, so this
        always returns ``(None, None, None)``.  Subclasses may override.
        May be deprecated when test_envs no longer requires it.

        Returns:
            Tuple ``(time, proprio_vec, proprio_dict)`` where all elements are
            ``None`` (no proprio keys configured).
        """
        return None, None, None

    def get_exteroception(self) -> dict:
        """Return exteroceptive (visual) observations.

        Compatibility shim for ``BaseV0.get_exteroception()``; used by test_envs.
        ``MyoGymnasiumEnv`` does not configure visual keys, so this always
        returns an empty dict.  Subclasses may override.
        May be deprecated when test_envs no longer requires it.

        Returns:
            Empty dict (no visual keys configured).
        """
        return {}

    def evaluate_success(
        self,
        paths,
        logger=None,
        successful_steps: int = 5,
    ) -> float:
        """Evaluate rollout success using the legacy helper.

        This compatibility shim preserves the API expected by
        ``myosuite.utils.examine_env`` for Gymnasium-native environments.
        """
        return _evaluate_success(
            self,
            paths=paths,
            logger=logger,
            successful_steps=successful_steps,
        )

    def examine_policy(
        self,
        policy: Any,
        horizon: int = 1000,
        num_episodes: int = 1,
        mode: str = "exploration",
        render: str | None = None,
        camera_name: str | None = None,
        frame_size: tuple[int, int] = (640, 480),
        output_dir: str = "/tmp/",
        filename: str = "newvid",
        device_id: int = 0,
    ) -> Any:
        """Roll out a policy and return a :class:`~myosuite.logger.grouped_datasets.Trace`.

        Args:
            policy: Policy object exposing ``get_action(obs)``.
            horizon: Maximum steps per episode.
            num_episodes: Number of episodes to collect.
            mode: ``"exploration"`` (stochastic) or ``"evaluation"`` (deterministic).
            render: ``"onscreen"`` / ``"offscreen"`` / ``None``.
            camera_name: Camera name for offscreen rendering.
            frame_size: ``(width, height)`` for offscreen frames.
            output_dir: Directory for saving MP4 files.
            filename: Base filename for saved videos.
            device_id: Unused; kept for signature compatibility.

        Returns:
            A :class:`~myosuite.logger.grouped_datasets.Trace` with rollout data
            grouped by episode (``Trial0``, ``Trial1``, …).
        """
        return _examine_policy(
            self,
            policy,
            horizon=horizon,
            num_episodes=num_episodes,
            mode=mode,
            render=render,
            camera_name=camera_name,
            frame_size=frame_size,
            output_dir=output_dir,
            filename=filename,
            device_id=device_id,
        )

    # ------------------------------------------------------------------
    # Backward-compatibility aliases (match old env_base.MujocoEnv API)
    # ------------------------------------------------------------------

    @property
    def mj_model(self) -> Any:
        """Backward-compatible alias for ``self.model``.

        .. deprecated::
            Use ``env.unwrapped.model`` instead.

        Old tutorials and scripts access ``env.unwrapped.mj_model``; this
        property bridges to the new ``self.model`` attribute.
        """
        warnings.warn(
            "env.mj_model is deprecated; use env.model instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model

    @property
    def mj_data(self) -> Any:
        """Backward-compatible alias for ``self.data``.

        .. deprecated::
            Use ``env.unwrapped.data`` instead.

        Old tutorials and scripts access ``env.mj_data``; this property
        bridges to the new ``self.data`` attribute.
        """
        warnings.warn(
            "env.mj_data is deprecated; use env.data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.data

    @property
    def mj_renderer(self) -> Any:
        """Lazy MJRenderer for backward-compatible offscreen rendering.

        .. deprecated::
            Use ``env.render()`` (Gymnasium standard) or instantiate
            ``MJRenderer(env.model, env.data)`` directly for offscreen frames.

        Old tutorials call ``env.mj_renderer.render_offscreen(...)``.
        The renderer is created on first access and reuses ``self.model``
        and ``self.data`` so it always reflects the current simulation state.

        Returns:
            ``MJRenderer`` instance bound to the current model/data.
        """
        warnings.warn(
            "env.mj_renderer is deprecated; use env.render() for Gymnasium-standard "
            "rendering, or instantiate MJRenderer(env.model, env.data) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from myosuite.viz.mj_renderer import MJRenderer

        if not hasattr(self, "_mj_renderer_compat") or self._mj_renderer_compat is None:
            self._mj_renderer_compat = MJRenderer(self.model, self.data)
        return self._mj_renderer_compat

    def mj_render(self) -> None:
        """Legacy realtime window renderer used by old policy utils."""
        if self._mj_renderer_compat is None:
            from myosuite.viz.mj_renderer import MJRenderer

            self._mj_renderer_compat = MJRenderer(self.model, self.data)
        self._mj_renderer_compat.render_to_window()

    @property
    def id(self) -> str:
        """Backward-compatible environment id string.

        Used by :func:`~myosuite.utils.policy_utils.examine_policy` to name the
        returned Trace.  For Gymnasium environments this can be obtained from
        ``env.spec.id`` when available.
        """
        spec = getattr(self, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return str(spec.id)
        return self.__class__.__name__

    @property
    def time(self) -> float:
        """Backward-compatible simulation time property."""
        return float(self.data.time) if self.data is not None else 0.0
