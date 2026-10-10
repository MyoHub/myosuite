# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Ordered-waypoint task on any scene and actor (CPU half).

The registered ``myoFullBodyWaypoint-v0`` (MyoFullBody, random routes on flat
ground) has an mjlab twin built from its registration. For a custom scene, pass a
model recipe and an ``edit_fn`` (twin-compatible) or a compiled model (CPU only).
The goal logic lives in :mod:`myosuite.terms.waypoint`, shared with the twin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.model_builder import build_from_recipe
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.terms.base_obs import joint_vel_obs, muscle_act_obs
from myosuite.terms.waypoint import (
    WaypointRewardCfg,
    WaypointRouteCfg,
    distance_to_next,
    perturb_qpos,
    quat_yaw,
    sample_route,
    waypoint_progress,
    waypoint_reward,
    waypoint_targets_obs,
)

OBS_KEYS = ("qpos", "qvel", "act", "waypoint_targets")


@dataclass(frozen=True)
class WaypointTaskCfg:
    """Goal, reward and reset of a waypoint task.

    Attributes:
        site_name: Site whose horizontal position must reach the waypoints.
        waypoints: Fixed world-frame ``(x, y)`` route in metres, start excluded;
            ``None`` draws a route from ``route`` at every reset.
        route: Random-route distribution (used when ``waypoints`` is ``None``).
        arrival_radius: Inclusive horizontal acceptance radius in metres.
        lookahead: Upcoming waypoints observed, in the heading frame.
        reward: Reward weights.
        min_site_height: Fail when the site drops below this height (``None``: never).
        reset_noise: Amplitude of uniform noise on the hinge/slide positions at reset.
        obs_keys: Observation keys, in order (subset of :data:`OBS_KEYS`).
    """

    site_name: str = "pelvis_mimic"
    waypoints: tuple[tuple[float, float], ...] | None = None
    route: WaypointRouteCfg = field(default_factory=WaypointRouteCfg)
    arrival_radius: float = 0.3
    lookahead: int = 2
    reward: WaypointRewardCfg = field(default_factory=WaypointRewardCfg)
    min_site_height: float | None = None
    reset_noise: float = 0.0
    obs_keys: tuple[str, ...] = OBS_KEYS

    def __post_init__(self) -> None:
        if self.waypoints is not None:
            points = np.asarray(self.waypoints, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
                raise ValueError("waypoints must be a nonempty (N, 2) sequence")
            if not np.isfinite(points).all():
                raise ValueError("waypoints must be finite")
        if not np.isfinite(self.arrival_radius) or self.arrival_radius <= 0:
            raise ValueError("arrival_radius must be finite and positive")
        if self.lookahead < 1 or self.route.num_waypoints < 1:
            raise ValueError("lookahead and route.num_waypoints must be positive")
        if self.reset_noise < 0:
            raise ValueError("reset_noise must be non-negative")
        if unknown := set(self.obs_keys) - set(OBS_KEYS):
            raise ValueError(f"Unknown obs keys {sorted(unknown)}; use {OBS_KEYS}")

    @property
    def num_waypoints(self) -> int:
        """Route length."""
        return (
            len(self.waypoints)
            if self.waypoints is not None
            else self.route.num_waypoints
        )


@dataclass(frozen=True)
class RootLayout:
    """The free root joint (if any) and the reset-noise bounds of a model.

    Attributes:
        has_free_root: Whether joint 0 is a free joint at ``qpos[0:7]``.
        noise_mask: ``(nq,)`` 1 on hinge/slide coordinates (perturbed at reset), else 0.
        low: ``(nq,)`` lower joint limits (``-inf`` where unlimited).
        high: ``(nq,)`` upper joint limits (``inf`` where unlimited).
    """

    has_free_root: bool
    noise_mask: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> RootLayout:
        free = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        mask = np.zeros(model.nq)
        low, high = np.full(model.nq, -np.inf), np.full(model.nq, np.inf)
        linear = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        for j in range(model.njnt):
            if model.jnt_type[j] in linear:
                adr = model.jnt_qposadr[j]
                mask[adr] = 1.0
                if model.jnt_limited[j]:
                    low[adr], high[adr] = model.jnt_range[j]
        return cls(bool(free), mask, low, high)


def qpos_obs(qpos: Any, has_free_root: bool) -> Any:
    """``qpos`` without the world ``x, y`` of a free root (translation invariant)."""
    return qpos[..., 2:] if has_free_root else qpos


def heading_yaw(xp: Any, qpos: Any, has_free_root: bool) -> Any:
    """Yaw of the free root, or zero (world frame) without one."""
    if has_free_root:
        return quat_yaw(xp, qpos[..., 3:7])
    return 0 * qpos[..., 0]


def initial_state(
    model: mujoco.MjModel, qpos: ArrayLike | None = None, qvel: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reset state: *qpos* / *qvel* (default zero), else keyframe 0, else ``qpos0``."""
    if qpos is not None:
        pos = np.array(qpos, dtype=float, copy=True)
        vel = (
            np.zeros(model.nv)
            if qvel is None
            else np.array(qvel, dtype=float, copy=True)
        )
    elif model.nkey > 0:
        pos, vel = model.key_qpos[0].copy(), model.key_qvel[0].copy()
    else:
        pos, vel = model.qpos0.copy(), np.zeros(model.nv)
    if pos.shape != (model.nq,) or not np.isfinite(pos).all():
        raise ValueError("initial_qpos must be finite and match model.nq")
    if vel.shape != (model.nv,) or not np.isfinite(vel).all():
        raise ValueError("initial_qvel must be finite and match model.nv")
    return pos, vel


def build_model(
    model_recipe: str | None = None,
    model_path: str | None = None,
    edit_fn: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjModel:
    """Compile the model of a registration: a recipe or an XML file, plus *edit_fn*."""
    if (model_recipe is None) == (model_path is None):
        raise ValueError("Pass exactly one of model_recipe and model_path")
    if model_recipe is not None:
        return build_from_recipe(model_recipe, edit_fn)[0]
    spec = mujoco.MjSpec.from_file(str(model_path))
    if edit_fn is not None:
        edit_fn(spec)
    return spec.compile()


class WaypointEnv(MyoGymnasiumEnv):
    """Reach ordered world-frame XY waypoints with a site of the actor.

    Args:
        model: Compiled scene and actor (CPU only). Otherwise built from
            ``model_recipe`` / ``model_path`` and ``edit_fn``, as the mjlab twin does.
        task: Goal, reward and reset settings.
        model_recipe: ``ModelBuilder`` recipe name (e.g. ``"musclemimic_fullbody"``).
        model_path: MJCF path (alternative to a recipe).
        edit_fn: In-place ``MjSpec`` edit adding the scene (terrain, obstacles).
        initial_qpos: Reset pose; default keyframe 0, else ``qpos0``.
        initial_qvel: Reset velocity with *initial_qpos* (default zero).
        frame_skip: Physics substeps per control step.
        render_mode: Standard render mode.

    Actions are raw actuator controls, clipped to the control ranges. Observations
    (:data:`OBS_KEYS`): ``qpos`` without the root's world ``x, y``, ``qvel *
    ctrl_dt``, muscle activations and the offsets to the next ``lookahead``
    waypoints in the heading frame. The reward (:func:`waypoint_reward`) pays
    progress and arrivals; the episode terminates on completion, a fall or a
    simulation divergence. Use ``TimeLimit`` (``max_episode_steps``) for a deadline.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        *,
        task: WaypointTaskCfg | None = None,
        model_recipe: str | None = None,
        model_path: str | None = None,
        edit_fn: Callable[[mujoco.MjSpec], None] | None = None,
        initial_qpos: ArrayLike | None = None,
        initial_qvel: ArrayLike | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        if (
            isinstance(frame_skip, bool)
            or not isinstance(frame_skip, (int, np.integer))
            or frame_skip < 1
        ):
            raise ValueError("frame_skip must be a positive integer")
        if model is None:
            model = build_model(model_recipe, model_path, edit_fn)
        elif model_recipe is not None or model_path is not None or edit_fn is not None:
            raise ValueError("Pass a compiled model or a model source, not both")
        self.task = task or WaypointTaskCfg()
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.task.site_name
        )
        if site_id < 0:
            raise ValueError(f"No site {self.task.site_name!r} in the model")
        super().__init__(frame_skip=int(frame_skip), render_mode=render_mode)
        self.seed(seed)
        self.model, self.data = model, mujoco.MjData(model)
        self._ctrl_dt = model.opt.timestep * self.frame_skip
        self.metadata = {
            **self.metadata,
            "render_fps": max(1, round(1 / self._ctrl_dt)),
        }
        self._site_id = site_id
        self._layout = RootLayout.from_model(model)
        self._init_qpos, self._init_qvel = initial_state(
            model, initial_qpos, initial_qvel
        )
        # Route start: the site and heading of the noise-free reset pose.
        self.data.qpos[:] = self._init_qpos
        self._start = self._site_xy_at_qpos()
        self._start_yaw = heading_yaw(np, self._init_qpos, self._layout.has_free_root)
        self.obs_keys = list(self.task.obs_keys)
        low = np.where(
            model.actuator_ctrllimited, model.actuator_ctrlrange[:, 0], -np.inf
        )
        high = np.where(
            model.actuator_ctrllimited, model.actuator_ctrlrange[:, 1], np.inf
        )
        self.action_space = gym.spaces.Box(
            low.astype(np.float32), high.astype(np.float32)
        )
        self._task_state = self.reset_task(self.np_random)
        mujoco.mj_forward(model, self.data)
        accessor = CpuEnvAccessor(model, self.data, self._ctrl_dt)
        self.observation_space = self._unbounded_obs_space(
            self._obs_dict_to_vec(self._get_obs_dict(accessor)).size
        )

    @property
    def waypoints(self) -> np.ndarray:
        """Route of the current episode, ``(W, 2)``."""
        return self._waypoints

    @property
    def next_waypoint(self) -> int:
        """Index of the active waypoint (``W`` when the route is complete)."""
        return int(self._task_state["next_index"])

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Restore the reset pose (plus joint noise) and draw or restore the route."""
        task = self.task
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        if task.reset_noise > 0:
            lay = self._layout
            self.data.qpos[:] = perturb_qpos(
                np,
                self._init_qpos,
                np_random.random(self.model.nq),
                task.reset_noise * lay.noise_mask,
                lay.low,
                lay.high,
            )
        if task.waypoints is not None:
            self._waypoints = np.asarray(task.waypoints, dtype=float)
        else:
            n = task.route.num_waypoints
            u = np_random.random((2, n))
            self._waypoints = sample_route(
                np, self._start, self._start_yaw, u[0], u[1], task.route
            )
        self._waypoints.flags.writeable = False
        next_index = np.asarray(0)
        return {
            "next_index": next_index,
            # From the noise-free start (the mjlab twin cannot read the reset site).
            "prev_distance": distance_to_next(
                np, self._start, self._waypoints, next_index
            ),
            "progress": np.asarray(0.0),
            "arrived": np.asarray(False),
            "failed": np.asarray(False),
            "solved": np.asarray(False),
        }

    def _site_xy_at_qpos(self) -> np.ndarray:
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self._site_id, :2].copy()

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        qpos = accessor.joint_pos()
        free = self._layout.has_free_root
        all_obs = {
            "qpos": qpos_obs(qpos, free),
            "qvel": joint_vel_obs(accessor),
            "act": muscle_act_obs(accessor),
            "waypoint_targets": waypoint_targets_obs(
                np,
                self.data.site_xpos[self._site_id, :2],
                heading_yaw(np, qpos, free),
                self._waypoints,
                self._task_state["next_index"],
                self.task.lookahead,
            ),
        }
        return self._select_obs_keys(all_obs)

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Reward of the last step's progress (reading it does not advance the route)."""
        state = self._task_state
        rwd = waypoint_reward(None, state, self.task.reward)
        rwd = {k: np.asarray(v).item() for k, v in rwd.items()}
        rwd["failed"] = bool(state["failed"])
        rwd["next_waypoint"] = int(state["next_index"])
        rwd["distance"] = float(state["prev_distance"])
        return rwd

    def _failed(self) -> bool:
        if self._check_mj_instability_termination():
            return True
        floor = self.task.min_site_height
        return floor is not None and bool(self.data.site_xpos[self._site_id, 2] < floor)

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance physics, then the route by at most one waypoint."""
        del kwargs
        if bool(self._task_state["solved"]) or bool(self._task_state["failed"]):
            raise gym.error.ResetNeeded("Episode ended; call reset() before step()")
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action must be finite and match action_space.shape")
        self.data.ctrl[:] = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        self._step_physics()
        state = self._task_state
        failed = np.asarray(self._failed())
        state.update(
            waypoint_progress(
                np,
                self.data.site_xpos[self._site_id, :2],
                self._waypoints,
                state["next_index"],
                state["prev_distance"],
                self.task.arrival_radius,
                failed,
            )
        )
        state["failed"] = failed
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._get_obs_dict(self._accessor)
        return self._finalize_step(obs, self.get_reward_dict(obs))
