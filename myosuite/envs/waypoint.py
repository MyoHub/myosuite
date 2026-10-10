# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental CPU waypoint task for a caller-built MuJoCo scene and actor.

Use ModelBuilder/MjSpec to construct the model, and Gymnasium TimeLimit for a
deadline. This class scores ordered XY arrival only: obstacle traversal, falls,
jump contacts and reference planning belong to task-specific extensions.
It is intentionally not registered until a matching GPU task is available.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv


class WaypointEnv(MyoGymnasiumEnv):
    """Track a body's origin through ordered world-frame XY targets (metres).

    Args:
        model: Compiled scene and actor; no geometry is added by this class.
        waypoints: Nonempty (N, 2) array of targets, excluding the spawn point.
        body_name: Body whose origin is evaluated, e.g. ``"pelvis"``.
        arrival_radius: Inclusive horizontal acceptance radius in metres.
        initial_qpos: Optional reset pose; defaults to ``model.qpos0``.
        frame_skip: Physics substeps per action.
        render_mode: Standard MyoGymnasiumEnv render mode.

    Success means visiting every target in order. At most one target is consumed
    per control step; merely crossing a target between samples does not count.
    Reward is minus distance to the target active at the start of the step.
    Actions are raw actuator controls, clipped to each actuator's control range.
    Model ownership stays with the caller; do not mutate it during an episode.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        waypoints: ArrayLike,
        *,
        body_name: str = "pelvis",
        arrival_radius: float = 0.2,
        initial_qpos: ArrayLike | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ) -> None:
        points = np.array(waypoints, dtype=float, copy=True)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            raise ValueError("waypoints must be a nonempty finite (N, 2) array")
        if not np.isfinite(arrival_radius) or arrival_radius <= 0:
            raise ValueError("arrival_radius must be finite and positive")
        if (
            isinstance(frame_skip, bool)
            or not isinstance(frame_skip, (int, np.integer))
            or frame_skip < 1
        ):
            raise ValueError("frame_skip must be a positive integer")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id <= 0:
            raise ValueError("body_name must identify a non-world body")
        qpos = np.array(
            model.qpos0 if initial_qpos is None else initial_qpos, copy=True
        )
        if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("initial_qpos must be finite and match model.nq")
        super().__init__(frame_skip=frame_skip, render_mode=render_mode)
        self.model, self.data = model, mujoco.MjData(model)
        self.waypoints, self.arrival_radius = points, float(arrival_radius)
        self.waypoints.flags.writeable = False
        self._body_id, self._initial_qpos = body_id, qpos
        self._ctrl_dt = model.opt.timestep * frame_skip
        self.metadata = {
            **self.metadata,
            "render_fps": max(1, round(1 / self._ctrl_dt)),
        }
        low = np.where(
            model.actuator_ctrllimited, model.actuator_ctrlrange[:, 0], -np.inf
        )
        high = np.where(
            model.actuator_ctrllimited, model.actuator_ctrlrange[:, 1], np.inf
        )
        self.action_space = gym.spaces.Box(
            low.astype(np.float32), high.astype(np.float32)
        )
        self.reset_task(self.np_random)
        mujoco.mj_forward(model, self.data)
        accessor = CpuEnvAccessor(model, self.data, self._ctrl_dt)
        self.observation_space = self._unbounded_obs_space(
            self._obs_dict_to_vec(self._get_obs_dict(accessor)).size
        )

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Restore the supplied pose and restart waypoint progress."""
        self.data.qpos[:] = self._initial_qpos
        self.next_waypoint = 0
        self._failed = False
        self._distance = 0.0
        return {}

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        target = self.waypoints[min(self.next_waypoint, len(self.waypoints) - 1)]
        return {
            "qpos": accessor.joint_pos(),
            "qvel": accessor.joint_vel(),
            "act": accessor.muscle_act(),
            "target_delta": target - self.data.xpos[self._body_id, :2],
            "remaining_waypoints": np.array([len(self.waypoints) - self.next_waypoint]),
            "arrival_radius": np.array([self.arrival_radius]),
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Return cached step metrics without advancing progress on reads."""
        solved = self.next_waypoint == len(self.waypoints)
        return {
            "dense": -self._distance,
            "solved": solved,
            "failed": self._failed,
            "done": solved or self._failed,
            "next_waypoint": self.next_waypoint,
            "distance": self._distance,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance real physics, then consume at most the next ordered target."""
        if self._failed or self.next_waypoint == len(self.waypoints):
            raise gym.error.ResetNeeded("Episode ended; call reset() before step()")
        action = np.asarray(action)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action must be finite and match action_space.shape")
        self.data.ctrl[:] = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        self._step_physics()
        self._failed = self._check_mj_instability_termination()
        self._distance = float(
            np.linalg.norm(
                self.waypoints[self.next_waypoint] - self.data.xpos[self._body_id, :2]
            )
        )
        if not self._failed and self._distance <= self.arrival_radius:
            self.next_waypoint += 1
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._get_obs_dict(self._accessor)
        return self._finalize_step(obs, self.get_reward_dict(obs))
