# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MDP terms of the waypoint twin; the goal logic is :mod:`myosuite.terms.waypoint`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (
    MjlabEntityAccessor,
    normalize_mjlab_env_ids,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.commands import RESAMPLE_ON_RESET_ONLY
from myosuite.envs.myo.backends.mjlab.tasks.mdp.events import write_cpu_state
from myosuite.envs.waypoint import WaypointTaskCfg, heading_yaw, qpos_obs
from myosuite.terms.base_obs import joint_vel_obs, muscle_act_obs
from myosuite.terms.waypoint import (
    distance_to_next,
    perturb_qpos,
    sample_route,
    waypoint_progress,
    waypoint_reward,
    waypoint_targets_obs,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Command term holding each env's route.
COMMAND = "route"


@dataclass(kw_only=True)
class WaypointCommandCfg(CommandTermCfg):
    """Per-env route and progress (CPU ``WaypointEnv`` task state).

    Attributes:
        task: Goal, reward and reset settings of the CPU registration.
        entity_name: Scene entity carrying the tracked site.
        start: Route start ``(x, y)``: the site at the noise-free reset pose.
        start_yaw: Heading yaw of the reset pose.
    """

    task: WaypointTaskCfg
    entity_name: str
    start: tuple[float, float]
    start_yaw: float
    resampling_time_range: tuple[float, float] = RESAMPLE_ON_RESET_ONLY

    def build(self, env: ManagerBasedRlEnv) -> WaypointCommand:
        return WaypointCommand(self, env)


class WaypointCommand(CommandTerm):
    """Routes drawn (or restored) at reset and advanced by :func:`advance_route`.

    The command is the route, ``(N, 2W)``. ``state`` holds the last step's
    :func:`~myosuite.terms.waypoint.waypoint_progress` result plus ``failed``.
    """

    cfg: WaypointCommandCfg

    def __init__(self, cfg: WaypointCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.task = cfg.task
        self._accessor = MjlabEntityAccessor(env, cfg.entity_name)
        self.site = self._accessor.site_id(cfg.task.site_name)
        self.has_free_root = not env.scene[cfg.entity_name].is_fixed_base
        n, w, dev = self.num_envs, cfg.task.num_waypoints, self.device
        self._start = torch.tensor(cfg.start, dtype=torch.float32, device=dev)
        self._start_yaw = torch.tensor(float(cfg.start_yaw), device=dev)
        fixed = cfg.task.waypoints
        self._fixed = (
            None
            if fixed is None
            else torch.tensor(fixed, dtype=torch.float32, device=dev)
        )
        self.waypoints = torch.zeros(n, w, 2, device=dev)
        self.next_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_distance = torch.zeros(n, device=dev)
        self.state: dict[str, torch.Tensor] = {
            "progress": torch.zeros(n, device=dev),
            "arrived": torch.zeros(n, dtype=torch.bool, device=dev),
            "failed": torch.zeros(n, dtype=torch.bool, device=dev),
            "solved": torch.zeros(n, dtype=torch.bool, device=dev),
        }

    @property
    def command(self) -> torch.Tensor:
        return self.waypoints.flatten(1)

    def site_pos(self) -> torch.Tensor:
        """Tracked site position ``(N, 3)`` in the CPU world frame."""
        return self._accessor.site_xpos([self.site])[:, 0]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        k = len(env_ids)
        if self._fixed is not None:
            route = self._fixed.expand(k, -1, -1)
        else:
            u = torch.rand(2, k, self.task.num_waypoints, device=self.device)
            route = sample_route(
                torch,
                self._start.expand(k, 2),
                self._start_yaw.expand(k),
                u[0],
                u[1],
                self.task.route,
            )
        self.waypoints[env_ids] = route
        self.next_index[env_ids] = 0
        self.prev_distance[env_ids] = distance_to_next(
            torch, self._start.expand(k, 2), route, self.next_index[env_ids]
        )
        for value in self.state.values():
            value[env_ids] = 0

    def advance(self) -> torch.Tensor:
        """One control step of every route (after ``sync_forward``); returns ``solved``."""
        pos = self.site_pos()
        failed = ~torch.isfinite(self._accessor.joint_pos()).all(-1)
        if self.task.min_site_height is not None:
            failed = failed | (pos[:, 2] < self.task.min_site_height)
        result = waypoint_progress(
            torch,
            pos[:, :2],
            self.waypoints,
            self.next_index,
            self.prev_distance,
            self.task.arrival_radius,
            failed,
        )
        self.next_index = result["next_index"]
        self.prev_distance = result["prev_distance"]
        self.state = {**result, "failed": failed}
        return result["solved"]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids  # Advanced by the termination term, before the rewards.

    def _update_metrics(self) -> None:
        pass


def _route(env: ManagerBasedRlEnv) -> WaypointCommand:
    return env.command_manager.get_term(COMMAND)


def advance_route(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Termination ``waypoints_done``: advance every route; the route is complete."""
    return _route(env).advance()


def failed(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Termination ``failed``: fall below ``min_site_height`` or non-finite state."""
    return _route(env).state["failed"]


def reward_component(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
    """One unweighted component of :func:`waypoint_reward` (the weight is the term's)."""
    return _route(env).state[key].float()


def solved(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Metric ``success``: the route is complete."""
    route = _route(env)
    return waypoint_reward(None, route.state, route.task.reward)["solved"].float()


def qpos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``qpos`` observation: without the root's world ``x, y``."""
    route = _route(env)
    return qpos_obs(
        MjlabEntityAccessor(env, asset_cfg.name).joint_pos(), route.has_free_root
    )


def qvel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``qvel`` observation: ``qvel * ctrl_dt``."""
    return joint_vel_obs(MjlabEntityAccessor(env, asset_cfg.name))


def act(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``act`` observation: muscle activations."""
    return muscle_act_obs(MjlabEntityAccessor(env, asset_cfg.name))


def waypoint_targets(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``waypoint_targets`` observation: next waypoints in the heading frame."""
    route = _route(env)
    qpos_full = MjlabEntityAccessor(env, asset_cfg.name).joint_pos()
    return waypoint_targets_obs(
        torch,
        route.site_pos()[:, :2],
        heading_yaw(torch, qpos_full, route.has_free_root),
        route.waypoints,
        route.next_index,
        route.task.lookahead,
    )


class reset_pose(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """CPU ``WaypointEnv.reset_task`` pose: reset state plus uniform joint noise.

    Params: ``asset_cfg``, ``qpos``/``qvel`` (CPU layout), ``noise``/``low``/``high``
    (per ``qpos`` coordinate, see :class:`~myosuite.envs.waypoint.RootLayout`).
    """

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p, dev = cfg.params, env.device
        self._qpos = torch.tensor(p["qpos"], dtype=torch.float32, device=dev)
        self._qvel = torch.tensor(p["qvel"], dtype=torch.float32, device=dev)
        self._noise = torch.tensor(p["noise"], dtype=torch.float32, device=dev)
        self._low = torch.tensor(p["low"], dtype=torch.float32, device=dev)
        self._high = torch.tensor(p["high"], dtype=torch.float32, device=dev)

    def __call__(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any
    ) -> None:
        ids = normalize_mjlab_env_ids(env, env_ids)
        q = self._qpos.repeat(len(ids), 1)
        q = perturb_qpos(
            torch, q, torch.rand_like(q), self._noise, self._low, self._high
        )
        write_cpu_state(
            env, params["asset_cfg"].name, ids, q, self._qvel.repeat(len(ids), 1)
        )
