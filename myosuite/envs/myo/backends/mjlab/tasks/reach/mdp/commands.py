# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reach target command (CPU ``ReachEnvV0`` target-site sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from myosuite.envs.myo.backends.mjlab.tasks.mdp.commands import (
    UniformVectorCommand,
    UniformVectorCommandCfg,
)
from myosuite.utils.reach_workspace import reachable_target_points

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReachTargetCommandCfg(UniformVectorCommandCfg):
    """Flattened ``3k`` world positions of the ``k`` tip-site targets.

    Attributes:
        tip_sites: Tip site names, in target order.
    """

    tip_sites: tuple[str, ...]

    def build(self, env: ManagerBasedRlEnv) -> ReachTargetCommand:
        return ReachTargetCommand(self, env)


class ReachTargetCommand(UniformVectorCommand):
    """Cartesian reach targets; logs the tip-to-target distance."""

    cfg: ReachTargetCommandCfg

    def __init__(self, cfg: ReachTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        ids, _ = env.scene[cfg.entity_name].find_sites(
            cfg.tip_sites, preserve_order=True
        )
        self._tip_ids = ids
        self.metrics["reach_error"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self) -> None:
        tip = self._accessor.site_xpos(self._tip_ids).reshape(self.num_envs, -1)
        self.metrics["reach_error"][:] = torch.linalg.norm(self._target - tip, dim=-1)


@dataclass(kw_only=True)
class RelativeReachTargetCommandCfg(ReachTargetCommandCfg):
    """Targets ``U(low, high)`` away from the tip sites' position at reset.

    ``low`` / ``high`` are the flattened ``3k`` offsets (CPU ``LegReachEnvV0``).
    """

    def build(self, env: ManagerBasedRlEnv) -> RelativeReachTargetCommand:
        return RelativeReachTargetCommand(self, env)


class RelativeReachTargetCommand(ReachTargetCommand):
    """Reach targets relative to where the tip sites start the episode."""

    cfg: RelativeReachTargetCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Reset-path resamples run before mjlab's forward(): refresh the site
        # positions from the qpos the reset events just wrote.
        env = self._env
        with wp.ScopedDevice(env.sim.wp_device):
            mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
        tip = self._accessor.site_xpos(self._tip_ids).reshape(self.num_envs, -1)
        u = torch.rand(len(env_ids), self._low.numel(), device=self.device)
        self._target[env_ids] = tip[env_ids] + self._low + u * (self._high - self._low)


@dataclass(kw_only=True)
class WorkspaceReachTargetCommandCfg(ReachTargetCommandCfg):
    """Targets the tip sites can reach (CPU ``target_sampling="workspace"``).

    ``low`` / ``high`` are the flattened ``3k`` bounds of the target boxes; targets are
    the tip positions over the joint ranges that lie inside them.
    """

    def build(self, env: ManagerBasedRlEnv) -> WorkspaceReachTargetCommand:
        return WorkspaceReachTargetCommand(self, env)


class WorkspaceReachTargetCommand(ReachTargetCommand):
    """Reach targets drawn from the reachable tip positions."""

    cfg: WorkspaceReachTargetCommandCfg

    def __init__(
        self, cfg: WorkspaceReachTargetCommandCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg, env)
        # The table is built on the scene model: map entity-local site ids to scene ids.
        scene_site_ids = env.scene[cfg.entity_name].indexing.site_ids
        sites = [int(scene_site_ids[i]) for i in self._tip_ids]
        points = reachable_target_points(
            env.sim.mj_model,
            sites,
            np.asarray(cfg.low).reshape(len(sites), 3),
            np.asarray(cfg.high).reshape(len(sites), 3),
        )
        self._points = torch.as_tensor(
            points.reshape(len(points), -1), dtype=torch.float32, device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        index = torch.randint(len(self._points), (len(env_ids),), device=self.device)
        self._target[env_ids] = self._points[index]
