# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Joint-pose target command (CPU ``PoseEnvV0`` target sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from myosuite.envs.myo.backends.mjlab.tasks.mdp.commands import (
    UniformVectorCommand,
    UniformVectorCommandCfg,
)
from myosuite.terms.base_event import fit_pose_target_to_couplings
from myosuite.terms.base_obs import target_joint_pos
from myosuite.utils.target_ranges import TargetCoupling

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointPoseCommandCfg(UniformVectorCommandCfg):
    """Joint target (CPU ``PoseEnvV0`` target sampling).

    Attributes:
        qpos_ids: ``qpos`` address of each entry; ``None`` for the leading
            ``len(low)`` entries.
        couplings: Coupled limits applied to each sample.
    """

    qpos_ids: tuple[int, ...] | None = None
    couplings: tuple[TargetCoupling, ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> JointPoseCommand:
        return JointPoseCommand(self, env)


class JointPoseCommand(UniformVectorCommand):
    """Joint-space target; logs the distance to it."""

    def __init__(self, cfg: JointPoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.metrics["pose_error"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if self.cfg.couplings:
            self._target[env_ids] = fit_pose_target_to_couplings(
                self._accessor,
                self._target[env_ids],
                self.cfg.low,
                self.cfg.high,
                self.cfg.couplings,
            )

    def _update_metrics(self) -> None:
        qpos = target_joint_pos(
            self._accessor, self._target.shape[-1], self.cfg.qpos_ids
        )
        self.metrics["pose_error"][:] = torch.linalg.norm(self._target - qpos, dim=-1)
