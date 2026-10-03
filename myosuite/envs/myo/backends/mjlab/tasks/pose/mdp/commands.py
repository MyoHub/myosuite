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

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointPoseCommandCfg(UniformVectorCommandCfg):
    """Target for the leading ``qpos`` entries (``len(low)`` of them)."""

    def build(self, env: ManagerBasedRlEnv) -> JointPoseCommand:
        return JointPoseCommand(self, env)


class JointPoseCommand(UniformVectorCommand):
    """Joint-space target; logs the distance to it."""

    def __init__(self, cfg: JointPoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.metrics["pose_error"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self) -> None:
        qpos = self._accessor.joint_pos()[:, : self._target.shape[-1]]
        self.metrics["pose_error"][:] = torch.linalg.norm(self._target - qpos, dim=-1)
