# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Episode-constant target commands (CPU ``reset_task`` target sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Resampling window long enough that commands only resample on episode reset.
RESAMPLE_ON_RESET_ONLY: tuple[float, float] = (1.0e9, 1.0e9)


@dataclass(kw_only=True)
class UniformVectorCommandCfg(CommandTermCfg):
    """Target vector sampled uniformly in ``[low, high]`` once per episode.

    ``low == high`` gives the fixed-target variants.

    Attributes:
        entity_name: Scene entity the target refers to.
        low: Lower bound per entry.
        high: Upper bound per entry.
    """

    entity_name: str
    low: tuple[float, ...]
    high: tuple[float, ...]
    resampling_time_range: tuple[float, float] = RESAMPLE_ON_RESET_ONLY

    def build(self, env: ManagerBasedRlEnv) -> UniformVectorCommand:
        return UniformVectorCommand(self, env)


class UniformVectorCommand(CommandTerm):
    """Per-env target, constant within an episode."""

    cfg: UniformVectorCommandCfg

    def __init__(self, cfg: UniformVectorCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._low = torch.tensor(cfg.low, dtype=torch.float32, device=self.device)
        self._high = torch.tensor(cfg.high, dtype=torch.float32, device=self.device)
        self._target = self._low.repeat(self.num_envs, 1)
        self._accessor = MjlabEntityAccessor(env, cfg.entity_name)

    @property
    def command(self) -> torch.Tensor:
        return self._target

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        u = torch.rand(len(env_ids), self._low.numel(), device=self.device)
        self._target[env_ids] = self._low + u * (self._high - self._low)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Defaulted: mjlab < 1.6 calls _update_command() without env_ids.
        del env_ids  # Target is constant within an episode.

    def _update_metrics(self) -> None:
        pass
