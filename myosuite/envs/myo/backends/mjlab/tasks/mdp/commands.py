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

    from myosuite.core.config import GoalSpec

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


@dataclass(kw_only=True)
class HeadingCommandCfg(CommandTermCfg):
    """Commanded planar heading (unit vector), fixed or sampled once per episode.

    Attributes:
        heading_dir: Heading of the fixed-direction tasks (and the initial value).
        randomize: Sample a fresh direction from the full unit circle at every reset
            (CPU ``randomize_heading``).
    """

    heading_dir: tuple[float, float] = (0.0, 1.0)
    randomize: bool = False
    resampling_time_range: tuple[float, float] = RESAMPLE_ON_RESET_ONLY

    def build(self, env: ManagerBasedRlEnv) -> HeadingCommand:
        return HeadingCommand(self, env)


class HeadingCommand(CommandTerm):
    """Per-env unit heading vector ``(dx, dy)``, constant within an episode."""

    cfg: HeadingCommandCfg

    def __init__(self, cfg: HeadingCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        fixed = torch.tensor(cfg.heading_dir, dtype=torch.float32, device=self.device)
        self._target = fixed.repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._target

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.randomize:
            return
        angle = 2.0 * torch.pi * torch.rand(len(env_ids), device=self.device)
        self._target[env_ids] = torch.stack([angle.cos(), angle.sin()], dim=-1)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids  # Target is constant within an episode.

    def _update_metrics(self) -> None:
        pass


def site_position_command_cfg(
    goal_spec: GoalSpec, entity_name: str
) -> UniformVectorCommandCfg:
    """mjlab command sampling the targets of a ``GoalSpec(target_type="site_positions")``.

    The command is the flattened ``(3 * n_sites,)`` position vector the CPU/MJX goal
    sampler returns as ``target_pos`` (sites in ``goal_spec.range`` order): uniform in
    the per-site bounds with ``randomize=True``, the lower bound otherwise.

    Args:
        goal_spec: A site-position goal with a non-empty ``range``.
        entity_name: Scene entity the targets refer to.

    Returns:
        The command config (read the values with ``command_manager.get_command``).

    Raises:
        ValueError: If the goal is not a ``site_positions`` goal or has no ``range``.
    """
    if goal_spec.target_type != "site_positions":
        raise ValueError(
            f"Expected a site_positions goal, got {goal_spec.target_type!r}."
        )
    _, lo, hi = goal_spec.site_bounds()
    if not goal_spec.randomize:
        hi = lo
    return UniformVectorCommandCfg(
        entity_name=entity_name,
        low=tuple(float(x) for x in lo.reshape(-1)),
        high=tuple(float(x) for x in hi.reshape(-1)),
    )
