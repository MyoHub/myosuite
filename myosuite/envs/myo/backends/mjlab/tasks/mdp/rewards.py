# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared reward terms for MyoSuite mjlab tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.manager_base import ManagerTermBase

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.distribution_reward import (
    DistributionMatchingCfg,
    distribution_matching_reward,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg


class DistributionMatchingReward(ManagerTermBase):
    """``log p*(x) - log q(x)`` of a feature for all envs
    (:mod:`myosuite.terms.distribution_reward`).

    ``RewardTermCfg.params``:

    - ``distribution``: the feature (read from a :class:`MjlabEntityAccessor` and the
      task state), the target and the estimator.
    - ``entity_name``: the scene entity the feature reads.
    - ``task_state``: optional; builds the task state from the env, as the task's other
      terms build theirs (for example
      ``{"target_angles": env.command_manager.get_command("pose")}``), with the context
      index of a per-context target under ``distribution.context_key``.

    While the estimator is not frozen it learns from every step of all envs, after
    scoring it. It is training state, kept across episodes; freeze it for play and
    evaluation. mjlab copies the env cfg, so the estimator that learns is
    ``env.reward_manager.get_term_cfg(name).params["distribution"].estimator``. A
    feature that reads derived quantities needs ``mdp.sync_forward`` as the first
    termination term.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._accessor = MjlabEntityAccessor(
            env, cfg.params.get("entity_name", "robot")
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        distribution: DistributionMatchingCfg,
        entity_name: str = "robot",
        task_state: Callable[[ManagerBasedRlEnv], dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        state = {} if task_state is None else task_state(env)
        return distribution_matching_reward(
            self._accessor, state, distribution=distribution
        )["dense"]
