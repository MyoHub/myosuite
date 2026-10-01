# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Termination helpers shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Termination key under which tasks register :func:`sync_forward`.
SYNC_TERM = "sync_forward"


def sync_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Refresh derived quantities before terminations and rewards; never terminates.

    mjlab evaluates terminations and rewards before its single post-step
    ``sim.forward()``, so there ``xpos``, ``site_xpos``, ``cvel``,
    ``actuator_*``, ``sensordata`` and contacts still lag ``qpos``/``qvel`` by
    one physics substep. CPU envs run ``mj_forward`` right after stepping, so
    tasks that score derived quantities register this as their first
    termination term (key :data:`SYNC_TERM`). Observations need nothing: mjlab
    computes them after its own forward.

    Args:
        env: The environment.

    Returns:
        All-false termination flags, shape ``(num_envs,)``.
    """
    env.sim.forward()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
