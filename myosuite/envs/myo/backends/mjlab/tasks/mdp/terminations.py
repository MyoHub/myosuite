# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Termination helpers shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Termination key under which tasks register :class:`sync_kinematics`.
SYNC_TERM = "sync_kinematics"

# Data fields CPU envs read after mj_step + mj_kinematics, i.e. still holding
# the value of the last physics substep (mj_kinematics does not recompute them).
STALE_FIELDS = ("cvel", "actuator_length", "actuator_velocity", "actuator_force")


class sync_kinematics(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """Reproduce the CPU post-step state for terminations, rewards and obs.

    CPU envs run ``mj_step`` then ``mj_kinematics``: positions/orientations
    (``xpos``, ``xquat``, ``xipos``, ``site_xpos``) match the new ``qpos``, while
    velocity- and actuator-dependent quantities (:data:`STALE_FIELDS`) keep
    the value of the last substep. mjlab evaluates terminations and rewards
    before its single post-step ``forward()`` (everything one substep stale)
    and observations after it (everything fresh).

    Registered as the first termination term (key :data:`SYNC_TERM`), this
    snapshots :data:`STALE_FIELDS` and then runs kinematics only; observation
    terms read the snapshot through :func:`cpu_post_step_field`. Never
    terminates.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self.stale = {f: getattr(env.sim.data, f).clone() for f in STALE_FIELDS}

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        for field, buf in self.stale.items():
            buf.copy_(getattr(env.sim.data, field))
        with wp.ScopedDevice(env.sim.wp_device):
            mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def cpu_post_step_field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    """``env.sim.data.<field>`` as a CPU env sees it (see :class:`sync_kinematics`).

    Envs mid-episode get the snapshot taken right after stepping; envs at
    episode start get the live value (CPU ``reset()`` runs ``mj_forward``).

    Args:
        env: Environment registering :class:`sync_kinematics` as ``SYNC_TERM``.
        field: One of :data:`STALE_FIELDS`.

    Returns:
        The field, shape ``(num_envs, ...)``.
    """
    live = getattr(env.sim.data, field)
    manager = getattr(env, "termination_manager", None)
    if manager is None:  # observation shapes are probed before the managers exist
        return live
    stale = manager.get_term_cfg(SYNC_TERM).func.stale[field]
    started = (env.episode_length_buf > 0).view(-1, *([1] * (live.ndim - 1)))
    return torch.where(started, stale, live)
