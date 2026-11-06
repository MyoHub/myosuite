# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward and metric terms shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.manager_base import ManagerTermBase

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.cocontraction import (
    MomentArmIndex,
    WastedTorqueCfg,
    wasted_torque_reward,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class wasted_torque(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """Wasted muscle torque per body mass (N·m/kg), shape ``(N,)``.

    The mjlab half of :func:`myosuite.terms.cocontraction.wasted_torque_reward`, usable as a
    reward term (with a negative weight) and as a metric. Reads ``actuator_force`` and
    ``actuator_moment``, so the task needs ``mdp.sync_forward`` as its first termination term.

    Params:
        entity_name: Scene entity of the muscles.
        cfg: :class:`~myosuite.terms.cocontraction.WastedTorqueCfg` (joints, body mass).
    """

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        name: str = cfg.params["entity_name"]
        cost: WastedTorqueCfg = cfg.params.get("cfg") or WastedTorqueCfg()
        index = MomentArmIndex.from_model(
            env.sim.mj_model,
            cost.joints,
            body_mass=cost.body_mass,
            name_prefix=f"{name}/",
        )
        self._index = index.to_backend(lambda a: torch.as_tensor(a, device=env.device))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        entity_name: str,
        cfg: WastedTorqueCfg | None = None,
    ) -> torch.Tensor:
        del cfg
        accessor = MjlabEntityAccessor(env, entity_name)
        return wasted_torque_reward(accessor, {}, moment_index=self._index)[
            "wasted_torque"
        ]
