# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Analytic MuJoCo muscle activation dynamics (``mju_muscleDynamics``) in torch."""

from __future__ import annotations

import mujoco
import numpy as np
import torch


class MuscleActivation(torch.nn.Module):
    """Activation after one control step of constant ``ctrl``, Euler-integrated.

    Matches MuJoCo's ``mjDYN_MUSCLE`` with Millard time constants, ``n_substeps``
    Euler updates of ``sim_dt`` and no ``actearly``. The world model uses it so the
    network never has to learn the first-order activation filter.

    Args:
        model: Compiled MuJoCo model whose actuators are all muscles.
        n_substeps: Physics steps per control step (the env's decimation).
        sim_dt: Physics timestep; defaults to ``model.opt.timestep`` (an mjlab
            env may simulate at another step than the compiled spec's option).
    """

    def __init__(
        self, model: mujoco.MjModel, n_substeps: int, sim_dt: float | None = None
    ) -> None:
        super().__init__()
        if not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE):
            raise ValueError("all actuators must be muscles")
        if np.any(model.actuator_actearly):
            raise ValueError("actearly muscles are not supported")
        if np.any(model.actuator_dynprm[:, 2] > 0):
            raise ValueError("smoothed muscle time constants are not supported")
        f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        self.register_buffer("tau_act", f(model.actuator_dynprm[:, 0]))
        self.register_buffer("tau_deact", f(model.actuator_dynprm[:, 1]))
        limited = model.actuator_actlimited.astype(bool)
        self.register_buffer(
            "act_lo", f(np.where(limited, model.actuator_actrange[:, 0], -np.inf))
        )
        self.register_buffer(
            "act_hi", f(np.where(limited, model.actuator_actrange[:, 1], np.inf))
        )
        self.dt = float(model.opt.timestep if sim_dt is None else sim_dt)
        self.n_substeps = int(n_substeps)

    def forward(self, act: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """Activation after ``n_substeps`` physics steps from ``act`` under ``ctrl``."""
        u = ctrl.clamp(0.0, 1.0)
        for _ in range(self.n_substeps):
            a = act.clamp(0.0, 1.0)
            tau = torch.where(
                u - act > 0,
                self.tau_act * (0.5 + 1.5 * a),
                self.tau_deact / (0.5 + 1.5 * a),
            )
            act = act + self.dt * (u - act) / tau.clamp_min(1e-15)
            act = torch.minimum(torch.maximum(act, self.act_lo), self.act_hi)
        return act
