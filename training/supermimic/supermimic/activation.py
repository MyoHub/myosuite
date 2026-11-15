# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""MuJoCo's muscle activation dynamics (``mjDYN_MUSCLE``) over one control step, in torch.

The activation is the only actuator state of the plant, and its dynamics are known exactly
(Millard et al. 2013 time constants, as ``mju_muscleDynamics``), so the world model does not learn
them: this module integrates them with the simulator's own explicit-Euler substeps, both to label
transitions and inside imagined rollouts. It also returns the substep-mean activation, which is what
drives the muscle forces during the control step and is the world model's action input.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/muscle.py``.
"""

from __future__ import annotations

import mujoco
import torch

MJ_MINVAL = 1e-15


class MuscleActivation:
    def __init__(self, m: mujoco.MjModel, substeps: int, dt: float, device: str):
        assert (m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).all()
        assert (
            not m.actuator_actearly.any()
        ), "actearly muscles use the next activation for force"
        assert (m.actuator_actnum == 1).all()
        prm = torch.tensor(m.actuator_dynprm[:, :3], dtype=torch.float32, device=device)
        self.tau_act, self.tau_deact, self.smooth = prm[:, 0], prm[:, 1], prm[:, 2]
        self.hard_switch = bool((self.smooth < MJ_MINVAL).all())
        self.limited = bool(m.actuator_actlimited.any())
        rng = torch.tensor(m.actuator_actrange, dtype=torch.float32, device=device)
        self.act_lo = torch.where(
            torch.tensor(m.actuator_actlimited, device=device).bool(),
            rng[:, 0],
            -torch.inf,
        )
        self.act_hi = torch.where(
            torch.tensor(m.actuator_actlimited, device=device).bool(),
            rng[:, 1],
            torch.inf,
        )
        self.substeps, self.dt = substeps, dt

    def act_dot(self, ctrl: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        ctrl = ctrl.clamp(0, 1)
        a = act.clamp(0, 1)
        tau_act = self.tau_act * (0.5 + 1.5 * a)
        tau_deact = self.tau_deact / (0.5 + 1.5 * a)
        dctrl = ctrl - act
        if self.hard_switch:
            tau = torch.where(dctrl > 0, tau_act, tau_deact)
        else:  # mju_muscleDynamicsTimescale with a quintic sigmoid of width `smooth`
            x = (dctrl / self.smooth.clamp(min=MJ_MINVAL) + 0.5).clamp(0, 1)
            s = x * x * x * (3 * x * (2 * x - 5) + 10)
            tau = tau_deact + (tau_act - tau_deact) * s
        return dctrl / tau.clamp(min=MJ_MINVAL)

    def step(
        self, act: torch.Tensor, ctrl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation after one control step under constant ``ctrl``, and its mean over the substeps
        (force at substep k uses the activation before that substep's integration)."""
        total = torch.zeros_like(act)
        for _ in range(self.substeps):
            total = total + act
            act = act + self.dt * self.act_dot(ctrl, act)
            if self.limited:
                act = torch.maximum(torch.minimum(act, self.act_hi), self.act_lo)
        return act, total / self.substeps
