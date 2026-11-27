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

Changed for MyoSuite: bodies with motors next to the muscles (a submodel with torque-driven
joints). The state is then the actuator *drive* in [0, 1], one per actuator: the muscles'
activation, a direct motor's normalized command (``(ctrl - lo) / (hi - lo)``, what drives its
force during the step) and a first-order (``mjDYN_FILTER``) motor's normalized activation,
integrated with the same Euler substeps.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/muscle.py``.
"""

from __future__ import annotations

import mujoco
import torch

MJ_MINVAL = 1e-15


class MuscleActivation:
    """Actuator dynamics over one control step.

    Args:
        m: The body.
        substeps: Physics steps per control step.
        dt: Physics step.
        device: Torch device.
        muscle_ids: Actuator ids of the muscles (``None``: every actuator).
    """

    def __init__(self, m: mujoco.MjModel, substeps: int, dt: float, device: str,
                 muscle_ids=None):  # fmt: skip
        nu = m.nu
        mus = list(range(nu)) if muscle_ids is None else [int(a) for a in muscle_ids]
        mot = [a for a in range(nu) if a not in set(mus)]
        self.all_muscles = not mot
        self.muscle_ids = torch.tensor(mus, dtype=torch.long, device=device)
        self.motor_ids = torch.tensor(mot, dtype=torch.long, device=device)
        # actuator order from cat(muscles, motors)
        self.order = torch.argsort(torch.tensor(mus + mot, device=device))
        dyn = m.actuator_dyntype
        assert (dyn[mus] == mujoco.mjtDyn.mjDYN_MUSCLE).all()
        assert (
            not m.actuator_actearly.any()
        ), "actearly actuators use the next activation for force"
        assert (m.actuator_actnum[mus] == 1).all()
        # int(): a NumPy scalar does not compare equal to a MuJoCo enum member (MuJoCo >= 3.17)
        filt = [a for a in mot if int(dyn[a]) == mujoco.mjtDyn.mjDYN_FILTER]
        assert all(
            int(dyn[a]) in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTER)
            for a in mot
        )
        self.motor_filter = torch.tensor([a in filt for a in mot], device=device)
        self.motor_tau = torch.tensor(
            [max(float(m.actuator_dynprm[a, 0]), MJ_MINVAL) for a in mot],
            dtype=torch.float32, device=device,
        )  # fmt: skip
        self.has_filter = bool(filt)
        # muscles' ctrl clamp: their control range (MuJoCo clamps ctrl first)
        cr = m.actuator_ctrlrange[mus]
        self.unit_ctrl = bool((cr[:, 0] == 0).all() and (cr[:, 1] == 1).all())
        self.ctrl_lo = torch.tensor(cr[:, 0], dtype=torch.float32, device=device)
        self.ctrl_hi = torch.tensor(cr[:, 1], dtype=torch.float32, device=device)
        m_prm = m.actuator_dynprm[mus]
        prm = torch.tensor(m_prm[:, :3], dtype=torch.float32, device=device)
        self.tau_act, self.tau_deact, self.smooth = prm[:, 0], prm[:, 1], prm[:, 2]
        self.hard_switch = bool((self.smooth < MJ_MINVAL).all())
        lim = m.actuator_actlimited[mus]
        self.limited = bool(lim.any())
        rng = torch.tensor(m.actuator_actrange[mus], dtype=torch.float32, device=device)
        self.act_lo = torch.where(
            torch.tensor(lim, device=device).bool(),
            rng[:, 0],
            -torch.inf,
        )
        self.act_hi = torch.where(
            torch.tensor(lim, device=device).bool(),
            rng[:, 1],
            torch.inf,
        )
        self.substeps, self.dt = substeps, dt

    def act_dot(self, ctrl: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        ctrl = (
            ctrl.clamp(0, 1)
            if self.unit_ctrl
            else torch.minimum(torch.maximum(ctrl, self.ctrl_lo), self.ctrl_hi)
        )
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
        """Drive after one control step under the constant action ``ctrl`` (in [0, 1]), and its mean over the
        substeps (force at substep k uses the activation before that substep's integration)."""
        if self.all_muscles:
            return self._muscles(act, ctrl)
        a1, mean = self._muscles(act[..., self.muscle_ids], ctrl[..., self.muscle_ids])
        x, u = act[..., self.motor_ids], ctrl[..., self.motor_ids].clamp(0, 1)
        if self.has_filter:  # first-order motors, normalized units (the affine map commutes with the filter)
            tot, y = torch.zeros_like(x), x
            for _ in range(self.substeps):
                tot = tot + y
                y = y + self.dt * (u - y) / self.motor_tau
            x1, xm = (
                torch.where(self.motor_filter, y, u),
                torch.where(self.motor_filter, tot / self.substeps, u),
            )
        else:
            x1, xm = u, u
        cat = lambda p, q: torch.cat((p, q), -1)[..., self.order]  # noqa: E731
        return cat(a1, x1), cat(mean, xm)

    def _muscles(
        self, act: torch.Tensor, ctrl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.zeros_like(act)
        for _ in range(self.substeps):
            total = total + act
            act = act + self.dt * self.act_dot(ctrl, act)
            if self.limited:
                act = torch.maximum(torch.minimum(act, self.act_hi), self.act_lo)
        return act, total / self.substeps
