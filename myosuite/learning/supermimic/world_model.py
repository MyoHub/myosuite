# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Deterministic world model ``s' = f(s, ctrl)`` for muscle-driven free-root bodies."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from torch import nn

from myosuite.learning.supermimic.kinematics import JointCoupling
from myosuite.learning.supermimic.muscle import MuscleActivation
from myosuite.learning.supermimic.state import StateSpec


@dataclass(frozen=True)
class WorldModelCfg:
    """World-model network and training settings."""

    hidden: tuple[int, ...] = (2048, 2048, 2048)
    window: int = 8  # autoregressive fit length k
    lr: float = 2e-4
    batch: int = 2048
    grad_clip: float = 1.0
    project_couplings: bool = True


class WorldModel(nn.Module):
    """Analytic activation + learned mechanics + analytic joint couplings.

    Args:
        model: Compiled MuJoCo model of the body.
        spec: State layout.
        n_substeps: Physics steps per control step.
        sim_dt: Physics timestep.
        cfg: Network settings.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        spec: StateSpec,
        n_substeps: int,
        sim_dt: float,
        cfg: WorldModelCfg = WorldModelCfg(),
    ) -> None:
        super().__init__()
        self.spec = spec
        self.cfg = cfg
        self.activation = MuscleActivation(model, n_substeps, sim_dt)
        self.coupling = JointCoupling(model)
        dims = [spec.feat_dim, *cfg.hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        head = nn.Linear(dims[-1], spec.delta_dim)
        # Zero output = constant-velocity extrapolation at initialisation.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.net = nn.Sequential(*layers, head)

    def step(self, s: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """Next state after one control step of *ctrl*."""
        sp = self.spec
        act_next = self.activation(s[..., sp.nq + sp.nv :], ctrl)
        s_next = sp.integrate(s, self.net(sp.features(s, act_next)), act_next)
        if not self.cfg.project_couplings:
            return s_next
        qpos, qvel, act = sp.split(s_next)
        qpos, qvel = self.coupling.project(qpos, qvel)
        return torch.cat([qpos, qvel, act], dim=-1)

    def rollout(self, s0: torch.Tensor, ctrls: torch.Tensor) -> torch.Tensor:
        """Open-loop states ``(k, N, D)`` under recorded controls ``(k, N, nu)``."""
        out, s = [], s0
        for u in ctrls:
            s = self.step(s, u)
            out.append(s)
        return torch.stack(out, 0)

    def window_loss(
        self, s0: torch.Tensor, ctrls: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean squared normalised error over a k-step window, and its per-step values."""
        pred = self.rollout(s0, ctrls)
        e = self.spec.error(pred, targets)
        per_step = (e * e).mean(-1).mean(-1)
        return per_step.mean(), per_step.detach()


class WindowSampler:
    """Uniform k-step windows of a recorded rollout that do not cross an episode end.

    Args:
        state: ``(T + 1, N, D)`` states.
        ctrl: ``(T, N, nu)`` controls.
        done: ``(T, N)`` episode ended after step ``t``.
        window: Window length k.
    """

    def __init__(
        self, state: torch.Tensor, ctrl: torch.Tensor, done: torch.Tensor, window: int
    ) -> None:
        self.state, self.ctrl, self.k = state, ctrl, window
        T = ctrl.shape[0]
        # valid[t, n]: no episode end within t .. t + k - 1.
        d = done.float()
        csum = torch.cat([torch.zeros_like(d[:1]), d.cumsum(0)], 0)
        ends = csum[window : T + 1] - csum[: T + 1 - window]
        self.starts = torch.nonzero(ends == 0)  # (M, 2) [t, n]

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def sample(
        self, batch: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``s0 (B, D)``, controls ``(k, B, nu)`` and target states ``(k, B, D)``."""
        idx = torch.randint(
            len(self), (batch,), generator=generator, device=self.starts.device
        )
        t, n = self.starts[idx, 0], self.starts[idx, 1]
        steps = torch.arange(self.k, device=t.device)[:, None]
        return (
            self.state[t, n],
            self.ctrl[t + steps, n].float(),
            self.state[t + 1 + steps, n],
        )
