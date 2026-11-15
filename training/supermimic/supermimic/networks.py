# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4).

Port of supermimic_vibed's ``networks.py`` (github.com/zita-ch/supermimic_vibed, MIT). The world model
predicts the body state; the muscle activations, the plant's actuator state, are integrated
analytically (``mdp.muscle``), and the world model's action input is their substep-mean over the step.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/rl/supermimic/networks.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from supermimic.heads import (
    invariant_features,
)
from supermimic.rotations import (
    mm3,
    mv3,
    so3_exp,
    so3_log,
    yaw_mat,
)


def mlp(
    inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``.

    ``no_sync`` (InstantTrack; the runner sets it from its cfg): the freeze test runs on the device (torch.where)
    instead of a host-side ``if``, so an update never synchronizes."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps
        self.no_sync = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if not self.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        if self.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
        else:
            self.mean.copy_(mean)
            self.var = var
            self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def explore(a: torch.Tensor, sigma: float, dist: str = "gaussian") -> torch.Tensor:
    """Exploration noise on excitations ``a`` in [0, 1] (the runner's ``action_noise`` / ``action_noise_dist``).

    gaussian: ``a + N(0, sigma^2)`` clipped to [0, 1]. Clipping is not mean-preserving: at a policy's typical
    near-silent excitations (a ~ 0.02) about 40 % of the draws fall below 0, and the executed excitation averages
    about 2.7x the intended one.
    beta: ``Beta(k a, k (1 - a))`` with ``k = 1 / (4 sigma^2) - 1``: mean exactly ``a``, variance ``a (1 - a) / (k + 1)``,
    so the standard deviation is ``sigma`` at ``a = 0.5`` and shrinks towards the bounds, which it never leaves. Drawn
    as ``X / (X + Y)`` from two Gamma variates; where both underflow (concentrations of ~1e-3 at a near 0 or 1) the
    excitation stays ``a``, as it does at 0 and 1 exactly.
    reflect: ``a + N(0, sigma^2)`` folded back into [0, 1] at both bounds (a triangle wave of period 2, so any number of
    folds): the Gaussian's step sizes, never outside the box and no mass piled on 0 or 1. Not mean-preserving either:
    at a = 0.02 the executed excitation averages 0.081 (clipping: 0.051), and 24 % of the draws exceed a + 0.1
    (clipping: 16 %, Beta: 1.4 %)."""
    if sigma <= 0:
        return a.clamp(0.0, 1.0)
    if dist == "gaussian":
        return (a + sigma * torch.randn_like(a)).clamp(0.0, 1.0)
    if dist == "reflect":
        x = torch.remainder(
            a.float() + sigma * torch.randn_like(a, dtype=torch.float32), 2.0
        )
        return (1.0 - (1.0 - x).abs()).clamp(0.0, 1.0)
    assert dist == "beta", dist
    k = 1.0 / (4.0 * sigma * sigma) - 1.0
    assert k > 0, "beta noise needs action_noise < 0.5"
    a = a.float().clamp(0.0, 1.0)
    x = torch._standard_gamma((k * a).clamp(min=1e-6))
    y = torch._standard_gamma((k * (1.0 - a)).clamp(min=1e-6))
    s = x + y
    return torch.where((s > 0) & (a > 0) & (a < 1), x / s.clamp(min=1e-30), a).clamp(
        0.0, 1.0
    )


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic body dynamics S_{t+1} = f(S_t, act_t, act_{t+1}) predicting frame-local state increments,
    with an auxiliary head for the foot touch of S_{t+1}. The activations before and after the step are inputs;
    contacts are the only observation the analytic heads cannot compute.

    ``structured`` (BlinkMimic's semi-implicit world model): the network predicts the velocity increments, and the
    positions follow the old and new velocities as the simulator's semi-implicit Euler substeps make them, plus
    small corrections the network predicts on top of that drift. ``dt`` is the control step and ``substeps`` its
    physics substeps."""

    def __init__(self, nq: int, num_muscles: int, num_touch: int, hidden=(2048, 2048, 2048), structured: bool = False,
                 dt: float = 0.01, substeps: int = 5):  # fmt: skip
        super().__init__()
        self.nq, self.num_touch = nq, num_touch
        self.structured, self.dt = structured, dt
        # weight of the new velocity in the position drift: n semi-implicit substeps with a linearly changing velocity
        # give sum_{k=1..n} (k / n) / n = (n + 1) / (2 n): 0.6 for 5 substeps (BlinkMimic: 0.625 for its 4)
        self.drift_w = (substeps + 1) / (2 * substeps)
        self.in_dim = 1 + 6 + 3 + 3 + 2 * nq + 2 * num_muscles
        self.out_dim = 3 + 3 + 3 + 3 + 2 * nq
        self.net = mlp(
            self.in_dim,
            list(hidden),
            self.out_dim + num_touch,
            act=nn.SiLU,
            layer_norm=True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)  # of the network's targets
        # of the plain frame-local increments, the loss normalization (equal to out_norm unless structured)
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.touch_norm = RunningNorm(num_touch)

    def inputs(self, S, act, act1):
        return torch.cat((invariant_features(S), act, act1), -1)

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1: root position in the yaw frame, rotation vector, linear and angular
        velocity in the root frame, joint positions and velocities."""
        p, R, v, w, q, qd = S[:6]
        p1, R1, v1, w1, q1, qd1 = S1[:6]
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv3(H.transpose(-1, -2), p1 - p),
                so3_log(mm3(Rt, R1)),
                mv3(Rt, v1 - v),
                mv3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the increments:
        the yaw frame for the root position, the root frame for the rotation vector."""
        _, R, v, w, _, qd = S[:6]
        H = yaw_mat(R)
        a, b = self.dt * (1 - self.drift_w), self.dt * self.drift_w
        return (
            mv3(H.transpose(-1, -2), a * v + b * v1),
            mv3(R.transpose(-1, -2), a * w + b * w1),
            a * qd + b * qd1,
        )

    def targets(self, S, S1):
        """The regression targets: the frame-local increments, or with ``structured`` the velocity increments plus
        the position corrections on top of the semi-implicit drift."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, self.nq, self.nq), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def integrate(self, S, d):
        """Next body state from the denormalized network output ``d`` (the layout of ``targets``)."""
        p, R, v, w, q, qd = S[:6]
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.nq, self.nq), -1)
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        qd1 = qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        return (p + mv3(H, dp), mm3(R, so3_exp(dth)), v1, w1, q + dq, qd1)

    def step(self, S, act, act1):
        """-> (body state of S_{t+1}, its normalized touch prediction)."""
        y = run_mlp(self.net, self.in_norm(self.inputs(S, act, act1)))
        d = y[..., : self.out_dim] * self.out_norm.std() + self.out_norm.mean
        return self.integrate(S, d), y[..., self.out_dim :]

    def touch(self, touch_n: torch.Tensor) -> torch.Tensor:
        """Normalized touch prediction -> touch (non-negative)."""
        return torch.relu(touch_n * self.touch_norm.std() + self.touch_norm.mean)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local body-state error components (the layout of ``increments``)."""
        return WorldModel.increments(S, S_hat)


class ActionMap(nn.Module):
    """The policy's outputs -> one excitation per muscle (in actuator order).

    ``identity`` (one output per muscle) is the only kind implemented. DynSyn muscle groups (He et al., ICML 2024: one
    output per group, each muscle executing its group's excitation) are planned as another kind: the map has no
    parameters, and a checkpoint records it as :meth:`spec` next to the policy weights, so groups need no new
    checkpoint layout (only a policy head with one output per group).
    """

    KINDS = ("identity",)

    def __init__(self, num_muscles: int, kind: str = "identity"):
        super().__init__()
        if kind not in self.KINDS:
            raise NotImplementedError(
                f"action map {kind!r}: only {self.KINDS} is implemented"
            )
        self.kind, self.num_muscles = kind, num_muscles
        self.num_outputs = num_muscles

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u

    def spec(self) -> dict:
        """The map as a checkpoint entry."""
        return {"kind": self.kind, "num_muscles": self.num_muscles}

    @classmethod
    def from_spec(cls, spec: dict | None, num_muscles: int) -> ActionMap:
        """The map of a checkpoint entry (``None``, a checkpoint from before action maps: identity).

        Raises:
            ValueError: If the entry is for another number of muscles.
        """
        spec = spec or {"kind": "identity", "num_muscles": num_muscles}
        if int(spec["num_muscles"]) != num_muscles:
            raise ValueError(
                f"action map for {spec['num_muscles']} muscles, the task has {num_muscles}"
            )
        return cls(num_muscles, str(spec["kind"]))


class Policy(nn.Module):
    """e = map(sigmoid(dec(enc(g), o_t))); deterministic, latent z in R^32 bounded by tanh.

    The decoder outputs one logit per action-map output (per muscle with the identity map); the env action is the
    excitation in [0, 1].
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        num_actions: int,
        enc_hidden=(2048, 1024, 512, 256),
        dec_hidden=(2048, 1024, 512, 256),
        z_dim: int = 32,
        init_excitation: float = 0.1,
        action_map: ActionMap | None = None,
    ):
        super().__init__()
        self.action_map = action_map or ActionMap(num_actions)
        if self.action_map.num_muscles != num_actions:
            raise ValueError("the action map must produce num_actions excitations")
        self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim, list(dec_hidden), self.action_map.num_outputs)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.fill_(math.log(init_excitation / (1 - init_excitation)))
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        """The latent of the reference-only goal (B, goal_dim)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))

    def decode(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """The excitations from a latent and the decoder observation (B, obs_dim)."""
        return self.action_map(
            torch.sigmoid(run_mlp(self.dec, torch.cat((z, self.obs_norm(obs)), -1)))
        )

    def logits(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) the decoder observation; goal: (B, goal_dim) the encoder's reference-only goal."""
        return run_mlp(self.dec, torch.cat((self.encode(goal), self.obs_norm(obs)), -1))

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.action_map(torch.sigmoid(self.logits(obs, goal)))


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
