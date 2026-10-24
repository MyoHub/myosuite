# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""World-model state layout for free-root muscle models.

State ``s = [qpos (nq) | qvel (nv) | act (na)]`` in MuJoCo layout, root first
(free joint: ``qvel[3:6]`` is the body-frame angular velocity).

Network input features (heading invariant): root height, rot6d of the root
orientation in the heading frame, root linear velocity in the heading frame,
body angular velocity, joint positions and velocities, activation now and after
the step. Network output: velocity increments ``(dv_h, dw_b, dqd)`` and position
corrections ``(dp_h, dth_b, dq)`` on top of explicit integration, so a zero
output extrapolates at constant velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from myosuite.physics.quat_math_torch import mul_quat, neg_quat, quat2mat, rot_vec_quat


@dataclass(frozen=True)
class StateScales:
    """Nominal scales of the state errors, inputs and network outputs."""

    # Error scales: root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s.
    err: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales: dv_h, dw_b, dqd, dp_h, dth_b, dq.
    out: tuple[float, ...] = (0.2, 0.5, 2.0, 0.005, 0.01, 0.01)
    # Input divisors: root z, rot6d, lin vel, ang vel, joint pos, joint vel.
    inp: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of rotation vector *v* (smooth at zero)."""
    angle = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) / angle * v], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector of unit quaternion *q*, short way round."""
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    s = torch.sqrt((q[..., 1:] ** 2).sum(-1, keepdim=True) + 1e-12)
    return 2 * torch.atan2(s, q[..., :1]) / s * q[..., 1:]


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion of a rotation by *yaw* about the world z axis."""
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], dim=-1)


class StateSpec:
    """Split, featurise, integrate and compare world-model states.

    Args:
        nq: Number of position coordinates (7 + joints).
        nv: Number of velocity coordinates (6 + joints).
        na: Number of activation states.
        dt: Control timestep in seconds.
        forward_axis: Root-frame axis that is horizontal when standing; its
            world projection defines the heading.
        scales: Nominal error / input / output scales.
    """

    def __init__(
        self,
        nq: int,
        nv: int,
        na: int,
        dt: float,
        forward_axis: torch.Tensor,
        scales: StateScales = StateScales(),
    ) -> None:
        self.nq, self.nv, self.na, self.nj = nq, nv, na, nv - 6
        self.dt = dt
        self.dim = nq + nv + na
        self.feat_dim = 1 + 6 + 3 + 3 + 2 * self.nj + 2 * na
        self.delta_dim = 2 * (6 + self.nj)
        self.forward_axis = forward_axis.float()
        self.scales = scales
        rep = lambda s, n: torch.cat([torch.full((k,), float(v)) for v, k in zip(s, n)])  # noqa: E731
        nj = self.nj
        self.err_scale = rep(scales.err, (3, 3, 3, 3, nj, nj))
        self.out_scale = rep(scales.out, (3, 3, nj, 3, 3, nj))
        self.in_scale = rep(scales.inp, (1, 6, 3, 3, nj, nj))

    def to(self, device: torch.device | str) -> StateSpec:
        """Move the constant tensors to *device*."""
        for name in ("forward_axis", "err_scale", "out_scale", "in_scale"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def split(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``qpos``, ``qvel``, ``act`` views of *s*."""
        return (
            s[..., : self.nq],
            s[..., self.nq : self.nq + self.nv],
            s[..., self.nq + self.nv :],
        )

    def heading(self, quat: torch.Tensor) -> torch.Tensor:
        """Yaw quaternion of the root's forward axis."""
        fwd = rot_vec_quat(self.forward_axis.expand(quat.shape[:-1] + (3,)), quat)
        return yaw_quat(torch.atan2(fwd[..., 1], fwd[..., 0]))

    def features(self, s: torch.Tensor, act_next: torch.Tensor) -> torch.Tensor:
        """Scaled heading-invariant network input for state *s*."""
        qpos, qvel, act = self.split(s)
        quat = qpos[..., 3:7] / qpos[..., 3:7].norm(dim=-1, keepdim=True)
        hq = self.heading(quat)
        local = mul_quat(neg_quat(hq), quat)
        rot6d = quat2mat(local)[..., :, :2].reshape(local.shape[:-1] + (6,))
        v_h = rot_vec_quat(qvel[..., :3], neg_quat(hq))
        mech = torch.cat(
            [qpos[..., 2:3], rot6d, v_h, qvel[..., 3:6], qpos[..., 7:], qvel[..., 6:]],
            -1,
        )
        return torch.cat([mech / self.in_scale, act, act_next], dim=-1)

    def integrate(
        self, s: torch.Tensor, delta: torch.Tensor, act_next: torch.Tensor
    ) -> torch.Tensor:
        """Next state from the unscaled network output *delta* and the next activation."""
        nj = self.nj
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6 : 6 + nj]
        dp_h, dth_b, dq = (
            d[..., 6 + nj : 9 + nj],
            d[..., 9 + nj : 12 + nj],
            d[..., 12 + nj :],
        )
        qpos, qvel, _ = self.split(s)
        quat = qpos[..., 3:7] / qpos[..., 3:7].norm(dim=-1, keepdim=True)
        hq = self.heading(quat)
        v = qvel[..., :3] + rot_vec_quat(dv_h, hq)
        w = qvel[..., 3:6] + dw_b
        qd = qvel[..., 6:] + dqd
        p = qpos[..., :3] + self.dt * v + rot_vec_quat(dp_h, hq)
        q = mul_quat(quat, quat_from_rotvec(self.dt * w + dth_b))
        q = q / q.norm(dim=-1, keepdim=True)
        qj = qpos[..., 7:] + self.dt * qd + dq
        return torch.cat([p, q, qj, v, w, qd, act_next], dim=-1)

    def error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised mechanical error ``[dp, rotvec, dv, dw, dq, dqd] / scale``."""
        qp1, qv1, _ = self.split(s_pred)
        qp2, qv2, _ = self.split(s_true)
        rot = rotvec_from_quat(mul_quat(neg_quat(qp2[..., 3:7]), qp1[..., 3:7]))
        e = torch.cat(
            [
                qp1[..., :3] - qp2[..., :3],
                rot,
                qv1[..., :3] - qv2[..., :3],
                qv1[..., 3:6] - qv2[..., 3:6],
                qp1[..., 7:] - qp2[..., 7:],
                qv1[..., 6:] - qv2[..., 6:],
            ],
            dim=-1,
        )
        return e / self.err_scale


def forward_axis_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Root-frame unit axis that is most horizontal under the standing orientation *quat*."""
    axes = torch.eye(3, dtype=quat.dtype, device=quat.device)
    world = rot_vec_quat(axes, quat.expand(3, 4))
    return axes[world[:, 2].abs().argmin()]
