# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Analytic MDP heads shared by the simulator and the world model.

The MDP is MuscleMimic's: its observation (root height and orientation, every joint position and velocity,
muscle length / velocity / force / excitation / activation, foot touch, and the root-relative positions,
rotations and velocities of the mimic sites), its goal (a concise lookahead of 5 reference frames 0.2 s apart
and the motion phase), its ``MimicReward`` and its deviation terminations. Tracked bodies stand in for
MuscleMimic's 17 mimic sites (pelvis, torso, head, shoulders, elbows, hands, hips, knees, ankles, toes): on
MyoFullBody they are the sites' own bodies (each site at its body's origin), on MS-Human-700 17 bodies whose
origins sit at the joints the sites mark. Quantities are computed the way MuscleMimic computes them, quirks
included (noted where they occur).

The state tuple is ``S = (p, R, v, w, q, qd, act, touch)``: root position, canonical root rotation
matrix, world linear and angular velocity of the root link frame, the independent joint positions and
velocities, the muscle activations and the four foot touch values. Every observation, reward and
termination term of the mjlab task is a function of a :class:`StepCtx` built on such a state, so the same
code labels simulator transitions (through the managers) and imagined ones (in the SuperMimic runner).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/heads.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .rotations import (
    mat_to_6d,
    mat_to_quat,
    mm3,
    mv3,
    quat_to_mat,
    so3_log,
    yaw_mat,
)

from .kinematics import SkeletonKinematics
from .motion import MotionLib
from .muscle_geometry import MuscleGeometry

BODY_STATE_DIM_EXTRA = 3 + 4 + 3 + 3  # p, quat, v, w (+ q, qd)
NUM_TOUCH = (
    4  # right heel, right toes, left heel, left toes (MyoFullBody's foot touch sensors)
)


@dataclass
class StepCtx:
    """What an analytic term may look at.

    ``S`` is the state, ``f`` the reference frame it is compared with, ``a`` the action that led to
    ``S`` (the excitation it observes) and ``a_prev`` the action before it.
    """

    S: tuple
    f: torch.Tensor
    a: torch.Tensor | None = None
    a_prev: torch.Tensor | None = None
    cache: dict = field(default_factory=dict)


def flatten_body(S) -> torch.Tensor:
    p, R, v, w, q, qd = S[:6]
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_body(x: torch.Tensor, nq: int):
    return (
        x[..., 0:3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13 : 13 + nq],
        x[..., 13 + nq : 13 + 2 * nq],
    )


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant body-state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S[:6]
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(mm3(H.transpose(-1, -2), R)),
            mv3(Rt, v),
            mv3(Rt, w),
            q,
            qd,
        ),
        -1,
    )


def rot_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations (via the relative quaternion: exact near 0, finite gradient)."""
    q = mat_to_quat(mm3(Ra.transpose(-1, -2), Rb))
    return 2 * torch.atan2(
        (q[..., 1:] ** 2).sum(-1).clamp(min=1e-30).sqrt(), q[..., 0].abs()
    )


def rel_sites(P, R, V, W):
    """MuscleMimic's ``calculate_relative_site_quantities`` with the first body (the pelvis) as the main site:
    world-frame position offsets, rotation vectors of R_main^T R_site, and its 6D relative velocity
    [ang, lin]. As in MuscleMimic, the linear part is R_main (v_main - v_site) and the angular part
    (R_main^T R_site)^T w_site - w_main."""
    P0, R0, V0, W0 = P[..., :1, :], R[..., :1, :, :], V[..., :1, :], W[..., :1, :]
    rrot = mm3(R0.transpose(-1, -2), R[..., 1:, :, :])
    rpos = P[..., 1:, :] - P0
    lin = mv3(R0, V0 - V[..., 1:, :])
    ang = mv3(rrot.transpose(-1, -2), W[..., 1:, :]) - W0
    return rpos, so3_log(rrot), torch.cat((ang, lin), -1)


_mv = mv3


class TrackingHeads:
    def __init__(self, kin: SkeletonKinematics, motion: MotionLib, geometry: MuscleGeometry, lookahead: int = 5,
                 stride: int = 20):  # fmt: skip
        assert kin.tracked_on_root[
            0
        ], "the first tracked body must be rigid with the root (MuscleMimic's main site)"
        self.kin, self.motion, self.geo, self.lookahead, self.stride = (
            kin,
            motion,
            geometry,
            lookahead,
            stride,
        )
        self.nq = kin.num_q
        self.body_state_dim = BODY_STATE_DIM_EXTRA + 2 * self.nq
        self.inv_dim = 1 + 6 + 3 + 3 + 2 * self.nq
        self.num_tracked = len(kin.tracked_idx)
        # per reference frame, in MuscleMimic's conventions: every joint, the free joint's qvel [v world, w body],
        # the root velocity of its reward [R^T v, w body] and the relative site quantities
        m = motion
        self.ref_q, self.ref_qd = kin.full_joint_state(m.q, m.qd)
        Rp = m.root_R @ kin.R_off.T
        w_body = _mv(Rp.transpose(-1, -2), m.root_w)
        self.ref_root_qvel = torch.cat((m.root_v, w_body), -1)
        self.ref_root_vel = torch.cat((_mv(Rp.transpose(-1, -2), m.root_v), w_body), -1)
        self.ref_rpos, self.ref_rangles, self.ref_rvel = rel_sites(
            m.body_p, m.body_R, m.body_v, m.body_w
        )

    # ---------------- shared pieces (cached per ctx) ----------------
    def fk(self, ctx: StepCtx):
        if "fk" not in ctx.cache:
            p, R, v, w, q, qd = ctx.S[:6]
            ctx.cache["fk"] = self.kin.forward(p, R, v, w, q, qd)
        return ctx.cache["fk"]

    def sites(self, ctx: StepCtx):
        if "sites" not in ctx.cache:
            ctx.cache["sites"] = rel_sites(*self.fk(ctx))
        return ctx.cache["sites"]

    def joints(self, ctx: StepCtx):
        if "joints" not in ctx.cache:
            ctx.cache["joints"] = self.kin.full_joint_state(ctx.S[4], ctx.S[5])
        return ctx.cache["joints"]

    def muscles(self, ctx: StepCtx):
        """Muscle lengths, velocities and forces of the state."""
        if "muscles" not in ctx.cache:
            L, Ld = self.geo.length_velocity(ctx.S[4], ctx.S[5])
            ctx.cache["muscles"] = (L, Ld, self.geo.force(L, Ld, ctx.S[6]))
        return ctx.cache["muscles"]

    def pelvis(self, ctx: StepCtx):
        """The free joint's rotation (its body's frame: MS-Human-700's pelvis, MyoFullBody's ``Full Body``, to
        which the pelvis is welded) and its qvel [v world, w body]."""
        if "pelvis" not in ctx.cache:
            Rp = mm3(ctx.S[1], self.kin.R_off.T)
            ctx.cache["pelvis"] = (
                Rp,
                torch.cat((ctx.S[2], _mv(Rp.transpose(-1, -2), ctx.S[3])), -1),
            )
        return ctx.cache["pelvis"]

    def precompute(self, S, muscles: bool = True) -> dict:
        """The pieces the terms of a state share -- kinematics, sites, joints, the free joint and, if asked, muscle
        geometry -- as one function for the simulator side to compile (imagination builds them inside its own
        compiled graphs). Returns a StepCtx cache."""
        ctx = StepCtx(S, None)
        self.fk(ctx), self.sites(ctx), self.joints(ctx), self.pelvis(ctx)
        if muscles:
            self.muscles(ctx)
        return ctx.cache

    # ---------------- observation pieces (MyoFullBody's observation, then the goal's state part) ----------------
    def root_pos(self, ctx: StepCtx) -> torch.Tensor:
        """Free joint position without x, y: height and quaternion (wxyz)."""
        return torch.cat((ctx.S[0][..., 2:3], mat_to_quat(self.pelvis(ctx)[0])), -1)

    def joint_pos(self, ctx: StepCtx) -> torch.Tensor:
        return self.joints(ctx)[0]

    def root_vel(self, ctx: StepCtx) -> torch.Tensor:
        return self.pelvis(ctx)[1]

    def joint_vel(self, ctx: StepCtx) -> torch.Tensor:
        return self.joints(ctx)[1]

    def muscle_length(self, ctx: StepCtx) -> torch.Tensor:
        return self.muscles(ctx)[0]

    def muscle_velocity(self, ctx: StepCtx) -> torch.Tensor:
        return self.muscles(ctx)[1]

    def muscle_force(self, ctx: StepCtx) -> torch.Tensor:
        return self.muscles(ctx)[2]

    def muscle_excitation(self, ctx: StepCtx) -> torch.Tensor:
        return ctx.a

    def muscle_activation(self, ctx: StepCtx) -> torch.Tensor:
        return ctx.S[6]

    def touch(self, ctx: StepCtx) -> torch.Tensor:
        return ctx.S[7]

    def site_rpos(self, ctx: StepCtx) -> torch.Tensor:
        return self.sites(ctx)[0].flatten(-2)

    def site_rangles(self, ctx: StepCtx) -> torch.Tensor:
        return self.sites(ctx)[1].flatten(-2)

    def site_rvel(self, ctx: StepCtx) -> torch.Tensor:
        return self.sites(ctx)[2].flatten(-2)

    # ---------------- goal: the reference-only part of GoalTrajMimic (concise lookahead + motion phase) ----------------
    def goal(self, ctx: StepCtx) -> torch.Tensor:
        m, f = self.motion, ctx.f
        c = m.clip_id[f]
        parts = [self.ref_rpos[f].flatten(-2)]
        for k in range(1, self.lookahead):
            g = torch.minimum(f + k * self.stride, m.clip_end[c])
            parts += [
                m.root_p[g] - m.root_p[f],
                self.ref_root_qvel[g] - self.ref_root_qvel[f],
                self.ref_rpos[g].flatten(-2),
            ]
        parts.append(
            ((f - m.clip_start[c]).float() / m.clip_len[c].float()).unsqueeze(-1)
        )
        return torch.cat(parts, -1)

    def goal_dim(self) -> int:
        n = 3 * (self.num_tracked - 1)
        return n + (self.lookahead - 1) * (3 + 6 + n) + 1

    # ---------------- critic: privileged reference errors ----------------
    def critic_extras(self, ctx: StepCtx) -> torch.Tensor:
        m, f = self.motion, ctx.f
        p, R = ctx.S[0], ctx.S[1]
        e_root = _mv(yaw_mat(R).transpose(-1, -2), m.root_p[f] - p)
        Rr = m.root_R[f]
        dyaw = torch.atan2(Rr[..., 1, 0], Rr[..., 0, 0]) - torch.atan2(
            R[..., 1, 0], R[..., 0, 0]
        )
        q, qd = self.joints(ctx)
        return torch.cat(
            (
                e_root,
                torch.sin(dyaw).unsqueeze(-1),
                torch.cos(dyaw).unsqueeze(-1),
                self.ref_q[f] - q,
                self.ref_qd[f] - qd,
            ),
            -1,
        )

    def critic_extras_dim(self) -> int:
        return 3 + 2 + 2 * self.ref_q.shape[-1]


# ---------------- analytic term implementations (signature: (heads, ctx, **params)) ----------------
# MimicReward (use_mean_exp_reward=False): exp(-w_exp * mean squared error) per component


def mimic_qpos(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    """Root position and every joint (squared errors, averaged) plus the root quaternion's angle."""
    f = ctx.f
    e = torch.cat(
        ((h.motion.root_p[f] - ctx.S[0]) ** 2, (h.ref_q[f] - h.joints(ctx)[0]) ** 2), -1
    ).mean(-1)
    return torch.exp(-w_exp * (e + rot_angle(h.motion.root_R[f], ctx.S[1])))


def mimic_qvel(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    f = ctx.f
    e = torch.cat(
        (
            (h.ref_root_qvel[f] - h.pelvis(ctx)[1]) ** 2,
            (h.ref_qd[f] - h.joints(ctx)[1]) ** 2,
        ),
        -1,
    ).mean(-1)
    return torch.exp(-w_exp * e)


def mimic_root_pos(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    return torch.exp(-w_exp * ((h.motion.root_p[ctx.f] - ctx.S[0]) ** 2).mean(-1))


def mimic_root_vel(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    """[R^T v, w body] (MuscleMimic calls the body-frame angular velocity of qvel global)."""
    Rp, qvel = h.pelvis(ctx)
    vel = torch.cat((_mv(Rp.transpose(-1, -2), qvel[..., :3]), qvel[..., 3:]), -1)
    return torch.exp(-w_exp * ((h.ref_root_vel[ctx.f] - vel) ** 2).mean(-1))


def mimic_rpos(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    return torch.exp(
        -w_exp * ((h.ref_rpos[ctx.f] - h.sites(ctx)[0]) ** 2).mean((-1, -2))
    )


def mimic_rquat(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    return torch.exp(
        -w_exp * ((h.ref_rangles[ctx.f] - h.sites(ctx)[1]) ** 2).mean((-1, -2))
    )


def mimic_rvel_rot(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    return torch.exp(
        -w_exp
        * ((h.ref_rvel[ctx.f][..., :3] - h.sites(ctx)[2][..., :3]) ** 2).mean((-1, -2))
    )


def mimic_rvel_lin(h: TrackingHeads, ctx: StepCtx, w_exp: float) -> torch.Tensor:
    return torch.exp(
        -w_exp
        * ((h.ref_rvel[ctx.f][..., 3:] - h.sites(ctx)[2][..., 3:]) ** 2).mean((-1, -2))
    )


# MuscleMimic's MeanRelativeSiteDeviationWithRootTerminalStateHandler, sigmoid-softened in imagination


def _exceeds(
    e: torch.Tensor, threshold: float, soft_width: float | None
) -> torch.Tensor:
    """Hard termination in the simulator; sigmoid-softened in imagination (``soft_width`` set)."""
    return (
        e > threshold
        if soft_width is None
        else torch.sigmoid((e - threshold) / soft_width)
    )


def _norm(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(-1).add(1e-12).sqrt()  # finite gradient at 0


def bad_body_pos_dev(
    h: TrackingHeads, ctx: StepCtx, threshold: float, soft_width: float | None = None
) -> torch.Tensor:
    """Mean distance between robot and reference site positions relative to the root (world frame)."""
    return _exceeds(
        _norm(h.ref_rpos[ctx.f] - h.sites(ctx)[0]).mean(-1), threshold, soft_width
    )


def bad_root_pos_dev(
    h: TrackingHeads, ctx: StepCtx, threshold: float, soft_width: float | None = None
) -> torch.Tensor:
    """Distance between robot and reference root positions (horizontal drift included)."""
    return _exceeds(_norm(h.motion.root_p[ctx.f] - ctx.S[0]), threshold, soft_width)


def bad_root_ori_dev(
    h: TrackingHeads, ctx: StepCtx, threshold: float, soft_width: float | None = None
) -> torch.Tensor:
    """Geodesic angle between robot and reference root orientations (heading included)."""
    return _exceeds(rot_angle(h.motion.root_R[ctx.f], ctx.S[1]), threshold, soft_width)


def analytic(fn):
    """Attach the analytic implementation ``fn(heads, ctx, **params)`` to a manager term function, so the
    SuperMimic runner can evaluate the term on imagined states with the env cfg's own params."""

    def deco(term):
        term.analytic = fn
        return term

    return deco
