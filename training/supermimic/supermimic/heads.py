# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""The task's MDP evaluated on states the method did not simulate.

The state tuple is ``S = (p, R, v, w, q, qd, act, touch)``: root position, canonical root rotation
matrix, world linear and angular velocity of the root link frame, the independent joint positions and
velocities, the actuator drive (the muscle activations; motors: see ``activation``) and the contact
bodies' touch values. A body with a fixed root keeps ``(p, R, v, w) = (0, I, 0, 0)``; its task state has
zero-width root arrays. :class:`TrackingHeads` turns it into the
task's :class:`~myosuite.terms.mimic_track.TrackState` with this method's differentiable kinematics and muscle
geometry, and :func:`analytic` maps each term of the task's env cfg (through its ``func.pure``) to the task's
pure function on that state, so the simulator and imagination score one MDP.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/heads.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from myosuite.envs.myo.tasks.mimic.tracking.motion import MotionLib
from myosuite.terms import mimic_track as mt

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
from .muscle_geometry import MuscleGeometry

BODY_STATE_DIM_EXTRA = 3 + 4 + 3 + 3  # p, quat, v, w (+ q, qd)
NUM_TOUCH = (
    4  # right heel, right toes, left heel, left toes (MyoFullBody's foot touch sensors)
)
"""Touch values of the full body (other bodies: their number of contact bodies)."""


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
    """The method's view of the task on states ``S``: kinematics, reference and the task's terms.

    Args:
        kin: Forward kinematics of the tracked bodies.
        motion: The task's motion library (torch arrays, the env's).
        geometry: Muscle geometry (length, velocity, force).
        lookahead: Goal frames, the current one included.
        stride: Control steps between goal frames.
    """

    def __init__(self, kin: SkeletonKinematics, motion: MotionLib, geometry: MuscleGeometry, lookahead: int = 5,
                 stride: int = 20, muscle_ids=None):  # fmt: skip
        self.fixed_root = kin.info.fixed_root
        assert (
            self.fixed_root or kin.tracked_on_root[0]
        ), "the first tracked body must be rigid with the root"
        self.kin, self.motion, self.geo = kin, motion, geometry
        # the muscles' columns of the drive (None: every actuator is a muscle)
        self.muscle_cols = muscle_ids
        self.lookahead, self.stride = lookahead, stride
        self.nq = kin.num_q
        self.body_state_dim = BODY_STATE_DIM_EXTRA + 2 * self.nq
        self.inv_dim = 1 + 6 + 3 + 3 + 2 * self.nq
        self.num_tracked = len(kin.tracked_idx)
        fr = motion.frames
        cols = torch.as_tensor(kin.info.indep_cols, device=fr["joint_pos"].device)
        # reference quantities per frame; the root rotation also in the canonical frame (x forward)
        self.root_p, self.root_R = fr["root_pos"], fr["root_rot"] @ kin.R_off
        self.ref_q, self.ref_qd = fr["joint_pos"], fr["joint_vel"]
        self.q = fr["joint_pos"][:, cols]
        self.body_p, self.body_v = fr["body_pos"], fr["body_lin_vel"]
        self.ref_root_qvel, self.ref_root_vel = fr["root_qvel"], fr["root_vel_local"]
        self.ref_rpos, self.ref_rangles, self.ref_rvel = (
            fr["site_rpos"],
            fr["site_rangles"],
            fr["site_rvel"],
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
            ctx.cache["muscles"] = (
                L,
                Ld,
                self.geo.force(L, Ld, self.activation(ctx.S[6])),
            )
        return ctx.cache["muscles"]

    def activation(self, drive: torch.Tensor) -> torch.Tensor:
        """The muscles' activations of a drive (the drive itself without motors)."""
        return drive if self.muscle_cols is None else drive[..., self.muscle_cols]

    def pelvis(self, ctx: StepCtx):
        """The free joint's rotation (its body's frame) and its qvel [v world, w body]."""
        if "pelvis" not in ctx.cache:
            Rp = mm3(ctx.S[1], self.kin.R_off.T)
            ctx.cache["pelvis"] = (
                Rp,
                torch.cat((ctx.S[2], _mv(Rp.transpose(-1, -2), ctx.S[3])), -1),
            )
        return ctx.cache["pelvis"]

    def precompute(self, S, muscles: bool = True) -> dict:
        """The pieces the terms of a state share, as one function for the simulator side to compile. Returns a
        StepCtx cache."""
        ctx = StepCtx(S, None)
        self.fk(ctx), self.sites(ctx), self.joints(ctx), self.pelvis(ctx)
        if muscles:
            self.muscles(ctx)
        return ctx.cache

    # ---------------- the task's state, features and reference ----------------
    def track(self, ctx: StepCtx) -> tuple[mt.TrackState, mt.TrackFeatures]:
        """The task's :class:`~myosuite.terms.mimic_track.TrackState` of ``ctx.S`` and its features."""
        if "track" not in ctx.cache:
            P, R, V, W = self.fk(ctx)
            q, qd = self.joints(ctx)
            Rp, qvel = self.pelvis(ctx)
            rpos, rangles, rvel = self.sites(ctx)
            if self.fixed_root:  # zero-width root, identity rotation
                empty = ctx.S[0][..., :0]
                root_pos = root_quat = qvel = local = empty
            else:
                root_pos, root_quat = ctx.S[0], mat_to_quat(Rp)
                local = torch.cat(
                    (_mv(Rp.transpose(-1, -2), qvel[..., :3]), qvel[..., 3:]), -1
                )
            s = mt.TrackState(
                root_pos=root_pos, root_quat=root_quat, root_qvel=qvel, joint_pos=q, joint_vel=qd,
                body_pos=P, body_rot=R, body_lin_vel=V, body_ang_vel=W, act=self.activation(ctx.S[6]),
                excitation=ctx.a, touch=ctx.S[7],
            )  # fmt: skip
            ctx.cache["track"] = (s, mt.TrackFeatures(Rp, local, rpos, rangles, rvel))
        return ctx.cache["track"]

    def ref(self, f: torch.Tensor) -> mt.TrackRef:
        """The task's reference at frames ``f``."""
        return self.motion.ref(f)

    def goal(self, ctx: StepCtx) -> torch.Tensor:
        """The task's goal (lookahead and phase) at ``ctx.f``."""
        m, f = self.motion, ctx.f
        c = m.clip_id[f]
        k = self.stride * torch.arange(1, self.lookahead, device=f.device)
        fut = torch.minimum(f[..., None] + k, m.clip_end[c][..., None])
        return mt.goal_observation(torch, m.ref(f), m.ref(fut), m.phase(f).float())

    def goal_dim(self) -> int:
        n = 3 * (self.num_tracked - 1)
        return n + (self.lookahead - 1) * ((0 if self.fixed_root else 9) + n) + 1

    # ---------------- critic: privileged reference errors ----------------
    def critic_extras(self, ctx: StepCtx) -> torch.Tensor:
        f = ctx.f
        if self.fixed_root:  # the root errors are constant
            q, qd = self.joints(ctx)
            return torch.cat((self.ref_q[f] - q, self.ref_qd[f] - qd), -1)
        p, R = ctx.S[0], ctx.S[1]
        e_root = _mv(yaw_mat(R).transpose(-1, -2), self.root_p[f] - p)
        Rr = self.root_R[f]
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
        return (0 if self.fixed_root else 3 + 2) + 2 * self.ref_q.shape[-1]


# ---------------- the task's terms on method states (signature: (heads, ctx, **params)) ----------------
_MUSCLE_OBS = {"muscle_length": 0, "muscle_velocity": 1, "muscle_force": 2}


def _observation(h: TrackingHeads, ctx: StepCtx, name: str) -> torch.Tensor:
    s, f = h.track(ctx)
    if name in _MUSCLE_OBS:
        s = dataclasses.replace(s, **{name: h.muscles(ctx)[_MUSCLE_OBS[name]]})
    return mt.observation(torch, s, f, name)


def _goal(h: TrackingHeads, ctx: StepCtx) -> torch.Tensor:
    return h.goal(ctx)


def _reward(h: TrackingHeads, ctx: StepCtx, name: str, kernel: float) -> torch.Tensor:
    s, f = h.track(ctx)
    return mt.reward(torch, s, f, h.ref(ctx.f), name, kernel)


def _failure(
    h: TrackingHeads,
    ctx: StepCtx,
    name: str,
    threshold: float,
    soft_width: float | None = None,
) -> torch.Tensor:
    """The task's failure rule; sigmoid-softened in imagination (``soft_width`` set)."""
    s, f = h.track(ctx)
    e = mt.deviation(torch, s, f, h.ref(ctx.f), name)
    return (
        e > threshold
        if soft_width is None
        else mt.soft_exceeds(torch, e, threshold, soft_width)
    )


_ANALYTIC = {
    mt.observation: _observation,
    mt.goal_observation: _goal,
    mt.reward: _reward,
    mt.exceeds: _failure,
}


def analytic(func: Any) -> Callable[..., torch.Tensor]:
    """The implementation ``fn(heads, ctx, **params)`` of a term of the task's env cfg on method states.

    Raises:
        ValueError: If the term has no pure function this method can evaluate.
    """
    fn = _ANALYTIC.get(getattr(func, "pure", None))
    if fn is None:
        raise ValueError(
            f"term {getattr(func, '__name__', func)!r} has no pure implementation"
        )
    return fn


MUSCLE_OBSERVATIONS = tuple(_MUSCLE_OBS)
"""Observation terms that need the muscle geometry."""
