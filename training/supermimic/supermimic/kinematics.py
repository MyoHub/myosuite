# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Differentiable forward kinematics of a musculoskeletal model in torch, built from the compiled MJCF.

Unlike the G1 (one hinge per link, at the link origin), the human skeletons have bodies with several
joints in sequence (MS-Human-700's knee carries 3 slides + 3 hinges), joints away from the body origin,
welded bodies (vertebrae, sternum) and joints that are polynomial functions of an independent joint
(``<equality>`` couplers, 42 in MS-Human-700; evaluated by
:func:`myosuite.physics.joint_coupling.projected_joint_couplings`). The FK here replicates ``mj_kinematics`` exactly for all
of that, takes only the independent coordinates, and propagates link-origin velocities alongside, so the
same function labels reference motions, simulator transitions and world-model predictions.

Only bodies on the path to a tracked body are evaluated; joint-less bodies on that path are folded
into their children's fixed offsets. A body with a fixed root (no free joint) is evaluated from the
world body, whose pose is the root state ``(p, R) = (0, I)`` with zero velocity.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/kinematics.py``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from myosuite.physics.joint_coupling import projected_joint_couplings

from myosuite.envs.myo.tasks.mimic.tracking.models import ModelInfo, root_offset
from .rotations import (
    axis_angle_to_mat,
    mm3,
    mv3,
    quat_to_mat,
)


class SkeletonKinematics:
    def __init__(
        self,
        tracked: list[str] | tuple[str, ...],
        device: str,
        m: mujoco.MjModel,
        dtype: torch.dtype = torch.float32,
    ):
        info = ModelInfo(m)
        self.info, self.device = info, device
        self.num_q = len(info.indep_ids)
        self.R_off = torch.tensor(root_offset(m), dtype=dtype, device=device)
        root = info.root_id  # the world for a fixed root
        tracked_ids = [m.body(n).id for n in tracked]
        # tracked bodies welded to the root (the root body itself, or a body without joints under it)
        self.tracked_on_root = [bool(m.body_weldid[b] == root) for b in tracked_ids]
        needed = {0} if info.fixed_root else set()
        for b in tracked_ids:
            while b != 0:
                needed.add(b)
                b = int(m.body_parentid[b])
        assert root in needed
        keep = {root} | set(tracked_ids) | {b for b in needed if m.body_jntnum[b] > 0}
        indep_col = {j: i for i, j in enumerate(info.indep_ids)}
        coupled_list = sorted(info.coupled)  # joint ids, evaluated once per forward
        coupled_col = {j: i for i, j in enumerate(coupled_list)}

        def body_offset(b):
            p, R = m.body_pos[b].copy(), np.zeros(9)
            mujoco.mju_quat2Mat(R, m.body_quat[b])
            return p, R.reshape(3, 3)

        # nodes in tree order (mujoco orders parents before children)
        self.body_names, self.parent, off_p, off_R, self.joints = [], [], [], [], []
        local = {}
        for b in range(0 if info.fixed_root else 1, m.nbody):
            if b not in keep:
                continue
            local[b] = len(self.body_names)
            self.body_names.append(m.body(b).name)
            if b == root:
                self.parent.append(-1)
                off_p.append(np.zeros(3)), off_R.append(np.eye(3))
                self.joints.append([])
                continue
            # fold the joint-less bodies between b and its nearest kept ancestor into b's offset
            chain, pb = [b], int(m.body_parentid[b])
            while pb not in local:
                assert (
                    m.body_jntnum[pb] == 0
                ), f"{m.body(pb).name} has joints but was folded"
                chain.append(pb)
                pb = int(m.body_parentid[pb])
            p, R = np.zeros(3), np.eye(3)
            for c in reversed(chain):
                pc, Rc = body_offset(c)
                p, R = p + R @ pc, R @ Rc
            self.parent.append(local[pb])
            off_p.append(p), off_R.append(R)
            js = []
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                jt = int(m.jnt_type[j])
                assert jt in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ), (m.joint(j).name, jt)
                src = ("q", indep_col[j]) if j in indep_col else ("c", coupled_col[j])
                jpos = m.jnt_pos[j].copy()
                js.append(
                    dict(
                        slide=jt == mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=torch.tensor(m.jnt_axis[j], dtype=dtype, device=device),
                        pos=None
                        if np.allclose(jpos, 0)
                        else torch.tensor(jpos, dtype=dtype, device=device),
                        src=src,
                        qpos0=float(m.qpos0[m.jnt_qposadr[j]]),
                    )
                )
            self.joints.append(js)
        self.off_p = torch.tensor(np.array(off_p), dtype=dtype, device=device)
        self.off_R = torch.tensor(np.array(off_R), dtype=dtype, device=device)
        self.num_bodies = len(self.body_names)
        self.tracked_idx = [self.body_names.index(n) for n in tracked]
        # coupled joints: written from the independent ones by MyoSuite's joint-coupling projection (in the
        # couplings' topological order, so chained couplings are exact), on full-length qpos / qvel vectors
        self.couplings = info.couplings.as_torch(device, dtype)
        adr = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)  # noqa: E731
        self._qpos0 = torch.tensor(m.qpos0, dtype=dtype, device=device)
        self._nv = m.nv
        self._indep_qadr, self._indep_dadr = (
            adr(a[info.indep_ids]) for a in (m.jnt_qposadr, m.jnt_dofadr)
        )
        self._coupled_qadr, self._coupled_dadr = (
            adr(a[coupled_list]) for a in (m.jnt_qposadr, m.jnt_dofadr)
        )
        # every non-root joint in model order, as a gather from cat(independent, coupled)
        self.full_index = torch.tensor(
            [
                indep_col[j] if j in indep_col else self.num_q + coupled_col[j]
                for j in info.joint_ids
            ],
            device=device,
        )
        self.q_lo = torch.tensor(info.q_lo, dtype=dtype, device=device)
        self.q_hi = torch.tensor(info.q_hi, dtype=dtype, device=device)
        # kin_unbind: joint coordinates read through one unbind (backward: one stack) instead of a select per joint
        # (backward: a zero-filled scatter per joint); the same values
        self.unbind = False

    def coupled(self, q: torch.Tensor, qd: torch.Tensor):
        """Positions and velocities of the coupled joints (in joint id order), from the independent coordinates."""
        batch = q.shape[:-1]
        qpos = self._qpos0.expand(*batch, -1).index_copy(-1, self._indep_qadr, q)
        qvel = qd.new_zeros(*batch, self._nv).index_copy(-1, self._indep_dadr, qd)
        qpos, qvel = projected_joint_couplings(self.couplings, qpos, qvel)
        return qpos[..., self._coupled_qadr], qvel[..., self._coupled_dadr]

    def full_joint_state(self, q: torch.Tensor, qd: torch.Tensor):
        """Independent coordinates (..., 37) -> every non-root joint (..., 79) in model (= entity) order."""
        if not self._coupled_qadr.numel():  # no couplings
            return q[..., self.full_index], qd[..., self.full_index]
        qc, qdc = self.coupled(q, qd)
        return torch.cat((q, qc), -1)[..., self.full_index], torch.cat((qd, qdc), -1)[
            ..., self.full_index
        ]

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), canonical root rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: independent joint positions/velocities (B,37) in ``ModelInfo.indep_names`` order.
        """
        qc, qdc = self.coupled(q, qd)
        n = self.num_bodies
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        if self.unbind:
            src = {
                "q": (q.unbind(-1), qd.unbind(-1)),
                "c": (qc.unbind(-1), qdc.unbind(-1)),
            }
        else:
            src = {"q": (q, qd), "c": (qc, qdc)}
        for i in range(n):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, mm3(R, self.R_off.T), v, w
                continue
            Rp, pp = Rs[par], P[par]
            pi = pp + mv3(Rp, self.off_p[i])
            Ri = mm3(Rp, self.off_R[i])
            vi = V[par] + torch.cross(W[par], pi - pp, dim=-1)
            wi = W[par]
            for jt in self.joints[i]:
                kind, col = jt["src"]
                xs, xds = src[kind]
                qj = (xs[col] if self.unbind else xs[..., col]) - jt["qpos0"]
                qdj = (xds[col] if self.unbind else xds[..., col]).unsqueeze(-1)
                axis_w = mv3(Ri, jt["axis"])
                if jt["slide"]:
                    dp = axis_w * qj.unsqueeze(-1)
                    pi = pi + dp
                    vi = vi + torch.cross(wi, dp, dim=-1) + axis_w * qdj
                    continue
                if jt["pos"] is None:
                    wi = wi + axis_w * qdj
                    Ri = mm3(Ri, axis_angle_to_mat(jt["axis"], qj))
                    continue
                anchor = pi + mv3(Ri, jt["pos"])
                va = vi + torch.cross(wi, anchor - pi, dim=-1)
                wi = wi + axis_w * qdj
                Ri = mm3(Ri, axis_angle_to_mat(jt["axis"], qj))
                pi = anchor - mv3(Ri, jt["pos"])
                vi = va + torch.cross(wi, pi - anchor, dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(n) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

    def root_body_quat(self, R: torch.Tensor) -> torch.Tensor:
        """Canonical root rotation -> root body rotation (the free joint's qpos quaternion is its wxyz)."""
        return R @ self.R_off.T

    def canonical_root(self, quat_body: torch.Tensor) -> torch.Tensor:
        """Free-joint quaternion (wxyz) of the root body -> canonical root rotation matrix."""
        return quat_to_mat(quat_body) @ self.R_off
