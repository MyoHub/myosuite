# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Differentiable torch forward kinematics built from a compiled ``MjModel``.

Reproduces ``mj_kinematics`` (body frames and site positions) for trees of free,
hinge and slide joints, batched over leading axes. Bodies are processed one tree
depth at a time, so a 100-body model costs about 20 batched steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from myosuite.physics.quat_math_torch import axis_angle2quat, mul_quat, rot_vec_quat


@dataclass
class _Level:
    """Bodies of one tree depth and their (padded) joints."""

    bodies: torch.Tensor  # (B,)
    parents: torch.Tensor  # (B,)
    body_pos: torch.Tensor  # (B, 3)
    body_quat: torch.Tensor  # (B, 4)
    jnt_qadr: torch.Tensor  # (B, J) qpos column, padded with the zero column
    jnt_qpos0: torch.Tensor  # (B, J)
    jnt_pos: torch.Tensor  # (B, J, 3)
    jnt_axis: torch.Tensor  # (B, J, 3)
    jnt_hinge: torch.Tensor  # (B, J) 1 hinge, 0 slide or padding
    jnt_slide: torch.Tensor  # (B, J) 1 slide


class TorchKinematics(torch.nn.Module):
    """Body poses and site positions as a differentiable function of ``qpos``.

    Args:
        model: Compiled MuJoCo model. Only free, hinge and slide joints are
            supported; a free joint must sit on a child of the world body.
        site_ids: Sites whose world positions :meth:`site_pos` returns.
        prune: Only compute the bodies on the paths from the root to the
            selected sites (others keep the identity pose).
        extra_bodies: Bodies kept by ``prune`` in addition to the sites' bodies.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_ids: np.ndarray | None = None,
        prune: bool = False,
        extra_bodies: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError("ball joints are not supported")
        nbody = model.nbody
        depth = np.zeros(nbody, dtype=np.int64)
        for b in range(1, nbody):
            depth[b] = depth[model.body_parentid[b]] + 1
        self.nbody = nbody
        self.nq = model.nq
        # The free root body (if any) is placed directly from qpos[0:7].
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) > 1:
            raise ValueError("at most one free joint is supported")
        self.free_body = int(model.jnt_bodyid[free[0]]) if len(free) else -1
        self.free_qadr = int(model.jnt_qposadr[free[0]]) if len(free) else -1
        if self.free_body >= 0 and model.body_parentid[self.free_body] != 0:
            raise ValueError("the free joint must be on a child of the world body")

        ids = np.arange(model.nsite) if site_ids is None else np.asarray(site_ids)
        keep = set(range(nbody))
        if prune:
            keep = set()
            for b in [*model.site_bodyid[ids], *extra_bodies]:
                while b > 0:
                    keep.add(int(b))
                    b = model.body_parentid[b]
        self.levels: list[_Level] = []
        for lvl in range(1, int(depth.max()) + 1):
            bodies = [
                b
                for b in range(nbody)
                if depth[b] == lvl and b != self.free_body and b in keep
            ]
            if not bodies:
                continue
            nj = max(1, max(int(model.body_jntnum[b]) for b in bodies))
            qadr = np.full((len(bodies), nj), model.nq, dtype=np.int64)
            qpos0 = np.zeros((len(bodies), nj))
            jpos = np.zeros((len(bodies), nj, 3))
            jaxis = np.zeros((len(bodies), nj, 3))
            jaxis[..., 2] = 1.0
            hinge = np.zeros((len(bodies), nj))
            slide = np.zeros((len(bodies), nj))
            for i, b in enumerate(bodies):
                for k in range(int(model.body_jntnum[b])):
                    j = int(model.body_jntadr[b]) + k
                    qadr[i, k] = model.jnt_qposadr[j]
                    qpos0[i, k] = model.qpos0[model.jnt_qposadr[j]]
                    jpos[i, k] = model.jnt_pos[j]
                    jaxis[i, k] = model.jnt_axis[j]
                    hinge[i, k] = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                    slide[i, k] = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
            f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
            self.levels.append(
                _Level(
                    bodies=torch.as_tensor(bodies),
                    parents=torch.as_tensor(
                        model.body_parentid[bodies].astype(np.int64)
                    ),
                    body_pos=f(model.body_pos[bodies]),
                    body_quat=f(model.body_quat[bodies]),
                    jnt_qadr=torch.as_tensor(qadr),
                    jnt_qpos0=f(qpos0),
                    jnt_pos=f(jpos),
                    jnt_axis=f(jaxis),
                    jnt_hinge=f(hinge),
                    jnt_slide=f(slide),
                )
            )
        self.register_buffer(
            "site_body", torch.as_tensor(model.site_bodyid[ids].astype(np.int64))
        )
        self.register_buffer(
            "site_local", torch.as_tensor(model.site_pos[ids], dtype=torch.float32)
        )
        self.register_buffer("free_index", torch.tensor([max(self.free_body, 0)]))

    def _apply(self, fn, recurse=True):  # keep the level tensors on the module's device
        super()._apply(fn, recurse)
        for lvl in self.levels:
            for name in lvl.__dataclass_fields__:
                setattr(lvl, name, fn(getattr(lvl, name)))
        return self

    def body_poses(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World body positions ``(..., nbody, 3)`` and quaternions ``(..., nbody, 4)``."""
        batch = qpos.shape[:-1]
        # Padded joints read this zero column (angle 0 = identity).
        qpad = torch.cat([qpos, qpos.new_zeros(batch + (1,))], dim=-1)
        pos_t = qpos.new_zeros(batch + (self.nbody, 3))
        quat_t = qpos.new_zeros(batch + (self.nbody, 4))
        quat_t[..., 0] = 1.0
        if self.free_body >= 0:
            a = self.free_qadr
            q = qpos[..., a + 3 : a + 7]
            pos_t = pos_t.index_copy(-2, self.free_index, qpos[..., None, a : a + 3])
            quat_t = quat_t.index_copy(
                -2, self.free_index, (q / q.norm(dim=-1, keepdim=True))[..., None, :]
            )
        for lvl in self.levels:
            ppos = pos_t.index_select(-2, lvl.parents)
            pquat = quat_t.index_select(-2, lvl.parents)
            xpos = ppos + rot_vec_quat(lvl.body_pos.expand_as(ppos), pquat)
            xquat = mul_quat(pquat, lvl.body_quat.expand_as(pquat))
            q = qpad[..., lvl.jnt_qadr] - lvl.jnt_qpos0  # (..., B, J)
            for k in range(lvl.jnt_qadr.shape[1]):
                jp = lvl.jnt_pos[:, k].expand_as(xpos)
                ax = lvl.jnt_axis[:, k].expand_as(xpos)
                qk = q[..., k]
                anchor = xpos + rot_vec_quat(jp, xquat)
                # Slide: translate along the world axis; hinge: rotate about the anchor.
                xpos = (
                    xpos
                    + rot_vec_quat(ax, xquat) * (qk * lvl.jnt_slide[:, k])[..., None]
                )
                xquat = mul_quat(xquat, axis_angle2quat(ax, qk * lvl.jnt_hinge[:, k]))
                hinge = lvl.jnt_hinge[:, k][:, None]
                xpos = hinge * (anchor - rot_vec_quat(jp, xquat)) + (1 - hinge) * xpos
            xquat = xquat / xquat.norm(dim=-1, keepdim=True)
            pos_t = pos_t.index_copy(-2, lvl.bodies, xpos)
            quat_t = quat_t.index_copy(-2, lvl.bodies, xquat)
        return pos_t, quat_t

    def site_pos(self, qpos: torch.Tensor) -> torch.Tensor:
        """World positions of the selected sites, ``(..., n_sites, 3)``."""
        pos, quat = self.body_poses(qpos)
        bpos = pos.index_select(-2, self.site_body)
        bquat = quat.index_select(-2, self.site_body)
        return bpos + rot_vec_quat(self.site_local.expand_as(bpos), bquat)


class JointCoupling(torch.nn.Module):
    """Analytic projection onto the model's ``mjEQ_JOINT`` polynomial couplings.

    MuJoCo enforces ``q1 - q1_0 = a0 + a1 x + ... + a4 x^4`` with ``x = q2 - q2_0``
    (or ``q1 - q1_0 = a0`` without a second joint) as a soft constraint. A learned
    model that predicts every joint independently drifts off it; projecting the
    dependent joints back keeps imagined states on the constraint manifold.

    Args:
        model: Compiled MuJoCo model.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__()
        eq = np.flatnonzero(
            (model.eq_type == mujoco.mjtEq.mjEQ_JOINT) & model.eq_active0.astype(bool)
        )
        j1 = model.eq_obj1id[eq]
        j2 = model.eq_obj2id[eq]
        has2 = j2 >= 0
        j2 = np.where(has2, j2, j1)
        self.register_buffer(
            "q1", torch.as_tensor(model.jnt_qposadr[j1].astype(np.int64))
        )
        self.register_buffer(
            "q2", torch.as_tensor(model.jnt_qposadr[j2].astype(np.int64))
        )
        self.register_buffer(
            "v1", torch.as_tensor(model.jnt_dofadr[j1].astype(np.int64))
        )
        self.register_buffer(
            "v2", torch.as_tensor(model.jnt_dofadr[j2].astype(np.int64))
        )
        f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        self.register_buffer("q1_0", f(model.qpos0[model.jnt_qposadr[j1]]))
        self.register_buffer("q2_0", f(model.qpos0[model.jnt_qposadr[j2]]))
        self.register_buffer("has2", f(has2))
        self.register_buffer("coef", f(model.eq_data[eq, :5]))
        mask = np.ones(model.nv, dtype=bool)
        mask[model.jnt_dofadr[j1]] = False
        self.register_buffer("independent_dofs", torch.as_tensor(np.flatnonzero(mask)))

    def project(
        self, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dependent joint positions and velocities recomputed from their masters."""
        x = (qpos[..., self.q2] - self.q2_0) * self.has2
        xd = qvel[..., self.v2] * self.has2
        c = self.coef
        y = c[:, 0] + x * (c[:, 1] + x * (c[:, 2] + x * (c[:, 3] + x * c[:, 4])))
        dy = c[:, 1] + x * (2 * c[:, 2] + x * (3 * c[:, 3] + x * 4 * c[:, 4]))
        qpos = qpos.index_copy(-1, self.q1, self.q1_0 + y)
        qvel = qvel.index_copy(-1, self.v1, dy * xd)
        return qpos, qvel
