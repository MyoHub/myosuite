# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""Left-right mirror of body states, joint coordinates, muscles and foot touch, for world-model data augmentation
(BlinkMimic's ``--wm_mirror``).

The mirror is the reflection ``M = diag(1, -1, 1)`` of the canonical frame's y axis (the sagittal plane): positions
and linear velocities map to ``M x``, the canonical root rotation to ``M R M``, angular velocities (pseudo-vectors)
to ``-M w``. Bodies and muscles are paired left / right by name (``_r`` / ``_l``, ``_R`` / ``_L``, ``_right`` /
``_left``, and a bare name with its ``_left`` / ``_L`` / ``_l`` twin; the rest are midline). The joint map (which
joint, which sign) is found from the forward kinematics: moving a joint by a small angle and mirroring the resulting
body poses must equal moving its partner by plus or minus that angle. A body's frame need not be the mirror image of
its partner's frame (OpenSim gives both sides the same conventions), so rotations are compared through a fixed
per-body offset taken from the default pose.

Every map is validated on random poses at construction: body positions, rotations and velocities through the
kinematics, muscle lengths and velocities through the muscle geometry. The residuals measure the model's own
asymmetry (``residuals``, ``body_residual``, ``muscle_residual``), and a model asymmetric beyond ``TOL`` raises,
since half of every mirrored batch would then be wrong. MS-Human-700's skeleton is symmetric to float precision and
MyoFullBody's to 2 mm (arms, head, toes); muscle paths are symmetric to well under 1 mm except LTpT_T12 (3-4 cm on
both models, an upstream path quirk), EDCL (1.3 cm) and two multifidus segments (2-6 mm) on MS-Human-700, DELT3
and PQ (1-3 mm) on MyoFullBody.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/mirror.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from supermimic.kinematics import (
    SkeletonKinematics,
)
from supermimic.muscle_geometry import (
    MuscleGeometry,
)

_SIDES = (
    ("_r", "_l"),
    ("_l", "_r"),
    ("_R", "_L"),
    ("_L", "_R"),
    ("_right", "_left"),
    ("_left", "_right"),
)
_BARE_TWINS = ("_left", "_L", "_l", "_right", "_R", "_r")


def _sides_of(pairs: Sequence[tuple[str, str]] | None) -> tuple:
    """Both directions of left / right suffix pairs (``None``: the default ones)."""
    if pairs is None:
        return _SIDES
    return tuple(x for a, b in pairs for x in ((a, b), (b, a)))


def partner_name(name: str, sides: Sequence[tuple[str, str]] | None = None) -> str:
    """The name of a body's left / right partner by its side suffix (itself for a name without one)."""
    for a, b in _sides_of(sides):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name


def partner_index(
    names: Sequence[str], what: str, sides: Sequence[tuple[str, str]] | None = None
) -> list[int]:
    """Each name's partner among ``names``: by side suffix, else a bare name's ``_left`` / ``_L`` / ``_l`` twin (or the
    twin's bare name), else itself (midline). Ambiguous or non-involutive pairings raise."""
    pos = {n: i for i, n in enumerate(names)}
    twins = _BARE_TWINS if sides is None else tuple(x for pair in sides for x in pair)
    out = []
    for n in names:
        p = partner_name(n, sides)
        if p != n:
            cands = (
                [p]
                if p in pos
                else [
                    n[: -len(s)] for s in twins if n.endswith(s) and n[: -len(s)] in pos
                ]
            )
        else:
            cands = [n + s for s in twins if n + s in pos]
        if len(cands) > 1:
            raise ValueError(
                f"mirror: {what} {n!r} has several left-right partners: {cands}"
            )
        out.append(pos[cands[0]] if cands else pos[n])
    if sorted(out) != list(range(len(names))) or any(
        out[j] != i for i, j in enumerate(out)
    ):
        bad = [names[i] for i, j in enumerate(out) if out[j] != i]
        raise ValueError(
            f"mirror: the {what} pairing is not a left-right involution at {bad[:8]}"
        )
    return out


class Mirror:
    """Mirror maps of one model, validated on random poses (``TOL``: the allowed residual per quantity, in m, matrix
    entries, m/s or rad/s at unit random velocities; ``tol`` scales them all)."""

    TOL: ClassVar[dict[str, float]] = {
        "default_pose": 1e-2, "joint_map": 2e-2, "positions": 1e-2, "rotations": 2e-2, "velocities": 1e-1,
        "muscle_length": 5e-2, "muscle_velocity": 5e-1, "muscle_force_scale": 1e-2,
    }  # fmt: skip

    def __init__(self, kin: SkeletonKinematics, geo: MuscleGeometry | None, touch_bodies: Sequence[str], device: str,
                 tol: float = 1.0, seed: int = 0, sides: Sequence[tuple[str, str]] | None = None):  # fmt: skip
        self.device = device
        M = self.M = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))
        names = kin.body_names
        bp = self.body_perm = torch.tensor(
            partner_index(names, "body", sides), device=device
        )
        self.touch_perm = torch.tensor(
            partner_index(list(touch_bodies), "touch body", sides),
            device=device,
            dtype=torch.long,
        )
        m, info = kin.info.m, kin.info
        nq = kin.num_q
        self.q0 = torch.tensor(
            [m.qpos0[m.jnt_qposadr[j]] for j in info.indep_ids],
            dtype=torch.float32,
            device=device,
        )
        gen = torch.Generator(device=device).manual_seed(seed)

        def fk(q, qd, p=None, R=None, v=None, w=None):
            B = q.shape[0]
            z = torch.zeros(B, 3, device=device)
            p, v, w = (z if x is None else x for x in (p, v, w))
            R = torch.eye(3, device=device).expand(B, 3, 3) if R is None else R
            return kin.forward(p, R, v, w, q, qd, all_bodies=True)

        # the default pose is symmetric, and defines each body's frame offset D_b from the mirror image of its
        # partner's frame: R_b = (M R_partner M) D_b in every pose of a symmetric model
        P0, R0, _, _ = (
            x[0] for x in fk(self.q0[None], torch.zeros(1, nq, device=device))
        )
        self.residuals = {"default_pose": float((P0[bp] - P0 @ M).abs().max())}
        self.Dt = ((M @ R0[bp] @ M).transpose(-1, -2) @ R0).transpose(
            -1, -2
        )  # (nb, 3, 3): D_b^T

        def image(P, Rb):
            """The mirror image of body poses (..., nb, 3), (..., nb, 3, 3): what the mirrored configuration's own FK
            must give, after its frame offsets."""
            return P[..., bp, :] @ M, M @ Rb[..., bp, :, :] @ M

        def canon(Rb):
            return Rb @ self.Dt

        # joint map: joint j moved by +delta, mirrored, equals its partner k = perm[j] moved by sign[j] * delta
        delta = 0.2
        Q = self.q0.repeat(2 * nq, 1)
        ar = torch.arange(nq, device=device)
        Q[ar, ar] += delta
        Q[nq + ar, ar] -= delta
        Pc, Rc, _, _ = fk(
            Q, torch.zeros_like(Q)
        )  # candidates: +delta on joint k, then -delta on joint k
        Pt, Rt = image(Pc[:nq], Rc[:nq])  # the images of +delta on joint j
        d = (Pt[:, None] - Pc[None]).flatten(2).norm(dim=-1) + (
            Rt[:, None] - canon(Rc)[None]
        ).flatten(2).norm(dim=-1)
        best2 = d.topk(2, dim=1, largest=False)
        best, margin = best2.indices[:, 0], best2.values[:, 1] - best2.values[:, 0]
        self.residuals["joint_map"] = float(best2.values[:, 0].max())
        perm, sign = best % nq, torch.where(best < nq, 1.0, -1.0)
        ambiguous = margin < 10 * best2.values[:, 0] + 1e-3
        if ambiguous.any():
            raise ValueError(
                f"mirror: the mirror image of these joints is ambiguous: {[info.indep_names[j] for j in ambiguous.nonzero().flatten().tolist()]}"
            )
        if sorted(perm.tolist()) != list(range(nq)) or not torch.equal(perm[perm], ar):
            raise ValueError(
                f"mirror: the joint map is not a left-right involution: {perm.tolist()}"
            )
        # q'[k] = q0[k] + sign[j] (q[j] - q0[j]) for k = perm[j]: as a gather by the output index k
        self.j_src = torch.empty(nq, dtype=torch.long, device=device)
        self.j_sign = torch.empty(nq, device=device)
        self.j_src[perm], self.j_sign[perm] = ar, sign
        # validation on random poses and root states: FK(mirror(S)) == the image of FK(S)
        B = 32
        u = torch.rand(B, nq, generator=gen, device=device)
        q = kin.q_lo + u * (kin.q_hi - kin.q_lo)
        qd = torch.randn(B, nq, generator=gen, device=device)
        p, v, w = (torch.randn(B, 3, generator=gen, device=device) for _ in range(3))
        R = _random_rotations(B, gen, device)
        S = (p, R, v, w, q, qd)
        P, Rb, V, W = fk(*S[4:], *S[:4])
        Sm = self.body(S)
        Pm, Rm, Vm, Wm = fk(*Sm[4:], *Sm[:4])
        Pt, Rt = image(P, Rb)
        self.body_residual = (Pm - Pt).abs().amax((0, 2))  # (nb,) over poses and axes
        self.residuals["positions"] = float(self.body_residual.max())
        self.residuals["rotations"] = float((canon(Rm) - Rt).abs().max())
        self.residuals["velocities"] = max(
            float((Vm - V[:, bp] @ M).abs().max()),
            float((Wm + W[:, bp] @ M).abs().max()),
        )
        # muscles: paired by name, validated by their lengths and velocities on the same poses
        self.muscle_perm = self.muscle_residual = None
        if geo is not None:
            mp = self.muscle_perm = torch.tensor(
                partner_index(info.muscle_names, "muscle", sides), device=device
            )
            L, Ld = geo.length_velocity(q, qd)
            Lm, Ldm = geo.length_velocity(Sm[4], Sm[5])
            self.muscle_residual = (Lm[:, mp] - L).abs().amax(0)  # (nm,) over poses
            self.residuals["muscle_length"] = float(self.muscle_residual.max())
            self.residuals["muscle_velocity"] = float((Ldm[:, mp] - Ld).abs().max())
            self.residuals["muscle_force_scale"] = float(
                (geo.F0[mp] - geo.F0).abs().max() / geo.F0.abs().max()
            )
        bad = {
            k: f"{v:.1e} > {tol * self.TOL[k]:.0e}"
            for k, v in self.residuals.items()
            if v > tol * self.TOL[k]
        }
        if bad:
            raise ValueError(f"mirror: the model is not left-right symmetric: {bad}")

    # ---------------------------------------------------------------- maps (any leading dims)
    def joints(self, q: torch.Tensor) -> torch.Tensor:
        return self.q0 + self.j_sign * (q[..., self.j_src] - self.q0[self.j_src])

    def joint_vel(self, qd: torch.Tensor) -> torch.Tensor:
        return self.j_sign * qd[..., self.j_src]

    def body(self, S):
        """(p, R, v, w, q, qd) -> its mirror image (the other entries of S are left out)."""
        p, R, v, w, q, qd = S[:6]
        M = self.M
        return (p @ M, M @ R @ M, v @ M, -(w @ M), self.joints(q), self.joint_vel(qd))

    def muscles(self, x: torch.Tensor) -> torch.Tensor:
        """Activations, excitations, lengths, ... in actuator order -> those of the mirror image."""
        return x[..., self.muscle_perm]

    def touch(self, t: torch.Tensor) -> torch.Tensor:
        return t[..., self.touch_perm]


def _random_rotations(B: int, gen: torch.Generator, device: str) -> torch.Tensor:
    from supermimic.rotations import (
        quat_to_mat,
    )

    return quat_to_mat(torch.randn(B, 4, generator=gen, device=device))
