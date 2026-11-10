# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Muscle-tendon lengths, velocities and forces of a musculoskeletal model in torch, from the independent joint coordinates.

Replicates MuJoCo's spatial tendons (``mj_tendon``: straight site-site segments and site-geom-site wraps
around spheres and cylinders, with side sites and inside wraps, as in MuJoCo-Warp's ``util_misc.wrap``)
and its Hill-type muscle curves (``muscle_gain`` / ``muscle_bias``), so the muscle length, velocity and
force observations of MuscleMimic can be evaluated on imagined states.

Lengths are differentiable: straight segments and outside wraps through their closed forms; inside wraps
(side site within the wrap geom, solved by Newton's method) hold the solved tangent point fixed in the geom
frame, which is exact to first order because the path length is stationary in it. Velocities follow
MuJoCo's ``ten_velocity = ten_J qvel``: every straight piece contributes the rate of change of its length
with its end points moving with their bodies (wrap tangent points move with the wrap geom), the arc on the
geom none. Tendon length does not depend on the root, so the root is held fixed at the origin.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/muscle_geometry.py``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from .rotations import mm3, mv3

from .kinematics import SkeletonKinematics

MINVAL = 1e-15  # mjMINVAL


def _sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with a finite gradient where x <= 0 (unselected torch.where branches would otherwise give NaN)."""
    return x.clamp(min=1e-30).sqrt()


def _norm(x: torch.Tensor) -> torch.Tensor:
    return _sqrt((x * x).sum(-1))


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / _norm(x).unsqueeze(-1)


def _dot2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _intersect(p1, p2, p3, p4) -> torch.Tensor:
    """Whether the 2D segments p1-p2 and p3-p4 intersect (util_misc.is_intersect)."""
    det = (p4[..., 1] - p3[..., 1]) * (p2[..., 0] - p1[..., 0]) - (
        p4[..., 0] - p3[..., 0]
    ) * (p2[..., 1] - p1[..., 1])
    ok = det.abs() >= MINVAL
    det = torch.where(ok, det, torch.ones_like(det))
    a = (
        (p4[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])
        - (p4[..., 1] - p3[..., 1]) * (p1[..., 0] - p3[..., 0])
    ) / det
    b = (
        (p2[..., 0] - p1[..., 0]) * (p1[..., 1] - p3[..., 1])
        - (p2[..., 1] - p1[..., 1]) * (p1[..., 0] - p3[..., 0])
    ) / det
    return ok & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)


def wrap_circle(
    e0: torch.Tensor,
    e1: torch.Tensor,
    side: torch.Tensor,
    has_side: torch.Tensor,
    r: torch.Tensor,
):
    """2D circle wrap of the segment e0-e1 around a circle of radius r at the origin (util_misc.wrap_circle).
    Returns (wrap length or -1, tangent point 0, tangent point 1)."""
    sq0, sq1, sqr = _dot2(e0, e0), _dot2(e1, e1), r * r
    dif = e1 - e0
    dd = _dot2(dif, dif)
    a = (-_dot2(dif, e0) / dd.clamp(min=MINVAL)).clamp(0.0, 1.0)
    near = a.unsqueeze(-1) * dif + e0
    nowrap = (sq0 < sqr) | (sq1 < sqr) | (r < MINVAL) | (dd < MINVAL)
    nowrap |= (_dot2(near, near) > sqr) & (~has_side | (_dot2(side, near) >= 0))
    s0, s1 = _sqrt(sq0 - sqr), _sqrt(sq1 - sqr)
    q0, q1 = sq0.clamp(min=MINVAL), sq1.clamp(min=MINVAL)

    def sol(sg):
        t0 = torch.stack(
            (
                (e0[..., 0] * sqr + sg * r * e0[..., 1] * s0) / q0,
                (e0[..., 1] * sqr - sg * r * e0[..., 0] * s0) / q0,
            ),
            -1,
        )
        t1 = torch.stack(
            (
                (e1[..., 0] * sqr - sg * r * e1[..., 1] * s1) / q1,
                (e1[..., 1] * sqr + sg * r * e1[..., 0] * s1) / q1,
            ),
            -1,
        )
        return t0, t1

    t00, t01 = sol(1.0)
    t10, t11 = sol(-1.0)
    good0 = torch.where(
        has_side, _dot2(_unit(t00 + t01), side), -_dot2(t00 - t01, t00 - t01)
    )
    good1 = torch.where(
        has_side, _dot2(_unit(t10 + t11), side), -_dot2(t10 - t11, t10 - t11)
    )
    good0 = torch.where(
        _intersect(e0, t00, e1, t01), torch.full_like(good0, -10000.0), good0
    )
    good1 = torch.where(
        _intersect(e0, t10, e1, t11), torch.full_like(good1, -10000.0), good1
    )
    first = good0 > good1
    p0 = torch.where(first.unsqueeze(-1), t00, t10)
    p1 = torch.where(first.unsqueeze(-1), t01, t11)
    nowrap |= _intersect(e0, p0, e1, p1)
    # length along the circle (util_misc.length_circle)
    ang = torch.acos(_dot2(_unit(p0), _unit(p1)).clamp(-1 + 1e-7, 1 - 1e-7))
    cross = p0[..., 1] * p1[..., 0] - p0[..., 0] * p1[..., 1]
    flip = ((cross > 0) & ~first) | ((cross < 0) & first)
    ang = torch.where(flip, 2 * math.pi - ang, ang)
    return torch.where(nowrap, -torch.ones_like(ang), r * ang), p0, p1


@torch.no_grad()
def wrap_inside(e0: torch.Tensor, e1: torch.Tensor, r: torch.Tensor, maxiter: int = 20, zinit: float = 1.0 - 1e-7,
                tol: float = 1e-6):  # fmt: skip
    """2D inside wrap (util_misc.wrap_inside): the single tangent point, without gradient. Returns
    (0 if wrapped else -1, point)."""
    len0, len1 = _norm(e0), _norm(e1)
    dif = e1 - e0
    dd = _dot2(dif, dif)
    nowrap = (
        (len0 <= r) | (len1 <= r) | (r < MINVAL) | (len0 < MINVAL) | (len1 < MINVAL)
    )
    a = -_dot2(dif, e0) / dd.clamp(min=MINVAL)
    seg = (dd > MINVAL) & (a > 0) & (a < 1)
    nowrap |= seg & (_norm(e0 + a.unsqueeze(-1) * dif) <= r)
    pnt = _unit(0.5 * (e0 + e1)) * r.unsqueeze(-1)  # default on numerical failure
    A, B = r / len0.clamp(min=MINVAL), r / len1.clamp(min=MINVAL)
    cosG = (len0 * len0 + len1 * len1 - dd) / (2 * len0 * len1).clamp(min=MINVAL)
    nowrap |= cosG < -1 + MINVAL
    done = cosG > 1 - MINVAL  # wrap at the default point
    G = torch.acos(cosG.clamp(-1, 1))

    def fun(z):
        return (
            torch.asin((A * z).clamp(-1, 1))
            + torch.asin((B * z).clamp(-1, 1))
            - 2 * torch.asin(z.clamp(-1, 1))
            + G
        )

    z = torch.full_like(A, zinit)
    f = fun(z)
    done |= f > 0
    for _ in range(maxiter):
        run = ~done & (f.abs() > tol)
        sz = z * z
        df = (A / (1 - sz * A * A).clamp(min=0).sqrt().clamp(min=MINVAL) + B / (1 - sz * B * B).clamp(min=0).sqrt().clamp(min=MINVAL)
              - 2 / (1 - sz).clamp(min=0).sqrt().clamp(min=MINVAL))  # fmt: skip
        bad = run & (df > -MINVAL)
        z1 = z - f / torch.where(df.abs() < MINVAL, torch.full_like(df, -MINVAL), df)
        bad |= run & (z1 > z)
        f1 = fun(z1)
        bad |= run & ~bad & (f1 > tol)
        done |= bad
        ok = run & ~bad
        z, f = torch.where(ok, z1, z), torch.where(ok, f1, f)
    # not converged within maxiter: default point
    done |= ~done & (f.abs() > tol)
    pos = e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0] > 0
    vec = _unit(torch.where(pos.unsqueeze(-1), e0, e1))
    ang = torch.asin(z.clamp(-1, 1)) - torch.asin(
        (torch.where(pos, A, B) * z).clamp(-1, 1)
    )
    c, s = torch.cos(ang), torch.sin(ang)
    rot = r.unsqueeze(-1) * torch.stack(
        (c * vec[..., 0] - s * vec[..., 1], s * vec[..., 0] + c * vec[..., 1]), -1
    )
    pnt = torch.where(done.unsqueeze(-1), pnt, rot)
    return torch.where(nowrap, -torch.ones_like(len0), torch.zeros_like(len0)), pnt


def muscle_gain_length(
    L: torch.Tensor, lmin: torch.Tensor, lmax: torch.Tensor
) -> torch.Tensor:
    a, b = 0.5 * (lmin + 1), 0.5 * (1 + lmax)
    x1 = (L - lmin) / (a - lmin).clamp(min=MINVAL)
    x2 = (1 - L) / (1 - a).clamp(min=MINVAL)
    x3 = (L - 1) / (b - 1).clamp(min=MINVAL)
    x4 = (lmax - L) / (lmax - b).clamp(min=MINVAL)
    out = torch.where(
        L <= a,
        0.5 * x1 * x1,
        torch.where(
            L <= 1,
            1 - 0.5 * x2 * x2,
            torch.where(L <= b, 1 - 0.5 * x3 * x3, 0.5 * x4 * x4),
        ),
    )
    return torch.where((L < lmin) | (L > lmax), torch.zeros_like(L), out)


class MuscleGeometry:
    """Muscle length, velocity and force of every muscle (actuator order) from (q, qd, act)."""

    def __init__(
        self, m: mujoco.MjModel, device: str, dtype: torch.dtype = torch.float32
    ):
        assert (m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON).all() and (
            m.actuator_gear[:, 0] == 1
        ).all()
        assert (
            not m.actuator_forcelimited.any()
            and (m.actuator_gainprm[:, 2] > 0).all()
            and (m.actuator_biasprm[:, 2] > 0).all()
        )
        S, SPH, CYL = (
            mujoco.mjtWrap.mjWRAP_SITE,
            mujoco.mjtWrap.mjWRAP_SPHERE,
            mujoco.mjtWrap.mjWRAP_CYLINDER,
        )
        pairs, triples, sites, geoms = [], [], {}, {}
        for col in range(m.nu):
            t = int(m.actuator_trnid[col, 0])
            adr, num = int(m.tendon_adr[t]), int(m.tendon_num[t])
            typ, obj = m.wrap_type[adr : adr + num], m.wrap_objid[adr : adr + num]
            assert (
                typ[0] == S
                and typ[-1] == S
                and not ((typ != S) & (typ != SPH) & (typ != CYL)).any()
            ), m.tendon(t).name
            for j in range(num - 1):
                if typ[j] == S and typ[j + 1] == S:
                    pairs.append(
                        (
                            col,
                            sites.setdefault(int(obj[j]), len(sites)),
                            sites.setdefault(int(obj[j + 1]), len(sites)),
                        )
                    )
                elif typ[j + 1] != S:  # site-geom-site
                    side = round(float(m.wrap_prm[adr + j + 1]))
                    triples.append((col, sites.setdefault(int(obj[j]), len(sites)), geoms.setdefault(int(obj[j + 1]), len(geoms)),
                                    sites.setdefault(int(obj[j + 2]), len(sites)), -1 if side < 0 else sites.setdefault(side, len(sites)),
                                    typ[j + 1] == CYL))  # fmt: skip
        site_ids, geom_ids = list(sites), list(geoms)
        bodies = sorted(
            {int(m.site_bodyid[s]) for s in site_ids}
            | {int(m.geom_bodyid[g]) for g in geom_ids}
        )
        self.kin = SkeletonKinematics(
            [m.body(b).name for b in bodies], device, m, dtype
        )
        col = {b: i for i, b in enumerate(bodies)}
        t = lambda x, **kw: torch.tensor(np.asarray(x), device=device, **kw)  # noqa: E731
        self.site_body = t([col[int(m.site_bodyid[s])] for s in site_ids])
        self.site_pos = t(m.site_pos[site_ids], dtype=dtype)
        self.geom_body = t([col[int(m.geom_bodyid[g])] for g in geom_ids])
        self.geom_pos = t(m.geom_pos[geom_ids], dtype=dtype)
        gR = np.zeros((len(geom_ids), 9))
        for i, g in enumerate(geom_ids):
            mujoco.mju_quat2Mat(gR[i], m.geom_quat[g])
        self.geom_R = t(gR.reshape(-1, 3, 3), dtype=dtype)
        self.geom_radius = t(m.geom_size[geom_ids, 0], dtype=dtype)
        p = np.array(pairs, dtype=np.int64).reshape(-1, 3)
        self.pair_col, self.pair_s0, self.pair_s1 = t(p[:, 0]), t(p[:, 1]), t(p[:, 2])
        tr = np.array([x[:5] for x in triples], dtype=np.int64)
        self.tri_col, self.tri_s0, self.tri_g, self.tri_s1, side = (
            t(tr[:, i]) for i in range(5)
        )
        self.tri_has_side = side >= 0
        self.tri_side = side.clamp(min=0)
        self.tri_cyl = t([x[5] for x in triples])
        self.num_muscles = m.nu
        # muscle curves: gainprm = (range0, range1, force, scale, lmin, lmax, vmax, fpmax, fvmax)
        g = m.actuator_gainprm
        self.range0, self.range1, self.F0 = (
            t(g[:, 0], dtype=dtype),
            t(g[:, 1], dtype=dtype),
            t(g[:, 2], dtype=dtype),
        )
        self.lmin, self.lmax, self.vmax = (
            t(g[:, 4], dtype=dtype),
            t(g[:, 5], dtype=dtype),
            t(g[:, 6], dtype=dtype),
        )
        self.fvmax = t(g[:, 8], dtype=dtype)
        # the passive force reads biasprm (same layout), which MyoFullBody's supraspinatus sets apart from gainprm
        b = m.actuator_biasprm
        self.b_range0, self.b_range1, self.b_F0 = (
            t(b[:, 0], dtype=dtype),
            t(b[:, 1], dtype=dtype),
            t(b[:, 2], dtype=dtype),
        )
        self.b_lmax, self.b_fpmax = t(b[:, 5], dtype=dtype), t(b[:, 7], dtype=dtype)
        self.lr0, self.lr1 = (
            t(m.actuator_lengthrange[:, 0], dtype=dtype),
            t(m.actuator_lengthrange[:, 1], dtype=dtype),
        )

    def length_velocity(
        self, q: torch.Tensor, qd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Muscle (= tendon) lengths and velocities, (..., nm) each, from the independent joints (..., nq)."""
        lead = q.shape[:-1]
        z = q.new_zeros(lead + (3,))
        R = self.kin.R_off.expand(lead + (3, 3))  # pelvis body frame = world
        P, Rb, V, W = self.kin.forward(z, R, z, z, q, qd)
        X = P[..., self.site_body, :] + mv3(
            Rb[..., self.site_body, :, :], self.site_pos
        )
        # site velocities: link-origin velocity + w x (x - origin)
        Xd = V[..., self.site_body, :] + torch.cross(
            W[..., self.site_body, :], X - P[..., self.site_body, :], dim=-1
        )
        L = q.new_zeros(lead + (self.num_muscles,))
        Ld = q.new_zeros(lead + (self.num_muscles,))
        # straight site-site segments
        d = X[..., self.pair_s1, :] - X[..., self.pair_s0, :]
        n = _norm(d)
        u = d / n.unsqueeze(-1)
        L = L.index_add(-1, self.pair_col, n)
        Ld = Ld.index_add(
            -1,
            self.pair_col,
            (u * (Xd[..., self.pair_s1, :] - Xd[..., self.pair_s0, :])).sum(-1),
        )
        # site-geom-site
        gb = self.geom_body[self.tri_g]
        gpos = P[..., gb, :] + mv3(Rb[..., gb, :, :], self.geom_pos[self.tri_g])
        gR = mm3(Rb[..., gb, :, :], self.geom_R[self.tri_g])
        x0, x1, sd = (
            X[..., self.tri_s0, :],
            X[..., self.tri_s1, :],
            X[..., self.tri_side, :],
        )
        wl, t0, t1 = self._wrap(x0, x1, gpos, gR, sd)
        wrapped = wl >= 0
        # tangent points move with the geom's body
        gv, gw, gp = V[..., gb, :], W[..., gb, :], P[..., gb, :]
        t0d = gv + torch.cross(gw, t0 - gp, dim=-1)
        t1d = gv + torch.cross(gw, t1 - gp, dim=-1)
        x0d, x1d = Xd[..., self.tri_s0, :], Xd[..., self.tri_s1, :]
        da, db, ds = t0 - x0, x1 - t1, x1 - x0
        na, nb, ns = _norm(da), _norm(db), _norm(ds)
        seg_len = torch.where(wrapped, na + wl + nb, ns)
        rate_w = (da / na.unsqueeze(-1) * (t0d - x0d)).sum(-1) + (
            db / nb.unsqueeze(-1) * (x1d - t1d)
        ).sum(-1)
        rate_s = (ds / ns.unsqueeze(-1) * (x1d - x0d)).sum(-1)
        L = L.index_add(-1, self.tri_col, seg_len)
        Ld = Ld.index_add(-1, self.tri_col, torch.where(wrapped, rate_w, rate_s))
        return L, Ld

    def _wrap(self, x0, x1, gpos, gR, side):
        """util_misc.wrap for every site-geom-site triple: (wrap length or -1, 3D tangent points in world)."""
        r = self.geom_radius[self.tri_g].expand(x0.shape[:-1])
        gRt = gR.transpose(-1, -2)
        p0, p1, sp = mv3(gRt, x0 - gpos), mv3(gRt, x1 - gpos), mv3(gRt, side - gpos)
        cyl = self.tri_cyl.expand(r.shape).unsqueeze(-1)
        # 2D frame: sphere - plane of (0, p0, p1); cylinder - local xy
        ax0 = _unit(p0)
        normal = torch.cross(p0, p1, dim=-1)
        par = _norm(normal) < MINVAL
        # (p0, p1) parallel: any normal to p0 (util_misc.wrap's choice)
        i = ax0.abs().argmax(-1, keepdim=True)
        alt = torch.ones_like(ax0).scatter(-1, i, 0.0)
        normal = torch.where(par.unsqueeze(-1), torch.cross(ax0, alt, dim=-1), normal)
        ax1 = _unit(torch.cross(_unit(normal), ax0, dim=-1))
        ex, ey = torch.zeros_like(p0), torch.zeros_like(p0)
        ex[..., 0], ey[..., 1] = 1.0, 1.0
        ax0, ax1 = torch.where(cyl, ex, ax0), torch.where(cyl, ey, ax1)
        e0 = torch.stack(((p0 * ax0).sum(-1), (p0 * ax1).sum(-1)), -1)
        e1 = torch.stack(((p1 * ax0).sum(-1), (p1 * ax1).sum(-1)), -1)
        has_side = self.tri_has_side.expand(r.shape)
        s2 = _unit(
            torch.stack(((sp * ax0).sum(-1), (sp * ax1).sum(-1)), -1)
        ) * r.unsqueeze(-1)
        inside = has_side & (_norm(sp) < r)
        wl_c, c0, c1 = wrap_circle(e0, e1, s2, has_side, r)
        wl_i, ci = wrap_inside(e0, e1, r)
        wl = torch.where(inside, wl_i, wl_c)
        q0 = torch.where(inside.unsqueeze(-1), ci, c0)
        q1 = torch.where(inside.unsqueeze(-1), ci, c1)
        tooclose = (_norm(p0) < MINVAL) | (_norm(p1) < MINVAL)
        wl = torch.where(tooclose, -torch.ones_like(wl), wl)
        res0 = ax0 * q0[..., :1] + ax1 * q0[..., 1:]
        res1 = ax0 * q1[..., :1] + ax1 * q1[..., 1:]
        # inside wraps: the solved point stays fixed in the geom frame (the length is stationary in it)
        res0 = torch.where(inside.unsqueeze(-1), res0.detach(), res0)
        res1 = torch.where(inside.unsqueeze(-1), res1.detach(), res1)
        # cylinder: heights of the tangent points by arc length, helix length
        L0 = _norm(p0[..., :2] - res0[..., :2])
        L1 = _norm(p1[..., :2] - res1[..., :2])
        tot = (L0 + wl + L1).clamp(min=MINVAL)
        z0 = p0[..., 2] + (p1[..., 2] - p0[..., 2]) * L0 / tot
        z1 = p0[..., 2] + (p1[..., 2] - p0[..., 2]) * (L0 + wl) / tot
        c = cyl.squeeze(-1)
        res0 = torch.cat(
            (res0[..., :2], torch.where(c, z0, res0[..., 2]).unsqueeze(-1)), -1
        )
        res1 = torch.cat(
            (res1[..., :2], torch.where(c, z1, res1[..., 2]).unsqueeze(-1)), -1
        )
        wl = torch.where(c & (wl >= 0), _sqrt(wl * wl + (z1 - z0) ** 2), wl)
        w0 = mv3(gR, res0) + gpos
        w1 = mv3(gR, res1) + gpos
        return wl, w0, w1

    def force(
        self, L: torch.Tensor, Ld: torch.Tensor, act: torch.Tensor
    ) -> torch.Tensor:
        """Actuator force gain(L, V) act + bias(L) (MuJoCo's muscle_gain on gainprm / muscle_bias on biasprm;
        negative = pulling)."""
        L0 = (self.lr1 - self.lr0) / (self.range1 - self.range0).clamp(min=MINVAL)
        Ln = self.range0 + (L - self.lr0) / L0.clamp(min=MINVAL)
        Vn = Ld / (L0 * self.vmax).clamp(min=MINVAL)
        FL = muscle_gain_length(Ln, self.lmin, self.lmax)
        y = self.fvmax - 1
        FV = torch.where(Vn <= -1, torch.zeros_like(Vn),
                         torch.where(Vn <= 0, (Vn + 1) ** 2,
                                     torch.where(Vn <= y, self.fvmax - (y - Vn) ** 2 / y.clamp(min=MINVAL), self.fvmax)))  # fmt: skip
        L0b = (self.lr1 - self.lr0) / (self.b_range1 - self.b_range0).clamp(min=MINVAL)
        Lnb = self.b_range0 + (L - self.lr0) / L0b.clamp(min=MINVAL)
        b = 0.5 * (1 + self.b_lmax)
        xb = (Lnb - 1) / (b - 1).clamp(min=MINVAL)
        FP = torch.where(Lnb <= 1, torch.zeros_like(Lnb),
                         torch.where(Lnb <= b, 0.5 * xb * xb, 0.5 + (Lnb - b) / (b - 1).clamp(min=MINVAL)))  # fmt: skip
        return -self.F0 * FL * FV * act - self.b_F0 * self.b_fpmax * FP
