# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Landmark IK of motion onto a body model, on the model's own joint couplings.

A :class:`ClipSource` turns a motion (AMASS SMPL+H: :mod:`.retarget_smplh`) into
landmark targets on the model; :class:`LandmarkIK` fits every frame on any ``MjModel``:

* targets: points on model bodies (an explicit :class:`LandmarkMap`: model sites or
  virtual sites, fixed offsets on a body) plus body orientations at a lower weight;
* unknowns: the free root and the independent joints only. Coupled joints follow the
  model's polynomials exactly (chain rule into the Jacobian, projection after every
  step), so the coupling residual is zero by construction. Couplings as soft costs
  (mink's ``EqualityConstraintTask``, as in GMR) leave clips off the model's couplings;
* solver: damped Gauss-Newton (Levenberg-Marquardt) with hard joint limits (active-set
  box steps), warm-started from the previous frames, with a small velocity-smoothing
  term (deviation from the constant-velocity prediction, so smooth motion is not
  biased towards standing still) and optional posture priors.

A new model version (myofullbody v2) is a rerun with its landmark map.

Solving the retargeting IK on the model's couplings with the motion's own body
placement as targets follows Mark Colley's ``refit_stale_couplings`` (MyoSuite PR
#651), extended from one limb to the whole body.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

REFIT_META_KEY = "refit_meta"
"""Key of the JSON metadata in a refitted clip's ``.npz``."""


@dataclass(frozen=True)
class Landmark:
    """A clip site tracked by a point of the model.

    Attributes:
        source: The clip's site name (``site_names``), whose ``site_xpos`` is the target.
        body: Model body carrying the point.
        site: Model site to use, if the model has it; otherwise a virtual site.
        pos: Offset of the virtual site in ``body``'s frame: the source body's site
            placement relative to the same-named body (checked against the clip).
        weight: Position weight (residual in metres times weight).
    """

    source: str
    body: str
    site: str | None = None
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    weight: float = 1.0


@dataclass(frozen=True)
class OrientationTarget:
    """A clip body orientation tracked by a model body.

    Attributes:
        source: The clip's body name (``body_names``), whose ``xquat`` is the target.
        body: Model body.
        weight: Metres of position error one radian is worth.
        axis: Body-frame unit axis (e.g. a limb's long axis) whose rotation component
            is weighted by ``axis_weight`` relative to the other two; ``None``: isotropic.
        axis_weight: Weight factor of the rotation about ``axis``.
    """

    source: str
    body: str
    weight: float = 0.05
    axis: tuple[float, float, float] | None = None
    axis_weight: float = 1.0


@dataclass(frozen=True)
class LandmarkMap:
    """Everything model-specific of a refit: targets, feet and the symmetry check.

    Attributes:
        name: Map name (dataset and model).
        version: Bumped whenever a target, offset or weight changes.
        landmarks: Site targets.
        orientations: Orientation targets.
        feet: ``(left, right)`` body tuples whose collision geoms are the soles.
        twist_pairs: ``(left, right)`` bodies of the axial-twist symmetry check.
        twist_reference: Body the axial twist is measured against (heading).
        coupling_term: Dependent joint whose polynomial value the twist asymmetry is
            regressed on (#651's measure).
        period_joint: Joint whose autocorrelation gives the stride period.
        posture: ``(joint, value, weight)`` priors on independent joints (metres per
            radian), for joints the targets leave open (the plane of elevation of a
            hanging arm); a source may give per-frame values (:class:`ClipTargets`).
    """

    name: str
    version: str
    landmarks: tuple[Landmark, ...]
    orientations: tuple[OrientationTarget, ...] = ()
    feet: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    twist_pairs: tuple[tuple[str, str], ...] = ()
    twist_reference: str = ""
    coupling_term: str = ""
    period_joint: str = ""
    posture: tuple[tuple[str, float, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (stored in every refitted clip)."""
        return asdict(self)


@dataclass(frozen=True)
class RefitConfig:
    """Solver options of :class:`LandmarkIK`.

    Attributes:
        smooth_weight: Weight of the deviation from the constant-velocity prediction
            ``q_prev + (q_prev - q_prev2)`` (metres per radian or metre).
        damping: Initial Levenberg-Marquardt damping (relative to ``diag(J^T J)``).
        max_iters: Gauss-Newton steps per frame.
        step_tol: Stop when the largest step is below this (rad or m).
    """

    smooth_weight: float = 0.03
    damping: float = 1e-3
    max_iters: int = 30
    step_tol: float = 1e-6


@dataclass(frozen=True)
class RefitThresholds:
    """Per-clip acceptance limits of :func:`validate_refit` (flagged when exceeded).

    Attributes:
        landmark_rmse_mm: Landmark RMSE over all frames and sites. The defaults suit
            SMPL+H on myofullbody: the model cannot follow SMPL's knee varus and twist,
            its fixed neck or its shoulder centres, so 1-3 cm are the norm.
        landmark_max_mm: Largest landmark error.
        coupling_residual: Largest distance of a coupled joint from its polynomial.
        limit_violations: Independent-joint samples outside their range.
        penetration_mm: Deepest foot penetration of the ground (z = 0).
        sliding_mps: Mean horizontal foot speed over the source's contact frames.
        accel_spike_floor: A velocity spike is a joint acceleration (second difference)
            above this (rad/s^2 or m/s^2).
        velocity_spikes: Spikes allowed (away from the joint limits).
        limit_stops: Spikes allowed where a joint runs into its limit.
    """

    landmark_rmse_mm: float = 40.0
    landmark_max_mm: float = 150.0
    coupling_residual: float = 1e-8
    limit_violations: int = 0
    penetration_mm: float = 5.0
    sliding_mps: float = 0.3
    accel_spike_floor: float = 500.0
    velocity_spikes: int = 0
    limit_stops: int = 0


# ----------------------------------------------------------------------------- clips


@dataclass
class ClipTargets:
    """A clip's landmark targets.

    Attributes:
        frequency: Frame rate (Hz).
        pos: ``(T, L, 3)`` landmark positions.
        quat: ``(T, O, 4)`` target orientations (wxyz).
        seed: ``(nq,)`` start pose of the first frame.
        contact: ``(T, 2)`` left/right foot contact of the source.
        posture: ``(T, P)`` per-frame values of the map's posture priors, or ``None``
            (the map's fixed values).
    """

    frequency: float
    pos: np.ndarray
    quat: np.ndarray
    seed: np.ndarray
    contact: np.ndarray
    posture: np.ndarray | None = None


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- solver


def _so3_log_quat(q: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a unit quaternion (wxyz)."""
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class LandmarkIK:
    """Landmark IK over the free root and the independent joints of a model.

    Args:
        model: Compiled model (any body; its active joint equalities are the couplings).
        lmap: Landmark map naming the model's bodies and sites.
        config: Solver options.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        lmap: LandmarkMap,
        config: RefitConfig | None = None,
    ) -> None:
        self.model, self.lmap = model, lmap
        self.config = config or RefitConfig()
        self.data = mujoco.MjData(model)
        m = model
        # landmark points: (body, local offset); model sites take precedence
        site_names = {m.site(i).name for i in range(m.nsite)}
        self.lm_body = np.empty(len(lmap.landmarks), np.int64)
        self.lm_off = np.empty((len(lmap.landmarks), 3))
        for i, lm in enumerate(lmap.landmarks):
            if lm.site is not None and lm.site in site_names:
                s = m.site(lm.site)
                self.lm_body[i], self.lm_off[i] = s.bodyid[0], s.pos
            else:
                self.lm_body[i], self.lm_off[i] = m.body(lm.body).id, lm.pos
        self.lm_w = np.array([lm.weight for lm in lmap.landmarks])
        self.or_body = np.array(
            [m.body(o.body).id for o in lmap.orientations], np.int64
        )
        self.or_w = np.array([o.weight for o in lmap.orientations])
        # anisotropic orientation weights: (target index, body-frame axis, 1 - factor)
        self.or_axes = [
            (k, np.asarray(o.axis, np.float64), 1.0 - o.axis_weight)
            for k, o in enumerate(lmap.orientations)
            if o.axis is not None and o.axis_weight != 1.0
        ]
        # reduced coordinates: every dof that no coupling or lock sets
        self.couplings = JointCouplings.from_model(m)
        c = self.couplings
        driven = np.zeros(m.nv, bool)
        driven[c.dep_dof] = driven[c.locked_dof] = True
        self.red_dof = np.flatnonzero(~driven)
        self.nr = len(self.red_dof)
        col = np.full(m.nv, -1)
        col[self.red_dof] = np.arange(self.nr)
        # box limits of the reduced hinge/slide joints (qpos address per column)
        self.lo = np.full(self.nr, -np.inf)
        self.hi = np.full(self.nr, np.inf)
        self.red_qadr = np.full(self.nr, -1)
        for j in range(m.njnt):
            if m.jnt_type[j] in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                k = col[m.jnt_dofadr[j]]
                if k >= 0:
                    self.red_qadr[k] = m.jnt_qposadr[j]
                    if m.jnt_limited[j]:
                        self.lo[k], self.hi[k] = m.jnt_range[j]
        self.bounded = np.flatnonzero(self.red_qadr >= 0)
        self.indep_qadr = self.red_qadr[self.bounded]
        self._jacp = np.empty((3, m.nv))
        self._jacr = np.empty((3, m.nv))
        # posture priors: (reduced column, qpos address, value, weight)
        self.post_col = np.array(
            [col[m.jnt_dofadr[m.joint(j).id]] for j, _, _ in lmap.posture], np.int64
        )
        self.post_qadr = np.array(
            [m.jnt_qposadr[m.joint(j).id] for j, _, _ in lmap.posture], np.int64
        )
        self.post_ref = np.array([v for _, v, _ in lmap.posture])
        self.post_w = np.array([w for _, _, w in lmap.posture])
        if (self.post_col < 0).any():
            raise ValueError("posture priors must be on independent joints")
        self.n_res = 3 * len(self.lm_body) + 3 * len(self.or_body) + len(self.post_w)

    # -- helpers
    def project(self, qpos: np.ndarray) -> np.ndarray:
        """``qpos`` with couplings on their polynomials and limits clamped (copy)."""
        q = np.array(qpos, np.float64)
        if len(self.bounded):
            a = self.indep_qadr
            q[..., a] = np.clip(q[..., a], self.lo[self.bounded], self.hi[self.bounded])
        project_joint_couplings(self.couplings, q)
        return q

    def _coupling_map(self, qpos: np.ndarray) -> np.ndarray:
        """``(nv, nr)`` derivative of the full dofs w.r.t. the reduced ones."""
        c, nv = self.couplings, self.model.nv
        C = np.zeros((nv, self.nr))
        C[self.red_dof, np.arange(self.nr)] = 1.0
        start = 0
        for end in c.level_ends:
            sl = slice(start, end)
            x = qpos[c.master_qpos[sl]] - c.master_ref[sl]
            coef = c.coef[sl]
            slope = coef[:, 1] + x * (
                2 * coef[:, 2] + x * (3 * coef[:, 3] + x * 4 * coef[:, 4])
            )
            C[c.dep_dof[sl]] = slope[:, None] * C[c.master_dof[sl]]
            start = end
        return C

    def landmarks(self, qpos: np.ndarray) -> np.ndarray:
        """``(L, 3)`` world landmark positions of a pose."""
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.lm_body].reshape(-1, 3, 3)
        return d.xpos[self.lm_body] + np.einsum("lij,lj->li", R, self.lm_off)

    def _evaluate(
        self, qpos: np.ndarray, pos: np.ndarray, quat: np.ndarray,
        prev: np.ndarray | None, jac: bool, posture: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:  # fmt: skip
        m, d = self.model, self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        if jac:
            mujoco.mj_comPos(m, d)
        R = d.xmat[self.lm_body].reshape(-1, 3, 3)
        p = d.xpos[self.lm_body] + np.einsum("lij,lj->li", R, self.lm_off)
        res = [((p - pos) * self.lm_w[:, None]).ravel()]
        qerr = np.empty(4)
        # per orientation target: world-frame weight matrix (axial component rescaled)
        W = self.or_w[:, None, None] * np.eye(3)
        for k, axis, drop in self.or_axes:
            a = d.xmat[self.or_body[k]].reshape(3, 3) @ axis
            W[k] -= self.or_w[k] * drop * np.outer(a, a)
        for k, b in enumerate(self.or_body):
            # world-frame rotation error R_model R_target^T
            neg = np.empty(4)
            mujoco.mju_negQuat(neg, quat[k])
            mujoco.mju_mulQuat(qerr, d.xquat[b], neg)
            res.append(W[k] @ _so3_log_quat(qerr))
        if len(self.post_w):
            ref = self.post_ref if posture is None else posture
            res.append(self.post_w * (qpos[self.post_qadr] - ref))
        if prev is not None:
            dq = np.empty(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, prev, qpos)
            res.append(self.config.smooth_weight * dq[self.red_dof])
        r = np.concatenate(res)
        if not jac:
            return r, None
        J = np.empty((self.n_res, m.nv))
        for i, b in enumerate(self.lm_body):
            mujoco.mj_jac(m, d, self._jacp, None, p[i], b)
            J[3 * i : 3 * i + 3] = self._jacp * self.lm_w[i]
        o = 3 * len(self.lm_body)
        for k, b in enumerate(self.or_body):
            mujoco.mj_jacBody(m, d, None, self._jacr, b)
            J[o + 3 * k : o + 3 * k + 3] = W[k] @ self._jacr
        Jr = J[: o + 3 * len(self.or_body)] @ self._coupling_map(qpos)
        if len(self.post_w):
            Jp = np.zeros((len(self.post_w), self.nr))
            Jp[np.arange(len(self.post_w)), self.post_col] = self.post_w
            Jr = np.vstack([Jr, Jp])
        if prev is not None:
            Jr = np.vstack([Jr, self.config.smooth_weight * np.eye(self.nr)])
        return r, Jr

    def _step(self, qpos: np.ndarray, delta: np.ndarray) -> np.ndarray:
        dq = np.zeros(self.model.nv)
        dq[self.red_dof] = delta
        # integrate the reduced dofs, then write the couplings exactly
        q = qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, 1.0)
        return self.project(q)

    def _box_step(self, H: np.ndarray, g: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Minimise ``0.5 d^T H d + g^T d`` inside the joint limits (active set)."""
        lo = np.full(self.nr, -np.inf)
        hi = np.full(self.nr, np.inf)
        b = self.bounded
        lo[b] = self.lo[b] - qpos[self.indep_qadr]
        hi[b] = self.hi[b] - qpos[self.indep_qadr]
        free = np.ones(self.nr, bool)
        # start with joints at a limit that the gradient pushes outward held
        free[(lo >= -1e-12) & (g > 0)] = False
        free[(hi <= 1e-12) & (g < 0)] = False
        delta = np.zeros(self.nr)
        for _ in range(self.nr + 1):
            fixed = ~free
            delta[fixed] = np.where(delta[fixed] <= lo[fixed], lo[fixed],
                                    np.where(delta[fixed] >= hi[fixed], hi[fixed], 0.0))  # fmt: skip
            rhs = -(g[free] + H[np.ix_(free, fixed)] @ delta[fixed])
            delta[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
            low, high = free & (delta < lo), free & (delta > hi)
            if low.any() or high.any():
                delta[low], delta[high] = lo[low], hi[high]
                free[low | high] = False
                continue
            # release held joints whose multiplier has the wrong sign
            grad = H @ delta + g
            release = fixed & (
                ((delta <= lo) & (grad < 0)) | ((delta >= hi) & (grad > 0))
            )
            release &= np.isfinite(lo) | np.isfinite(hi)
            if not release.any():
                break
            free[release] = True
        return delta

    def solve(
        self, seed: np.ndarray, pos: np.ndarray, quat: np.ndarray,
        prev: np.ndarray | None = None, posture: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:  # fmt: skip
        """Fit one frame.

        Args:
            seed: ``(nq,)`` start pose.
            pos: ``(L, 3)`` landmark targets.
            quat: ``(O, 4)`` orientation targets.
            prev: Predicted pose of the smoothing term (constant velocity from the
                previous frames), or ``None``.
            posture: ``(P,)`` values of the posture priors (``None``: the map's).

        Returns:
            ``(qpos, iterations)``.
        """
        cfg = self.config
        q = self.project(seed)
        r, J = self._evaluate(q, pos, quat, prev, True, posture)
        cost = r @ r
        lam = cfg.damping
        it = 0
        for it in range(1, cfg.max_iters + 1):
            H = J.T @ J
            g = J.T @ r
            Hd = H + lam * np.diag(np.diag(H)) + 1e-9 * np.eye(self.nr)
            delta = self._box_step(Hd, g, q)
            q_new = self._step(q, delta)
            r_new, J_new = self._evaluate(q_new, pos, quat, prev, True, posture)
            cost_new = r_new @ r_new
            if cost_new <= cost:
                q, r, J, cost = q_new, r_new, J_new, cost_new
                lam = max(lam / 3, 1e-7)
                if np.abs(delta).max() < cfg.step_tol:
                    break
            else:
                lam *= 5
                if lam > 1e6:
                    break
        return q, it

    def fit(
        self, pos: np.ndarray, quat: np.ndarray, seed: np.ndarray,
        posture: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        """Fit a clip frame by frame, warm-started from the previous frame.

        Args:
            pos: ``(T, L, 3)`` landmark targets.
            quat: ``(T, O, 4)`` orientation targets.
            seed: ``(nq,)`` start pose of the first frame (later frames start from
                the constant-velocity prediction).
            posture: ``(T, P)`` per-frame posture values (``None``: the map's).

        Returns:
            ``(qpos (T, nq), iterations (T,))``.
        """
        T = len(pos)
        out = np.empty((T, self.model.nq))
        iters = np.empty(T, np.int64)
        m = self.model
        vel = np.empty(m.nv)
        for t in range(T):
            if t == 0:
                pred = None
            elif t == 1:
                pred = out[0]
            else:  # constant-velocity prediction from the last two frames
                mujoco.mj_differentiatePos(m, vel, 1.0, out[t - 2], out[t - 1])
                pred = out[t - 1].copy()
                mujoco.mj_integratePos(m, pred, vel, 1.0)
            start = seed if pred is None else pred
            ref = None if posture is None else posture[t]
            out[t], iters[t] = self.solve(start, pos[t], quat[t], pred, ref)
        return out, iters


# ------------------------------------------------------------------------- validation


def _orientation_errors(
    ik: LandmarkIK, qpos: np.ndarray, quat: np.ndarray
) -> np.ndarray:
    """``(T, O)`` orientation errors (rad)."""
    m, d = ik.model, ik.data
    out = np.empty((len(qpos), len(ik.or_body)))
    qerr, neg = np.empty(4), np.empty(4)
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for k, b in enumerate(ik.or_body):
            mujoco.mju_negQuat(neg, quat[t, k])
            mujoco.mju_mulQuat(qerr, d.xquat[b], neg)
            out[t, k] = np.linalg.norm(_so3_log_quat(qerr))
    return out


def _geom_lowest_z(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    """Lowest world z of a collision geom (sphere, capsule, cylinder, ellipsoid, box, mesh)."""
    t, s = m.geom_type[g], m.geom_size[g]
    R, z = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g, 2]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        return z - s[0]
    if t == G.mjGEOM_CAPSULE:
        return z - abs(R[2, 2]) * s[1] - s[0]
    if t == G.mjGEOM_CYLINDER:
        return z - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0.0, 1 - R[2, 2] ** 2))
    if t == G.mjGEOM_ELLIPSOID:
        return z - np.sqrt(((R[2] * s) ** 2).sum())
    if t == G.mjGEOM_BOX:
        return z - (np.abs(R[2]) * s).sum()
    if t == G.mjGEOM_MESH:
        a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
        return z + (m.mesh_vert[a : a + n] @ R[2]).min()
    return z


def _foot_kinematics(
    m: mujoco.MjModel, lmap: LandmarkMap, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 2)`` lowest sole z per foot and ``(T, 2, 3)`` foot origin (mean of the bodies)."""
    d = mujoco.MjData(m)
    feet = [[m.body(b).id for b in side] for side in lmap.feet]
    geoms = [
        [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids
         and (m.geom_contype[g] or m.geom_conaffinity[g])]
        for ids in feet
    ]  # fmt: skip
    low = np.empty((len(qpos), 2))
    org = np.empty((len(qpos), 2, 3))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for s in range(2):
            low[t, s] = min(_geom_lowest_z(m, d, g) for g in geoms[s])
            org[t, s] = d.xpos[feet[s]].mean(0)
    return low, org


def _sliding(low: np.ndarray, org: np.ndarray, contact: np.ndarray, dt: float) -> float:
    """Mean horizontal foot speed (m/s) over the contact frames."""
    v = np.linalg.norm(np.gradient(org[..., :2], dt, axis=0), axis=-1)
    return float(v[contact].mean()) if contact.any() else 0.0


def _yaw(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _stride_period(x: np.ndarray, dt: float) -> int | None:
    """Frames of one stride from the autocorrelation of ``x`` (0.5 - 2 s), or ``None``."""
    x = x - x.mean()
    lo, hi = int(0.5 / dt), min(int(2.0 / dt), len(x) // 2)
    if hi <= lo:
        return None
    ac = np.array([(x[:-k] * x[k:]).mean() for k in range(lo, hi)])
    k = int(np.argmax(ac))
    return lo + k if ac[k] > 0.3 * (x * x).mean() else None


def twist_asymmetry(
    m: mujoco.MjModel, lmap: LandmarkMap, qpos: np.ndarray, dt: float
) -> dict[str, Any] | None:
    """Left/right axial-twist asymmetry against the knee coupling term (#651's measure).

    Each segment's internal twist is its heading relative to the reference body (left
    mirrored). Left at ``t`` and right half a stride later are compared, and their
    difference is regressed on the left coupling term (the model's polynomial value of
    ``lmap.coupling_term``). A symmetric gait has slopes near zero.

    Returns:
        ``{"period_s", "slope": {left body: slope}, "x", "y": {...}}`` (the samples, for
        pooling), or ``None`` without a stride.
    """
    if not lmap.twist_pairs:
        return None
    jp = m.jnt_qposadr[m.joint(lmap.period_joint).id]
    period = _stride_period(qpos[:, jp], dt)
    if period is None:
        return None
    half = period // 2
    d = mujoco.MjData(m)
    names = [b for pair in lmap.twist_pairs for b in pair]
    ids = [m.body(b).id for b in names]
    ref = m.body(lmap.twist_reference).id
    yaw = np.empty((len(qpos), len(ids)))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R = d.xmat[ids].reshape(-1, 3, 3)
        yaw[t] = _yaw(R) - _yaw(d.xmat[ref].reshape(3, 3))
    yaw = np.angle(np.exp(1j * yaw))
    proj = np.array(qpos, np.float64)
    project_joint_couplings(JointCouplings.from_model(m), proj)
    x = proj[:-half, m.jnt_qposadr[m.joint(lmap.coupling_term).id]]
    x = x - m.qpos0[m.jnt_qposadr[m.joint(lmap.coupling_term).id]]
    out: dict[str, Any] = {"period_s": period * dt, "slope": {}, "x": x, "y": {}}
    for i, (left, _right) in enumerate(lmap.twist_pairs):
        y = -yaw[:-half, 2 * i] - yaw[half:, 2 * i + 1]
        y = np.angle(np.exp(1j * y))
        A = np.stack([x, np.ones_like(x)], 1)
        out["slope"][left] = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
        out["y"][left] = y
    return out


def validate_refit(
    ik: LandmarkIK,
    targets: ClipTargets,
    qpos: np.ndarray,
    thresholds: RefitThresholds | None = None,
) -> dict[str, Any]:
    """Per-clip checks of a fit against its targets.

    Args:
        ik: The solver (model and landmark map) the fit used.
        targets: The clip's targets and source contacts.
        qpos: ``(T, nq)`` fitted poses.
        thresholds: Limits; a failed check is listed in ``"flags"``.

    Returns:
        JSON-serialisable metrics and ``"flags"`` (empty when the clip passes).
    """
    th = thresholds or RefitThresholds()
    m, lmap = ik.model, ik.lmap
    dt = 1.0 / targets.frequency
    err = np.stack([np.linalg.norm(ik.landmarks(q) - p, axis=-1)
                    for q, p in zip(qpos, targets.pos)]) * 1e3  # fmt: skip
    ori = np.degrees(_orientation_errors(ik, qpos, targets.quat))
    c = ik.couplings
    proj = qpos.copy()
    project_joint_couplings(c, proj)
    coupling = float(np.abs(proj - qpos).max())
    a = ik.indep_qadr
    lo, hi = ik.lo[ik.bounded], ik.hi[ik.bounded]
    viol = int(((qpos[:, a] < lo - 1e-9) | (qpos[:, a] > hi + 1e-9)).sum())
    # dependent joints the polynomials carry outside their own range (informational)
    dep_j = [int(np.flatnonzero(m.jnt_qposadr == q)[0]) for q in c.dep_qpos]
    dep_lim = np.array([m.jnt_limited[j] for j in dep_j], bool)
    dep_lo, dep_hi = m.jnt_range[dep_j].T
    dq = qpos[:, c.dep_qpos]
    dep_viol = int((dep_lim & ((dq < dep_lo - 1e-6) | (dq > dep_hi + 1e-6))).sum())
    # feet: penetration of the ground plane (z = 0), sliding and floating in contact
    low, org = _foot_kinematics(m, lmap, qpos)
    contact = np.asarray(targets.contact, bool)
    pen = float(max(0.0, -low.min()) * 1e3)
    slide = _sliding(low, org, contact, dt)
    float_mm = np.where(contact, np.maximum(low, 0.0), np.nan) * 1e3
    # velocity spikes: joint accelerations above the floor, off and at the limits
    v = np.gradient(qpos[:, a], dt, axis=0)
    acc = np.abs(np.diff(qpos[:, a], 2, axis=0)) / dt**2
    at = (qpos[:, a] <= lo + 1e-9) | (qpos[:, a] >= hi - 1e-9)
    at_limit = at[:-2] | at[1:-1] | at[2:]  # frames of each second difference
    spikes = int(((acc > th.accel_spike_floor) & ~at_limit).sum())
    stops = int(((acc > th.accel_spike_floor) & at_limit).sum())
    out: dict[str, Any] = {
        "frames": len(qpos),
        "landmark_rmse_mm": float(np.sqrt((err**2).mean())),
        "landmark_max_mm": float(err.max()),
        "landmark_max_by_site_mm": {
            lm.source: float(e) for lm, e in zip(lmap.landmarks, err.max(0))
        },
        "orientation_mean_deg": float(ori.mean()),
        "orientation_max_deg": float(ori.max()),
        "coupling_residual": coupling,
        "limit_violations": viol,
        "dependent_limit_violations": dep_viol,
        "penetration_mm": pen,
        "sliding_mps": slide,
        "contact_float_mm_median": float(np.nanmedian(float_mm))
        if contact.any()
        else 0.0,
        "contact_float_mm_p95": float(np.nanpercentile(float_mm, 95))
        if contact.any()
        else 0.0,
        "joint_speed_max_rps": float(np.abs(v).max()),
        "velocity_spikes": spikes,
        "limit_stops": stops,
    }
    checks = {
        "landmark_rmse": out["landmark_rmse_mm"] > th.landmark_rmse_mm,
        "landmark_max": out["landmark_max_mm"] > th.landmark_max_mm,
        "coupling_residual": coupling > th.coupling_residual,
        "limit_violations": viol > th.limit_violations,
        "penetration": pen > th.penetration_mm,
        "sliding": slide > th.sliding_mps,
        "velocity_spikes": spikes > th.velocity_spikes,
        "limit_stops": stops > th.limit_stops,
    }
    flags = [k for k, bad in checks.items() if bad]
    out["flags"] = flags
    return out


# ----------------------------------------------------------------------- sources


class ClipSource(Protocol):
    """Where a fit's targets come from (e.g. :class:`.retarget_smplh.SMPLHSource`).

    Attributes:
        name: Source format tag stored in the metadata.
    """

    name: str

    def targets(
        self, path: str | Path, model: mujoco.MjModel, lmap: LandmarkMap
    ) -> ClipTargets:
        """The clip's targets on *model*."""
        ...

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Post-process a fit (e.g. ground height); returns qpos, targets and metadata."""
        ...


# ----------------------------------------------------------------------- output files


def model_tag(model_name: str, version: str) -> str:
    """Cache folder of a model version: ``<model>-<version>``."""
    return f"{model_name}-{version}"


def default_cache_root(model_name: str, version: str) -> Path:
    """``~/.cache/myosuite/mimic/<model>-<version>`` (the refitted clips' root)."""
    return (
        Path.home() / ".cache" / "myosuite" / "mimic" / model_tag(model_name, version)
    )


def find_cache_root(
    model_name: str, version: str, fingerprint: str, probe: str
) -> tuple[Path | None, list[Path]]:
    """The cache folder holding clips that fit the installed body.

    Folders keep the ``<model>-<version>`` name they were fitted under. A release of
    the model package that leaves the body's kinematics alone (same
    :func:`kinematic_fingerprint`) keeps its clips valid, so when the current version's
    folder lacks *probe*, the sibling ``<model>-*`` folders are scanned (sorted) for
    one whose *probe* clip carries *fingerprint* in its metadata.

    Args:
        model_name: Model name.
        version: The installed model version (the myo_sim version).
        fingerprint: The installed body's kinematic fingerprint.
        probe: A clip path relative to the root (e.g. the first listed clip).

    Returns:
        ``(root, tried)``: the folder (``None`` if none fits) and every folder looked at.
    """
    cache = default_cache_root(model_name, version)
    tried = [cache]
    if (cache / probe).is_file():
        return cache, tried
    for sib in sorted(cache.parent.glob(f"{model_name}-*")):
        if sib == cache or not sib.is_dir():
            continue
        tried.append(sib)
        clip = sib / probe
        if clip.is_file():
            meta = read_refit_meta(clip) or {}
            if (meta.get("model"), meta.get("model_fingerprint")) == (
                model_name, fingerprint,
            ):  # fmt: skip
                return sib, tried
    return None, tried


def save_refit(
    path: str | Path, model: mujoco.MjModel, qpos: np.ndarray, frequency: float,
    meta: dict[str, Any],
) -> None:  # fmt: skip
    """Write a refitted clip in the MuscleMimic trajectory layout the motion library reads.

    Args:
        path: Output ``.npz``.
        model: The model the clip is fitted on (joint names and types).
        qpos: ``(T, nq)`` poses.
        frequency: Frame rate (Hz).
        meta: Metadata stored as JSON under :data:`REFIT_META_KEY`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        qpos=qpos.astype(np.float64),
        joint_names=np.array([model.joint(j).name for j in range(model.njnt)]),
        jnt_type=model.jnt_type.astype(np.int32),
        frequency=np.float64(frequency),
        split_points=np.array([0, len(qpos)], np.int32),
        **{REFIT_META_KEY: np.array(json.dumps(meta, sort_keys=True))},
    )
    tmp.replace(path)


def read_refit_meta(path: str | Path) -> dict[str, Any] | None:
    """A clip's refit metadata, or ``None`` for a clip that is not a refit."""
    with np.load(path, allow_pickle=False) as d:
        if REFIT_META_KEY not in d.files:
            return None
        return json.loads(str(d[REFIT_META_KEY]))


def check_refit_meta(
    path: str | Path, model_name: str, version: str, fingerprint: str | None = None
) -> None:
    """Refuse a clip without refit metadata, refitted on another model, or on a body
    whose kinematics differ.

    A clip carrying ``model_fingerprint`` is accepted when it equals *fingerprint*,
    whatever version it was fitted under. A clip without one (fitted before the
    fingerprint was recorded), or a check without *fingerprint*, falls back to the
    version. Either mismatch names the myo_sim version the clip was fitted for.

    Args:
        path: Clip ``.npz``.
        model_name: The env's model name.
        version: The env's model version (the installed myo_sim version).
        fingerprint: The env body's :func:`kinematic_fingerprint`.

    Raises:
        ValueError: On missing metadata or a mismatch.
    """
    meta = read_refit_meta(path)
    if meta is None:
        raise ValueError(f"{path}: not a refitted clip (no {REFIT_META_KEY})")
    got, got_v = meta.get("model"), meta.get("model_version")
    if got != model_name:
        raise ValueError(
            f"{path}: refitted on {got} {got_v}, the env runs {model_name} {version}"
        )
    got_fp = meta.get("model_fingerprint")
    script = "scripts/mimic/retarget_amass.py"
    if fingerprint is not None and got_fp is not None:
        if got_fp == fingerprint:
            return
        why = (f"its body (kinematic fingerprint {got_fp}) differs from the installed "
               f"one ({fingerprint})")  # fmt: skip
        fix = f"retarget the clips again ({script})"
    elif got_v == version:
        return
    else:
        why = "the clip carries no kinematic fingerprint to show that the body is unchanged"
        fix = (f"with myo_sim {got_v} installed, record the fingerprint "
               f"({script} --stamp-fingerprint <clip folder>)")  # fmt: skip
    raise ValueError(
        f"{path}: refitted on {model_name} from myo_sim {got_v}, but myo_sim {version} "
        f"is installed and {why}. Install myo_sim=={got_v}, or {fix}."
    )


def kinematic_fingerprint(
    m: mujoco.MjModel,
    landmark_bodies: Sequence[str],
    feet: Sequence[Sequence[str]],
) -> str:
    """Hash of everything that makes a refitted clip valid for a compiled body.

    A clip stores qpos fitted to the body's geometry, so the hash covers:

    * joints, in qpos order: name, type, ``jnt_qposadr``, body, anchor (``jnt_pos``),
      axis, limited flag and range; and ``qpos0``;
    * the body tree: name, parent and ``pos``/``quat`` offset of every body;
    * the coupling equalities: every active ``mjEQ_JOINT`` with its two joint names and
      polynomial coefficients;
    * the landmark bodies (the retargeting targets' carriers): world position and
      orientation in the default pose;
    * the foot collision geoms (the soles the ground height is fitted to): name, body,
      type, size, pos and quat.

    Dynamics that leave a clip's meaning alone (masses, inertia, muscles, tendons,
    actuators, contact solver parameters, colours) are left out, as are the source's
    inputs (recorded separately, e.g. ``smplh_map_version``). Floats are rounded to
    1e-6 (integers of micro-units), so MuJoCo versions and platforms agree.

    Args:
        m: The compiled body.
        landmark_bodies: Bodies carrying the landmark and orientation targets.
        feet: Body groups whose collision geoms are the soles.

    Returns:
        The first 16 hex digits of the SHA-256 of a sorted-key JSON serialisation.
    """

    def r(x: Any) -> list[int]:
        return (
            np.rint(np.asarray(x, np.float64) * 1e6).astype(np.int64).ravel().tolist()
        )

    def name(obj: mujoco.mjtObj, i: int) -> str | None:
        return mujoco.mj_id2name(m, obj, int(i)) if i >= 0 else None

    J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
    joints = sorted(range(m.njnt), key=lambda j: int(m.jnt_qposadr[j]))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    foot_ids = {m.body(b).id for side in feet for b in side}
    doc = {
        "joints": [
            [name(J, j), int(m.jnt_type[j]), int(m.jnt_qposadr[j]),
             name(B, m.jnt_bodyid[j]), r(m.jnt_pos[j]), r(m.jnt_axis[j]),
             int(m.jnt_limited[j]), r(m.jnt_range[j])]
            for j in joints
        ],
        "qpos0": r(m.qpos0),
        "bodies": [
            [name(B, b), int(m.body_parentid[b]), r(m.body_pos[b]), r(m.body_quat[b])]
            for b in range(m.nbody)
        ],
        "couplings": [
            [name(J, m.eq_obj1id[e]), name(J, m.eq_obj2id[e]), r(m.eq_data[e, :5])]
            for e in range(m.neq)
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]
        ],
        "landmarks": [
            [b, r(d.xpos[m.body(b).id]), r(d.xquat[m.body(b).id])]
            for b in sorted(set(landmark_bodies))
        ],
        "feet": [
            [m.geom(g).name, name(B, m.geom_bodyid[g]), int(m.geom_type[g]),
             r(m.geom_size[g]), r(m.geom_pos[g]), r(m.geom_quat[g])]
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in foot_ids
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ],
    }  # fmt: skip
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stamp_fingerprint(
    path: str | Path, model_name: str, version: str, fingerprint: str
) -> bool:
    """Record *fingerprint* in a clip's metadata without refitting it.

    Only a clip refitted on *model_name* at *version* (the installed one, whose body
    *fingerprint* describes) is stamped; every array is rewritten unchanged and the
    qpos is checked byte for byte before the file is replaced.

    Args:
        path: Clip ``.npz``.
        model_name: Model name.
        version: The installed model version (the myo_sim version).
        fingerprint: The installed body's :func:`kinematic_fingerprint`.

    Returns:
        ``True`` if the clip was stamped, ``False`` if it already carried *fingerprint*.

    Raises:
        ValueError: For a clip without refit metadata, of another model or version, or
            carrying another fingerprint.
    """
    path = Path(path)
    meta = read_refit_meta(path)
    if meta is None:
        raise ValueError(f"{path}: not a refitted clip (no {REFIT_META_KEY})")
    if (meta.get("model"), meta.get("model_version")) != (model_name, version):
        raise ValueError(
            f"{path}: refitted on {meta.get('model')} {meta.get('model_version')}, "
            f"not on the installed {model_name} {version}"
        )
    if "model_fingerprint" in meta:
        if meta["model_fingerprint"] == fingerprint:
            return False
        raise ValueError(
            f"{path}: already carries fingerprint {meta['model_fingerprint']}"
        )
    meta["model_fingerprint"] = fingerprint
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    arrays[REFIT_META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp, allow_pickle=False) as d:
        q = d["qpos"]
        same = q.dtype == arrays["qpos"].dtype and q.shape == arrays["qpos"].shape
        if not (same and q.tobytes() == arrays["qpos"].tobytes()):
            tmp.unlink()
            raise RuntimeError(f"{path}: qpos changed on rewrite; left untouched")
    tmp.replace(path)
    return True


def refit_clip(
    src: str | Path,
    model: mujoco.MjModel,
    model_name: str,
    model_version: str,
    lmap: LandmarkMap,
    source: ClipSource,
    config: RefitConfig | None = None,
    thresholds: RefitThresholds | None = None,
    ik: LandmarkIK | None = None,
    model_fingerprint: str | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit one clip onto the model and validate it.

    Args:
        src: Source clip.
        model: Target model.
        model_name: Its name (metadata).
        model_version: Its version (metadata, e.g. the myo_sim version).
        lmap: Landmark map.
        source: Target source.
        config: Solver options.
        thresholds: Validation limits.
        ik: A solver to reuse (built from *model*, *lmap*, *config* otherwise).
        model_fingerprint: The body's :func:`kinematic_fingerprint` (metadata).

    Returns:
        ``(qpos, frequency, meta)``; ``meta`` holds the source, its hash, the model
        (name, version and fingerprint),
        the landmark map version, solver stats and :func:`validate_refit`'s metrics.
    """
    import time  # noqa: PLC0415

    ik = ik or LandmarkIK(model, lmap, config)
    targets = source.targets(src, model, lmap)
    t0 = time.perf_counter()
    qpos, iters = ik.fit(targets.pos, targets.quat, targets.seed, targets.posture)
    qpos, targets, extra = source.finish(ik, targets, qpos)
    seconds = time.perf_counter() - t0
    metrics = validate_refit(ik, targets, qpos, thresholds)
    meta = {
        "source": str(src),
        "source_format": source.name,
        "source_sha256": file_sha256(src),
        "model": model_name,
        "model_version": model_version,
        **({"model_fingerprint": model_fingerprint} if model_fingerprint else {}),
        "landmark_map": lmap.name,
        "landmark_map_version": lmap.version,
        "solver": asdict(ik.config),
        "fit_seconds": seconds,
        "iterations_mean": float(iters.mean()),
        "iterations_max": int(iters.max()),
        "metrics": metrics,
        **extra,
    }
    return qpos, targets.frequency, meta


def summarize(metas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-clip metadata: medians, worst clips and the flagged list."""
    ms = [x["metrics"] for x in metas]

    def col(k: str) -> np.ndarray:
        return np.array([x[k] for x in ms], np.float64)

    keys = ("landmark_rmse_mm", "landmark_max_mm", "orientation_mean_deg",
            "coupling_residual", "limit_violations", "penetration_mm", "sliding_mps",
            "contact_float_mm_median", "contact_float_mm_p95", "velocity_spikes",
            "limit_stops", "joint_speed_max_rps")  # fmt: skip
    stats = {
        k: {"median": float(np.median(col(k))), "p95": float(np.percentile(col(k), 95)),
            "max": float(col(k).max())}
        for k in keys
    }  # fmt: skip
    frames = sum(x["frames"] for x in ms)
    flagged = [
        {"source": x["source"], "flags": x["metrics"]["flags"]}
        for x in metas if x["metrics"]["flags"]
    ]  # fmt: skip
    return {
        "clips": len(metas),
        "frames": frames,
        "fit_ms_per_frame": 1e3 * sum(x["fit_seconds"] for x in metas) / max(frames, 1),
        "stats": stats,
        "flagged": flagged,
    }


__all__ = [
    "REFIT_META_KEY",
    "ClipSource",
    "ClipTargets",
    "Landmark",
    "LandmarkIK",
    "LandmarkMap",
    "OrientationTarget",
    "RefitConfig",
    "RefitThresholds",
    "check_refit_meta",
    "default_cache_root",
    "file_sha256",
    "find_cache_root",
    "kinematic_fingerprint",
    "model_tag",
    "read_refit_meta",
    "refit_clip",
    "save_refit",
    "stamp_fingerprint",
    "summarize",
    "twist_asymmetry",
    "validate_refit",
]
