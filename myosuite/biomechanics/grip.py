# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""A hand holding a cylindrical handle: the handle geom and the grip posture.

:class:`GripHandle` adds a handle (a cylinder on a mocap body) to a model spec.
:func:`solve_handle_grip` closes the hand around it: finger and thumb angles and
the handle's place in the hand such that every listed body touches the handle,
no hand geom enters it and neighbouring fingers do not pass through each other.
The static optimization then takes the hand-handle contact forces as unknowns
(:class:`~myosuite.biomechanics.static_optimization.GripContactConfig`)::

    GripHandle(radius=0.0175).add_to(spec)
    model = spec.compile()
    grip = solve_handle_grip(model, qpos, hand_body="thirdmc_r", joints=..., contact_bodies=..., axis=...)
    qpos = grip.apply(model, qpos)
    cfg = StaticOptimizationConfig(grip=grip.contact_config(loaded_joint="flexion_r"))
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myosuite.biomechanics.emg import project_tendon_nodes
from myosuite.biomechanics.static_optimization import GripContactConfig
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

# The handle waits here until a solver moves it to the hand, clear of any body.
_PARKED_POS = (0.0, 0.0, 10.0)


@dataclass(frozen=True)
class GripHandle:
    """A cylindrical handle on a mocap body, held in the hand.

    Attributes:
        radius: Handle radius [m].
        half_length: Half of the handle length [m].
        friction: Sliding friction of skin on the handle. The handle's geom
            priority makes it the contact's friction, whatever the hand geoms say.
        name: Name of the handle body and of its geom.
    """

    radius: float = 0.0175
    half_length: float = 0.07
    friction: float = 1.0
    name: str = "grip_handle"

    def add_to(self, spec: mujoco.MjSpec) -> None:
        """Add the handle to *spec* as a mocap body, parked away from the model.

        The geom only collides with geoms whose ``contype`` has bit 0 set (the
        hand's collision geoms), not with other static geoms.
        """
        body = spec.worldbody.add_body(name=self.name, mocap=True, pos=_PARKED_POS)
        body.add_geom(
            name=self.name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.radius, self.half_length, 0.0],
            contype=0,
            conaffinity=1,
            condim=3,
            priority=1,
            friction=[self.friction, 0.005, 0.0001],
            rgba=[0.55, 0.55, 0.6, 1.0],
        )


@dataclass(frozen=True)
class HandleGrip:
    """A solved grip: hand joint angles and the handle's pose in the hand.

    Attributes:
        handle_body: Name of the handle's mocap body.
        hand_body: Body whose frame holds the handle pose.
        joint_qpos: ``(name, angle)`` of every solved hand joint [rad].
        handle_pos: Handle centre in the hand body's frame [m].
        handle_quat: Handle orientation in the hand body's frame (w, x, y, z).
        gaps: ``(body, gap)`` of each contact body: its closest geom's distance
            to the handle [m], negative inside it.
        max_penetration: Deepest overlap of a hand geom with the handle or of
            two neighbouring fingers [m] (zero when clear).
    """

    handle_body: str
    hand_body: str
    joint_qpos: tuple[tuple[str, float], ...]
    handle_pos: tuple[float, float, float]
    handle_quat: tuple[float, float, float, float]
    gaps: tuple[tuple[str, float], ...]
    max_penetration: float

    def apply(self, model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
        """*qpos* ``(nq,)`` or ``(T, nq)`` with the grip's joint angles, couplings projected."""
        out = np.array(qpos, dtype=np.float64, copy=True)
        for name, value in self.joint_qpos:
            out[..., model.jnt_qposadr[model.joint(name).id]] = value
        project_joint_couplings(JointCouplings.from_model(model), out)
        return project_tendon_nodes(model, out)

    def contact_config(
        self, loaded_joint: str | None = None, **kwargs: float
    ) -> GripContactConfig:
        """Static-optimization settings for this grip (see :class:`GripContactConfig`)."""
        return GripContactConfig(
            handle_body=self.handle_body,
            hand_body=self.hand_body,
            handle_pos=self.handle_pos,
            handle_quat=self.handle_quat,
            loaded_joint=loaded_joint,
            **kwargs,
        )


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """Bodies in the subtree of *root*, *root* included."""
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


def _related(model: mujoco.MjModel, b1: int, b2: int) -> bool:
    """Whether one body is the other or an ancestor of it."""
    return b1 == b2 or b1 in _subtree(model, b2) or b2 in _subtree(model, b1)


def solve_handle_grip(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    hand_body: str,
    joints: Sequence[str],
    contact_bodies: Sequence[str],
    axis: np.ndarray,
    handle_body: str = "grip_handle",
    ranges: Mapping[str, tuple[float, float]] | None = None,
    starts: int = 8,
    seed: int = 0,
    max_nfev: int = 300,
) -> HandleGrip:
    """Close the hand around the handle, solved on the hand alone.

    Unknowns are the hinge *joints* (dependents of joint couplings follow their
    masters) and the handle centre; the handle axis stays along *axis*. A least
    squares fit, from several starts, puts every *contact_bodies* body on the
    handle surface (its closest geom), keeps every other collidable hand geom
    out of the handle and neighbouring fingers out of each other, and keeps the
    handle centred on the contact bodies along its axis. Joint ranges bound the
    angles.

    Args:
        model: Compiled model with a :class:`GripHandle`; it is not modified.
        qpos: ``(nq,)`` pose of the rest of the body. Flexion joints start (and
            are weakly held) half way to their flexion limit, the others at *qpos*.
        hand_body: Body whose frame holds the returned handle pose.
        joints: Hand joints to solve.
        contact_bodies: Bodies that must touch the handle (phalanges, palm).
        axis: World direction of the handle axis at *qpos*.
        handle_body: Name of the handle's mocap body.
        ranges: Optional narrower ranges of some joints [rad], e.g. finger
            flexion between neutral and its limit (no hyperextension).
        starts: Number of starts (the first two unperturbed).
        seed: Seed of the perturbed starts.
        max_nfev: Function evaluations per start.

    Returns:
        The best grip of all starts.
    """
    m = copy.deepcopy(model)
    d = mujoco.MjData(m)
    handle = m.body(handle_body).id
    mocap = int(m.body_mocapid[handle])
    if mocap < 0:
        raise ValueError(f"body {handle_body!r} is not a mocap body")
    hgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == handle]
    couplings = JointCouplings.from_model(m)
    dependent = {int(a) for a in np.asarray(couplings.dep_qpos)}
    jids = [m.joint(n).id for n in joints]
    jids = [
        j
        for j in jids
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and int(m.jnt_qposadr[j]) not in dependent
    ]
    adr = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
    lo = np.where(m.jnt_limited[jids], m.jnt_range[jids, 0], -np.pi)
    hi = np.where(m.jnt_limited[jids], m.jnt_range[jids, 1], np.pi)
    for k, j in enumerate(jids):
        if ranges and m.joint(j).name in ranges:
            r0, r1 = ranges[m.joint(j).name]
            lo[k], hi[k] = max(lo[k], r0), min(hi[k], r1)
    q0 = np.asarray(qpos, dtype=np.float64).copy()
    # Start with every flexion joint half way to its flexion limit.
    flex = np.array(["flex" in m.joint(j).name for j in jids], dtype=bool)
    half = np.where((lo < 0.0) & (hi > 0.0), 0.5 * hi, 0.5 * (lo + hi))
    guess = np.clip(np.where(flex, half, q0[adr]), lo, hi)
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, axis)

    def geoms_of(bodies: set[int]) -> list[int]:
        return [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in bodies
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]

    contact = [m.body(n).id for n in contact_bodies]
    by_body = {b: geoms_of({b}) for b in contact}
    hand = set()
    for j in jids:
        hand |= _subtree(m, int(m.jnt_bodyid[j]))
    hand |= set(contact)
    hand_geoms = geoms_of(hand)

    def dist(g1: int, g2: int, cap: float = 0.05) -> float:
        return float(mujoco.mj_geomDistance(m, d, g1, g2, cap, np.zeros(6)))

    def place(x: np.ndarray) -> None:
        q = q0.copy()
        q[adr] = x[: len(adr)]
        project_joint_couplings(couplings, q)
        d.qpos[:] = project_tendon_nodes(m, q)
        d.mocap_pos[mocap] = x[len(adr) :]
        d.mocap_quat[mocap] = quat
        mujoco.mj_kinematics(m, d)

    # Finger pairs that may collide: different, unrelated bodies, clear of each
    # other at the guess (neighbouring palm skins may overlap by design).
    place(np.r_[guess, np.zeros(3)])
    pairs = [
        (g1, g2)
        for i, g1 in enumerate(hand_geoms)
        for g2 in hand_geoms[i + 1 :]
        if not _related(m, int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2]))
        and dist(g1, g2) >= 0.0
    ]
    # The handle starts at the contact bodies' centre, and one radius beyond
    # their geoms' centre on the side flexion moves them to (the palmar side).
    contact_geoms = [g for b in contact for g in by_body[b]]
    centres = [np.mean(d.xpos[contact], axis=0)]
    bent = np.mean(d.geom_xpos[contact_geoms], axis=0)
    place(np.r_[np.clip(0.0, lo, hi), np.zeros(3)])
    palmar = bent - np.mean(d.geom_xpos[contact_geoms], axis=0)
    palmar -= (palmar @ axis) * axis
    if np.linalg.norm(palmar) > 0.0:
        radius = m.geom_size[hgeoms[0], 0]
        centres.append(bent + radius * palmar / np.linalg.norm(palmar))

    def residual(x: np.ndarray) -> np.ndarray:
        place(x)
        h = hgeoms[0]
        touch = [min(dist(g, h) for g in by_body[b]) for b in contact]
        clear = [min(0.0, dist(g, h)) for g in hand_geoms]
        apart = [min(0.0, dist(g1, g2, 0.01)) for g1, g2 in pairs]
        centre = np.mean([d.xpos[b] for b in contact], axis=0)
        along = axis @ (x[len(adr) :] - centre)
        return np.concatenate(
            [
                300.0 * np.asarray(touch),
                300.0 * np.asarray(clear),
                300.0 * np.asarray(apart),
                [10.0 * along],
                0.3 * (x[: len(adr)] - guess),
            ]
        )

    rng = np.random.default_rng(seed)
    bounds = (np.r_[lo, np.full(3, -np.inf)], np.r_[hi, np.full(3, np.inf)])
    best = None
    for k in range(starts):
        x0 = np.r_[guess, centres[k % len(centres)]]
        if k >= len(centres):
            x0 = (
                x0
                + np.r_[rng.uniform(-0.3, 0.3, len(adr)), rng.uniform(-0.02, 0.02, 3)]
            )
        x0[: len(adr)] = np.clip(x0[: len(adr)], lo + 1e-9, hi - 1e-9)
        sol = least_squares(
            residual, x0, bounds=bounds, max_nfev=max_nfev, diff_step=1e-6
        )
        if best is None or sol.cost < best.cost:
            best = sol
    place(best.x)
    h = hgeoms[0]
    gaps = tuple((m.body(b).name, min(dist(g, h) for g in by_body[b])) for b in contact)
    overlap = [dist(g, h) for g in hand_geoms] + [dist(a, b) for a, b in pairs]
    hb = m.body(hand_body).id
    rot = d.xmat[hb].reshape(3, 3)
    pos = rot.T @ (d.mocap_pos[mocap] - d.xpos[hb])
    inv, rel = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(inv, d.xquat[hb])
    mujoco.mju_mulQuat(rel, inv, quat)
    hand_joints = sorted(
        {m.joint(j).name for j in range(m.njnt) if m.jnt_bodyid[j] in hand}
    )
    return HandleGrip(
        handle_body=handle_body,
        hand_body=hand_body,
        joint_qpos=tuple(
            (n, float(d.qpos[m.jnt_qposadr[m.joint(n).id]])) for n in hand_joints
        ),
        handle_pos=tuple(float(v) for v in pos),
        handle_quat=tuple(float(v) for v in rel),
        gaps=gaps,
        max_penetration=float(max(0.0, -min(overlap, default=0.0))),
    )
