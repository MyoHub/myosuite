# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Map measured wrenches and gravity loads to MuJoCo generalized forces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import mujoco
import numpy as np


class ForcePlateWrench(Protocol):
    """Frame-aligned force-plate wrench in world SI units (N, m, N m)."""

    force: np.ndarray
    cop: np.ndarray
    free_torque: np.ndarray


def _vector(value: np.ndarray | None, name: str) -> np.ndarray:
    out = np.zeros(3) if value is None else np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite three-vector")
    return out


def _data(model: mujoco.MjModel, qpos: np.ndarray) -> mujoco.MjData:
    q = np.asarray(qpos, dtype=float)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError(f"qpos must contain {model.nq} finite positions")
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_fwdPosition(model, data)
    return data


def _apply(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: int,
    force: np.ndarray,
    torque: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    qfrc = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, force, torque, point, body, qfrc)
    return qfrc


def force_at_body(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body: str,
    force: np.ndarray,
    *,
    point: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    frame: Literal["world", "body"] = "world",
) -> np.ndarray:
    """Map a wrench applied to a body into external generalized force.

    Args:
        model: Compiled model, unchanged.
        qpos: Synchronized pose ``(nq,)``.
        body: Body receiving the force.
        force: Force [N], expressed in ``frame``.
        point: Application position [m], world coordinates for ``world`` or local
            body coordinates for ``body``. Default is the body frame origin.
        torque: Free torque [N m] about the application point, in ``frame``.
        frame: Coordinate frame for force, point and torque.

    Returns:
        ``(nv,)`` external generalized force to supply to inverse dynamics.
    """
    if frame not in ("world", "body"):
        raise ValueError("frame must be 'world' or 'body'")
    data = _data(model, qpos)
    bid = model.body(body).id
    f, t = _vector(force, "force"), _vector(torque, "torque")
    p = data.xpos[bid].copy() if point is None else _vector(point, "point")
    if frame == "body":
        rot = data.xmat[bid].reshape(3, 3)
        f, t = rot @ f, rot @ t
        p = data.xpos[bid] + rot @ _vector(point, "point")
    return _apply(model, data, bid, f, t, p)


def force_at_site(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    site: str,
    force: np.ndarray,
    *,
    torque: np.ndarray | None = None,
    frame: Literal["world", "site"] = "world",
) -> np.ndarray:
    """Map a wrench applied at a site's origin into generalized force.

    Args:
        model: Compiled model, unchanged.
        qpos: Synchronized pose ``(nq,)``.
        site: Site at the force application point.
        force: Force [N], in world or site coordinates.
        torque: Free torque [N m], in the same frame as force.
        frame: ``world`` or the site's local ``site`` frame.

    Returns:
        ``(nv,)`` external generalized force; site pose sets the lever arm.
    """
    if frame not in ("world", "site"):
        raise ValueError("frame must be 'world' or 'site'")
    data = _data(model, qpos)
    sid = model.site(site).id
    f, t = _vector(force, "force"), _vector(torque, "torque")
    if frame == "site":
        rot = data.site_xmat[sid].reshape(3, 3)
        f, t = rot @ f, rot @ t
    return _apply(model, data, int(model.site_bodyid[sid]), f, t, data.site_xpos[sid])


def held_mass_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    mass: float,
    *,
    body: str | None = None,
    site: str | None = None,
    point: np.ndarray | None = None,
) -> np.ndarray:
    """Map a held object's gravity to generalized force at its center of mass.

    Args:
        model: Compiled model, unchanged; gravity comes from ``model.opt.gravity``.
        qpos: Synchronized pose ``(nq,)``.
        mass: Nonnegative, finite held mass [kg].
        body: Body receiving the load, mutually exclusive with ``site``.
        site: Site at the object's center of mass.
        point: World center-of-mass position [m] when using ``body``; default
            is that body's frame origin.

    Returns:
        ``(nv,)`` gravity load. This quasi-static approximation excludes object
        acceleration, rotational inertia and contacts; use measured wrenches or
        an explicit object model for dynamic tasks. Do not double-count objects
        already included in the model's masses.
    """
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("mass must be nonnegative and finite")
    if (body is None) == (site is None) or (site is not None and point is not None):
        raise ValueError("choose body (with optional point) or site")
    force = float(mass) * model.opt.gravity
    if site is not None:
        return force_at_site(model, qpos, site, force)
    return force_at_body(model, qpos, body, force, point=point)


def force_plates_to_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    plates: Sequence[ForcePlateWrench],
    *,
    contact_bodies: Sequence[str] = ("calcn_r", "calcn_l"),
    plate_bodies: Mapping[int, str] | None = None,
    min_force_n: float = 5.0,
    max_distance_m: float = 0.35,
) -> np.ndarray:
    """Map world-frame force-plate wrenches onto contacting model bodies as ``qfrc``.

    Args:
        model: Compiled MuJoCo model.
        qpos: Synchronized poses ``(T, nq)`` or single pose ``(nq,)``.
        plates: Force-plate records providing ``force`` [N], ``cop`` [m], and
            ``free_torque`` [N m] in world coordinates.
        contact_bodies: Candidate body names matched by proximity to ``cop`` when
            ``plate_bodies`` does not explicitly assign a plate.
        plate_bodies: Optional ``{plate_index: body_name}`` assignment override.
        min_force_n: Force-magnitude threshold [N] below which a plate is unloaded.
        max_distance_m: Maximum CoP-to-body-origin distance [m] for automatic matching.

    Returns:
        External generalized force ``(T, nv)`` (or ``(nv,)`` for a single pose).
    """
    poses = np.asarray(qpos, dtype=float)
    single = poses.ndim == 1
    if single:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq}) or ({model.nq},)")

    cand_ids = [model.body(name).id for name in contact_bodies]
    data = mujoco.MjData(model)
    out = np.zeros((len(poses), model.nv))

    for t in range(len(poses)):
        data.qpos[:] = poses[t]
        mujoco.mj_fwdPosition(model, data)
        for p_idx, plate in enumerate(plates):
            f = plate.force[t]
            if np.linalg.norm(f) < min_force_n or not np.isfinite(plate.cop[t]).all():
                continue
            if plate_bodies is not None and p_idx in plate_bodies:
                body_name = plate_bodies[p_idx]
            else:
                dists = [
                    np.linalg.norm(data.xpos[bid] - plate.cop[t]) for bid in cand_ids
                ]
                best = int(np.argmin(dists))
                if dists[best] > max_distance_m:
                    continue
                body_name = contact_bodies[best]
            out[t] += force_at_body(
                model,
                poses[t],
                body_name,
                f,
                point=plate.cop[t],
                torque=plate.free_torque[t],
                frame="world",
            )
    return out[0] if single else out
