# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reachable targets of reach and site-tracking tasks, shared by the CPU envs and the mjlab twins."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

_N_SAMPLES = 100_000
_N_POSTURES = 10_000
_JOINT_TYPES = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
_CACHE: dict[tuple, np.ndarray] = {}
_POSTURE_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _uniform_joint_qpos(
    model: mujoco.MjModel, rng: np.random.Generator, n_samples: int
) -> np.ndarray:
    """``qpos0`` with every limited hinge and slide joint uniform within its range."""
    qpos = np.tile(model.qpos0, (n_samples, 1))
    for j in range(model.njnt):
        if model.jnt_limited[j] and int(model.jnt_type[j]) in _JOINT_TYPES:
            lo, hi = model.jnt_range[j]
            qpos[:, model.jnt_qposadr[j]] = rng.uniform(lo, hi, n_samples)
    return qpos


def _site_positions(
    model: mujoco.MjModel, qpos: np.ndarray, ids: tuple[int, ...]
) -> np.ndarray:
    """World positions ``(n, k, 3)`` of the sites *ids* in each posture of *qpos*."""
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), len(ids), 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        positions[i] = data.site_xpos[list(ids)]
    return positions


def _kinematic_model_key(model: mujoco.MjModel, ids: tuple[int, ...]) -> bytes:
    """Digest of all kinematic and joint-coupling fields affecting *ids*."""
    idx = list(ids)
    h = hashlib.sha256()
    h.update(
        repr(
            (model.nq, model.nv, model.njnt, model.nbody, model.nsite, model.neq, ids)
        ).encode()
    )
    for arr in (
        model.qpos0,
        model.body_parentid,
        model.body_jntadr,
        model.body_jntnum,
        model.body_pos,
        model.body_quat,
        model.jnt_type,
        model.jnt_bodyid,
        model.jnt_qposadr,
        model.jnt_dofadr,
        model.jnt_limited,
        model.jnt_range,
        model.jnt_pos,
        model.jnt_axis,
        model.site_bodyid[idx],
        model.site_pos[idx],
        model.site_quat[idx],
        model.eq_type,
        model.eq_obj1id,
        model.eq_obj2id,
        model.eq_objtype,
        model.eq_active0,
        model.eq_data,
    ):
        contig = np.ascontiguousarray(arr)
        h.update(f"{contig.dtype.str}:{contig.shape}:".encode())
        h.update(contig.tobytes())
    return h.digest()


def reachable_target_points(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    low: np.ndarray,
    high: np.ndarray,
    n_samples: int = _N_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """World positions of the sites over the joint ranges, restricted to the target box.

    Joint configurations are drawn uniformly within the joint limits (other joints stay at
    ``qpos0``); a configuration is kept when every site lies inside its box. Sampling
    targets from the result guarantees that a policy can reach them. The result is
    deterministic and cached, so the CPU env and the mjlab twin see the same table.

    Args:
        model: Compiled model.
        site_ids: Ids of the sites that must reach their targets.
        low: Box lower corners, shape ``(k, 3)``.
        high: Box upper corners, shape ``(k, 3)``.
        n_samples: Number of joint configurations to draw.
        seed: Seed of the sampling.

    Returns:
        Array ``(M, k, 3)``: for each kept configuration, the position of every site.

    Raises:
        ValueError: If no configuration puts the sites inside their boxes.
    """
    ids = tuple(int(i) for i in site_ids)
    low, high = (
        np.asarray(low, float).reshape(len(ids), 3),
        np.asarray(high, float).reshape(len(ids), 3),
    )
    key = (
        _kinematic_model_key(model, ids),
        low.tobytes(),
        high.tobytes(),
        n_samples,
        seed,
    )
    if key not in _CACHE:
        qpos = _uniform_joint_qpos(model, np.random.default_rng(seed), n_samples)
        positions = _site_positions(model, qpos, ids)
        inside = np.all((positions >= low) & (positions <= high), axis=(1, 2))
        if not inside.any():
            raise ValueError("no joint configuration reaches the target box")
        _CACHE[key] = positions[inside]
    return _CACHE[key]


def reachable_postures(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    n_samples: int = _N_POSTURES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random postures on the joint couplings and the world positions of sites in them.

    Every limited hinge and slide joint is drawn uniformly within its range (other
    joints keep ``qpos0``) and the coupled joints are put on their polynomials;
    postures whose coupled joints then leave their own range are drawn again (MuJoCo
    would hold such a joint at its limit, against the coupling). Each posture is one
    the model can take, so its site positions are a target set a policy can match
    exactly. A free root (joint 0) keeps its ``qpos0`` pose, raised
    or lowered so that the lowest site is as high as in ``qpos0``: the posture rests
    on the floor ``qpos0`` stands on instead of floating above it or sinking into it.
    The result is deterministic and cached, so the CPU env and the mjlab twin draw
    from the same table.

    Args:
        model: Compiled model.
        site_ids: Ids of the tracked sites.
        n_samples: Number of postures.
        seed: Seed of the sampling.

    Returns:
        ``(qpos, site_pos)``: the postures ``(n_samples, nq)`` and the positions of
        the sites in them ``(n_samples, k, 3)``.

    Raises:
        ValueError: If no drawn posture keeps the coupled joints within their ranges.
    """
    ids = tuple(int(i) for i in site_ids)
    key = (
        _kinematic_model_key(model, ids),
        n_samples,
        seed,
    )
    if key not in _POSTURE_CACHE:
        couplings = JointCouplings.from_model(model)
        limited = np.flatnonzero(
            (model.jnt_limited != 0) & np.isin(model.jnt_type, _JOINT_TYPES)
        )
        adr, (lo, hi) = model.jnt_qposadr[limited], model.jnt_range[limited].T
        rng, batches, n_kept = np.random.default_rng(seed), [], 0
        while n_kept < n_samples:
            batch = _uniform_joint_qpos(model, rng, n_samples)
            project_joint_couplings(couplings, batch)
            batch = batch[np.all((batch[:, adr] >= lo) & (batch[:, adr] <= hi), axis=1)]
            if not len(batch):
                raise ValueError("no posture keeps the coupled joints in their ranges")
            batches.append(batch)
            n_kept += len(batch)
        qpos = np.concatenate(batches)[:n_samples]
        positions = _site_positions(model, qpos, ids)
        if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            rest = model.qpos0.copy()
            project_joint_couplings(couplings, rest)
            floor = _site_positions(model, rest[None], ids)[0, :, 2].min()
            lift = floor - positions[:, :, 2].min(axis=1)
            qpos[:, model.jnt_qposadr[0] + 2] += lift
            positions[:, :, 2] += lift[:, None]
        _POSTURE_CACHE[key] = (qpos, positions)
    return _POSTURE_CACHE[key]
