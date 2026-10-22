# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Simultaneous marker IK using SciPy bounds and MuJoCo configuration integration.

The legacy single-site physics-wrapper solver cannot enforce scalar joint limits
or fit multiple markers together. Here SciPy differentiates the complete residual
through ``mj_integratePos``, including free/ball quaternion coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


@dataclass(frozen=True)
class MarkerIKResult:
    """Solved configuration and convergence evidence, in metres for marker error."""

    qpos: np.ndarray
    max_marker_error_m: float
    max_equality_error: float
    evaluations: int
    success: bool


def qpos_from_marker_targets(
    model: mujoco.MjModel,
    targets: Mapping[str, ArrayLike],
    *,
    qpos: ArrayLike | None = None,
    joint_names: Sequence[str] | None = None,
    tolerance_m: float = 1e-5,
    max_evaluations: int = 200,
    equality_weight: float = 10.0,
) -> MarkerIKResult:
    """Fit site positions jointly, preserving quaternion and scalar-limit semantics.

    Active equality errors are included and checked separately; this is kinematic
    fitting, not dynamic/contact-constrained trajectory optimization. Limited
    ball joints are refused because scalar optimizer bounds cannot express their
    cone constraint. Joint selection leaves other coordinates at the seed pose.
    """
    if not targets:
        raise ValueError("At least one marker target is required.")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or max_evaluations < 1:
        raise ValueError("Positive tolerance and evaluation budget are required.")
    if not np.isfinite(equality_weight) or equality_weight <= 0:
        raise ValueError("equality_weight must be finite and positive.")
    target_positions = np.asarray(list(targets.values()), dtype=float)
    if target_positions.shape != (len(targets), 3) or not np.all(
        np.isfinite(target_positions)
    ):
        raise ValueError("Marker targets must be finite 3-vectors.")
    site_ids = [model.site(name).id for name in targets]
    seed = np.array(model.qpos0 if qpos is None else qpos, dtype=float, copy=True)
    if seed.shape != (model.nq,) or not np.all(np.isfinite(seed)):
        raise ValueError(f"qpos must have finite shape ({model.nq},).")
    mujoco.mj_normalizeQuat(model, seed)
    if joint_names is None:
        joint_ids = list(range(model.njnt))
    else:
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be nonempty and unique.")
        joint_ids = [model.joint(name).id for name in joint_names]
    dofs, lower, upper = [], [], []
    for joint in joint_ids:
        kind = int(model.jnt_type[joint])
        if kind == mujoco.mjtJoint.mjJNT_BALL and model.jnt_limited[joint]:
            raise NotImplementedError(
                "Limited ball-joint marker IK needs a cone-constrained solver."
            )
        width = (
            6
            if kind == mujoco.mjtJoint.mjJNT_FREE
            else 3
            if kind == mujoco.mjtJoint.mjJNT_BALL
            else 1
        )
        start = int(model.jnt_dofadr[joint])
        dofs.extend(range(start, start + width))
        lo, hi = np.full(width, -np.inf), np.full(width, np.inf)
        if width == 1 and model.jnt_limited[joint]:
            address = int(model.jnt_qposadr[joint])
            seed[address] = np.clip(seed[address], *model.jnt_range[joint])
            lo[0], hi[0] = model.jnt_range[joint] - seed[address]
        lower.extend(lo)
        upper.extend(hi)
    if not dofs:
        raise ValueError("The model has no movable coordinates to retarget.")
    data = mujoco.MjData(model)
    indices = np.array(dofs, dtype=int)

    def evaluate(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(model.nv)
        velocity[indices] = delta
        data.qpos[:] = seed
        mujoco.mj_integratePos(model, data.qpos, velocity, 1.0)
        mujoco.mj_forward(model, data)
        marker_error = data.site_xpos[site_ids] - target_positions
        equality = data.efc_pos[: data.nefc][
            data.efc_type[: data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        ].copy()
        return marker_error, equality

    def residual(delta: np.ndarray) -> np.ndarray:
        marker_error, equality = evaluate(delta)
        return np.concatenate((marker_error.ravel(), equality_weight * equality))

    result = least_squares(
        residual,
        np.zeros(len(dofs)),
        bounds=(lower, upper),
        jac="3-point",
        max_nfev=max_evaluations,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    error, equality = evaluate(result.x)
    marker_max = float(np.linalg.norm(error, axis=1).max())
    equality_max = float(np.max(np.abs(equality), initial=0.0))
    return MarkerIKResult(
        data.qpos.copy(),
        marker_max,
        equality_max,
        result.nfev,
        bool(
            result.success and marker_max <= tolerance_m and equality_max <= tolerance_m
        ),
    )
