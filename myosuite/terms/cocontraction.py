# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Wasted muscle torque (co-contraction cost).

Muscle ``m`` applies the torque ``tau_mj = actuator_moment[m, j] * actuator_force[m]`` about
DOF ``j``. Antagonists cancel each other's torque, so the torque wasted at ``j`` is
``sum_m |tau_mj| - |sum_m tau_mj|``: zero when all muscles crossing ``j`` push the same way,
positive under co-contraction. The cost sums this over a set of DOFs and divides by body mass
(N·m/kg), as in the analysis of MyoSuite PR #573.

MuJoCo stores ``actuator_moment`` sparse (rows ``moment_rowadr`` / ``moment_rownnz``, columns
``moment_colind``). For tendon and joint transmissions the columns of a row are those of the
tendon Jacobian (``ten_J_colind``) or the joint DOF, which are fixed by the model on CPU MuJoCo
and mujoco-warp; only the row start differs (mujoco-warp allocates rows per world). The
:class:`MomentArmIndex` therefore stores, per muscle and DOF, the offset inside the row, and the
term gathers ``actuator_moment[rowadr[m] + offset]`` on either backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import mujoco
import numpy as np

from myosuite.core.protocols import EnvAccessor

FULLBODY_LEG_JOINTS: tuple[str, ...] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "knee_angle_l",
    "ankle_angle_l",
)
"""Default DOFs of the cost: the sagittal and frontal hip, knee and ankle of both legs."""


class MuscleTorqueAccessor(EnvAccessor, Protocol):
    """Accessor with the muscle forces and sparse moments (CPU and mjlab accessors)."""

    def muscle_force(self) -> Any:
        """Actuator forces in N, shape ``(..., nu)``."""
        ...

    def actuator_moment_sparse(self) -> tuple[Any, Any]:
        """``(actuator_moment, moment_rowadr)``, shapes ``(..., nJmom)`` and ``(..., nu)``."""
        ...


@dataclass(frozen=True)
class WastedTorqueCfg:
    """Opt-in co-contraction cost.

    Attributes:
        weight: Penalty per N·m/kg, subtracted from the reward every control step. ``0`` logs
            the metric without changing the reward.
        joints: Hinge or slide joints whose DOFs are scored.
        body_mass: Normaliser in kg; ``None`` uses the model's total mass.
    """

    weight: float = 0.0
    joints: tuple[str, ...] = FULLBODY_LEG_JOINTS
    body_mass: float | None = None


@dataclass(frozen=True)
class MomentArmIndex:
    """Gather plan from MuJoCo's sparse ``actuator_moment`` to dense muscle x DOF moment arms.

    Attributes:
        actuator_ids: Muscle actuators that cross at least one scored DOF, shape ``(n_m,)``.
        offset: Position of each scored DOF in the muscle's moment row, shape ``(n_m, n_d)``;
            ``0`` where the muscle does not cross the DOF (masked by ``valid``).
        valid: Whether the muscle crosses the DOF, shape ``(n_m, n_d)``, float ``0`` / ``1``.
        body_mass: Normaliser in kg.
        dof_ids: Scored DOF addresses, shape ``(n_d,)`` (informational).
    """

    actuator_ids: Any
    offset: Any
    valid: Any
    body_mass: float
    dof_ids: Any = field(default=None)

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        joints: tuple[str, ...] = FULLBODY_LEG_JOINTS,
        *,
        body_mass: float | None = None,
        name_prefix: str = "",
    ) -> MomentArmIndex:
        """Build the plan from a compiled model.

        Args:
            model: Compiled model (CPU env model or ``env.sim.mj_model`` of an mjlab scene).
            joints: Hinge or slide joint names.
            body_mass: Normaliser in kg; ``None`` uses ``mj_getTotalmass(model)``.
            name_prefix: Prefix of the joint names in *model* (mjlab entity, e.g. ``"robot/"``).

        Returns:
            The gather plan as numpy arrays.

        Raises:
            ValueError: If a joint is missing or not 1-DOF, no muscle crosses the joints, or a
                muscle uses a transmission other than a tendon or joint.
        """
        dof_ids = []
        for name in joints:
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name_prefix + name
            )
            if jid < 0:
                raise ValueError(
                    f"Joint {name_prefix + name!r} not found in the model."
                )
            if model.jnt_type[jid] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"Joint {name!r} is not a hinge or slide joint.")
            dof_ids.append(int(model.jnt_dofadr[jid]))
        dof_pos = {d: k for k, d in enumerate(dof_ids)}

        actuators, offsets, valids = [], [], []
        for a in np.flatnonzero(
            model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
        ):
            trn, tid = int(model.actuator_trntype[a]), int(model.actuator_trnid[a, 0])
            if trn == mujoco.mjtTrn.mjTRN_TENDON:
                adr = int(model.ten_J_rowadr[tid])
                cols = model.ten_J_colind[adr : adr + int(model.ten_J_rownnz[tid])]
            elif trn == mujoco.mjtTrn.mjTRN_JOINT:
                cols = np.asarray([model.jnt_dofadr[tid]])
            else:
                raise ValueError(
                    f"Muscle {a} uses an unsupported transmission ({trn})."
                )
            off = np.zeros(len(dof_ids), dtype=np.int64)
            ok = np.zeros(len(dof_ids), dtype=np.float32)
            for slot, col in enumerate(cols):
                if int(col) in dof_pos:
                    off[dof_pos[int(col)]] = slot
                    ok[dof_pos[int(col)]] = 1.0
            if ok.any():
                actuators.append(int(a))
                offsets.append(off)
                valids.append(ok)
        if not actuators:
            raise ValueError(f"No muscle crosses the joints {joints}.")
        return cls(
            actuator_ids=np.asarray(actuators, dtype=np.int64),
            offset=np.stack(offsets),
            valid=np.stack(valids),
            body_mass=float(
                mujoco.mj_getTotalmass(model) if body_mass is None else body_mass
            ),
            dof_ids=np.asarray(dof_ids, dtype=np.int64),
        )

    def to_backend(self, convert: Callable[[Any], Any]) -> MomentArmIndex:
        """Return a copy with the arrays converted, e.g. to torch tensors on the sim device."""
        return replace(
            self,
            actuator_ids=convert(self.actuator_ids),
            offset=convert(self.offset),
            valid=convert(self.valid),
            dof_ids=convert(self.dof_ids),
        )


def muscle_moment_arms(
    xp: Any, actuator_moment: Any, moment_rowadr: Any, index: MomentArmIndex
) -> Any:
    """Dense moment arms of the indexed muscles about the indexed DOFs.

    Args:
        xp: Array module (numpy or torch).
        actuator_moment: Sparse moment values, shape ``(..., nJmom)``.
        moment_rowadr: Row start per actuator, shape ``(..., nu)``.
        index: Gather plan with arrays of the same backend.

    Returns:
        Moment arms ``dL/dq`` (m for hinge DOFs), shape ``(..., n_m, n_d)``.
    """
    rowadr = moment_rowadr[..., index.actuator_ids]
    idx = rowadr[..., :, None] + index.offset
    flat = idx.reshape(*idx.shape[:-2], -1)
    if getattr(xp, "__name__", "") == "torch":
        vals = xp.take_along_dim(actuator_moment, flat.long(), dim=-1)
    else:
        vals = xp.take_along_axis(actuator_moment, flat, axis=-1)
    return vals.reshape(idx.shape) * index.valid


def wasted_muscle_torque(xp: Any, moment_arms: Any, force: Any) -> Any:
    """Torque the muscles spend against each other, summed over DOFs.

    Args:
        xp: Array module (numpy or torch).
        moment_arms: Shape ``(..., n_m, n_d)``.
        force: Muscle forces in N, shape ``(..., n_m)``.

    Returns:
        ``sum_j (sum_m |tau_mj| - |sum_m tau_mj|)`` in N·m, shape ``(...)``.
    """
    tau = moment_arms * force[..., :, None]
    return (xp.abs(tau).sum(-2) - xp.abs(tau.sum(-2))).sum(-1)


def wasted_torque_reward(
    accessor: MuscleTorqueAccessor,
    task_state: dict[str, Any],
    *,
    moment_index: MomentArmIndex,
    weight: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Co-contraction cost: wasted muscle torque per body mass.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for the uniform call signature.
        moment_index: Gather plan, with arrays of the accessor's backend.
        weight: Penalty per N·m/kg.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``wasted_torque`` (N·m/kg), ``dense`` (``-weight * wasted_torque``),
        ``solved`` and ``done`` (both ``False``).
    """
    del task_state, kwargs
    xp = accessor.array_module()
    moment, rowadr = accessor.actuator_moment_sparse()
    arms = muscle_moment_arms(xp, moment, rowadr, moment_index)
    force = accessor.muscle_force()[..., moment_index.actuator_ids]
    wasted = wasted_muscle_torque(xp, arms, force) / moment_index.body_mass
    return {
        "wasted_torque": wasted,
        "dense": -weight * wasted,
        "solved": False,
        "done": False,
    }


__all__ = [
    "FULLBODY_LEG_JOINTS",
    "MomentArmIndex",
    "MuscleTorqueAccessor",
    "WastedTorqueCfg",
    "muscle_moment_arms",
    "wasted_muscle_torque",
    "wasted_torque_reward",
]
