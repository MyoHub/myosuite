# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Refit clips that were retargeted on superseded joint couplings.

The GMR clips of ``amathislab/musclemimic-retargeted`` were fitted on
``musclemimic_models`` 1.0.5 or earlier. There, the ``knee_angle_rotation3_l``
polynomial has the opposite sign of 1.0.6's. The joint axes did not change, so
forward kinematics of a clip frame still give the bodies GMR placed. GMR
(``gmr_plus``) treats the joint couplings as a soft cost next to its body
targets. Each frame therefore shares the wrong knee twist between thigh and
shank. Projecting the dependent joints onto the model's polynomials
(:func:`~myosuite.physics.joint_coupling.project_joint_couplings`) keeps the
thigh's share and turns the shank and foot further the other way.

:func:`refit_stale_couplings` instead solves GMR's objective again on the
model's couplings, for the limbs whose couplings are stale, with the clip's
own body poses as targets. Each clip frame is GMR's least-squares compromise
on the old coupling, and the old and new couplings differ only by the sign of
the same twist. Solving the same problem on the new coupling therefore returns,
to first order, what GMR would have fitted on the model. Run it before the
couplings are projected or the clip is differentiated::

    qpos = expand_motion_clip_to_model(clip, model).qpos
    qpos = refit_stale_couplings(model, qpos)
    kin = clip_kinematics(model, qpos, 1 / clip.frequency_hz, lowpass_hz=6.0)

The fit runs on `mink <https://github.com/kevinzakka/mink>`_, as GMR does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

#: gmr_plus's final IK stage for MyoFullBody: body -> (position cost, orientation
#: cost). Source: ``ik_match_table2`` of
#: ``loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody.json`` in
#: amathislab/musclemimic (Apache-2.0).
GMR_MYOFULLBODY_COSTS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "pelvis": (100.0, 5.0),
        "lumbar1": (40.0, 5.0),
        "head": (45.0, 5.0),
        "humerus_l": (0.0, 5.0),
        "ulna_l": (0.0, 5.0),
        "lunate_l": (0.0, 5.0),
        "humerus_r": (0.0, 5.0),
        "ulna_r": (40.0, 5.0),
        "lunate_r": (40.0, 5.0),
        "femur_l": (0.0, 5.0),
        "tibia_l": (0.0, 5.0),
        "calcn_l": (50.0, 5.0),
        "toes_l": (50.0, 5.0),
        "femur_r": (0.0, 5.0),
        "tibia_r": (0.0, 5.0),
        "calcn_r": (50.0, 5.0),
        "toes_r": (50.0, 5.0),
    }
)

_NQ = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}
_NV = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


@dataclass(frozen=True)
class CouplingRefitConfig:
    """Options of :func:`refit_stale_couplings`.

    Attributes:
        frame_costs: Body name -> ``(position cost, orientation cost)`` of the
            body-frame targets, the costs of GMR's IK table. Bodies outside the
            refitted limbs or missing from the model are ignored.
        equality_cost: Cost of each joint-equality violation (gmr_plus: 5).
        sign_tol: A coupling is stale when its dependent joint stays within
            this distance (rad or m) of the negated polynomial over the whole
            clip, and some frame is more than ten times as far from the model's
            polynomial.
        max_iters: Maximum Gauss-Newton steps per frame.
    """

    frame_costs: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: GMR_MYOFULLBODY_COSTS
    )
    equality_cost: float = 5.0
    sign_tol: float = 5e-3
    max_iters: int = 50

    def __post_init__(self) -> None:
        for name, value in (
            ("equality_cost", self.equality_cost),
            ("sign_tol", self.sign_tol),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {self.max_iters}")


def _checked_qpos(model: mujoco.MjModel, qpos: Any) -> np.ndarray:
    """``qpos`` as a float64 ``(T, nq)`` copy."""
    qpos = np.array(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must be (T, nq={model.nq}), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos must be finite")
    return qpos


def stale_joint_couplings(
    model: mujoco.MjModel, qpos: Any, tol: float = 5e-3
) -> list[str]:
    """Dependent joints that a clip keeps on the negated polynomial.

    Args:
        model: Compiled model whose couplings (active ``mjEQ_JOINT``
            equalities) are the reference.
        qpos: ``(T, nq)`` clip positions.
        tol: Distance (rad or m) within which the clip follows the negated
            polynomial in every frame; some frame must also be more than
            ``10 * tol`` from the model's polynomial.

    Returns:
        Names of the dependent joints of the stale couplings, in coupling order.

    Raises:
        ValueError: If *qpos* is not a finite ``(T, nq)`` array.
    """
    qpos = _checked_qpos(model, qpos)
    couplings = JointCouplings.from_model(model)
    on_model = qpos.copy()
    project_joint_couplings(couplings, on_model)
    offset = qpos[:, couplings.dep_qpos] - couplings.dep_ref
    poly = on_model[:, couplings.dep_qpos] - couplings.dep_ref
    stale = (np.abs(offset + poly).max(axis=0) <= tol) & (
        np.abs(offset - poly).max(axis=0) > 10 * tol
    )
    joint_at = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
    return [model.joint(joint_at[int(a)]).name for a in couplings.dep_qpos[stale]]


def _limb_bodies(model: mujoco.MjModel, joints: Sequence[int]) -> np.ndarray:
    """Mask of the limbs holding *joints*.

    A limb is the subtree of the topmost body whose parent is welded to the
    root body (MyoFullBody: ``femur_l`` under the pelvis, which is welded to
    the free-joint body).
    """
    limb_roots = set()
    for j in joints:
        body = int(model.jnt_bodyid[j])
        base = model.body_weldid[model.body_rootid[body]]
        if model.body_weldid[body] == base:
            raise ValueError(f"Joint {model.joint(j).name!r} is on the root body")
        while model.body_weldid[model.body_parentid[body]] != base:
            body = int(model.body_parentid[body])
        limb_roots.add(body)
    mask = np.zeros(model.nbody, dtype=bool)
    for b in range(1, model.nbody):  # MuJoCo lists parents before children
        mask[b] = b in limb_roots or mask[model.body_parentid[b]]
    return mask


class _LimbFit:
    """GMR's IK objective with every joint outside the masked bodies held."""

    def __init__(
        self,
        model: mujoco.MjModel,
        bodies: np.ndarray,
        config: CouplingRefitConfig,
    ) -> None:
        try:
            import mink
        except ImportError as err:
            raise ImportError(
                "refit_stale_couplings needs mink. Install with: "
                "pip install mink or pip install 'MyoSuite[examples]'."
            ) from err
        self._mink = mink
        self._max_iters = config.max_iters
        joints = np.flatnonzero(bodies[model.jnt_bodyid])
        self.qpos_addr = np.concatenate(
            [
                np.arange(
                    model.jnt_qposadr[j], model.jnt_qposadr[j] + _NQ[model.jnt_type[j]]
                )
                for j in joints
            ]
        ).astype(np.int64)
        dofs = np.concatenate(
            [
                np.arange(
                    model.jnt_dofadr[j], model.jnt_dofadr[j] + _NV[model.jnt_type[j]]
                )
                for j in joints
            ]
        )
        self._frozen = np.setdiff1d(np.arange(model.nv), dofs)
        self.frames = [
            mink.FrameTask(
                model.body(b).name,
                "body",
                position_cost=config.frame_costs[model.body(b).name][0],
                orientation_cost=config.frame_costs[model.body(b).name][1],
            )
            for b in np.flatnonzero(bodies)
            if any(config.frame_costs.get(model.body(b).name, (0.0, 0.0)))
        ]
        if not self.frames:
            raise ValueError(
                "None of the refitted bodies has a frame cost: "
                f"{[model.body(b).name for b in np.flatnonzero(bodies)]}"
            )
        self.body_ids = [model.body(t.frame_name).id for t in self.frames]
        equalities = np.flatnonzero(
            (model.eq_type == mujoco.mjtEq.mjEQ_JOINT) & (model.eq_active0 != 0)
        )
        self._tasks = [*self.frames]
        if len(equalities):
            self._tasks.append(
                mink.EqualityConstraintTask(
                    model, cost=config.equality_cost, equalities=equalities.tolist()
                )
            )
        # Only the moving joints that no coupling sets are range-limited.
        held = np.setdiff1d(
            np.arange(model.njnt),
            np.setdiff1d(joints, model.eq_obj1id[equalities]),
        )
        limited = model.jnt_limited.copy()
        try:
            model.jnt_limited[held] = 0
            self._configuration = mink.Configuration(model)
            self._limits = [mink.ConfigurationLimit(model)]
        finally:
            model.jnt_limited[:] = limited
        self._constraints = (
            [mink.DofFreezingTask(model, dof_indices=self._frozen.tolist())]
            if len(self._frozen)
            else []
        )

    def solve(
        self, seed: np.ndarray, targets: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Pose closest to the ``(position, wxyz quaternion)`` targets of the frames."""
        mink = self._mink
        for task, (pos, quat) in zip(self.frames, targets):
            task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat, pos])))
        self._configuration.update(seed.copy())
        for _ in range(self._max_iters):
            velocity = mink.solve_ik(
                self._configuration,
                self._tasks,
                1.0,
                "daqp",
                damping=1e-4,
                limits=self._limits,
                constraints=self._constraints,
            )
            velocity[self._frozen] = 0.0
            self._configuration.integrate_inplace(velocity, 1.0)
            if np.abs(velocity).max() < 1e-9:
                break
        return self._configuration.q.copy()


def refit_stale_couplings(
    model: mujoco.MjModel,
    qpos: Any,
    config: CouplingRefitConfig | None = None,
) -> np.ndarray:
    """Refit the limbs whose couplings a clip keeps on the negated polynomial.

    For every limb (a subtree hanging from the bodies welded to the root, such
    as one leg) holding a stale coupling (:func:`stale_joint_couplings`), each
    frame is solved again on the
    model's couplings, with the frame's own body poses as targets. The targets
    are taken from forward kinematics of the frame on *model*, so the clip must
    have been fitted on the same joint axes. The default costs are gmr_plus's
    for MyoFullBody (:data:`GMR_MYOFULLBODY_COSTS`).

    Args:
        model: Compiled model whose couplings the result follows. It is
            restored before return.
        qpos: ``(T, nq)`` clip positions.
        config: Costs and detection tolerance; defaults to
            :class:`CouplingRefitConfig`.

    Returns:
        ``(T, nq)`` positions, a copy of *qpos* if no coupling is stale. Only
        the joints of the refitted limbs change, and their dependent joints
        lie on the model's polynomials.

    Raises:
        ImportError: If mink is not installed and a limb needs refitting.
        ValueError: If *qpos* is not a finite ``(T, nq)`` array, or a refitted
            limb has no body with a frame cost.
    """
    config = config or CouplingRefitConfig()
    qpos = _checked_qpos(model, qpos)
    stale = stale_joint_couplings(model, qpos, config.sign_tol)
    if not stale:
        return qpos
    fit = _LimbFit(
        model, _limb_bodies(model, [model.joint(n).id for n in stale]), config
    )
    data = mujoco.MjData(model)
    out = qpos.copy()
    for t, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        targets = [(data.xpos[b].copy(), data.xquat[b].copy()) for b in fit.body_ids]
        out[t, fit.qpos_addr] = fit.solve(frame, targets)[fit.qpos_addr]
    projected = out.copy()
    project_joint_couplings(JointCouplings.from_model(model), projected)
    out[:, fit.qpos_addr] = projected[:, fit.qpos_addr]
    return out
