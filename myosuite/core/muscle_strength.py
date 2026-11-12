# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Subject-specific muscle strength: scaling and calibration to measured torques.

Geometric scaling of a model changes its segments and muscle paths, not how
strong the muscles are. :func:`calibrate_muscle_strength` fits a strength factor
per muscle so that the model's maximal joint torques match measured maximal
voluntary torques (e.g. from a dynamometer), and :func:`scale_muscle_strength`
applies the factors to a compiled model (or :func:`scale_muscle_strength_spec`
to a spec)::

    trials = [MaxTorqueMeasurement("elbow_flexion", qpos_90deg, torque=62.0)]
    cal = calibrate_muscle_strength(model, trials, prior=strength_factor_from_body_mass(68.0, 75.2))
    scale_muscle_strength(model, cal.factors)

A strength factor multiplies a muscle's peak active and passive force, as a
change of its physiological cross-section would. Model torques are taken along
the joint's coupled motion (dependent joints follow their polynomials), at rest
(isometric), with the agonists fully active and the antagonists passive, and
without gravity, as dynamometers report gravity-corrected torque. Real maximal
efforts co-contract and activate below 100 %, so the factors describe effective
strength.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from myosuite.core.muscle_conditions import _peak_force as _peak_force
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


def _muscles(model: mujoco.MjModel) -> np.ndarray:
    return model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE


def scale_muscle_strength(model: mujoco.MjModel, factors: float | np.ndarray) -> None:
    """Multiply each muscle's peak active and passive force, in place.

    Args:
        model: Compiled model; non-muscle actuators are left unchanged.
        factors: One factor for all muscles, or ``(nu,)`` per actuator.

    Raises:
        ValueError: If a muscle's factor is not positive and finite.
        ImportError: If the installed myo-sim lacks the model-only strength API.
    """
    try:
        from myo_sim.physics.muscle_strength import scale_muscle_strength as scale
    except ImportError as exc:
        raise ImportError(
            "Muscle strength scaling requires a myo-sim build containing "
            "MyoHub/myo_sim#186 (myo_sim.physics.muscle_strength). "
            "Install that development build or a release containing it; 0.2.5 lacks this API."
        ) from exc
    scale(model, factors)


def scale_muscle_strength_spec(
    spec: mujoco.MjSpec, factors: float | np.ndarray
) -> mujoco.MjSpec:
    """Spec-level :func:`scale_muscle_strength`: writes explicit peak forces.

    myo-sim compiles a copy to resolve automatic peak forces (``force="-1"``).

    Args:
        spec: Model spec, modified in place.
        factors: One factor for all muscles, or ``(nu,)`` per actuator.

    Returns:
        The same spec.

    Raises:
        ImportError: If the installed myo-sim lacks the model-only strength API.
    """
    try:
        from myo_sim.physics.muscle_strength import scale_muscle_strength_spec as scale
    except ImportError as exc:
        raise ImportError(
            "Spec strength scaling requires a myo-sim build containing "
            "MyoHub/myo_sim#186 (myo_sim.physics.muscle_strength). "
            "Install that development build or a release containing it; 0.2.5 lacks this API."
        ) from exc
    return scale(spec, factors)


def strength_factor_from_body_mass(mass: float, reference_mass: float) -> float:
    """Strength factor of a subject relative to the model's reference subject.

    Total lower-limb muscle volume scales with height x mass (Handsfield et al.
    2014, J Biomech 47:631, R^2 = 0.92) and fibre length with height, so the
    cross-section, and with it peak force, scales with body mass. The intercept
    of the published regression is ignored; use measured torques
    (:func:`calibrate_muscle_strength`) where available.

    Args:
        mass: Subject body mass [kg].
        reference_mass: Body mass of the model's reference subject [kg].

    Returns:
        ``mass / reference_mass``.
    """
    if mass <= 0 or reference_mass <= 0:
        raise ValueError("Masses must be positive")
    return float(mass / reference_mass)


@dataclass(frozen=True)
class MaxTorqueMeasurement:
    """One measured maximal voluntary joint torque.

    Attributes:
        joint: Name of the measured joint (the master of any couplings).
        qpos: ``(nq,)`` posture of the trial; coupled joints are projected.
        torque: Gravity-corrected maximal torque [N m], positive.
        direction: ``+1`` for torque along the joint's positive axis, ``-1``
            for the opposite direction (e.g. extension of a flexion joint).
    """

    joint: str
    qpos: np.ndarray
    torque: float
    direction: float = 1.0


@dataclass(frozen=True)
class StrengthCalibration:
    """Fitted strength factors and the torques they reproduce.

    Attributes:
        factors: ``(nu,)`` strength factor per actuator (1 for non-muscles,
            ``prior`` for muscles no measurement involves).
        involved: ``(nu,)`` muscles whose force acts on some measurement.
        measured: ``(m,)`` measured torques [N m].
        nominal: ``(m,)`` maximal torques of the unscaled model [N m].
        predicted: ``(m,)`` maximal torques with ``factors`` applied [N m].
    """

    factors: np.ndarray
    involved: np.ndarray
    measured: np.ndarray
    nominal: np.ndarray
    predicted: np.ndarray


def _coupled_direction(
    model: mujoco.MjModel, couplings: JointCouplings, qpos: np.ndarray, joint: int
) -> np.ndarray:
    """``(nv,)`` velocity of all dofs per unit velocity of *joint*, couplings included."""
    h = 1e-6
    lo, hi = qpos.copy(), qpos.copy()
    lo[model.jnt_qposadr[joint]] -= h
    hi[model.jnt_qposadr[joint]] += h
    project_joint_couplings(couplings, lo)
    project_joint_couplings(couplings, hi)
    e = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, e, 2 * h, lo, hi)
    return e


def _validate_mvc_model(model: mujoco.MjModel) -> None:
    unsupported = [
        model.actuator(i).name
        for i in range(model.nu)
        if model.actuator_forcelimited[i] or model.actuator_actnum[i] > 1
    ] + [model.joint(int(j)).name for j in np.flatnonzero(model.jnt_actfrclimited)]
    if unsupported:
        raise ValueError(
            f"Force clamps and multi-state actuators are not supported for MVC: {unsupported}"
        )
    if not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
        restricted = (
            (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
            & (model.actuator_actadr < 0)
            & (model.actuator_ctrllimited != 0)
            & (
                (model.actuator_ctrlrange[:, 0] > 0)
                | (model.actuator_ctrlrange[:, 1] < 1)
            )
        )
        names = [model.actuator(int(i)).name for i in np.flatnonzero(restricted)]
        if names:
            raise ValueError(
                f"stateless muscle control clamps must include [0, 1] for MVC: {names}"
            )
    if np.any(model.body_gravcomp != 0):
        raise ValueError(
            "gravity compensation is not supported for gravity-corrected MVC"
        )


def max_torque_terms(
    model: mujoco.MjModel, measurement: MaxTorqueMeasurement
) -> tuple[np.ndarray, np.ndarray, float]:
    """Each actuator's active and passive torque, and the joints' passive torque, in a trial.

    Args:
        model: Compiled model.
        measurement: The trial (joint, posture, direction).

    Returns:
        ``(active, passive, joints)``: ``(nu,)`` torque of each actuator at full
        activation beyond its passive force, in the measured direction
        (antagonists negative); ``(nu,)`` torque of each actuator's passive
        force; and the torque of the joints' own springs [N m].

    Raises:
        ValueError: If force clamps, multi-state actuators, gravity compensation
            or stateless muscle control clamps excluding [0, 1] prevent the
            gravity-corrected, affine full-activation probe.
    """
    _validate_mvc_model(model)
    if np.any(model.actuator_actearly):
        model = copy.deepcopy(model)
        model.actuator_actearly[:] = 0
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, measurement.joint)
    # Plain ints: tuple membership of a NumPy scalar in MuJoCo enums is False on
    # some MuJoCo 3.17 wheels (Linux), so compare integer values.
    if joint < 0 or int(model.jnt_type[joint]) not in (
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ):
        raise ValueError(
            f"{measurement.joint!r} is not a hinge or slide joint of the model"
        )
    couplings = JointCouplings.from_model(model)
    qpos = np.array(measurement.qpos, dtype=np.float64)
    project_joint_couplings(couplings, qpos)
    e = measurement.direction * _coupled_direction(model, couplings, qpos, joint)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    stateful = model.actuator_actadr >= 0
    forces = []
    for level in (0.0, 1.0):
        data.act[model.actuator_actadr[stateful]] = level
        data.ctrl[~stateful] = level
        mujoco.mj_fwdActuation(model, data)
        forces.append(data.actuator_force.copy())
    moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        moment,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    arm = (
        moment @ e
    )  # generalized force along the trial direction per unit actuator force
    return arm * (forces[1] - forces[0]), arm * forces[0], float(e @ data.qfrc_passive)


def calibrate_muscle_strength(
    model: mujoco.MjModel,
    measurements: Sequence[MaxTorqueMeasurement],
    *,
    prior: float = 1.0,
    regularization: float = 1e-3,
    bounds: tuple[float, float] = (0.25, 4.0),
) -> StrengthCalibration:
    """Fit per-muscle strength factors to measured maximal torques.

    Minimises ``sum_j ((T_j(k) - T_meas_j) / T_meas_j)**2 + regularization *
    sum_i (k_i - prior)**2`` within *bounds* (bounded linear least squares);
    ``T_j(k)`` is the model's maximal torque of trial ``j`` with every muscle's
    peak force, active and passive, multiplied by ``k``. Net torques do not
    separate synergists: the small default regularization only decides how a
    trial's change is shared among them, without pulling the fit to the prior.

    Args:
        model: Compiled model (not modified).
        measurements: Trials; at least one.
        prior: Factor of muscles no trial involves, and the target of the
            regularization (e.g. :func:`strength_factor_from_body_mass`).
        regularization: Weight of the prior against the relative torque errors;
            keep it small (a weight of 0.1 already absorbs about a quarter of a
            correction shared by three synergists).
        bounds: Lower and upper bounds of each factor.

    Returns:
        The factors and the measured, nominal and predicted torques.

    Raises:
        ValueError: If there are no measurements, a torque is not positive,
            or the model cannot support the steady probe in :func:`max_torque_terms`.
    """
    if not measurements or any(m.torque <= 0 for m in measurements):
        raise ValueError("Need measurements with positive torques")
    terms = [max_torque_terms(model, m) for m in measurements]
    muscles = _muscles(model)
    # Torque per unit strength factor: agonists fully active (antagonists and
    # other actuators off) plus every muscle's passive force; joint springs fixed.
    per_unit = (
        np.maximum([a for a, _, _ in terms], 0.0) + [p for _, p, _ in terms]
    ) * muscles
    joints = np.array([j for _, _, j in terms])
    measured = np.array([m.torque for m in measurements], dtype=np.float64)
    involved = muscles & (per_unit != 0).any(axis=0)

    n = int(involved.sum())
    A = np.vstack(
        [per_unit[:, involved] / measured[:, None], np.sqrt(regularization) * np.eye(n)]
    )
    b = np.r_[
        (measured - joints) / measured, np.full(n, np.sqrt(regularization) * prior)
    ]
    factors = np.where(muscles, prior, 1.0)
    factors[involved] = lsq_linear(A, b, bounds=bounds).x
    return StrengthCalibration(
        factors=factors,
        involved=involved,
        measured=measured,
        nominal=per_unit.sum(axis=1) + joints,
        predicted=per_unit @ factors + joints,
    )
