# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Participant strength, mass and geometry calibration of compiled MuJoCo models.

MVC torques are gravity-corrected, signed joint torques: agonists are maximally
active, antagonists relaxed, and passive muscle and joint forces retained.
Torques follow independent joints' coupled motion by virtual work. Calibration
estimates effective group strength; net torque cannot identify synergists.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import mujoco
import numpy as np

from myosuite.core.muscle_strength import (
    MaxTorqueMeasurement,
    max_torque_terms,
    scale_muscle_strength,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


def _muscles(model: mujoco.MjModel, muscles: Sequence[str] | None) -> np.ndarray:
    mask = model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
    if muscles is None:
        return np.flatnonzero(mask)
    ids = np.array([model.actuator(name).id for name in muscles], dtype=int)
    if len(ids) != len(np.unique(ids)) or not np.all(mask[ids]):
        raise ValueError("muscles must name distinct muscle actuators")
    return ids


def _posture(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    q = np.array(qpos, dtype=float, copy=True)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError(f"qpos must contain {model.nq} finite positions")
    project_joint_couplings(JointCouplings.from_model(model), q)
    return q


def posture_at(model: mujoco.MjModel, joints: Mapping[str, float]) -> np.ndarray:
    """Return the first keyframe (else qpos0) with named coordinates set.

    Args:
        model: Compiled model, unchanged.
        joints: Independent hinge/slide joint names to positions [rad or m].

    Returns:
        Coupling-consistent ``(nq,)`` posture.
    """
    q = (model.key_qpos[0] if model.nkey else model.qpos0).copy()
    for name, value in joints.items():
        joint = model.joint(name).id
        if int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(f"{name!r} must be a hinge or slide joint")
        q[model.jnt_qposadr[joint]] = value
    return _posture(model, q)


def _torque_terms(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str
) -> tuple[np.ndarray, np.ndarray, float]:
    q = _posture(model, qpos)
    jid = model.joint(joint).id
    if int(model.jnt_type[jid]) not in (
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ):
        raise ValueError(f"{joint!r} must be a hinge or slide joint")
    couplings = JointCouplings.from_model(model)
    adr = model.jnt_qposadr[jid]
    if adr in couplings.dep_qpos or adr in couplings.locked_qpos:
        raise ValueError("MVC joint must be independent of the joint equalities")
    return max_torque_terms(model, MaxTorqueMeasurement(joint, q, torque=1.0))


def max_joint_torque(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint: str,
    direction: float = 1.0,
    *,
    max_activation: np.ndarray | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Predict isometric net MVC, optionally with fatigued activation caps.

    Args:
        model: Compiled model, unchanged.
        qpos: Probe posture ``(nq,)`` [rad/m, quaternions for ball/free joints].
        joint: Independent hinge/slide joint about which torque is measured.
        direction: ``+1`` or ``-1`` along the joint coordinate.
        max_activation: ``(nu,)`` muscle activation caps in [0, 1]; e.g. ``1-MF``.
            Non-muscle entries are ignored. Passive forces are not capped.

    Returns:
        Signed torque [N m for hinges, N for slides] and agonist actuator names.
        Includes passive muscle/joint forces; excludes gravity and external loads.

    Raises:
        ValueError: If the model has force clamps, multi-state actuators or gravity
            compensation, or enabled stateless muscle control clamps exclude part
            of [0, 1]. Force must be affine in activation and strength.
    """
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be +1 or -1")
    active, passive, springs = _torque_terms(model, qpos, joint)
    ids = _muscles(model, None)
    cap = np.ones(model.nu) if max_activation is None else np.asarray(max_activation)
    if cap.shape != (model.nu,) or not np.all(
        np.isfinite(cap[ids]) & (cap[ids] >= 0) & (cap[ids] <= 1)
    ):
        raise ValueError(
            "max_activation must have shape (nu,) and muscle caps in [0,1]"
        )
    agonists = ids[direction * active[ids] > 0]
    torque = passive.sum() + springs + active[agonists] @ cap[agonists]
    return float(torque), tuple(model.actuator(int(i)).name for i in agonists)


@dataclass(frozen=True)
class StrengthCalibration:
    """One group strength scale and the signed MVC used to fit it.

    Attributes:
        joint: Independent hinge/slide joint.
        posture: ``(nq,)`` probe posture.
        measured_torque: Signed gravity-corrected MVC [N m or N].
        model_torque: Net MVC of the unscaled model at the probe posture.
        scale: Factor for ``agonists``, including their active and passive force.
        agonists: Only these actuators should be scaled with this calibration.
    """

    joint: str
    posture: np.ndarray
    measured_torque: float
    model_torque: float
    scale: float
    agonists: tuple[str, ...]

    def record(self) -> dict:
        """Return JSON-serialisable calibration fields."""
        out = asdict(self)
        out["posture"] = self.posture.tolist()
        return out


def strength_from_mvc(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str, measured_torque: float
) -> StrengthCalibration:
    """Fit agonist strength, retaining unscaled antagonist and spring torques.

    Args:
        model: Compiled model, unchanged.
        qpos: MVC posture ``(nq,)``.
        joint: Independent hinge/slide joint measured by the dynamometer.
        measured_torque: Finite, nonzero signed gravity-corrected MVC [N m or N].

    Returns:
        Calibration to apply with ``scale_strength(model, cal.scale, cal.agonists)``.

    Raises:
        ValueError: If the measurement cannot be fitted with a positive scale.
    """
    if not np.isfinite(measured_torque) or measured_torque == 0:
        raise ValueError("measured_torque must be finite and nonzero")
    total, names = max_joint_torque(model, qpos, joint, float(np.sign(measured_torque)))
    active, passive, springs = _torque_terms(model, qpos, joint)
    selected = _muscles(model, names)
    scaled = float((active[selected] + passive[selected]).sum())
    fixed = float(passive.sum() - passive[selected].sum() + springs)
    if not names or abs(scaled) < 1e-12:
        raise ValueError("model has no scalable agonist torque in this direction")
    scale = (measured_torque - fixed) / scaled
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("measurement cannot be matched with a positive agonist scale")
    return StrengthCalibration(
        joint, _posture(model, qpos), float(measured_torque), total, float(scale), names
    )


def scale_strength(
    model: mujoco.MjModel, scale: float, muscles: Sequence[str] | None = None
) -> mujoco.MjModel:
    """Return a copy with selected muscles' peak active and passive forces scaled.

    Args:
        model: Compiled model, unchanged; automatic peak forces are resolved first.
        scale: Positive, finite effective strength factor.
        muscles: Actuator names; ``None`` selects all muscle actuators.

    Returns:
        Independent model copy. Non-muscle actuators retain their parameters.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("strength scale must be positive and finite")
    out = copy.deepcopy(model)
    ids = _muscles(model, muscles)
    factors = np.ones(model.nu)
    factors[ids] = scale
    scale_muscle_strength(out, factors)
    selected = np.zeros(model.nu, dtype=bool)
    selected[ids] = True
    out.actuator_gainprm[~selected] = model.actuator_gainprm[~selected]
    bias = selected & (model.actuator_biastype == mujoco.mjtBias.mjBIAS_MUSCLE)
    out.actuator_biasprm[~bias] = model.actuator_biasprm[~bias]
    return out


@dataclass(frozen=True)
class StrengthFit:
    """Shared selected-muscle scale fitted across several signed MVC postures."""

    joint: str
    scale: float
    muscles: tuple[str, ...]
    postures: tuple[tuple[float, ...], ...]
    measured: tuple[float, ...]
    predicted: tuple[float, ...]
    rmse: float

    def record(self) -> dict:
        """Return JSON-serialisable calibration fields."""
        return asdict(self)


def fit_strength_from_mvcs(
    model: mujoco.MjModel,
    postures: Sequence[np.ndarray],
    joint: str,
    measured_torques: Sequence[float],
    *,
    muscles: Sequence[str] | None = None,
) -> StrengthFit:
    """Fit one identifiable group scale by least squares across MVC postures.

    Args:
        model: Compiled model, unchanged.
        postures: At least two ``(nq,)`` probe postures.
        joint: Independent measured hinge/slide joint.
        measured_torques: Signed gravity-corrected torques [N m or N].
        muscles: Group to scale; default is the union of trial agonists.

    Returns:
        Fitted scale, predictions and torque RMSE. Residuals expose posture mismatch.
    """
    measured = np.asarray(measured_torques, dtype=float)
    if measured.ndim != 1 or len(postures) != len(measured) or len(measured) < 2:
        raise ValueError("need at least two paired MVC postures and torques")
    if not np.all(np.isfinite(measured) & (measured != 0)):
        raise ValueError("MVC torques must be finite and nonzero")
    terms = [_torque_terms(model, q, joint) for q in postures]
    ids = _muscles(model, None)
    if muscles is None:
        selected = ids[
            np.any(
                [np.sign(t) * a[ids] > 0 for t, (a, _, _) in zip(measured, terms)],
                axis=0,
            )
        ]
    else:
        selected = _muscles(model, muscles)
    coefficients, fixed = [], []
    for torque, (active, passive, springs) in zip(measured, terms):
        recruited = np.zeros(model.nu)
        agonists = ids[np.sign(torque) * active[ids] > 0]
        recruited[agonists] = active[agonists]
        coeff = float((recruited[selected] + passive[selected]).sum())
        coefficients.append(coeff)
        fixed.append(float(recruited.sum() + passive.sum() + springs - coeff))
    c, offset = np.array(coefficients), np.array(fixed)
    if c @ c < 1e-24:
        raise ValueError("selected muscles have no scalable torque in these trials")
    scale = float(c @ (measured - offset) / (c @ c))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MVC trials require a nonpositive strength scale")
    predicted = offset + c * scale
    return StrengthFit(
        joint,
        scale,
        tuple(model.actuator(int(i)).name for i in selected),
        tuple(tuple(_posture(model, q).tolist()) for q in postures),
        tuple(measured.tolist()),
        tuple(predicted.tolist()),
        float(np.sqrt(np.mean((predicted - measured) ** 2))),
    )


def scale_body_mass(
    model: mujoco.MjModel,
    represented_mass: float | None = None,
    *,
    participant_mass: float | None = None,
    reference_mass: float | None = None,
    bodies: Sequence[str] | None = None,
) -> mujoco.MjModel:
    """Return a mass/inertia-scaled copy without changing muscle strength.

    Args:
        model: Compiled model, unchanged. Segment geometry remains unchanged.
        represented_mass: Target sum [kg] of selected model bodies, explicitly
            not a participant's whole-body mass for a partial-body model.
        participant_mass: Participant whole-body mass [kg]. Requires the model's
            reference participant ``reference_mass``; scales by their ratio.
        reference_mass: Whole-body mass [kg] of the model's reference participant.
        bodies: Bodies to scale, e.g. anatomical bodies excluding held objects.
            ``None`` selects all non-world bodies, including world-fixed bodies.

    Returns:
        Model copy with refreshed MuJoCo constants and original automatic muscle
        strengths frozen explicitly. Supply exactly one mass-scaling mode.
    """
    ratio_mode = participant_mass is not None or reference_mass is not None
    if (represented_mass is None) == (not ratio_mode):
        raise ValueError(
            "choose represented_mass or participant_mass with reference_mass"
        )
    ids = (
        np.arange(1, model.nbody)
        if bodies is None
        else np.array([model.body(name).id for name in bodies], dtype=int)
    )
    if not len(ids) or 0 in ids or len(ids) != len(np.unique(ids)):
        raise ValueError("bodies must select distinct non-world bodies")
    if ratio_mode:
        if participant_mass is None or reference_mass is None:
            raise ValueError("participant_mass requires explicit reference_mass")
        values = np.array([participant_mass, reference_mass], dtype=float)
        ratio = participant_mass / reference_mass if np.all(values > 0) else np.nan
    else:
        values = np.array([represented_mass], dtype=float)
        total = float(model.body_mass[ids].sum())
        ratio = represented_mass / total if total > 0 else np.nan
    if not np.all(np.isfinite(values) & (values > 0)) or not np.isfinite(ratio):
        raise ValueError("masses must be positive and finite and selected mass nonzero")
    out = scale_strength(model, 1.0)
    out.body_mass[ids] *= ratio
    out.body_inertia[ids] *= ratio
    mujoco.mj_setConst(out, mujoco.MjData(out))
    return out


@dataclass(frozen=True)
class SegmentLengthMeasurement:
    """Measured distance [m] between named anatomical body/site landmarks.

    Body origins and sites must first be placed at the measured landmarks.
    A generic body origin need not coincide with a joint center.
    """

    proximal: str
    distal: str
    length: float
    object_type: Literal["body", "site"] = "site"


@dataclass(frozen=True)
class SegmentLengthCheck:
    """Model/measured landmark distance and an explicit acceptance tolerance."""

    proximal: str
    distal: str
    object_type: str
    measured_length: float
    model_length: float
    error: float
    tolerance: float
    within_tolerance: bool

    def record(self) -> dict:
        """Return JSON-serialisable anthropometry check fields."""
        return asdict(self)


def check_segment_lengths(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    measurements: Sequence[SegmentLengthMeasurement],
    *,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.005,
) -> tuple[SegmentLengthCheck, ...]:
    """Verify an externally scaled model against measured segment lengths.

    Args:
        model: Compiled subject model, unchanged.
        qpos: Calibration posture ``(nq,)`` matching the measurement protocol.
        measurements: Named landmark endpoint distances [m]; at least one.
        relative_tolerance: Allowed fraction of measured length.
        absolute_tolerance: Allowed absolute difference [m].

    Returns:
        Checks accepting differences up to ``max(absolute, relative * length)``.
        This verifies geometry; it does not resize segments or muscle paths.
        Landmark placement and tolerances require study-specific validation.
    """
    tolerances = np.array([relative_tolerance, absolute_tolerance])
    if not measurements or not np.all(np.isfinite(tolerances) & (tolerances >= 0)):
        raise ValueError("need measurements and finite nonnegative tolerances")
    data = mujoco.MjData(model)
    data.qpos[:] = _posture(model, qpos)
    mujoco.mj_fwdPosition(model, data)
    checks = []
    for measurement in measurements:
        if not np.isfinite(measurement.length) or measurement.length <= 0:
            raise ValueError("measured segment lengths must be positive and finite")
        if measurement.object_type == "site":
            proximal = data.site_xpos[model.site(measurement.proximal).id]
            distal = data.site_xpos[model.site(measurement.distal).id]
        elif measurement.object_type == "body":
            proximal = data.xpos[model.body(measurement.proximal).id]
            distal = data.xpos[model.body(measurement.distal).id]
        else:
            raise ValueError("segment object_type must be 'body' or 'site'")
        length = float(np.linalg.norm(distal - proximal))
        error = length - measurement.length
        tolerance = max(absolute_tolerance, relative_tolerance * measurement.length)
        checks.append(
            SegmentLengthCheck(
                measurement.proximal,
                measurement.distal,
                measurement.object_type,
                measurement.length,
                length,
                error,
                tolerance,
                abs(error) <= tolerance,
            )
        )
    return tuple(checks)
