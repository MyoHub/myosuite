# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Participant strength, mass, geometry and fatigue calibration of compiled models.

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
from scipy.optimize import minimize

from myosuite.core.muscle_conditions import CumulativeFatigue
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

    Raises:
        ImportError: If the installed myo-sim lacks the selected-body mass API.
    """
    try:
        from myo_sim.physics.body_mass import scale_body_mass as scale
    except ModuleNotFoundError as exc:
        if exc.name != "myo_sim.physics.body_mass":
            raise
        raise ImportError(
            "Mass scaling requires a myo-sim build containing MyoHub/myo_sim#187 "
            "(myo_sim.physics.body_mass). Install that development build or a "
            "release containing it; 0.2.5 lacks this API."
        ) from exc
    return scale(
        model,
        represented_mass,
        participant_mass=participant_mass,
        reference_mass=reference_mass,
        bodies=bodies,
    )


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


_ONE_MUSCLE = (
    """<mujoco><worldbody>{bodies}</worldbody><actuator>{muscles}</actuator></mujoco>"""
)


def _fatigue_bank(n: int) -> CumulativeFatigue:
    """3CC-r state for *n* independent muscles (one per load)."""
    bodies = "".join(
        f'<body><joint name="j{i}" axis="0 1 0" range="0 1"/>'
        '<geom size=".1" mass="1"/></body>'
        for i in range(n)
    )
    muscles = "".join(
        f'<muscle name="m{i}" joint="j{i}" lengthrange="0 1"/>' for i in range(n)
    )
    model = mujoco.MjModel.from_xml_string(
        _ONE_MUSCLE.format(bodies=bodies, muscles=muscles)
    )
    return CumulativeFatigue(model)


def endurance_time(
    loads: Sequence[float],
    F: float,
    R: float,
    r: float,
    *,
    dt: float = 0.25,
    max_time: float = 1800.0,
) -> np.ndarray:
    """3CC-r time to task failure of sustained contractions (Frey-Law et al. 2012).

    A hold of ``load`` (fraction of MVC) fails when the residual capacity
    ``1 - MF`` falls below the load; the crossing is interpolated between steps.

    Args:
        loads: Target loads in (0, 1].
        F, R, r: 3CC-r fatigue, recovery and rest-recovery coefficients.
        dt: Step between failure checks [s] (integrated in MyoSuite's sub-steps).
        max_time: Give up after this long [s]; such holds return ``inf``.

    Returns:
        ``(len(loads),)`` endurance times [s].
    """
    loads = np.asarray(loads, dtype=float)
    if (
        loads.ndim != 1
        or not len(loads)
        or not np.all(np.isfinite(loads) & (loads > 0) & (loads <= 1))
    ):
        raise ValueError("loads must be a nonempty vector in (0,1]")
    if (
        not np.all(np.isfinite([F, R, r, dt, max_time]))
        or min(F, R, r, dt, max_time) <= 0
    ):
        raise ValueError("coefficients, dt and max_time must be positive and finite")
    bank = _fatigue_bank(len(loads))
    for setter, value in (
        (bank.set_FatigueCoefficient, F),
        (bank.set_RecoveryCoefficient, R),
        (bank.set_RecoveryMultiplier, r),
    ):
        setter(value)
    out = np.full(len(loads), np.inf)
    margin = (1.0 - bank.MF) - loads
    t = 0.0
    while t < max_time and np.isinf(out).any():
        step = min(dt, max_time - t)
        bank.compute_act(loads, dt=step)
        t += step
        new = (1.0 - bank.MF) - loads
        crossed = np.isinf(out) & (new < 0)
        out[crossed] = t - step * new[crossed] / (new[crossed] - margin[crossed])
        margin = new
    return out


@dataclass(frozen=True)
class FatigueFit:
    """3CC-r parameters fitted to endurance times.

    Attributes:
        F, R, r: Fitted (or fixed) coefficients.
        loads: Loads of the holds (fraction of MVC).
        observed: Observed endurance times [s].
        predicted: Endurance times of the fitted model [s].
        rmse_log: Root-mean-square error of the log endurance times.
        fitted: Names of the fitted coefficients.
    """

    F: float
    R: float
    r: float
    loads: tuple[float, ...]
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    rmse_log: float
    fitted: tuple[str, ...]

    def record(self) -> dict:
        """JSON-serialisable form for the session record."""
        return asdict(self)


def fit_fatigue_parameters(
    loads: Sequence[float],
    endurance_times: Sequence[float],
    *,
    F: float = 0.01,
    R: float = 0.002,
    r: float = 15.0,
    fit_recovery: bool = False,
) -> FatigueFit:
    """Fit 3CC-r's F (and optionally R) to a participant's sustained holds.

    Minimises the squared error of log endurance times (Nelder-Mead in log
    parameters). ``r`` acts only at rest, so sustained holds cannot inform it.

    Args:
        loads: Loads of the holds (fraction of the calibrated MVC).
        endurance_times: Observed times to task failure [s].
        F, R, r: Starting (or fixed) coefficients, e.g. MyoSuite's group row.
        fit_recovery: Also fit R (needs holds at three or more loads).

    Returns:
        The fit.
    """
    loads = np.asarray(loads, dtype=float)
    observed = np.asarray(endurance_times, dtype=float)
    if (
        loads.ndim != 1
        or observed.shape != loads.shape
        or not len(loads)
        or not np.all(np.isfinite(loads) & (loads > 0) & (loads < 1))
        or not np.all(np.isfinite(observed) & (observed > 0))
    ):
        raise ValueError(
            "need paired finite loads in (0,1) and positive endurance times"
        )
    if not np.all(np.isfinite([F, R, r])) or min(F, R, r) <= 0:
        raise ValueError("starting coefficients must be positive and finite")
    if fit_recovery and len(np.unique(loads)) < 3:
        raise ValueError("fitting recovery needs holds at three distinct loads")
    max_time = 3.0 * float(observed.max())
    start_F = F
    for _ in range(16):
        initial = endurance_time(loads, start_F, R, r, max_time=max_time)
        if np.all(np.isfinite(initial) & (initial > 0)):
            break
        start_F *= 2
    else:
        raise ValueError(
            "could not initialize finite failure times; check holds and starting rates"
        )

    def unpack(x: np.ndarray) -> tuple[float, float]:
        return float(np.exp(x[0])), float(np.exp(x[1])) if fit_recovery else R

    def loss(x: np.ndarray) -> float:
        f_, r_cap = unpack(x)
        pred = endurance_time(loads, f_, r_cap, r, max_time=max_time)
        pred = np.where(np.isfinite(pred), pred, max_time)
        return float(np.mean((np.log(pred) - np.log(observed)) ** 2))

    x0 = np.log([start_F, R]) if fit_recovery else np.log([start_F])
    res = minimize(
        loss,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-10, "maxfev": 400},
    )
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"fatigue parameter fitting failed: {res.message}")
    f_, r_cap = unpack(res.x)
    pred = endurance_time(loads, f_, r_cap, r, max_time=max_time)
    if not np.all(np.isfinite(pred) & (pred > 0)):
        raise ValueError(
            "fit did not predict finite failure times; check holds and starting rates"
        )
    return FatigueFit(
        F=f_,
        R=r_cap,
        r=float(r),
        loads=tuple(loads.tolist()),
        observed=tuple(observed.tolist()),
        predicted=tuple(pred.tolist()),
        rmse_log=float(np.sqrt(res.fun)),
        fitted=("F", "R") if fit_recovery else ("F",),
    )


def fatigue_params(
    model: mujoco.MjModel, muscles: Sequence[str], fit: FatigueFit
) -> tuple[tuple[str, float, float, float], ...]:
    """Return validated per-muscle ``(name, F, R, r)`` coefficient rows."""
    _muscles(model, muscles)
    return tuple((m, fit.F, fit.R, fit.r) for m in muscles)
