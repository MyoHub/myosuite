# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""C3D capture adapter and hybrid EMG-assisted static-optimization workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from myosuite.biomechanics.emg import EMGFilterConfig
from myosuite.biomechanics.external_loads import force_at_body
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.muscle_calibration import (
    CalibrationError,
    CalibrationTrial,
    MuscleCalibrationResult,
    compute_calibrated_emg_activations,
    prepare_calibration_trial,
)
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    static_optimization,
)


@dataclass(frozen=True)
class EMGAssistedPipelineResult:
    """Output of hybrid EMG-assisted static optimization on a calibrated subject model."""

    calibrated_model: mujoco.MjModel
    emg_activation: np.ndarray
    so_result: StaticOptimizationResult
    emg_tracking_rmse: float


def trial_from_c3d(
    c3d_trial: Any,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    channel_to_actuators: Mapping[int | str, str | Sequence[str]],
    *,
    plate_bodies: Sequence[str] | Mapping[int, str] | None = None,
    mvc_peaks: np.ndarray | None = None,
    lowpass_hz: float | None = 6.0,
) -> CalibrationTrial:
    """Build a :class:`CalibrationTrial` from a C3D capture record."""
    time = np.asarray(c3d_trial.time, dtype=np.float64)
    dt = float(np.median(np.diff(time)))
    kin = clip_kinematics(
        model, np.asarray(qpos, dtype=np.float64), dt, lowpass_hz=lowpass_hz
    )

    ext_qfrc: np.ndarray | None = None
    plates = getattr(c3d_trial, "force_plates", ())
    if plates and plate_bodies is not None:
        ext_qfrc = np.zeros((len(time), model.nv), dtype=np.float64)
        body_map = (
            dict(plate_bodies)
            if isinstance(plate_bodies, Mapping)
            else {i: b for i, b in enumerate(plate_bodies)}
        )
        for p_idx, body_name in body_map.items():
            if p_idx >= len(plates):
                continue
            fp = plates[p_idx]
            fp_t = np.asarray(fp.time, dtype=np.float64)
            f_interp = np.column_stack(
                [np.interp(time, fp_t, fp.force[:, k]) for k in range(3)]
            )
            m_interp = np.column_stack(
                [np.interp(time, fp_t, fp.moment[:, k]) for k in range(3)]
            )
            c_interp = np.column_stack(
                [np.interp(time, fp_t, fp.cop[:, k]) for k in range(3)]
            )
            for k in range(len(time)):
                ext_qfrc[k] += force_at_body(
                    model,
                    kin.qpos[k],
                    body_name,
                    f_interp[k],
                    point=c_interp[k],
                    torque=m_interp[k],
                    frame="world",
                )

    return CalibrationTrial(
        time=time,
        qpos=kin.qpos,
        qvel=kin.qvel,
        qacc=kin.qacc,
        external_qfrc=ext_qfrc,
        emg_signals=np.asarray(c3d_trial.analog, dtype=np.float64),
        emg_fs=float(c3d_trial.analog_rate),
        is_raw_emg=True,
        mvc_peaks=mvc_peaks,
        channel_labels=tuple(str(x) for x in c3d_trial.analog_labels),
        channel_to_actuators=channel_to_actuators,
    )


def run_emg_assisted_pipeline(
    trial: CalibrationTrial,
    calibration: MuscleCalibrationResult | mujoco.MjModel,
    *,
    emg_filter: EMGFilterConfig | None = None,
    lowpass_hz: float | None = None,
    so_config: StaticOptimizationConfig | None = None,
    allow_unconverged: bool = False,
) -> EMGAssistedPipelineResult:
    """Run calibrated EMG-to-activation mapping + hybrid EMG-assisted static optimization."""
    if isinstance(calibration, MuscleCalibrationResult):
        if not calibration.converged and not allow_unconverged:
            raise CalibrationError(
                f"Refusing to run EMG-assisted pipeline on unconverged calibration "
                f"(status={calibration.status}: {calibration.message}). Pass "
                "allow_unconverged=True to override."
            )
        cal_model = calibration.calibrated_model
        shape_a = calibration.shape_factor_a
        delay_s = calibration.delay_s
    else:
        cal_model = calibration
        shape_a = 0.0
        delay_s = 0.0

    _, dt, qpos, qvel, qacc, ext, env, act_to_col = prepare_calibration_trial(
        cal_model, trial, emg_filter or EMGFilterConfig(), lowpass_hz
    )
    emg_act = compute_calibrated_emg_activations(
        cal_model,
        env,
        dt,
        act_to_col,
        shape_factor_a=shape_a,
        delay_s=delay_s,
    )
    cfg_so = so_config or StaticOptimizationConfig(
        emg_weight=25.0, emg_effort_weight=1.0
    )
    so_res = static_optimization(
        cal_model,
        qpos,
        qvel,
        qacc,
        external_qfrc=ext,
        emg_activation=emg_act,
        config=cfg_so,
    )
    tracked_mask = np.isfinite(emg_act)
    tracking_rmse = (
        float(
            np.sqrt(
                np.mean((so_res.activation[tracked_mask] - emg_act[tracked_mask]) ** 2)
            )
        )
        if tracked_mask.any()
        else 0.0
    )
    return EMGAssistedPipelineResult(
        calibrated_model=cal_model,
        emg_activation=emg_act,
        so_result=so_res,
        emg_tracking_rmse=tracking_rmse,
    )


__all__ = [
    "EMGAssistedPipelineResult",
    "run_emg_assisted_pipeline",
    "trial_from_c3d",
]
