# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for CEINMS-style multi-trial muscle calibration and C3D + EMG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.emg import (
    EMGFilterConfig,
    excitation_to_activation,
    shift_signal,
)
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.muscle_calibration import (
    CalibrationTrial,
    MuscleCalibrationConfig,
    apply_muscle_parameters,
    calibrate_subject_muscles,
    nonlinear_excitation_shape,
    run_emg_assisted_pipeline,
    trial_from_c3d,
)

pytestmark = pytest.mark.tier1


def _build_elbow_3muscle_model(*, include_deep: bool = False) -> mujoco.MjModel:
    deep_tendon = (
        """
        <spatial name="t_bra">
          <site site="s_hum_ant"/>
          <site site="s_ulna_prox"/>
        </spatial>
        """
        if include_deep
        else ""
    )
    deep_act = (
        '<muscle name="BRA" tendon="t_bra" lengthrange="0.28 0.40" force="260"/>'
        if include_deep
        else ""
    )
    xml = f"""
    <mujoco model="elbow_emg">
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <body name="humerus" pos="0 0 1.0">
          <site name="s_bic_orig" pos="0.04 0 0.14" size="0.005"/>
          <site name="s_hum_ant" pos="0.035 0 0.08" size="0.005"/>
          <site name="s_tri_orig" pos="-0.04 0 0.14" size="0.005"/>
          <body name="ulna" pos="0 0 -0.28">
            <joint name="elbow" type="hinge" axis="0 -1 0" range="-0.2 1.6"/>
            <inertial pos="0.14 0 0" mass="1.6" diaginertia="0.012 0.012 0.003"/>
            <site name="s_bic_ins" pos="0.06 0 0.0" size="0.005"/>
            <site name="s_ulna_prox" pos="0.05 0 0.0" size="0.005"/>
            <site name="s_tri_ins" pos="-0.045 0 0.0" size="0.005"/>
            <site name="s_hand" pos="0.28 0 0" size="0.01"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t_bic">
          <site site="s_bic_orig"/>
          <site site="s_bic_ins"/>
        </spatial>
        <spatial name="t_tri">
          <site site="s_tri_orig"/>
          <site site="s_tri_ins"/>
        </spatial>
        {deep_tendon}
      </tendon>
      <actuator>
        <muscle name="BIC" tendon="t_bic" lengthrange="0.34 0.46" force="300"/>
        <muscle name="TRI" tendon="t_tri" lengthrange="0.38 0.48" force="280"/>
        {deep_act}
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _simulate_inverse_trial(
    gt_model: mujoco.MjModel,
    time: np.ndarray,
    qpos: np.ndarray,
    emg_env: np.ndarray,
    *,
    shape_a: float,
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = float(time[1] - time[0])
    kin = clip_kinematics(gt_model, qpos, dt)
    u_shifted = shift_signal(emg_env, dt, delay_s)
    u_shaped = nonlinear_excitation_shape(u_shifted, shape_a)
    act = excitation_to_activation(u_shaped, dt, model=gt_model)

    # Compute external_qfrc so that gt_model + act satisfies inverse dynamics exactly!
    from myosuite.biomechanics.inverse_dynamics import required_qfrc  # noqa: PLC0415

    d = mujoco.MjData(gt_model)
    mom = np.zeros((gt_model.nu, gt_model.nv))
    ext = np.zeros((len(time), gt_model.nv))
    for k in range(len(time)):
        d.qpos[:], d.qvel[:] = kin.qpos[k], kin.qvel[k]
        mujoco.mj_fwdPosition(gt_model, d)
        mujoco.mj_fwdVelocity(gt_model, d)
        d.act[:] = act[k]
        mujoco.mj_fwdActuation(gt_model, d)
        mujoco.mju_sparse2dense(
            mom, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        tau_musc = mom.T @ d.actuator_force
        ext[k] = required_qfrc(gt_model, d, kin.qacc[k]) - tau_musc
    return kin.qpos, kin.qvel, kin.qacc, ext


def test_multi_trial_ceinms_calibration_recovers_strength_and_slashes_rmse() -> None:
    nominal = _build_elbow_3muscle_model(include_deep=False)
    orig_gain = nominal.actuator_gainprm[:, 2].copy()

    gt_model = apply_muscle_parameters(
        nominal,
        strength_scales={0: 1.45, 1: 0.68},
        length_shifts_m={0: 0.010, 1: -0.008},
        span_scales={0: 1.05, 1: 0.95},
    )
    t = np.linspace(0.0, 1.2, 61)
    q1 = (0.35 + 0.25 * np.sin(2.0 * np.pi * t / 1.2))[:, None]
    emg1 = np.column_stack(
        [
            0.45 + 0.35 * np.sin(2.0 * np.pi * t / 1.2),
            0.20 + 0.15 * np.cos(2.0 * np.pi * t / 1.2),
        ]
    )
    q2 = (0.45 + 0.20 * np.cos(2.0 * np.pi * t / 0.8))[:, None]
    emg2 = np.column_stack(
        [
            0.25 + 0.20 * np.cos(2.0 * np.pi * t / 0.8),
            0.55 + 0.30 * np.sin(2.0 * np.pi * t / 0.8),
        ]
    )

    qp1, qv1, qa1, ext1 = _simulate_inverse_trial(
        gt_model, t, q1, emg1, shape_a=-1.5, delay_s=0.02
    )
    qp2, qv2, qa2, ext2 = _simulate_inverse_trial(
        gt_model, t, q2, emg2, shape_a=-1.5, delay_s=0.02
    )

    ch_map = {0: "BIC", 1: "TRI"}
    trials = [
        CalibrationTrial(t, qp1, emg1, ch_map, qvel=qv1, qacc=qa1, external_qfrc=ext1),
        CalibrationTrial(t, qp2, emg2, ch_map, qvel=qv2, qacc=qa2, external_qfrc=ext2),
    ]
    res = calibrate_subject_muscles(
        nominal,
        trials,
        config=MuscleCalibrationConfig(reg_weight=1e-4, max_nfev=80),
    )

    # Caller's nominal model is never mutated.
    np.testing.assert_allclose(nominal.actuator_gainprm[:, 2], orig_gain)
    assert res.initial_torque_rmse > 1.2
    assert res.final_torque_rmse < 0.15 * res.initial_torque_rmse
    assert res.strength_scales["BIC"] == pytest.approx(1.45, abs=0.18)
    assert res.strength_scales["TRI"] == pytest.approx(0.68, abs=0.18)


@dataclass
class _MockForcePlate:
    time: np.ndarray
    force: np.ndarray
    moment: np.ndarray
    cop: np.ndarray


@dataclass
class _MockC3DTrial:
    time: np.ndarray
    analog: np.ndarray
    analog_labels: tuple[str, ...]
    analog_rate: float
    force_plates: tuple[_MockForcePlate, ...]


def test_c3d_trial_adapter_and_hybrid_emg_assisted_deep_muscle_synthesis() -> None:
    model = _build_elbow_3muscle_model(include_deep=True)
    t_mocap = np.linspace(0.0, 0.8, 41)
    qpos = np.full((len(t_mocap), 1), 0.25)

    # High-rate 1000 Hz raw band-limited EMG on superficial BIC and TRI only (BRA unmeasured).
    fs_emg = 1000.0
    t_emg = np.linspace(0.0, 0.8, 801)
    carrier = np.sin(2.0 * np.pi * 80.0 * t_emg)
    # MVC peak of a rectified 80 Hz unit sine is 2/pi ~= 0.6366; command BIC ~0.10 and TRI ~0.25.
    raw_analog = np.column_stack([0.10 * carrier, 0.25 * carrier])
    mvc_peaks = np.array([2.0 / np.pi, 2.0 / np.pi])

    d = mujoco.MjData(model)
    d.qpos[:] = qpos[0]
    mujoco.mj_fwdPosition(model, d)
    hand_pos = d.site_xpos[model.site("s_hand").id].copy()
    fp = _MockForcePlate(
        time=t_mocap,
        force=np.tile([0.0, 0.0, -8.0], (len(t_mocap), 1)),
        moment=np.zeros((len(t_mocap), 3)),
        cop=np.tile(hand_pos, (len(t_mocap), 1)),
    )
    c3d = _MockC3DTrial(
        time=t_mocap,
        analog=raw_analog,
        analog_labels=("EMG_BIC", "EMG_TRI"),
        analog_rate=fs_emg,
        force_plates=(fp,),
    )

    trial = trial_from_c3d(
        c3d,
        model,
        qpos,
        {"EMG_BIC": "BIC", "EMG_TRI": "TRI"},
        plate_bodies={0: "ulna"},
        mvc_peaks=mvc_peaks,
        lowpass_hz=6.0,
    )
    out = run_emg_assisted_pipeline(
        trial, model, emg_filter=EMGFilterConfig(normalize="mvc")
    )

    bra_id = model.actuator("BRA").id
    tri_id = model.actuator("TRI").id
    assert np.isnan(out.emg_activation[:, bra_id]).all()
    assert np.all(out.so_result.status == "solved")
    assert out.emg_tracking_rmse < 0.05
    # Unmeasured deep BRA is recruited while antagonist TRI co-contraction is preserved!
    assert float(out.so_result.activation[10:, tri_id].mean()) > 0.18
    assert float(out.so_result.activation[10:, bra_id].mean()) > 0.25
