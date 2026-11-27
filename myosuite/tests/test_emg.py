# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``myosuite.biomechanics.emg``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics import emg as emg_mod
from myosuite.biomechanics.emg import (
    EMGFilterConfig,
    _muscle_timescale,
    activation_to_excitation,
    excitation_to_activation,
    phase_normalize,
    process_emg,
    project_tendon_nodes,
    shift_signal,
)

pytestmark = pytest.mark.tier1

_TWO_MUSCLE_XML = """
<mujoco model="two_muscle_hinge">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <site name="flex_orig" pos="0.04 0 0.12"/>
      <site name="ext_orig" pos="-0.04 0 0.12"/>
      <body name="forearm" pos="0 0 0">
        <inertial pos="0 0 -0.15" mass="1.2" diaginertia="0.012 0.012 0.002"/>
        <joint name="elbow_flexion" type="hinge" axis="0 -1 0" range="-1.5 1.5"/>
        <site name="flex_ins" pos="0.035 0 -0.05"/>
        <site name="ext_ins" pos="-0.035 0 -0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="bic_tendon"><site site="flex_orig"/><site site="flex_ins"/></spatial>
    <spatial name="tri_tendon"><site site="ext_orig"/><site site="ext_ins"/></spatial>
  </tendon>
  <actuator>
    <muscle name="biceps_r" tendon="bic_tendon" force="120" lengthrange="0.09 0.20"
            timeconst="0.01 0.04" tausmooth="0.2"/>
    <muscle name="triceps_r" tendon="tri_tendon" force="120" lengthrange="0.09 0.20"/>
  </actuator>
</mujoco>
"""

_SLIDE_TENDON_NODE_XML = """
<mujoco model="slide_tendon_node">
  <compiler angle="radian"/>
  <worldbody>
    <body name="palm" pos="0 0 0">
      <site name="fdp_orig" pos="0.02 0 0.08"/>
      <body name="node_lum" pos="0.02 0 0.02">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="node_LUM1_r" type="slide" axis="0 0 1" range="-0.05 0.05"/>
      </body>
      <body name="finger" pos="0 0 0">
        <inertial pos="0 0 -0.04" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="mcp2_flexion_r" type="hinge" axis="0 -1 0" range="-0.2 1.4"/>
        <site name="fdp_ins" pos="0.015 0 -0.03"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="node_LUM1_t"><joint joint="node_LUM1_r" coef="1.0"/></fixed>
    <spatial name="fdp2_tendon"><site site="fdp_orig"/><site site="fdp_ins"/></spatial>
  </tendon>
  <equality>
    <tendon name="lum_tie" tendon1="node_LUM1_t" tendon2="fdp2_tendon" polycoef="0 1.0 0 0 0"/>
  </equality>
</mujoco>
"""


def test_process_emg_recovers_envelope_and_suppresses_notch_hum() -> None:
    fs = 2000.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    rng = np.random.default_rng(42)
    true_env = 0.5 * (1.0 - np.cos(2.0 * np.pi * 1.0 * t)) ** 2
    carrier = rng.standard_normal(len(t))
    hum = 2.5 * np.sin(2.0 * np.pi * 50.0 * t)
    raw = np.column_stack([true_env * carrier + hum, hum])

    out = process_emg(
        raw,
        fs,
        EMGFilterConfig(
            bandpass_hz=(20.0, 450.0),
            notch_hz=50.0,
            lowpass_hz=4.0,
            normalize="none",
        ),
    )
    sl = slice(200, -200)
    assert np.corrcoef(out[sl, 0], true_env[sl])[0, 1] > 0.94
    assert float(np.sqrt(np.mean(out[sl, 1] ** 2))) < 0.05

    mvc = process_emg(
        raw[:, :1],
        fs,
        EMGFilterConfig(notch_hz=50.0, lowpass_hz=4.0, normalize="mvc"),
        mvc_peaks=np.array([2.0]),
    )
    peak = process_emg(
        raw[:, :1],
        fs,
        EMGFilterConfig(notch_hz=50.0, lowpass_hz=4.0, normalize="peak"),
    )
    assert np.isclose(peak.max(), 1.0)
    assert np.isclose(mvc.max(), out[:, 0].max() / 2.0, rtol=1e-5)


def test_activation_excitation_roundtrip_shift_and_phase_normalize() -> None:
    model = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_XML)
    dt = 0.005
    t = np.arange(0.0, 2.0, dt)
    u = np.column_stack(
        [
            0.15 + 0.65 * np.sin(np.pi * t / 2.0) ** 2,
            0.10 + 0.50 * np.sin(2.0 * np.pi * t / 2.0) ** 2,
        ]
    )
    a = excitation_to_activation(u, dt, model=model)
    u_rec = activation_to_excitation(a, dt, model=model)
    assert float(np.mean(np.abs(u_rec[10:-10] - u[10:-10]))) < 0.02

    shifted = shift_signal(u[:, 0], dt, 0.05)
    assert np.allclose(shifted[20:], u[10:-10, 0], atol=1e-12)

    events = np.array([[0.0, 1.0], [1.0, 1.995]])
    cyc = phase_normalize(a, t, events, points=101)
    assert cyc.shape == (2, 101, 2)


def test_tausmooth_matches_mujoco_muscle_dynamics_and_delegates_to_hill() -> None:
    model = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_XML)
    data = mujoco.MjData(model)
    ta = model.actuator_dynprm[:, 0].copy()
    td = model.actuator_dynprm[:, 1].copy()
    ts = model.actuator_dynprm[:, 2].copy()
    assert ts[0] == pytest.approx(0.2)
    assert ts[1] == pytest.approx(0.0)

    rng = np.random.default_rng(7)
    for _ in range(30):
        ctrl = rng.uniform(0.0, 1.0, size=2)
        act = rng.uniform(0.0, 1.0, size=2)
        data.ctrl[:] = ctrl
        data.act[:] = act
        mujoco.mj_forward(model, data)
        dctrl = ctrl - act
        tau = _muscle_timescale(dctrl, act, ta, td, ts)
        assert np.allclose(dctrl / tau, data.act_dot, atol=1e-14)

    # Constant excitation step matches mj_step Euler update with tausmooth=0.2.
    dt = float(model.opt.timestep)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.act[:] = np.array([0.42, 0.42])
    u_const = np.tile(np.array([[0.48, 0.48]]), (25, 1))
    a_pred = excitation_to_activation(
        u_const, dt, initial=np.array([0.42, 0.42]), model=model
    )
    a_mj = np.empty_like(a_pred)
    a_mj[0] = data.act
    for k in range(1, len(u_const)):
        data.ctrl[:] = u_const[k - 1]
        mujoco.mj_step(model, data)
        a_mj[k] = data.act
    assert np.allclose(a_pred, a_mj, atol=1e-14)
    # Smooth vs abrupt switching differ inside |u - a| < 0.5 * tausmooth.
    assert not np.allclose(a_pred[:, 0], a_pred[:, 1], atol=1e-4)

    calls: list[float] = []

    def fake_step(act_in, ctrl_in, params, h, substeps=1):
        calls.append(float(h))
        dc = ctrl_in - act_in
        tau = _muscle_timescale(
            dc, act_in, params.tau_act, params.tau_deact, params.tausmooth
        )
        nxt = np.clip(act_in + h * dc / tau, 0.0, 1.0)
        return nxt, dc / tau

    fake_lib = SimpleNamespace(
        MuscleParams=lambda **kw: SimpleNamespace(**kw),
        step_muscle_activation=fake_step,
    )
    with patch.object(emg_mod, "_hill_library", return_value=fake_lib):
        a_delegated = excitation_to_activation(
            u_const, dt, initial=np.array([0.42, 0.42]), model=model
        )
    assert len(calls) == len(u_const) - 1
    assert np.allclose(a_delegated, a_pred, atol=1e-15)


def test_project_tendon_nodes_satisfies_tendon_equality() -> None:
    model = mujoco.MjModel.from_xml_string(_SLIDE_TENDON_NODE_XML)
    q = np.zeros((5, model.nq), dtype=np.float64)
    q[:, 1] = np.linspace(0.0, 1.1, 5)
    q_proj = project_tendon_nodes(model, q)

    data = mujoco.MjData(model)
    for row in q_proj:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        eq_res = float(np.abs(data.efc_pos[0]))
        assert eq_res < 1e-10
