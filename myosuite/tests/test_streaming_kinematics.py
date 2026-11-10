# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Live fixed-lag kinematics against exact polynomial motion and the offline estimate."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.streaming_kinematics import StreamingKinematics
from myosuite.tests.support.joint_couplings import joint_equality_residuals

pytestmark = pytest.mark.tier1

_XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size=".1 .05 .02" mass="2"/>
      <body name="arm" pos=".1 0 0">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".5"/>
        <body name="hand" pos=".2 0 0">
          <joint name="ball" type="ball"/>
          <geom type="sphere" size=".03" mass=".2"/>
          <body name="finger" pos=".05 0 0">
            <joint name="follower" axis="0 0 1"/>
            <geom type="capsule" fromto="0 0 0 .05 0 0" size=".01" mass=".05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="follower" joint2="hinge" polycoef="0.1 0.5 -0.3 0 0"/>
  </equality>
</mujoco>
"""

_AXIS_ROOT = np.array([1.0, 2.0, 2.0]) / 3.0
_AXIS_BALL = np.array([0.0, 0.6, 0.8])
_CUBIC = np.array([0.2, -0.7, 1.1, -0.9])  # c0 + c1 t + c2 t^2 + c3 t^3


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


def _quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None] * axis], axis=1
    )


def _cubic_motion(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every coordinate cubic in time (rotations about fixed axes): exact for polyorder 3."""
    p = np.polynomial.polynomial.polyval(t, _CUBIC)
    d = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(_CUBIC))
    dd = np.polynomial.polynomial.polyval(
        t, np.polynomial.polynomial.polyder(_CUBIC, 2)
    )
    col = lambda x, k: k * x[:, None]  # noqa: E731
    qpos = np.concatenate(
        [
            col(p, np.array([1.0, -0.5, 0.3])),
            _quat(_AXIS_ROOT, p),
            col(p, 1.0),
            _quat(_AXIS_BALL, -p),
            col(p, 0.7),
        ],
        axis=1,
    )
    qvel = np.concatenate(
        [
            col(d, np.array([1.0, -0.5, 0.3])),
            col(d, _AXIS_ROOT),
            col(d, 1.0),
            col(-d, _AXIS_BALL),
            col(d, 0.7),
        ],
        axis=1,
    )
    qacc = np.concatenate(
        [
            col(dd, np.array([1.0, -0.5, 0.3])),
            col(dd, _AXIS_ROOT),
            col(dd, 1.0),
            col(-dd, _AXIS_BALL),
            col(dd, 0.7),
        ],
        axis=1,
    )
    return qpos, qvel, qacc


@pytest.mark.parametrize("lag", [0, 3, 5, 10])
def test_cubic_motion_is_recovered_exactly_on_jittered_dropped_frames(
    model: mujoco.MjModel, lag: int
) -> None:
    rng = np.random.default_rng(lag)
    t = np.cumsum(rng.uniform(0.008, 0.014, 60))  # 90 Hz with jitter
    t = np.delete(t, [20, 21, 35])  # dropped tracker frames
    qpos, qvel, qacc = _cubic_motion(t)
    stream = StreamingKinematics(model, window=11, lag=lag, project_couplings=False)
    states = [stream.push(ti, q) for ti, q in zip(t, qpos)]
    assert all(s is None for s in states[:10]) and all(
        s is not None for s in states[10:]
    )
    for k, s in enumerate(states[10:]):
        i = k + 10 - lag
        assert s.time == t[i]
        np.testing.assert_allclose(s.qpos, qpos[i], atol=1e-9)
        np.testing.assert_allclose(s.qvel, qvel[i], atol=1e-7)
        np.testing.assert_allclose(s.qacc, qacc[i], atol=1e-5)


def test_estimates_lie_on_the_couplings(model: mujoco.MjModel) -> None:
    t = np.arange(0.0, 0.5, 1 / 90)
    qpos = _cubic_motion(t)[0]
    qpos[:, -1] += 0.05 * np.sin(9 * t)  # off the coupling
    stream = StreamingKinematics(model)
    for s in filter(None, (stream.push(ti, q) for ti, q in zip(t, qpos))):
        pos_res, vel_res = joint_equality_residuals(model, s.qpos, s.qvel)
        assert np.abs(pos_res).max() < 1e-12 and np.abs(vel_res).max() < 1e-12


def test_centred_lag_matches_offline_and_beats_no_lag(model: mujoco.MjModel) -> None:
    dt = 1 / 90
    t = np.arange(0.0, 2.0, dt)
    qpos = _cubic_motion(np.sin(2.0 * t))[0]  # smooth, not polynomial
    offline = clip_kinematics(model, qpos, dt, project_couplings=False)

    def max_acc_error(lag: int) -> float:
        stream = StreamingKinematics(model, window=11, lag=lag, project_couplings=False)
        err = 0.0
        for i, (ti, q) in enumerate(zip(t, qpos)):
            s = stream.push(ti, q)
            if s is not None and 20 <= i - lag < len(t) - 20:
                err = max(err, np.abs(s.qacc - offline.qacc[i - lag]).max())
        return err

    peak = np.abs(offline.qacc[20:-20]).max()  # about 12 rad/s^2
    centred, causal = max_acc_error(5), max_acc_error(0)
    assert centred < 0.01 * peak
    assert causal > 3 * centred  # no delay costs accuracy


def test_noise_is_attenuated(model: mujoco.MjModel) -> None:
    dt = 1 / 90
    t = np.arange(0.0, 3.0, dt)
    clean = _cubic_motion(0.5 * np.sin(2.0 * t))[0]
    noisy = clean.copy()
    noisy[:, 7] += np.radians(0.5) * np.random.default_rng(0).standard_normal(
        len(t)
    )  # 0.5 deg tracking noise
    truth = clip_kinematics(model, clean, dt, project_couplings=False).qacc[:, 6]
    raw = clip_kinematics(model, noisy, dt, project_couplings=False).qacc[:, 6]
    stream = StreamingKinematics(model, window=15, project_couplings=False)
    est = np.array(
        [
            s.qacc[6]
            for s in (stream.push(ti, q) for ti, q in zip(t, noisy))
            if s is not None
        ]
    )
    inner = slice(20, -20)
    raw_rms = np.sqrt(np.mean((raw[inner] - truth[inner]) ** 2))
    est_rms = np.sqrt(np.mean((est[inner] - truth[7:-7][inner]) ** 2))
    assert est_rms < raw_rms / 20


def test_reset_and_invalid_inputs(model: mujoco.MjModel) -> None:
    qpos = _cubic_motion(np.arange(0.0, 0.2, 0.01))[0]
    stream = StreamingKinematics(model, window=5)
    for i in range(5):
        state = stream.push(0.01 * i, qpos[i])
    assert state is not None
    stream.reset()
    assert stream.push(1.0, qpos[0]) is None
    with pytest.raises(ValueError, match="increase"):
        stream.push(1.0, qpos[1])
    with pytest.raises(ValueError):
        stream.push(2.0, qpos[1, :5])
    for kwargs in (
        {"window": 3, "polyorder": 3},
        {"polyorder": 1},
        {"window": 5, "lag": 5},
    ):
        with pytest.raises(ValueError):
            StreamingKinematics(model, **kwargs)
