# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Clip kinematics and inverse dynamics against analytic motion and MuJoCo's forward dynamics."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.physics.inverse_dynamics import clip_kinematics, inverse_dynamics
from myosuite.tests.support.joint_couplings import joint_equality_residuals

pytestmark = pytest.mark.tier1

# Free root, a damped and sprung hinge with armature, a ball joint, and a hinge
# coupled to the first one by a joint equality.
_XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size=".1 .05 .02" mass="2"/>
      <body name="arm" pos=".1 0 0">
        <joint name="hinge" axis="0 1 0" damping="0.3" stiffness="2" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".5"/>
        <body name="hand" pos=".2 0 0">
          <joint name="ball" type="ball" damping="0.1"/>
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


def _axis_angle_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None] * axis], axis=1
    )


def _analytic(model: mujoco.MjModel, t: np.ndarray) -> tuple[np.ndarray, ...]:
    """Positions and their exact velocities/accelerations (fixed-axis rotations)."""
    th, th_d, th_dd = (
        0.8 * np.sin(1.5 * t),
        1.2 * np.cos(1.5 * t),
        -1.8 * np.sin(1.5 * t),
    )
    h, h_d, h_dd = (
        0.5 * np.sin(2.5 * t),
        1.25 * np.cos(2.5 * t),
        -3.125 * np.sin(2.5 * t),
    )
    ph, ph_d, ph_dd = 0.6 * np.cos(2 * t), -1.2 * np.sin(2 * t), -2.4 * np.cos(2 * t)
    c = model.eq_data[0, :3]
    f = c[0] + c[1] * h + c[2] * h**2
    f_d = (c[1] + 2 * c[2] * h) * h_d
    f_dd = 2 * c[2] * h_d**2 + (c[1] + 2 * c[2] * h) * h_dd
    pos = np.stack([0.3 * np.sin(2 * t), 0.1 * t**2, 1 + 0.05 * np.cos(3 * t)], axis=1)
    vel = np.stack([0.6 * np.cos(2 * t), 0.2 * t, -0.15 * np.sin(3 * t)], axis=1)
    acc = np.stack([-1.2 * np.sin(2 * t), 0.2 + 0 * t, -0.45 * np.cos(3 * t)], axis=1)
    qpos = np.concatenate(
        [
            pos,
            _axis_angle_quat(_AXIS_ROOT, th),
            h[:, None],
            _axis_angle_quat(_AXIS_BALL, ph),
            f[:, None],
        ],
        axis=1,
    )
    qvel = np.concatenate(
        [
            vel,
            th_d[:, None] * _AXIS_ROOT,
            h_d[:, None],
            ph_d[:, None] * _AXIS_BALL,
            f_d[:, None],
        ],
        axis=1,
    )
    qacc = np.concatenate(
        [
            acc,
            th_dd[:, None] * _AXIS_ROOT,
            h_dd[:, None],
            ph_dd[:, None] * _AXIS_BALL,
            f_dd[:, None],
        ],
        axis=1,
    )
    return qpos, qvel, qacc


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


def _max_errors(model: mujoco.MjModel, dt: float) -> tuple[float, float]:
    t = np.arange(0.0, 2.0, dt)
    qpos, qvel, qacc = _analytic(model, t)
    kin = clip_kinematics(model, qpos, dt)
    return float(np.abs(kin.qvel - qvel).max()), float(np.abs(kin.qacc - qacc).max())


def test_derivatives_are_second_order_on_the_manifold(model: mujoco.MjModel) -> None:
    v_coarse, a_coarse = _max_errors(model, 0.01)
    v_fine, a_fine = _max_errors(model, 0.005)
    # Every frame, the first and last included, converges at second order.
    assert v_coarse / v_fine > 3.5 and a_coarse / a_fine > 3.5
    assert v_fine < 1e-4 and a_fine < 1e-3


def test_quaternion_sign_flips_do_not_change_the_kinematics(
    model: mujoco.MjModel,
) -> None:
    t = np.arange(0.0, 1.0, 0.01)
    qpos = _analytic(model, t)[0]
    flipped = qpos.copy()
    flipped[1::2, 3:7] *= -1
    flipped[::3, 8:12] *= -1
    # Filtering mixes neighbouring frames, so it must see continuous signs.
    a = clip_kinematics(model, qpos, 0.01, lowpass_hz=6.0)
    b = clip_kinematics(model, flipped, 0.01, lowpass_hz=6.0)
    np.testing.assert_allclose(b.qvel, a.qvel, atol=1e-9)
    np.testing.assert_allclose(b.qacc, a.qacc, atol=1e-7)


def test_couplings_are_projected_before_differentiation(model: mujoco.MjModel) -> None:
    t = np.arange(0.0, 1.0, 0.005)
    qpos, qvel, _ = _analytic(model, t)
    qpos[:, -1] += 0.05 * np.sin(7 * t)  # off the coupling
    kin = clip_kinematics(model, qpos, 0.005)
    for q, v in zip(kin.qpos[::20], kin.qvel[::20]):
        pos_res, vel_res = joint_equality_residuals(model, q, v)
        assert np.abs(pos_res).max() < 1e-12
        assert np.abs(vel_res).max() < 1e-4
    raw = clip_kinematics(model, qpos, 0.005, project_couplings=False)
    assert np.abs(joint_equality_residuals(model, raw.qpos[40])[0]).max() > 1e-2


def test_lowpass_is_zero_phase_and_removes_noise(model: mujoco.MjModel) -> None:
    dt = 0.005
    t = np.arange(0.0, 3.0, dt)
    qpos, _, qacc = _analytic(model, t)
    noisy = qpos.copy()
    noisy[:, 7] += 1e-3 * np.sin(2 * np.pi * 40 * t)  # 1 mrad of 40 Hz noise
    raw = clip_kinematics(model, noisy, dt)
    kin = clip_kinematics(model, noisy, dt, lowpass_hz=6.0)
    inner = slice(40, -40)  # 1.2 cut-off periods from either end
    assert np.abs(raw.qacc[inner, 6] - qacc[inner, 6]).max() > 50
    # A lag of one frame would leave a position error of 6e-3 rad.
    assert np.abs(kin.qpos[inner, 7] - qpos[inner, 7]).max() < 2e-4
    assert np.abs(kin.qacc[inner, 6] - qacc[inner, 6]).max() < 0.1
    np.testing.assert_allclose(
        np.linalg.norm(kin.qpos[:, 3:7], axis=1), 1.0, atol=1e-12
    )


def test_inverse_dynamics_matches_mujoco_forward_dynamics() -> None:
    env = make_env("myoArmReachRandom-v0")  # 50 muscles, 11 joint equalities, limits
    model = env.unwrapped.model
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    frames = {k: [] for k in ("qpos", "qvel", "qacc", "qfrc")}
    for _ in range(60):
        data.ctrl[:] = rng.uniform(0, 1, model.nu)
        data.qfrc_applied[:] = rng.normal(0, 0.5, model.nv)
        mujoco.mj_forward(model, data)
        frames["qpos"].append(data.qpos.copy())
        frames["qvel"].append(data.qvel.copy())
        frames["qacc"].append(data.qacc.copy())
        frames["qfrc"].append(
            data.qfrc_actuator + data.qfrc_applied + data.qfrc_constraint
        )
        mujoco.mj_step(model, data)
    qpos, qvel, qacc, qfrc = (
        np.array(frames[k]) for k in ("qpos", "qvel", "qacc", "qfrc")
    )
    assert np.abs(qvel).max() > 1.0  # the motion is not trivial

    tau = inverse_dynamics(model, qpos, qvel, qacc)
    np.testing.assert_allclose(tau, qfrc, rtol=1e-8, atol=1e-8)

    # Same as mj_inverse with the constraint solver off.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    for t in (0, 30, 59):
        data.qpos[:], data.qvel[:], data.qacc[:] = qpos[t], qvel[t], qacc[t]
        mujoco.mj_inverse(model, data)
        np.testing.assert_allclose(tau[t], data.qfrc_inverse, rtol=1e-10, atol=1e-10)
    env.close()


def test_external_forces_are_subtracted(model: mujoco.MjModel) -> None:
    qpos, qvel, qacc = _analytic(model, np.arange(0.0, 0.1, 0.01))
    external = np.random.default_rng(1).normal(size=qvel.shape)
    tau = inverse_dynamics(model, qpos, qvel, qacc)
    np.testing.assert_allclose(
        inverse_dynamics(model, qpos, qvel, qacc, external_qfrc=external),
        tau - external,
    )
    # A free body at rest needs its weight on the root's vertical dof.
    still = qpos[:1].copy()
    still[0, 7:] = 0.0
    still[0, 8] = 1.0
    rest = inverse_dynamics(
        model, still, np.zeros((1, model.nv)), np.zeros((1, model.nv))
    )
    weight = model.body_subtreemass[1] * -model.opt.gravity[2]
    np.testing.assert_allclose(rest[0, :3], [0.0, 0.0, weight], atol=1e-9)
    with pytest.raises(ValueError):
        inverse_dynamics(model, qpos, qvel, qacc, external_qfrc=external[:, :3])


@pytest.mark.parametrize(
    "kwargs",
    [{"dt": 0.0}, {"dt": 0.01, "lowpass_hz": 60.0}, {"dt": 0.01, "lowpass_hz": -1.0}],
)
def test_rejects_invalid_inputs(model: mujoco.MjModel, kwargs: dict) -> None:
    qpos = _analytic(model, np.arange(0.0, 0.5, 0.01))[0]
    with pytest.raises(ValueError):
        clip_kinematics(model, qpos, **kwargs)
    with pytest.raises(ValueError):
        clip_kinematics(model, qpos[:3], 0.01)
