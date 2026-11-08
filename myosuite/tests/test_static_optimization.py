# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Static optimization against MuJoCo's own actuator forces and closed-form optima."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizer,
    static_optimization,
)

pytestmark = pytest.mark.tier1


def _qfrc_actuator(model: mujoco.MjModel, qpos, qvel, activation) -> np.ndarray:
    """Generalized actuator force MuJoCo computes for these activations."""
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = qpos, qvel, activation
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)
    return data.qfrc_actuator.copy()


@pytest.fixture(scope="module")
def arm_motion() -> tuple[mujoco.MjModel, dict[str, np.ndarray]]:
    """myoArm driven by known activations, without contacts or limits."""
    env = make_env("myoArmReachRandom-v0")  # 50 muscles, 11 joint equalities
    model = env.unwrapped.model
    sim_model = copy.deepcopy(model)
    sim_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    sim_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    data = mujoco.MjData(sim_model)
    rng = np.random.default_rng(0)
    rec = {k: [] for k in ("qpos", "qvel", "qacc", "act")}
    for _ in range(40):
        data.ctrl[:] = rng.uniform(0.0, 0.6, model.nu)
        mujoco.mj_forward(sim_model, data)
        for k in rec:
            rec[k].append(getattr(data, k).copy())
        for _ in range(5):
            mujoco.mj_step(sim_model, data)
    env.close()
    return model, {k: np.array(v) for k, v in rec.items()}


def test_activations_reproduce_the_inverse_dynamics(arm_motion) -> None:
    model, rec = arm_motion
    res = static_optimization(model, rec["qpos"], rec["qvel"], rec["qacc"])
    assert (res.status == "solved").all()
    assert res.activation.min() >= 0.0 and res.activation.max() <= 1.0
    for t in range(len(res.activation)):
        qfrc = _qfrc_actuator(model, rec["qpos"][t], rec["qvel"][t], res.activation[t])
        np.testing.assert_allclose(
            qfrc + res.reserve[t] + res.constraint_force[t],
            res.joint_torque[t],
            atol=1e-4,
        )
    # Coupling forces act only on coupled dofs.
    coupled = np.zeros(model.nv, dtype=bool)
    c = JointCouplings.from_model(model)
    coupled[np.r_[c.dep_dof, c.master_dof, c.locked_dof]] = True
    assert np.abs(res.constraint_force[:, ~coupled]).max() < 1e-9
    assert np.abs(res.constraint_force[:, coupled]).max() > 1e-2


def test_optimum_is_no_worse_than_the_true_activations(arm_motion) -> None:
    model, rec = arm_motion
    res = static_optimization(model, rec["qpos"], rec["qvel"], rec["qacc"])
    # The true activations are feasible without reserves; the optimum costs at most as much.
    cost = (res.activation**2).sum(axis=1) + (res.reserve**2).sum(axis=1)
    assert (cost <= (rec["act"] ** 2).sum(axis=1) + 1e-6).all()
    # Reserves trade off against activations (r = F^2 mu / 2 at the optimum): with
    # F = 1 N m they stay under 0.2 N m while joint torques reach 20 N m.
    assert np.abs(res.reserve).max() < 0.2
    assert np.abs(res.joint_torque).max() > 10


def test_elbow_meets_the_optimality_conditions() -> None:
    env = make_env("myoElbowPose1D6MRandom-v0")
    model = env.unwrapped.model
    qpos, zero = np.array([1.0]), np.zeros(1)
    so = StaticOptimizer(model)
    frame = so.solve(qpos, zero, zero)  # hold the forearm against gravity at 57 degrees
    # Torque per unit activation of each muscle, from MuJoCo's generalized forces.
    base = _qfrc_actuator(model, qpos, zero, np.zeros(model.nu))
    c = np.array(
        [
            _qfrc_actuator(model, qpos, zero, np.eye(model.nu)[i])[0] - base[0]
            for i in range(model.nu)
        ]
    )
    mu = (
        2 * frame.reserve[0] / so.config.reserve_optimal_force**2
    )  # multiplier of the torque row
    # Stationarity: a = mu c / 2 where that lies in [0, 1], else the bound.
    np.testing.assert_allclose(
        frame.activation, np.clip(mu * c / 2, 0.0, 1.0), atol=1e-5
    )
    assert (frame.activation > 1e-3).sum() >= 2 and (frame.activation < 1e-6).sum() >= 1
    env.close()


def test_motor_controls_match_the_closed_form() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="j" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/></body></worldbody>
          <actuator><motor joint="j" gear="10"/></actuator></mujoco>"""
    )
    frame = StaticOptimizer(model).solve(np.zeros(1), np.zeros(1), np.zeros(1))
    tau = frame.joint_torque[0]  # weight of the rod about the hinge
    # min u^2 + r^2 s.t. 10 u + r = tau  ->  u = 10 tau / 101, r = tau / 101.
    np.testing.assert_allclose(frame.activation, [10 * tau / 101], rtol=1e-6)
    np.testing.assert_allclose(frame.reserve, [tau / 101], rtol=1e-5)
    np.testing.assert_allclose(frame.actuator_force, [10 * tau / 101], rtol=1e-6)


def test_floating_base_residuals_carry_the_root_wrench() -> None:
    env = make_env("myoLegWalk-v0")  # free root, 80 muscles, knee couplings
    model = env.unwrapped.model
    rng = np.random.default_rng(1)
    qpos = np.tile(model.key_qpos[0] if model.nkey else model.qpos0, (5, 1))
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    qpos[:, hinges] += rng.normal(0, 0.05, (5, len(hinges)))
    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qvel, qacc = rng.normal(0, 0.3, (5, model.nv)), rng.normal(0, 1.0, (5, model.nv))
    res = static_optimization(model, qpos, qvel, qacc)
    assert (res.status == "solved").all()
    root = slice(0, 6)
    # Muscles and couplings exert no net force on the free root.
    np.testing.assert_allclose(
        res.reserve[:, root], res.joint_torque[:, root], rtol=1e-6, atol=1e-6
    )
    assert (
        np.abs(res.joint_torque[:, 2]).min() > 300
    )  # body weight on the vertical residual
    assert np.abs(res.reserve[:, 6:]).max() < 0.5
    # The body weight on the root must not swamp the solver's relative tolerance.
    loose = static_optimization(
        model, qpos, qvel, qacc, config=StaticOptimizationConfig(tolerance=1e-4)
    )
    assert np.abs(loose.activation - res.activation).max() < 2e-3
    env.close()


def test_streaming_equals_batch_and_is_deterministic(arm_motion) -> None:
    model, rec = arm_motion
    batch = static_optimization(
        model, rec["qpos"][:10], rec["qvel"][:10], rec["qacc"][:10]
    )
    so = StaticOptimizer(model)
    for t in range(10):
        frame = so.solve(rec["qpos"][t], rec["qvel"][t], rec["qacc"][t])
        np.testing.assert_array_equal(frame.activation, batch.activation[t])


def test_rejects_clamped_actuator_forces() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="j"/><geom size=".1" mass="1"/></body></worldbody>
          <actuator><motor joint="j" forcerange="-1 1"/></actuator></mujoco>"""
    )
    with pytest.raises(ValueError, match="not supported"):
        StaticOptimizer(model)


def test_config_reserve_weight_trades_off_activations(arm_motion) -> None:
    model, rec = arm_motion
    cheap = StaticOptimizationConfig(reserve_optimal_force=100.0)
    a = static_optimization(model, rec["qpos"][:5], rec["qvel"][:5], rec["qacc"][:5])
    b = static_optimization(
        model, rec["qpos"][:5], rec["qvel"][:5], rec["qacc"][:5], config=cheap
    )
    assert np.abs(b.reserve).sum() > 10 * np.abs(a.reserve).sum()
    assert (b.activation**2).sum() < (a.activation**2).sum()
