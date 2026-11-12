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
from myosuite.core.model_builder import build_from_recipe
from myosuite.core.muscle_conditions import _peak_force
from myosuite.envs.myo.myoedits import edit_fn_arm_reaching
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.biomechanics.static_optimization import (
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
    # The myoArmReach model (50 muscles, 11 joint equalities), built directly: two
    # modules register its env id with different models, so the id is order-dependent.
    model = build_from_recipe("full_arm", edit_fn=edit_fn_arm_reaching)[0]
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
        2 * frame.reserve[0] / so.config.reserve_optimal_force[0] ** 2
    )  # multiplier of the torque row
    # Stationarity: a = mu c / 2 where that lies in [0, 1], else the bound.
    np.testing.assert_allclose(
        frame.activation, np.clip(mu * c / 2, 0.0, 1.0), atol=1e-5
    )
    assert (frame.activation > 1e-3).sum() >= 2 and (frame.activation < 1e-6).sum() >= 1
    env.close()


@pytest.mark.parametrize("reserve", [None, 1.0])
def test_motor_controls_match_the_closed_form(reserve: float | None) -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="j" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/></body></worldbody>
          <actuator><motor joint="j" gear="10" ctrlrange="-2 1"/></actuator></mujoco>"""
    )
    so = StaticOptimizer(model, StaticOptimizationConfig(reserve_optimal_force=reserve))
    frame = so.solve(np.zeros(1), np.zeros(1), np.zeros(1))
    tau = frame.joint_torque[0]  # weight of the rod about the hinge
    # Default reserve: 0.005 of the motor's peak torque (gear 10 at control -2).
    f = 0.1 if reserve is None else reserve
    np.testing.assert_allclose(so.config.reserve_optimal_force, [f], rtol=1e-12)
    # min u^2 + (r / f)^2 s.t. 10 u + r = tau  ->  u = 10 tau / (100 + f^2).
    np.testing.assert_allclose(frame.activation, [10 * tau / (100 + f**2)], rtol=1e-6)
    np.testing.assert_allclose(frame.reserve, [f**2 * tau / (100 + f**2)], rtol=1e-5)
    np.testing.assert_allclose(
        frame.actuator_force, frame.activation, rtol=1e-12
    )  # before the gear


@pytest.mark.parametrize("reference", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("gear", [1.0, -2.0])
def test_servo_peak_torque_includes_native_bias(reference: float, gear: float) -> None:
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
      <worldbody><body><joint name="j" ref="{reference}"/><geom size=".1"/></body></worldbody>
      <actuator><position joint="j" kp="100" gear="{gear}" ctrlrange="1 2"/></actuator></mujoco>
    """)
    data = mujoco.MjData(model)
    endpoints = []
    for ctrl in model.actuator_ctrlrange[0]:
        data.ctrl[0] = ctrl
        mujoco.mj_forward(model, data)
        endpoints.append(data.qfrc_actuator.copy())
    expected = np.max(np.abs(endpoints), axis=0)
    optimizer = StaticOptimizer(model)
    np.testing.assert_allclose(optimizer.peak_actuator_qfrc(), expected, rtol=1e-12)
    np.testing.assert_allclose(
        optimizer.config.reserve_optimal_force,
        StaticOptimizationConfig().reserve_fraction * expected,
        rtol=1e-12,
    )


@pytest.mark.parametrize("fraction", [0.005, 0.1])
def test_servo_reserves_match_the_biased_closed_form(fraction: float) -> None:
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
      <worldbody><body><joint name="j" ref="2"/><geom size=".1"/></body></worldbody>
      <actuator><position joint="j" kp="10" ctrlrange="-1 1"/></actuator></mujoco>
    """)
    optimizer = StaticOptimizer(
        model, StaticOptimizationConfig(reserve_fraction=fraction)
    )
    frame = optimizer.solve(model.qpos0, np.zeros(1), np.zeros(1), np.array([19.0]))
    # Native force is 10 u - 20, so its peak magnitude on [-1, 1] is 30.
    # min u^2 + (r / F)^2, 10 u - 20 + r = -19 -> u=10/(100+F^2).
    reserve = fraction * 30.0
    assert frame.status == "solved"
    np.testing.assert_allclose(
        frame.activation, [10.0 / (100.0 + reserve**2)], rtol=1e-6
    )
    np.testing.assert_allclose(
        frame.reserve, [reserve**2 / (100.0 + reserve**2)], rtol=1e-5
    )
    np.testing.assert_allclose(
        frame.actuator_force + frame.reserve, frame.joint_torque, atol=1e-8
    )


def _scaled_strength(model: mujoco.MjModel, k: float) -> mujoco.MjModel:
    """*model* without gravity or joint springs, its muscles *k* times as strong."""
    m = copy.deepcopy(model)
    muscle = m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
    force = _peak_force(m)[muscle] * k
    m.actuator_gainprm[muscle, 2] = m.actuator_biasprm[muscle, 2] = force
    m.opt.gravity[:] = 0.0
    m.jnt_stiffness[:] = 0.0
    return m


@pytest.mark.parametrize("reserve", [None, 1.0])
def test_default_reserves_follow_the_muscle_strength(reserve: float | None) -> None:
    """A 100 times weaker elbow under a 100 times smaller load: scaled reserves give
    the same activations; OpenSim's fixed 1 N m reserve takes over the load."""
    env = make_env("myoElbowPose1D6MRandom-v0")
    qpos, zero, load = np.array([1.2]), np.zeros(1), np.array([-20.0])
    config = StaticOptimizationConfig(reserve_optimal_force=reserve)
    frames, optimizers = [], []
    for k in (1.0, 0.01):
        so = StaticOptimizer(_scaled_strength(env.unwrapped.model, k), config)
        frames.append(so.solve(qpos, zero, zero, k * load))
        optimizers.append(so)
    strong, weak = frames
    np.testing.assert_allclose(weak.joint_torque, 0.01 * strong.joint_torque, rtol=1e-9)
    share = [abs(f.reserve[0] / f.joint_torque[0]) for f in frames]
    assert share[0] < 0.01  # muscles carry the strong elbow's load either way
    if reserve is None:
        # 0.005 of sum_i |moment arm_i| * peak isometric force_i at qpos0.
        model = optimizers[0].model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        arms = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            arms,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
        peak = np.abs(arms[:, 0]) @ np.abs(model.actuator_gainprm[:, 2])
        assert (model.actuator_gainprm[:, 2] > 0).all()  # explicit forces
        np.testing.assert_allclose(
            optimizers[0].config.reserve_optimal_force, [0.005 * peak], rtol=1e-12
        )
        np.testing.assert_allclose(
            optimizers[1].config.reserve_optimal_force,
            0.01 * np.asarray(optimizers[0].config.reserve_optimal_force),
            rtol=1e-9,
        )
        np.testing.assert_allclose(weak.activation, strong.activation, atol=1e-6)
    else:
        assert share[1] > 0.5
    # The resolved settings carry the weights to another model unchanged.
    again = StaticOptimizer(env.unwrapped.model, optimizers[0].config)
    assert (
        again.config.reserve_optimal_force == optimizers[0].config.reserve_optimal_force
    )
    env.close()


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
    # The body weight on the root must not swamp the solver's relative tolerance:
    # polished at 1e-5, the solve lands on the default one (with the root residuals
    # in the cost it stays about 2e-5 off; at 1e-4 platforms differ by 2-5e-3).
    loose = static_optimization(
        model, qpos, qvel, qacc, config=StaticOptimizationConfig(tolerance=1e-5)
    )
    assert np.abs(loose.activation - res.activation).max() < 1e-6
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


def test_max_activation_bounds_one_frame() -> None:
    env = make_env("myoElbowPose1D6MRandom-v0")
    model = env.unwrapped.model
    qpos, zero, load = np.array([1.2]), np.zeros(1), np.array([-20.0])
    so = StaticOptimizer(model)
    free = so.solve(qpos, zero, zero, load)
    i = int(np.argmax(free.activation))
    cap = np.full(model.nu, np.inf)
    cap[i] = 0.5 * free.activation[i]
    bounded = so.solve(qpos, zero, zero, load, max_activation=cap)
    np.testing.assert_allclose(bounded.activation[i], cap[i], atol=1e-6)
    # Synergists take over; the joint torque to balance is unchanged.
    rest = np.arange(model.nu) != i
    assert bounded.activation[rest].sum() > free.activation[rest].sum() + 0.05
    np.testing.assert_array_equal(bounded.joint_torque, free.joint_torque)
    # Optimality with the bound: a_j = clip(mu c_j / 2, 0, cap_j) for every muscle.
    base = _qfrc_actuator(model, qpos, zero, np.zeros(model.nu))
    c = np.array(
        [
            _qfrc_actuator(model, qpos, zero, np.eye(model.nu)[j])[0] - base[0]
            for j in range(model.nu)
        ]
    )
    mu = 2 * bounded.reserve[0] / so.config.reserve_optimal_force[0] ** 2
    np.testing.assert_allclose(
        bounded.activation, np.clip(mu * c / 2, 0.0, np.minimum(cap, 1.0)), atol=1e-5
    )
    # A floor lifts an idle muscle to it; the others relax, the torque is unchanged.
    j = int(np.argmin(free.activation))
    floor = np.zeros(model.nu)
    floor[j] = 0.3
    lifted = so.solve(qpos, zero, zero, load, min_activation=floor)
    np.testing.assert_allclose(lifted.activation[j], 0.3, atol=1e-6)
    np.testing.assert_allclose(lifted.joint_torque, free.joint_torque)
    # Optimality with the floor: a_k = clip(mu c_k / 2, floor_k, 1).
    mu = 2 * lifted.reserve[0] / so.config.reserve_optimal_force[0] ** 2
    np.testing.assert_allclose(
        lifted.activation, np.clip(mu * c / 2, floor, 1.0), atol=1e-5
    )
    # A floor above the frame's cap is held at the cap.
    both = so.solve(
        qpos,
        zero,
        zero,
        load,
        max_activation=cap,
        min_activation=np.full(model.nu, 0.9),
    )
    np.testing.assert_allclose(both.activation[i], cap[i], atol=1e-6)
    # A bound below the actuator range clamps to it; the next frame is unbounded again.
    low = so.solve(qpos, zero, zero, load, max_activation=np.full(model.nu, -1.0))
    np.testing.assert_array_equal(low.activation, np.zeros(model.nu))
    np.testing.assert_allclose(
        so.solve(qpos, zero, zero, load).activation, free.activation, atol=1e-6
    )
    env.close()
