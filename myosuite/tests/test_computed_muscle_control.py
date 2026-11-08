# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Computed muscle control tracks forward-simulated and kinematic motions."""

from __future__ import annotations

import copy

import mujoco
import myo_sim
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.model_builder import ModelBuilder
from myosuite.physics.computed_muscle_control import (
    ComputedMuscleControl,
    ComputedMuscleControlConfig,
    computed_muscle_control,
)
from myosuite.physics.inverse_dynamics import clip_kinematics
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1


def _reaching_motion(model: mujoco.MjModel, rate: float, duration: float) -> np.ndarray:
    """Random minimum-jerk reaches (0.5-0.9 s, 0.3 s holds) of every independent hinge."""
    c = JointCouplings.from_model(model)
    coupled = set(c.dep_dof.tolist()) | set(c.locked_dof.tolist())
    joints = [
        j
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and model.jnt_dofadr[j] not in coupled
    ]
    lo, hi = model.jnt_range[joints].T
    lo, hi = lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)
    rng = np.random.default_rng(0)
    start = model.key_qpos[0] if model.nkey else model.qpos0
    cur, poses = np.clip(start[model.jnt_qposadr[joints]], lo, hi), []
    while sum(len(p) for p in poses) < duration * rate:
        goal, n = rng.uniform(lo, hi), int(rng.uniform(0.5, 0.9) * rate)
        s = np.arange(n) / n
        poses.append(cur + (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None] * (goal - cur))
        poses.append(np.repeat(goal[None], int(0.3 * rate), 0))
        cur = goal
    qpos = np.tile(start, (int(duration * rate), 1))
    qpos[:, model.jnt_qposadr[joints]] = np.concatenate(poses)[: len(qpos)]
    project_joint_couplings(c, qpos)
    return qpos


def _elbow(dofs: int, forearm_scale: float = 1.0) -> mujoco.MjModel:
    rel = f"legacy/elbow/myoelbow_{dofs}dof6muscles.xml"
    model = ModelBuilder.from_xml_file(myo_sim.get_path(rel)).build()[0]
    body = model.body("r_ulna_radius_hand").id
    model.body_mass[body] *= forearm_scale
    model.body_inertia[body] *= forearm_scale
    return model


def _simulated_motion(model: mujoco.MjModel, every: int) -> dict[str, np.ndarray]:
    """3 s driven by smooth excitations around a mid-range hold, sampled every *every* steps."""
    lo, hi = model.jnt_range.T
    data = mujoco.MjData(model)
    data.qpos[:] = lo + 0.5 * (hi - lo)
    cmc = ComputedMuscleControl(model)
    zero = np.zeros(model.nv)
    base = cmc.optimizer.solve(data.qpos, zero, zero).activation + 0.1
    rng = np.random.default_rng(0)
    freq, phase = rng.uniform(0.2, 1.0, model.nu), rng.uniform(0, 2 * np.pi, model.nu)
    data.act[:] = base
    rec = {k: [] for k in ("qpos", "qvel", "qacc")}
    for i in range(round(3.0 / model.opt.timestep) + 1):
        if i % every == 0:
            mujoco.mj_forward(model, data)
            for k in rec:
                rec[k].append(getattr(data, k).copy())
        data.ctrl[:] = base + 0.1 * np.sin(2 * np.pi * freq * data.time + phase)
        mujoco.mj_step(model, data)
    return {k: np.array(v) for k, v in rec.items()}


@pytest.mark.parametrize(("every", "tolerance_deg"), [(1, 0.25), (5, 0.6)])
def test_tracks_a_forward_simulated_elbow_motion(
    every: int, tolerance_deg: float
) -> None:
    """2-dof elbow, a 30-deg motion its own muscles made, at 500 and 100 Hz control."""
    model = _elbow(2)
    ref = _simulated_motion(model, every)
    res = computed_muscle_control(
        model, ref["qpos"], ref["qvel"], ref["qacc"], every * model.opt.timestep
    )
    assert (res.optimization.status == "solved").all()
    assert np.degrees(np.ptp(ref["qpos"][:, 1])) > 25.0
    assert np.degrees(np.abs(res.tracking_error)).max() < tolerance_deg


def test_tracks_myoarm_reaches() -> None:
    """myoArm (50 muscles, 11 shoulder couplings): minimum-jerk reaches at 100 Hz."""
    env = make_env("myoArmReachRandom-v0")
    model = copy.deepcopy(env.unwrapped.model)
    env.close()
    qpos = _reaching_motion(model, 100.0, 3.0)
    kin = clip_kinematics(model, qpos, 0.01)
    config = ComputedMuscleControlConfig(kp=400.0)
    res = computed_muscle_control(
        model, kin.qpos, kin.qvel, kin.qacc, kin.dt, config=config
    )
    assert (res.optimization.status == "solved").all()
    error = np.degrees(np.abs(res.tracking_error))
    assert error.max() < 4.5 and np.sqrt((error**2).mean()) < 0.8


def test_optimised_activations_are_reached() -> None:
    """With one physics step per control step the inverted dynamics are exact and
    every target lies within the activations reachable in the step."""
    model = _elbow(2)
    ref = _simulated_motion(model, 1)
    res = computed_muscle_control(model, ref["qpos"], ref["qvel"], ref["qacc"], 0.002)
    np.testing.assert_allclose(
        res.activation, res.optimization.activation, rtol=0, atol=1e-9
    )
    assert (np.ptp(res.activation, axis=0) > 0.1).sum() >= 3


def test_damped_motor_follows_the_desired_acceleration() -> None:
    """A motor acts at once, so one Euler step realises qacc* exactly, damping included."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="j" axis="0 1 0" damping="2"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/></body></worldbody>
          <actuator><motor joint="j" gear="5"/></actuator></mujoco>"""
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    cmc = ComputedMuscleControl(model)
    for target in (0.3, -0.2, 0.5):
        qvel = data.qvel.copy()
        step = cmc.step(data, np.array([target]), np.zeros(1), np.zeros(1))
        np.testing.assert_allclose(
            (data.qvel - qvel) / model.opt.timestep, step.qacc_desired, rtol=1e-9
        )


def test_reserves_carry_what_the_muscles_cannot() -> None:
    """A forearm 100 times too heavy: tracked on its reserves, lost without them."""
    model = _elbow(1, forearm_scale=100.0)
    t = np.arange(101) * 0.01
    w = np.pi
    qpos = np.radians(70.0 + 20.0 * np.sin(w * t))[:, None]
    qvel = np.radians(20.0 * w * np.cos(w * t))[:, None]
    qacc = np.radians(-20.0 * w * w * np.sin(w * t))[:, None]
    held = computed_muscle_control(model, qpos, qvel, qacc, 0.01)
    assert np.abs(held.optimization.reserve).min() > 100.0
    assert np.degrees(np.abs(held.tracking_error)).max() < 1.0
    config = ComputedMuscleControlConfig(apply_reserves=False)
    lost = computed_muscle_control(model, qpos, qvel, qacc, 0.01, config=config)
    assert np.degrees(np.abs(lost.tracking_error)).max() > 30.0


def test_applied_loads_are_known_and_kept() -> None:
    """A 2 kg weight in the hand and a joint torque: held, and left in place."""
    model = _elbow(1)
    data = mujoco.MjData(model)
    data.qpos[0] = np.radians(90.0)
    data.xfrc_applied[model.body("r_ulna_radius_hand").id, 2] = -19.6
    data.qfrc_applied[0] = 0.5
    xfrc, qfrc = data.xfrc_applied.copy(), data.qfrc_applied.copy()
    mujoco.mj_forward(model, data)
    cmc, zero = ComputedMuscleControl(model), np.zeros(1)
    steps = [
        cmc.step(data, data.qpos.copy(), zero, zero, n_substeps=5) for _ in range(100)
    ]
    assert np.degrees(np.abs(steps[-1].tracking_error[0])) < 1e-3
    np.testing.assert_array_equal(data.xfrc_applied, xfrc)
    np.testing.assert_array_equal(data.qfrc_applied, qfrc)
    capped = cmc.step(
        data, data.qpos.copy(), zero, zero, max_activation=np.full(6, 0.05)
    )
    assert capped.optimization.activation.max() <= 0.05 + 1e-9
    assert abs(capped.optimization.reserve[0]) > 1.0


def test_rejects_unsupported_set_ups() -> None:
    with pytest.raises(ValueError, match="whole number of physics steps"):
        computed_muscle_control(_elbow(1), *np.zeros((3, 4, 1)), 1 / 60)
    filtered = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="j"/><geom size=".1" mass="1"/></body></worldbody>
          <actuator><general joint="j" dyntype="filter" dynprm=".1"/></actuator></mujoco>"""
    )
    with pytest.raises(ValueError, match="Only muscle activation dynamics"):
        ComputedMuscleControl(filtered)
