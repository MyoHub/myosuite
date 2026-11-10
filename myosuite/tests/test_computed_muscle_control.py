# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Computed muscle control tracks forward-simulated and kinematic motions."""

from __future__ import annotations

import copy
from collections.abc import Callable

import mujoco
import myo_sim
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.envs.myo.myoedits import edit_fn_arm_reaching
from myosuite.biomechanics.computed_muscle_control import (
    ComputedMuscleControl,
    ComputedMuscleControlConfig,
    computed_muscle_control,
)
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.biomechanics.static_optimization import StaticOptimizationConfig

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


# A torso with a limb swung by two muscles.
_FLOATING = """
<mujoco>
  <worldbody>
    <body name="torso" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .2" mass="10"/>
      <site name="a" pos=".1 0 -.1"/><site name="b" pos="-.1 0 -.1"/>
      <body name="limb" pos="0 0 -.2"><joint name="hip" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04" mass="3"/>
        <site name="c" pos=".04 0 -.12"/><site name="d" pos="-.04 0 -.12"/>
      </body>
    </body>
  </worldbody>
  <tendon><spatial name="front"><site site="a"/><site site="c"/></spatial>
    <spatial name="back"><site site="b"/><site site="d"/></spatial></tendon>
  <actuator><muscle tendon="front" force="400"/><muscle tendon="back" force="400"/></actuator>
</mujoco>"""

# A hinge driven through filter, exact-filter and integrator activation dynamics.
_FILTERS = """
<mujoco><worldbody><body><joint name="j" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/></body></worldbody>
  <actuator>
    <general joint="j" dyntype="filter" dynprm=".03" gainprm="4" ctrlrange="-1 1"/>
    <general joint="j" dyntype="filterexact" dynprm=".05" gainprm="4" ctrlrange="-1 1"/>
    <general joint="j" dyntype="integrator" gainprm="4" ctrlrange="-5 5" actrange="-.05 .05"/>
  </actuator></mujoco>"""


def _elbow(dofs: int, forearm_scale: float = 1.0) -> mujoco.MjModel:
    rel = f"legacy/elbow/myoelbow_{dofs}dof6muscles.xml"
    model = ModelBuilder.from_xml_file(myo_sim.get_path(rel)).build()[0]
    body = model.body("r_ulna_radius_hand").id
    model.body_mass[body] *= forearm_scale
    model.body_inertia[body] *= forearm_scale
    return model


def _record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: Callable[[float], np.ndarray],
    seconds: float,
    every: int,
) -> dict[str, np.ndarray]:
    """Forward-simulate under ``ctrl(time)``, sampling every *every* physics steps."""
    rec = {k: [] for k in ("qpos", "qvel", "qacc")}
    for i in range(round(seconds / model.opt.timestep) + 1):
        if i % every == 0:
            mujoco.mj_forward(model, data)
            for k in rec:
                rec[k].append(getattr(data, k).copy())
        data.ctrl[:] = ctrl(data.time)
        mujoco.mj_step(model, data)
    return {k: np.array(v) for k, v in rec.items()}


def _simulated_motion(model: mujoco.MjModel, every: int) -> dict[str, np.ndarray]:
    """3 s driven by smooth excitations around a mid-range hold."""
    lo, hi = model.jnt_range.T
    data = mujoco.MjData(model)
    data.qpos[:] = lo + 0.5 * (hi - lo)
    zero = np.zeros(model.nv)
    base = ComputedMuscleControl(model).optimizer.solve(data.qpos, zero, zero)
    base = base.activation + 0.1
    rng = np.random.default_rng(0)
    freq, phase = rng.uniform(0.2, 1.0, model.nu), rng.uniform(0, 2 * np.pi, model.nu)
    data.act[:] = base
    return _record(
        model,
        data,
        lambda t: base + 0.1 * np.sin(2 * np.pi * freq * t + phase),
        3.0,
        every,
    )


@pytest.mark.parametrize(("every", "tolerance_deg"), [(1, 0.1), (5, 0.4)])
def test_tracks_a_forward_simulated_elbow_motion(
    every: int, tolerance_deg: float
) -> None:
    """2-dof elbow, a 37-deg motion its own muscles made, at 500 and 100 Hz control."""
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
    # The myoArmReach model, built directly: its env id is registered twice.
    model = build_from_recipe("full_arm", edit_fn=edit_fn_arm_reaching)[0]
    qpos = _reaching_motion(model, 100.0, 3.0)
    kin = clip_kinematics(model, qpos, 0.01)
    res = computed_muscle_control(model, kin.qpos, kin.qvel, kin.qacc, kin.dt)
    assert (res.optimization.status == "solved").all()
    error = np.degrees(np.abs(res.tracking_error))
    assert error.max() < 4.5
    assert np.sqrt((error**2).mean()) < 0.4


def test_standing_leg_on_estimated_ground_reactions() -> None:
    """MyoLeg (free root, 80 muscles) stands on the optimization's ground reactions."""
    env = make_env("myoLegWalk-v0")
    model = copy.deepcopy(env.unwrapped.model)
    env.close()
    qpos = model.key_qpos[0].copy()
    qpos[2] -= 0.005  # feet 2 mm into the floor, where MuJoCo's contacts would push
    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qpos, zero = np.tile(qpos, (101, 1)), np.zeros((101, model.nv))
    config = ComputedMuscleControlConfig(
        optimization=StaticOptimizationConfig(contact_forces=True)
    )
    res = computed_muscle_control(model, qpos, zero, zero, 0.01, config=config)
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    assert np.abs(res.tracking_error[:, :3]).max() < 1.5e-4  # m
    assert np.degrees(np.abs(res.tracking_error[:, 3:])).max() < 0.1
    assert np.abs(res.optimization.reserve[:, :3]).max() < 0.003 * weight
    np.testing.assert_allclose(
        res.optimization.contact_force[:, 1:, 2].sum(axis=1), weight, rtol=1e-3
    )


def test_free_falling_body_needs_no_residuals() -> None:
    """A falling torso swinging a muscle-driven limb: no external wrench to supply."""
    model = mujoco.MjModel.from_xml_string(_FLOATING)
    data = mujoco.MjData(model)
    data.act[:] = 0.2
    freq, phase = np.array([1.0, 1.3]), np.array([0.0, 1.0])
    ref = _record(
        model, data, lambda t: 0.2 + 0.15 * np.sin(2 * np.pi * freq * t + phase), 1.0, 5
    )
    res = computed_muscle_control(model, ref["qpos"], ref["qvel"], ref["qacc"], 0.01)
    assert np.degrees(np.ptp(ref["qpos"][:, 7])) > 90.0
    assert np.abs(res.tracking_error[:, :3]).max() < 6e-4  # m
    assert np.degrees(np.abs(res.tracking_error[:, 3:])).max() < 0.5
    assert np.abs(res.optimization.reserve[:, :6]).max() < 0.6  # N, N m


def test_optimised_activations_are_reached() -> None:
    """With one physics step per control step every target is reached exactly."""
    model = _elbow(2)
    ref = _simulated_motion(model, 1)
    res = computed_muscle_control(model, ref["qpos"], ref["qvel"], ref["qacc"], 0.002)
    np.testing.assert_allclose(
        res.activation, res.optimization.activation, rtol=0, atol=1e-9
    )
    assert (np.ptp(res.activation, axis=0) > 0.1).sum() >= 3


def _smoothed_elbow() -> mujoco.MjModel:
    model = _elbow(1)
    model.actuator_dynprm[:, 2] = 0.2  # tausmooth
    return model


@pytest.mark.parametrize(
    "make", [lambda: mujoco.MjModel.from_xml_string(_FILTERS), _smoothed_elbow]
)
def test_any_activation_dynamics_is_inverted_exactly(
    make: Callable[[], mujoco.MjModel],
) -> None:
    """Filter, exact filter, integrator and smoothed muscle dynamics over 5 physics
    steps: replaying the excitations gives the optimised mean activations."""
    model = make()
    data = mujoco.MjData(model)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    cmc, zero = ComputedMuscleControl(model), np.zeros(1)
    stateful = np.flatnonzero(model.actuator_actadr >= 0)
    lo, hi = model.actuator_ctrlrange[stateful].T
    reached = 0
    for k in range(40):
        replay = mujoco.MjData(cmc.sim_model)
        mujoco.mj_copyData(replay, cmc.sim_model, data)
        target = np.array([1.0 + 0.3 * np.sin(0.2 * k)])
        step = cmc.step(data, target, zero, zero, n_substeps=5)
        replay.ctrl[:] = step.excitation
        replay.qfrc_applied += step.optimization.reserve
        mean = np.zeros(model.na)
        for _ in range(5):
            mujoco.mj_step(cmc.sim_model, replay)
            mean += replay.act / 5
        u = step.excitation[stateful]
        free = (u > lo + 1e-9) & (u < hi - 1e-9)
        np.testing.assert_allclose(
            mean[model.actuator_actadr[stateful]][free],
            step.optimization.activation[stateful][free],
            rtol=0,
            atol=1e-9,
        )
        reached += free.sum()
    assert reached > 40


def test_damping_is_realised_exactly() -> None:
    """Joint damping (implicit in Euler) and a velocity servo (explicit) on a weightless
    hinge: the step realises qacc* exactly over 1 and 5 physics steps."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><option gravity="0 0 0"/><worldbody><body>
          <joint name="j" axis="0 1 0" damping="2"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/></body></worldbody>
          <actuator><motor joint="j" gear="5"/><velocity joint="j" kv="3"/></actuator></mujoco>"""
    )
    zero = np.zeros(1)
    for n in (1, 5):
        data = mujoco.MjData(model)
        data.qvel[0] = 1.0
        mujoco.mj_forward(model, data)
        cmc = ComputedMuscleControl(model)
        for target in (0.3, -0.2, 0.5):
            qvel = data.qvel.copy()
            step = cmc.step(data, np.array([target]), zero, zero, n_substeps=n)
            realised = (data.qvel - qvel) / (n * model.opt.timestep)
            np.testing.assert_allclose(realised, step.qacc_desired, rtol=1e-9)


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
    assert np.degrees(np.abs(held.tracking_error)).max() < 0.05
    config = ComputedMuscleControlConfig(apply_reserves=False)
    lost = computed_muscle_control(model, qpos, qvel, qacc, 0.01, config=config)
    assert np.degrees(np.abs(lost.tracking_error)).max() > 30.0


def test_applied_loads_are_known_and_kept() -> None:
    """A 2 kg weight in the hand and a joint torque: held, and left in place."""
    model = _elbow(1)
    data = mujoco.MjData(model)
    hold = np.radians([90.0])
    data.xfrc_applied[model.body("r_ulna_radius_hand").id, 2] = -19.6
    data.qfrc_applied[0] = 0.5
    xfrc, qfrc = data.xfrc_applied.copy(), data.qfrc_applied.copy()
    cmc, zero = ComputedMuscleControl(model), np.zeros(1)
    start = cmc.reset(data, hold, zero, zero)
    np.testing.assert_array_equal(data.act, start.activation)
    unloaded = cmc.reset(mujoco.MjData(model), hold, zero, zero)
    assert start.joint_torque[0] - unloaded.joint_torque[0] > 2.0  # the weight's pull
    steps = [cmc.step(data, hold, zero, zero, n_substeps=5) for _ in range(100)]
    error = np.degrees(np.abs([s.tracking_error[0] for s in steps]))
    assert error.max() < 1e-3  # the muscles start in equilibrium
    np.testing.assert_array_equal(data.xfrc_applied, xfrc)
    np.testing.assert_array_equal(data.qfrc_applied, qfrc)
    capped = cmc.step(data, hold, zero, zero, max_activation=np.full(6, 0.05))
    assert capped.optimization.activation.max() <= 0.05 + 1e-9
    assert abs(capped.optimization.reserve[0]) > 1.0


def test_rejects_unsupported_set_ups() -> None:
    with pytest.raises(ValueError, match="whole number of physics steps"):
        computed_muscle_control(_elbow(1), *np.zeros((3, 4, 1)), 1 / 60)
    unbounded = _FILTERS.replace('ctrlrange="-5 5" ', "")
    with pytest.raises(ValueError, match="need a ctrlrange"):
        ComputedMuscleControl(mujoco.MjModel.from_xml_string(unbounded))


@pytest.mark.parametrize(
    "actearly",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason="known limitation: the step plans with the post-update activation; "
                "without actearly the force lags one physics step (see ComputedMuscleControl)",
            ),
        ),
        True,
    ],
)
def test_one_physics_step_realises_the_requested_acceleration(actearly: bool) -> None:
    """1-DoF elbow at 90 deg, one 2 ms step towards ``qpos + 0.01``: the realised
    acceleration equals ``qacc*`` only when the force uses the updated activation."""
    model = _elbow(1)
    model.actuator_actearly[:] = int(actearly)
    cmc, data = ComputedMuscleControl(model), mujoco.MjData(model)
    zero = np.zeros(model.nv)
    qpos = model.qpos0.copy()
    qpos[0] = np.pi / 2
    cmc.reset(data, qpos, zero, zero)
    qvel0 = data.qvel.copy()
    res = cmc.step(data, qpos + 0.01, zero, zero, n_substeps=1)
    realised = (data.qvel - qvel0) / model.opt.timestep
    assert np.abs(res.qacc_desired).max() > 1.0
    np.testing.assert_allclose(realised, res.qacc_desired, rtol=1e-3, atol=1e-6)
