# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU <-> mjlab parity of the basic-suite task twins.

The mjlab twin of an env id must score the same state the same way as the
CPU env. Open-loop rollouts diverge (float32 Warp vs float64 MuJoCo in a
chaotic muscle system), so parity is checked one step at a time: the mjlab
env is synced to the CPU state, both take the same action, and observation,
reward and termination are compared.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.core.muscle_conditions import (  # noqa: E402
    CumulativeFatigue,
    TorchFatigueState,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state  # noqa: E402
from myosuite.envs.myo.tasks.basic.arm.pose import PoseEnvV0  # noqa: E402
from myosuite.envs.myo.tasks.basic.leg.reach import LegReachEnvV0  # noqa: E402
from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

# Entry points of the basic suite that have an mjlab twin.
_PORTED_ENTRY_POINTS = (
    "arm.pose:PoseEnvV0",
    "torso.pose:TorsoEnvV0",
    "arm.reach:ReachEnvV0",
    "leg.reach:LegReachEnvV0",
    "leg.walk:LegWalkEnvV0",
)

PARITY_IDS = (
    "myoFingerPoseRandom-v0",
    "motorFingerPoseFixed-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoElbowPose1D6MExoRandom-v0",
    "myoHandPoseRandom-v0",
    "myoHandPose3Fixed-v0",
    "myoSarcFingerPoseRandom-v0",
    "myoFatiElbowPose1D6MRandom-v0",
    "myoReafHandPoseRandom-v0",
    "myoTorsoPoseFixed-v0",
    "myoFatiTorsoPoseFixed-v0",
    "myoFingerReachRandom-v0",
    "motorFingerReachRandom-v0",
    "myoHandReachRandom-v0",
    "myoArmReachRandom-v0",
    "myoReafHandReachFixed-v0",
    "myoSarcArmReachFixed-v0",
    "myoLegStandRandom-v0",
    "myoLegWalk-v0",
    "myoSarcLegWalk-v0",
    "myoFatiLegWalk-v0",
)

# Ported families whose multi-floating-body models still need the
# entity-splitting scene builder (mjlab allows one freejoint per entity).
_PENDING_TWINS: set[str] = set()


# MuJoCo Warp computes a different wrapped length than C MuJoCo for a few
# side-site wraps at some poses, where the CPU length is smooth (not a float32
# branch flip; Warp does implement wrap_inside). Measured over random
# joint-range poses: hand EDC* ~2e-4 m (1 of 39 tendons), arm DELT2/DELT3/
# PECM1/FCR up to 2.6e-3 m, which shows up as up to ~8e-3 in qvel*dt after a step
# (muscle forces differ). The hand and arm models get a looser tolerance.
_WARP_WRAP_DIFF_OBS_ATOL = 1e-2
_WARP_WRAP_DIFF_REW_ATOL = 5e-2


# Foot-contact events: one step can differ by ~5e-3 in velocities although the muscle
# lengths agree to 1e-5 (constraint-solver ordering; the other steps match to ~1e-6).
_CONTACT_OBS_ATOL, _CONTACT_REW_ATOL = 1e-2, 1e-2


def _tolerances(env_id: str) -> tuple[float, float]:
    if "Hand" in env_id or "Arm" in env_id:
        return _WARP_WRAP_DIFF_OBS_ATOL, _WARP_WRAP_DIFF_REW_ATOL
    if "LegWalk" in env_id:
        return _CONTACT_OBS_ATOL, _CONTACT_REW_ATOL
    return 5e-4, 5e-3


def _basic_suite_ids(entry_points: tuple[str, ...]) -> list[str]:
    return sorted(
        env_id
        for env_id, spec in gym.registry.items()
        if any(str(spec.entry_point).endswith(ep) for ep in entry_points)
        and "tasks.basic" in str(spec.entry_point)
    )


def _make_pair(env_id: str) -> tuple[gym.Env, ManagerBasedRlEnv]:
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    mj = ManagerBasedRlEnv(cfg=load_env_cfg(env_id), device="cpu")
    mj.reset()
    return cpu, mj


def _sync(cpu: gym.Env, mj: ManagerBasedRlEnv) -> None:
    """Copy CPU state (physics, task target, fatigue, perturbations) into mjlab."""

    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x)[None], dtype=torch.float32)

    robot = mj.scene["robot"]
    write_cpu_state(  # CPU layout: a free root joint first, if any
        mj,
        "robot",
        torch.zeros(1, dtype=torch.long),
        _t(cpu.data.qpos),
        _t(cpu.data.qvel),
    )
    if cpu.model.na:
        mj.sim.data.act[:] = _t(cpu.data.act)
    if "pose" in mj.command_manager.active_terms:
        mj.command_manager.get_term("pose")._target[:] = _t(cpu.target_jnt_value)
    elif hasattr(cpu, "target_sids"):
        target = np.concatenate([cpu.data.site_xpos[s] for s in cpu.target_sids])
        mj.command_manager.get_term("reach")._target[:] = _t(target)

    fatigue = mj.action_manager.get_term("muscles")._fatigue
    if fatigue is not None:
        fatigue.MA[:], fatigue.MR[:], fatigue.MF[:] = (
            _t(cpu.muscle_fatigue.MA),
            _t(cpu.muscle_fatigue.MR),
            _t(cpu.muscle_fatigue.MF),
        )
    if getattr(cpu, "weight_bodyname", None) is not None:
        body = cpu.model.body(cpu.weight_bodyname).id
        geom = cpu.model.body_geomadr[body]
        mj_body = int(
            robot.indexing.body_ids[robot.find_bodies(cpu.weight_bodyname)[0][0]]
        )
        mj.sim.model.body_mass[:, mj_body] = float(cpu.model.body_mass[body])
        mj.sim.model.geom_size[:, int(mj.sim.mj_model.body_geomadr[mj_body]), 0] = (
            float(cpu.model.geom_size[geom, 0])
        )
    mj.episode_length_buf[:] = round(cpu.data.time / mj.step_dt)
    mj.sim.forward()


@pytest.mark.parametrize("env_id", PARITY_IDS)
def test_one_step_parity(env_id: str) -> None:
    """Same state + same action -> same obs, reward and termination."""
    cpu, mj = _make_pair(env_id)
    assert mj.action_manager.total_action_dim == cpu.action_space.shape[0]
    assert (
        mj.observation_manager.group_obs_dim["actor"][0]
        == cpu.observation_space.shape[0]
    )

    _sync(cpu, mj)
    obs0 = mj.observation_manager.compute_group("actor")[0].numpy()
    if isinstance(cpu, LegReachEnvV0 | LegWalkEnvV0):
        expected = cpu._obs_dict_to_vec(cpu._get_obs_dict(cpu._accessor))
    else:
        expected = cpu.get_obs()
    if isinstance(cpu, PoseEnvV0 | LegReachEnvV0 | LegWalkEnvV0):  # clip to +-10
        expected = expected.clip(-10, 10)
    np.testing.assert_allclose(obs0, expected, atol=1e-5)

    obs_atol, rew_atol = _tolerances(env_id)
    rng = np.random.default_rng(0)
    for _ in range(25):
        _sync(cpu, mj)
        action = rng.uniform(-1.2, 1.2, cpu.action_space.shape).astype(np.float32)
        cpu_obs, cpu_rew, cpu_term, _, cpu_info = cpu.step(action)
        mj_obs, mj_rew, mj_term, mj_trunc, _ = mj.step(torch.as_tensor(action[None]))
        assert not bool(mj_trunc[0])
        assert bool(mj_term[0]) == cpu_term
        if cpu_term:
            break
        np.testing.assert_allclose(mj_obs["actor"][0].numpy(), cpu_obs, atol=obs_atol)
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=rew_atol)
        values = dict(mj.metrics_manager.get_active_iterable_terms(0))
        assert bool(values["success"][0]) == bool(cpu_info["solved"])


def test_fatigue_torch_matches_numpy() -> None:
    """Batched torch 3CC-r fatigue reproduces the CPU model step by step."""
    import mujoco  # noqa: PLC0415

    model = gym.make("myoFatiElbowPose1D6MRandom-v0").unwrapped.model
    cpu = CumulativeFatigue(model, frame_skip=10)
    gpu = TorchFatigueState.from_mj_model(model, num_envs=1)
    rng = np.random.default_rng(0)
    dt = model.opt.timestep * 10
    for _ in range(200):
        excitation = rng.uniform(0, 1, cpu.na)
        cpu_ctrl, _, _ = cpu.compute_act(excitation.copy(), dt=dt)
        gpu_ctrl = gpu.step(torch.as_tensor(excitation[None], dtype=torch.float32), dt)
        np.testing.assert_allclose(gpu_ctrl[0].numpy(), cpu_ctrl, atol=1e-5)
    assert int(sum(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)) == cpu.na


def test_every_ported_cpu_env_has_mjlab_twin() -> None:
    """Every basic-suite CPU id of a ported family is registered with mjlab."""
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415

    registered = set(list_tasks())
    missing = [
        e
        for e in _basic_suite_ids(_PORTED_ENTRY_POINTS)
        if e not in registered and e not in _PENDING_TWINS
    ]
    assert not missing, f"CPU envs without mjlab twin: {missing}"


@pytest.mark.parametrize(
    "env_id",
    [
        "myoTorsoExoPoseFixed-v0",
        "myoSarcTorsoExoPoseFixed-v0",
        "myoFatiTorsoExoPoseFixed-v0",
    ],
)
def test_torso_exo_observation_matches_cpu(env_id: str) -> None:
    """The exo twin's 6-DoF chains are converted back to the CPU ``qpos``/``qvel``.

    Both envs start from the same reset and get the same random actions (free run, no
    state sync); the CPU observation layout and values must agree.
    """
    cpu = gym.make(env_id).unwrapped
    obs_cpu, _ = cpu.reset(seed=0)
    mj = ManagerBasedRlEnv(cfg=load_env_cfg(env_id), device="cpu")
    obs_mj = mj.reset()[0]["actor"][0].numpy()
    assert obs_mj.shape == obs_cpu.shape
    np.testing.assert_allclose(obs_mj, obs_cpu, atol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = rng.uniform(-1, 1, cpu.action_space.shape).astype(np.float32)
        obs_cpu = cpu.step(action)[0]
        obs_mj = mj.step(torch.as_tensor(action[None]))[0]["actor"][0].numpy()
        np.testing.assert_allclose(obs_mj, obs_cpu, atol=2e-3)


def test_every_twin_logs_a_success_metric() -> None:
    """Success rate is a standard metric (``Episode_Metrics/success``) of every twin."""
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415

    registered = set(list_tasks())
    twins = [e for e in _basic_suite_ids(_PORTED_ENTRY_POINTS) if e in registered]
    # Legacy leg/torso twins are not in the basic-suite entry-point list.
    twins += [e for e in registered if "Leg" in e or "Torso" in e]
    assert twins
    missing = [
        e
        for e in twins
        if "success" not in load_env_cfg(e).metrics
        or load_env_cfg(e).metrics["success"].reduce != "last"
    ]
    assert not missing, f"twins without a 'last'-reduced success metric: {missing}"
