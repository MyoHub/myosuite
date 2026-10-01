# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests: ``reset(seed=...)`` fully determines an episode.

Covers the challenge envs whose reset leaked state across episodes or drew
from an unseeded RNG (fatigue, Relocate's mocap goal, Bimanual's reset order,
Baoding's targets, the Soccer goalkeeper, rough RunTrack terrain) and the
fatigue reset of every env class that owns a ``CumulativeFatigue``.
"""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core.muscle_conditions import CumulativeFatigue
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.envs.heightfields import TrackTypes
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

SEED = 123
N_STEPS = 12

CHALLENGE_IDS = [
    "myoChallengeChaseTagP1-v0",
    "myoChallengeChaseTagP2-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeSoccerP2-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeOslRunRandom-v0",
    "myoChallengeBimanual-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeRelocateP2-v0",
    "myoChallengeRelocateP2eval-v0",
    "myoChallengeBaodingP1-v1",
    "myoChallengeBaodingP2-v1",
]
# One id per basic env class that owns a fatigue model.
BASIC_FATIGUE_IDS = [
    "myoFatiFingerReachRandom-v0",
    "myoFatiElbowPose1D6MRandom-v0",
    "myoFatiHandKeyTurnRandom-v0",
    "myoFatiHandObjHoldRandom-v0",
    "myoFatiHandPenTwirlRandom-v0",
    "myoFatiLegStandRandom-v0",
    "myoFatiLegWalk-v0",
    "myoFatiLegHillyTerrainWalk-v0",
    "myoFatiTorsoPoseFixed-v0",
]


def _fati(env_id: str) -> str:
    return "myoFati" + env_id[len("myo") :]


# (env_id, gym.make kwargs). The fatigue variants draw a random initial
# fatigue state, which must come from the env seed too.
ROLLOUT_CASES = (
    [(env_id, {}) for env_id in CHALLENGE_IDS]
    + [("myoChallengeSoccerP2-v0", {"goalkeeper_probabilities": (0.0, 1.0, 0.0)})]
    + [(_fati(env_id), {"fatigue_reset_random": True}) for env_id in CHALLENGE_IDS]
    + [(env_id, {"fatigue_reset_random": True}) for env_id in BASIC_FATIGUE_IDS]
)
FATIGUE_IDS = [
    _fati(env_id)
    for env_id in (
        "myoChallengeChaseTagP1-v0",
        "myoChallengeChaseTagFBP2-v0",
        "myoChallengeSoccerP1-v0",
        "myoChallengeOslRunFixed-v0",
        "myoChallengeBimanual-v0",
        "myoChallengeRelocateP1-v0",
        "myoChallengeBaodingP1-v1",
        "myoChallengeDieReorientP2-v0",
    )
] + BASIC_FATIGUE_IDS


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """These tests reseed the global np.random; restore it for later tests."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _case_id(case: tuple[str, dict]) -> str:
    env_id, kwargs = case
    return env_id + "".join(f"-{k}" for k in kwargs)


def _rollout(env: gym.Env, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the obs and reward trajectory of ``reset(seed=SEED)`` + actions."""
    obs, _ = env.reset(seed=SEED)
    obs_traj, rwd_traj = [np.asarray(obs, dtype=np.float64)], []
    for action in actions:
        obs, rwd, terminated, truncated, _ = env.step(action)
        obs_traj.append(np.asarray(obs, dtype=np.float64))
        rwd_traj.append(float(rwd))
        if terminated or truncated:
            break
    return np.stack(obs_traj), np.asarray(rwd_traj)


def _actions(env: gym.Env) -> np.ndarray:
    space = env.action_space
    rng = np.random.default_rng(7)
    return rng.uniform(space.low, space.high, size=(N_STEPS, *space.shape)).astype(
        np.float32
    )


def _assert_same(ref: tuple, other: tuple, what: str) -> None:
    np.testing.assert_array_equal(other[0], ref[0], err_msg=f"{what}: obs differ")
    np.testing.assert_array_equal(other[1], ref[1], err_msg=f"{what}: reward differ")


@pytest.mark.parametrize(
    ("env_id", "kwargs"), ROLLOUT_CASES, ids=map(_case_id, ROLLOUT_CASES)
)
def test_same_instance_reset_is_reproducible(env_id: str, kwargs: dict) -> None:
    """reset(seed) -> actions -> reset(seed) -> same actions repeats the episode."""
    env = gym.make(env_id, **kwargs)
    try:
        actions = _actions(env)
        # Warm-up reset with another seed: an episode must not depend on the
        # previous one (Bimanual also skips its object scaling on reset #1).
        env.reset(seed=SEED + 1)
        first = _rollout(env, actions)
        np.random.seed(4242)
        _assert_same(first, _rollout(env, actions), f"{env_id}: second reset(seed)")
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_id", "kwargs"), ROLLOUT_CASES, ids=map(_case_id, ROLLOUT_CASES)
)
def test_fresh_instances_ignore_global_rng(env_id: str, kwargs: dict) -> None:
    """Fresh instances built under different global np.random states agree."""
    trajs = []
    for global_seed in (0, 999):
        np.random.seed(global_seed)
        env = gym.make(env_id, **kwargs)
        try:
            trajs.append(_rollout(env, _actions(env)))
        finally:
            env.close()
    _assert_same(trajs[0], trajs[1], f"{env_id}: fresh instance")


@pytest.mark.parametrize("env_id", FATIGUE_IDS)
def test_reset_restores_fatigue_state(env_id: str) -> None:
    """A driven fatigue model is back at MA=0, MR=1, MF=0 after reset()."""
    env = gym.make(env_id)
    try:
        fatigue = env.unwrapped.muscle_fatigue
        env.reset(seed=SEED)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        for _ in range(N_STEPS):
            *_, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert fatigue.MA.max() > 0.1, "the fatigue model was not driven"
        env.reset(seed=SEED)
        np.testing.assert_array_equal(fatigue.MA, 0.0)
        np.testing.assert_array_equal(fatigue.MR, 1.0)
        np.testing.assert_array_equal(fatigue.MF, 0.0)
    finally:
        env.close()


def test_fatigue_random_reset_draws_from_given_generator() -> None:
    """reset(np_random=...) uses that generator; without it the own RNG is used."""
    env = gym.make("myoFatiElbowPose1D6MRandom-v0")
    model = env.unwrapped.model
    env.close()
    a, b = CumulativeFatigue(model, seed=1), CumulativeFatigue(model, seed=2)
    for fatigue in (a, b):
        fatigue.reset(fatigue_reset_random=True, np_random=np.random.default_rng(0))
    np.testing.assert_array_equal(a.MF, b.MF)
    np.testing.assert_allclose(a.MA + a.MR + a.MF, 1.0)

    c, d = CumulativeFatigue(model, seed=1), CumulativeFatigue(model, seed=1)
    c.reset(fatigue_reset_random=True)
    d.reset(fatigue_reset_random=True)
    np.testing.assert_array_equal(c.MF, d.MF)


@pytest.mark.parametrize(
    "env_id",
    [
        "myoChallengeRelocateP1-v0",
        "myoChallengeRelocateP2-v0",
        "myoChallengeRelocateP2eval-v0",
    ],
)
def test_relocate_simulates_the_sampled_goal(env_id: str) -> None:
    """The mocap goal of an episode is the pose that episode sampled."""
    env = gym.make(env_id)
    u = env.unwrapped
    try:
        for seed in (SEED, SEED + 1):
            env.reset(seed=seed)
            np.testing.assert_allclose(
                u.data.xpos[u.goal_bid], u.model.body_pos[u.goal_bid], atol=1e-12
            )
            np.testing.assert_allclose(
                u.data.xquat[u.goal_bid], u.model.body_quat[u.goal_bid], atol=1e-9
            )
    finally:
        env.close()


def test_bimanual_reset_builds_obs_from_the_reset_state() -> None:
    """Obs and lift baselines come from the forwarded, teleported reset state."""
    env = gym.make("myoChallengeBimanual-v0")
    u = env.unwrapped
    try:
        u.max_force = 1e3
        obs, _ = env.reset(seed=SEED)
        ref = mujoco.MjData(u.model)
        ref.qpos[:], ref.qvel[:], ref.act[:] = u.data.qpos, u.data.qvel, u.data.act
        ref.mocap_pos[:], ref.mocap_quat[:] = u.data.mocap_pos, u.data.mocap_quat
        mujoco.mj_forward(u.model, ref)
        assert u.init_obj_z == pytest.approx(ref.site_xpos[u.obj_sid][2])
        assert u.init_palm_z == pytest.approx(ref.site_xpos[u.palm_sid][2])
        assert u.max_force == pytest.approx(max(0.0, ref.sensordata[0]))

        mujoco.mj_forward(u.model, u.data)
        expected = u._obs_dict_to_vec(
            u._get_obs_dict(CpuEnvAccessor(u.model, u.data, u._ctrl_dt))
        )
        np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-6)
    finally:
        env.close()


def test_bimanual_success_does_not_end_next_episode() -> None:
    """A solved episode must not make the next episode terminate on step 1."""
    env = gym.make("myoChallengeBimanual-v0")
    u = env.unwrapped
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    adr = u.model.jnt_qposadr[u.model.body(u.obj_bid).jntadr[0]]
    try:
        env.reset(seed=SEED)
        for _ in range(3):
            u.data.qpos[adr : adr + 3] = [u.goal_pos[0], u.goal_pos[1], 1.12]
            u.data.qvel[:] = 0.0
            u.goal_touch = 100
            *_, terminated, _, info = env.step(zero)
            if terminated:
                break
        assert terminated and info["rwd_dict"]["solved"], "episode was not solved"
        env.reset(seed=SEED + 1)
        *_, terminated, _, _ = env.step(zero)
        assert not terminated
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_id", ["myoChallengeBaodingP1-v1", "myoChallengeBaodingP2-v1"]
)
def test_baoding_reset_places_first_targets(env_id: str) -> None:
    """reset() shows the targets that the first step() aims for."""
    env = gym.make(env_id)
    u = env.unwrapped
    sids = [u.target1_sid, u.target2_sid]
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    try:
        for seed in (SEED, SEED + 1):
            env.reset(seed=seed)
            at_reset = u.model.site_pos[sids].copy()
            for _ in range(N_STEPS):
                env.step(zero)
                if u.counter == 1:
                    np.testing.assert_array_equal(u.model.site_pos[sids], at_reset)
    finally:
        env.close()


def test_soccer_goalkeeper_noise_matches_episode_speed() -> None:
    """The random-walk noise is scaled by this episode's goalkeeper speed."""
    env = gym.make("myoChallengeSoccerP2-v0", goalkeeper_probabilities=(0.0, 1.0, 0.0))
    keeper = env.unwrapped.goalkeeper
    try:
        for seed in (SEED, SEED + 1):
            env.reset(seed=seed)
            assert keeper.noise_process.scale == keeper.block_velocity
    finally:
        env.close()


def test_rough_track_terrain_ignores_global_rng() -> None:
    """The rough RunTrack terrain is drawn from the env seed only."""
    env = gym.make("myoChallengeOslRunRandom-v0")
    u = env.unwrapped
    try:
        for seed in range(50):
            env.reset(seed=seed)
            if u.trackfield.terrain_type == TrackTypes.ROUGH:
                break
        else:
            pytest.fail("no rough track among the first 50 seeds")
        hfield = u.model.hfield_data.copy()
        np.random.seed(1)
        env.reset(seed=seed)
        np.testing.assert_array_equal(u.model.hfield_data, hfield)
    finally:
        env.close()


def test_bimanual_pillars_follow_the_sampled_start_and_goal() -> None:
    """The mocap pillars sit at this episode's sampled start/goal, not at the XML centres."""
    env = gym.make("myoChallengeBimanual-v0")
    u = env.unwrapped
    try:
        positions = []
        for seed in (SEED, SEED + 1):
            env.reset(seed=seed)
            for body, sampled in ((u.start_bid, u.start_pos), (u.goal_bid, u.goal_pos)):
                np.testing.assert_allclose(u.data.xpos[body], sampled, atol=1e-9)
            positions.append(u.data.xpos[u.start_bid].copy())
        assert not np.allclose(*positions)  # the sample actually moves the pillar
        obj_adr = u.model.body(u.obj_bid).jntadr[0]
        np.testing.assert_allclose(
            u.data.qpos[obj_adr : obj_adr + 2], u.start_pos[:2], atol=1e-9
        )
    finally:
        env.close()
