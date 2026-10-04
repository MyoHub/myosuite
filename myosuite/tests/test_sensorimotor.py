# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU sensorimotor delay and observation noise (``SensorimotorCfg``).

Delays are checked against an undelayed env of the same id: an observation
delay only shifts the observation stream, an action delay equals feeding the
undelayed env the action stream shifted by ``k`` (raw 0 after every reset).
The mjlab twin is tested in ``test_sensorimotor_mjlab.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core.sensorimotor import FixedLagBuffer, SensorimotorCfg
from myosuite.envs.myo.tasks.basic.arm.pose import PoseEnvV0
from myosuite.terms.base_action import sigmoid_muscle_activation

pytestmark = pytest.mark.tier1

ENV_ID = "myoElbowPose1D6MRandom-v0"
EPISODES = ((1, 25), (2, 25))  # (reset seed, steps): the 2nd reset is mid-run


def _actions(env: gym.Env, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, *env.action_space.shape)).astype(np.float32)


def _rollout(
    env: gym.Env, actions: np.ndarray, action_delay: int = 0
) -> tuple[list[list[np.ndarray]], list[list[float]], list[list[np.ndarray]]]:
    """Per-episode observations, rewards and applied ``ctrl`` over :data:`EPISODES`.

    With ``action_delay`` the env is fed the action stream shifted by that many
    steps, zero after every reset (the oracle of an action delay).
    """
    obs, rew, ctrl, i = [], [], [], 0
    for seed, steps in EPISODES:
        obs.append([env.reset(seed=seed)[0]])
        rew.append([])
        ctrl.append([])
        queue = deque(np.zeros_like(actions[0]) for _ in range(action_delay))
        for _ in range(steps):
            queue.append(actions[i])
            o, r, *_ = env.step(queue.popleft())
            obs[-1].append(o)
            rew[-1].append(r)
            ctrl[-1].append(env.unwrapped.data.ctrl.copy())
            i += 1
    return obs, rew, ctrl


def test_cfg_validation_and_default_off() -> None:
    assert not SensorimotorCfg().enabled
    assert SensorimotorCfg.coerce(None) == SensorimotorCfg()
    assert SensorimotorCfg(obs_noise_std=0.1).enabled
    assert SensorimotorCfg(obs_delay_steps=np.int64(2)).obs_delay_steps == 2
    for bad in ({"obs_delay_steps": -1}, {"obs_noise_std": float("nan")}):
        with pytest.raises(ValueError):
            SensorimotorCfg(**bad)
    for bad in ({"action_delay_steps": 1.5}, {"obs_delay_steps": True}):
        with pytest.raises(TypeError):
            SensorimotorCfg(**bad)
    with pytest.raises(TypeError):
        SensorimotorCfg.coerce({"obs_delay_steps": 1})  # type: ignore[arg-type]


def test_fixed_lag_buffer() -> None:
    buf = FixedLagBuffer(2, np.zeros((3, 1)))
    frames = [np.full((3, 1), float(t)) for t in range(1, 5)]
    out = [buf.push(f) for f in frames[:2]]
    assert all(np.array_equal(o, np.zeros((3, 1))) for o in out)
    assert np.array_equal(buf.push(frames[2]), frames[0])
    buf.refill(np.full((3, 1), -1.0), rows=np.array([1]))  # row 1 restarts
    assert np.array_equal(buf.push(frames[3]).ravel(), [2.0, -1.0, 2.0])
    frames[3][:] = 99.0  # pushed frames are copied
    assert np.array_equal(buf.push(frames[0]).ravel(), [3.0, -1.0, 3.0])
    assert np.array_equal(buf.push(frames[1]).ravel(), [4.0, 4.0, 4.0])
    assert np.array_equal(FixedLagBuffer(0, frames[0]).push(frames[1]), frames[1])


def test_obs_delay_shifts_observations() -> None:
    """obs_t = undelayed obs_{max(0, t-k)} per episode; dynamics and rewards unchanged."""
    k = 3
    base = gym.make(ENV_ID)
    delayed = gym.make(ENV_ID, sensorimotor=SensorimotorCfg(obs_delay_steps=k))
    actions = _actions(base, sum(n for _, n in EPISODES))
    obs0, rew0, ctrl0 = _rollout(base, actions)
    obsd, rewd, ctrld = _rollout(delayed, actions)
    for ep0, epd in zip(obs0, obsd, strict=True):
        for t, o in enumerate(epd):
            np.testing.assert_array_equal(o, ep0[max(0, t - k)])
        # After the mid-run reset the history holds the reset obs, not the old episode.
        assert all(np.array_equal(o, ep0[0]) for o in epd[: k + 1])
    assert rewd == rew0
    np.testing.assert_array_equal(np.concatenate(ctrld), np.concatenate(ctrl0))


@pytest.mark.parametrize("env_id", [ENV_ID, "myoFatiElbowPose1D6MRandom-v0"])
def test_action_delay_applies_shifted_actions(env_id: str) -> None:
    """Action delay k == the undelayed env fed the stream shifted by k (fatigue included)."""
    k = 2
    base = gym.make(env_id)
    delayed = gym.make(env_id, sensorimotor=SensorimotorCfg(action_delay_steps=k))
    actions = _actions(base, sum(n for _, n in EPISODES))
    obs0, rew0, ctrl0 = _rollout(base, actions, action_delay=k)
    obsd, rewd, ctrld = _rollout(delayed, actions)
    np.testing.assert_array_equal(np.concatenate(obsd), np.concatenate(obs0))
    np.testing.assert_array_equal(np.concatenate(ctrld), np.concatenate(ctrl0))
    assert rewd == rew0
    if (
        "Fati" not in env_id
    ):  # first k steps of every episode: excitation of raw action 0
        rest = sigmoid_muscle_activation(np.float32(0.0), np)
        for ep in ctrld:
            np.testing.assert_allclose(np.asarray(ep[:k]), rest, rtol=1e-6)


def test_obs_noise_seeded_unbiased_and_after_delay() -> None:
    """N(0, sigma^2) from np_random: seeded, unbiased, applied after the delay."""
    sigma, k = 0.05, 3
    cfg = SensorimotorCfg(obs_delay_steps=k, obs_noise_std=sigma)
    base = gym.make(ENV_ID)
    noisy = gym.make(ENV_ID, sensorimotor=cfg)
    actions = _actions(base, 99)  # one episode (TimeLimit 100)
    clean = [base.reset(seed=7)[0]] + [base.step(a)[0] for a in actions]
    run = [noisy.reset(seed=7)[0]] + [noisy.step(a)[0] for a in actions]
    again = gym.make(ENV_ID, sensorimotor=cfg)
    rerun = [again.reset(seed=7)[0]] + [again.step(a)[0] for a in actions[:20]]
    np.testing.assert_array_equal(np.asarray(rerun), np.asarray(run[:21]))
    other = gym.make(ENV_ID, sensorimotor=cfg).reset(seed=8)[0]
    assert not np.array_equal(other, run[0])

    noise = np.asarray([o - clean[max(0, t - k)] for t, o in enumerate(run)])
    n = noise.size
    assert abs(noise.std() - sigma) < 0.1 * sigma
    assert abs(noise.mean()) < 4 * sigma / np.sqrt(n)
    lag1 = np.corrcoef(noise[1:].ravel(), noise[:-1].ravel())[0, 1]
    assert abs(lag1) < 0.1  # fresh noise every step, also during the reset fill
    assert len({o.tobytes() for o in run[: k + 1]}) == k + 1


def test_default_off_is_untouched() -> None:
    """No kwarg, ``None`` and ``SensorimotorCfg()`` give bit-identical rollouts and RNG."""
    envs = [
        gym.make(ENV_ID),
        gym.make(ENV_ID, sensorimotor=None),
        gym.make(ENV_ID, sensorimotor=SensorimotorCfg()),
    ]
    actions = _actions(envs[0], sum(n for _, n in EPISODES))
    runs = [_rollout(env, actions) for env in envs]
    for obs, rew, ctrl in runs[1:]:
        np.testing.assert_array_equal(np.concatenate(obs), np.concatenate(runs[0][0]))
        assert rew == runs[0][1]
    states = [env.unwrapped.np_random.bit_generator.state for env in envs]
    assert states[1] == states[0] and states[2] == states[0]
    assert all(env.unwrapped._sensorimotor is None for env in envs)


class _SuperStepPose(PoseEnvV0):
    """A task whose ``step``/``reset`` call the parent's (must delay once)."""

    def step(self, action: np.ndarray, **kwargs: Any):  # type: ignore[override]
        return super().step(action, **kwargs)

    def reset(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        return super().reset(*args, **kwargs)


def test_delay_applies_once_and_without_gym_make() -> None:
    """Direct construction, ``.unwrapped`` stepping and ``super().step`` chains."""
    cfg = SensorimotorCfg(obs_delay_steps=2, action_delay_steps=1)
    kwargs = dict(gym.spec(ENV_ID).kwargs)
    nested = _SuperStepPose(**kwargs, sensorimotor=cfg)
    reference = gym.make(ENV_ID, sensorimotor=cfg).unwrapped
    assert nested.sensorimotor == cfg
    actions = _actions(reference, 12)
    o1 = [nested.reset(seed=3)[0]] + [nested.step(a)[0] for a in actions]
    o2 = [reference.reset(seed=3)[0]] + [reference.step(a)[0] for a in actions]
    np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))
