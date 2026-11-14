# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Step-local reuse must not change tracking outputs or state-reading hooks."""

from __future__ import annotations

from types import MethodType

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID
from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body
from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip
from myosuite.terms import mimic_track as mt

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("state_reuse") / "synthetic.npz"
    write_synthetic_clip(path, compiled_body("myofullbody"), 1.0, 0.0, 0.02, 0)
    return str(path)


def _env(clip: str, *, ctrl_dt: float = 0.02, **overrides):
    return make_env(
        EnvConfig(
            MIMIC_ENV_ID,
            ctrl_dt=ctrl_dt,
            task_kwargs={"overrides": {"motion.clips": clip, **overrides}},
        )
    )


def test_step_reuses_features_but_external_reads_are_fresh(clip, monkeypatch) -> None:
    env = _env(clip)
    u = env.unwrapped
    env.reset(seed=0, options={"clip": 0})
    calls = []
    features = mt.features

    def counted(xp, state):
        calls.append(state)
        return features(xp, state)

    monkeypatch.setattr(mt, "features", counted)
    try:
        env.step(np.full(u.model.nu, 0.05))
        assert len(calls) == 1
        assert u._step_state is None and not u._reuse_step_state
        before = u.track_state()
        u.data.qpos[u.info.root_qadr + 2] += 0.25
        mujoco.mj_forward(u.model, u.data)
        obs = u._get_obs_dict(u._accessor)
        np.testing.assert_allclose(obs["root_pos"][0], before.root_pos[2] + 0.25)
        assert len(calls) == 2
        np.testing.assert_array_equal(obs["joint_pos"], u.data.qpos[u.info.joint_qadr])
        env.reset(seed=0, options={"clip": 0})
        assert len(calls) == 3
        assert u.track_state().root_pos[2] != before.root_pos[2] + 0.25
    finally:
        env.close()


@pytest.mark.parametrize("dt", [0.01, 0.02])
def test_reuse_matches_fresh_reads_at_multiple_control_rates(clip, dt) -> None:
    cached, fresh = _env(clip, ctrl_dt=dt), _env(clip, ctrl_dt=dt)
    u = fresh.unwrapped
    original = u.track_state
    u.track_state = MethodType(lambda self: original(), u)
    try:
        for seed in (0, 13):
            a = cached.reset(seed=seed, options={"clip": 0})
            b = fresh.reset(seed=seed, options={"clip": 0})
            np.testing.assert_array_equal(a[0], b[0])
            for action in np.random.default_rng(seed).uniform(
                -0.1, 0.2, (15, u.model.nu)
            ):
                out_a, out_b = cached.step(action), fresh.step(action)
                np.testing.assert_array_equal(out_a[0], out_b[0])
                assert out_a[1:4] == out_b[1:4]
                assert out_a[4]["rwd_dict"] == out_b[4]["rwd_dict"]
                for name, value in out_a[4]["obs_dict"].items():
                    np.testing.assert_array_equal(value, out_b[4]["obs_dict"][name])
                np.testing.assert_array_equal(cached.unwrapped.data.qpos, u.data.qpos)
                if out_a[2] or out_a[3]:
                    break
    finally:
        cached.close()
        fresh.close()


def test_mutating_custom_reward_cannot_change_observed_state(clip) -> None:
    def reward(state, ref, dt):
        state.joint_pos[:] = 123.0
        state.body_rot[:] = 0.0
        return 7.0

    env = _env(clip, reward_fn=reward)
    u = env.unwrapped
    try:
        env.reset(seed=0, options={"clip": 0})
        obs, rew, *_ = env.step(np.zeros(u.model.nu))
        assert rew == 7.0
        expected = u._obs_dict_to_vec(u._get_obs_dict(u._accessor)).astype(np.float32)
        np.testing.assert_array_equal(obs, expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    "hook", ["get_reward_dict", "transition_terms", "_get_obs_dict", "track_state"]
)
def test_overridden_hooks_keep_fresh_reads(clip, hook, monkeypatch) -> None:
    env = _env(clip)
    u = env.unwrapped
    original = getattr(type(u), hook)

    def edited(self, *args):
        result = original(self, *args)
        if hook == "track_state":
            result.joint_pos[:] = 0.125
        else:
            self.data.qpos[self.info.root_qadr + 2] += 0.1
            mujoco.mj_forward(self.model, self.data)
        return result

    class HookEnv(type(u)):
        pass

    setattr(HookEnv, hook, edited)
    u.__class__ = HookEnv
    calls = []
    features = mt.features

    def counted(xp, state):
        calls.append(state)
        return features(xp, state)

    try:
        env.reset(seed=0, options={"clip": 0})
        monkeypatch.setattr(mt, "features", counted)
        obs, *_ = env.step(np.zeros(u.model.nu))
        assert len(calls) == 2
        assert u._step_state is None and not u._reuse_step_state
        if hook in ("get_reward_dict", "transition_terms"):
            np.testing.assert_allclose(obs[0], u.data.qpos[u.info.root_qadr + 2])
        # A subsequent ordinary read must use the current simulator, including edits
        # made by the overridden hook after calling super.
        state, _ = u._state_features()
        if hook == "track_state":
            np.testing.assert_array_equal(state.joint_pos, 0.125)
        else:
            np.testing.assert_array_equal(state.root_pos, u.data.qpos[:3])
    finally:
        env.close()


def test_exception_clears_step_state(clip, monkeypatch) -> None:
    env = _env(clip)
    u = env.unwrapped
    env.reset(seed=0, options={"clip": 0})
    goal = mt.goal_observation

    def broken_goal(*args):
        raise RuntimeError("observation failed")

    monkeypatch.setattr(mt, "goal_observation", broken_goal)
    try:
        with pytest.raises(RuntimeError, match="observation failed"):
            env.step(np.zeros(u.model.nu))
        assert u._step_state is None and not u._reuse_step_state
        monkeypatch.setattr(mt, "goal_observation", goal)
        env.reset(seed=0, options={"clip": 0})
        env.step(np.zeros(u.model.nu))
        assert u._step_state is None
    finally:
        env.close()
