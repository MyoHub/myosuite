# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for Soccer dense reward learnability."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier1


def test_soccer_act_reg_weight_is_not_catastrophic() -> None:
    from myosuite.envs.myo.tasks.challenge.soccer import SoccerEnv

    weights = SoccerEnv.DEFAULT_RWD_KEYS_AND_WEIGHTS
    assert weights["act_reg"] == pytest.approx(-0.1)
    assert abs(weights["act_reg"]) < 1.0
    assert "alive" in weights
    assert weights["alive"] == pytest.approx(0.5)


def test_soccer_time_cost_is_per_step_not_cumulative() -> None:
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeSoccerP1-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        obs_dict = unwrapped._get_obs_dict(unwrapped._accessor)
        # Fake a late-episode clock; time_cost value must stay 1.0.
        obs_dict = dict(obs_dict)
        obs_dict["time"] = np.array([50.0], dtype=np.float64)
        rwd = unwrapped.get_reward_dict(obs_dict)
        assert float(rwd["time_cost"]) == pytest.approx(1.0)
        assert float(rwd["alive"]) in (0.0, 1.0)
    finally:
        env.close()


def test_soccer_longer_active_rollout_improves_return() -> None:
    """Regression: act_reg=-100 made longer episodes catastrophically worse."""
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeSoccerP1-v0")
    try:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        totals = []
        for horizon in (3, 25):
            env.reset(seed=2)
            total = 0.0
            for _ in range(horizon):
                _obs, reward, terminated, truncated, _info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            totals.append(total)
        # Surviving longer must not crater returns by thousands.
        assert totals[1] > totals[0] - 5.0
        assert totals[1] > -50.0
    finally:
        env.close()


def test_soccer_pain_matches_per_joint_limit_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pain takes one J^T f product and equals the old per-joint computation."""
    import mujoco

    import myosuite  # noqa: F401  (registers the envs)
    from myosuite.utils import gym

    env = gym.make("myoChallengeSoccerP1-v0")
    u = env.unwrapped
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        space = env.action_space
        nonzero = 0
        for _ in range(30):
            env.step(rng.uniform(space.low, space.high).astype(space.dtype))
            expected = 0.0
            for joint in u.JNT_OVEREXT:
                efc_force = u.data.efc_force.copy()
                is_limit = u.data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
                efc_force[~is_limit] = 0.0
                qfrc = np.zeros(u.model.nv)
                mujoco.mj_mulJacTVec(u.model, u.data, qfrc, efc_force)
                frc = qfrc[u.model.joint(joint).dofadr].squeeze()
                expected += np.abs(np.clip(frc, -1000, 1000)) / 1000
            expected /= len(u.JNT_OVEREXT)
            pain = u._get_jnt_limit_violation()
            assert np.asarray(pain).tobytes() == np.asarray(expected).tobytes()
            nonzero += pain != 0.0
        assert nonzero > 0

        calls: list[int] = []
        mul_jac_t_vec = mujoco.mj_mulJacTVec

        def counting(*args: object) -> None:
            calls.append(1)
            mul_jac_t_vec(*args)

        monkeypatch.setattr(mujoco, "mj_mulJacTVec", counting)
        u._get_jnt_limit_violation()
        assert len(calls) == 1
    finally:
        env.close()
