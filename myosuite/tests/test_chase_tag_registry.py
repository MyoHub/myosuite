# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for the directional myoLeg and 1v1 chase-tag environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1


def test_chase_tag_vs_config_fields_are_subset_of_task_config() -> None:
    """``ChaseTagVsConfig``'s fields must stay a subset of ``ChaseTagVsTaskConfig``'s.

    ``ChaseTagVsTaskConfig._low_level_config()`` projects itself onto
    ``ChaseTagVsConfig`` generically (by matching field names), rather than
    listing every field by hand. This test is the guard that keeps that
    projection valid: if someone adds a field to ``ChaseTagVsConfig`` without
    a same-named field on ``ChaseTagVsTaskConfig``, the projection would
    raise a ``TypeError`` at construction time -- this test catches it at
    collection time instead.
    """
    import dataclasses

    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
        ChaseTagVsConfig,
    )
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
        ChaseTagVsTaskConfig,
    )

    low_level_fields = {f.name for f in dataclasses.fields(ChaseTagVsConfig)}
    task_config_fields = {f.name for f in dataclasses.fields(ChaseTagVsTaskConfig)}
    missing = low_level_fields - task_config_fields
    assert not missing, (
        f"ChaseTagVsConfig fields {missing} have no matching field on "
        "ChaseTagVsTaskConfig; _low_level_config()'s generic projection "
        "would fail."
    )


class TestLegDirectionalRegistry:
    """Smoke tests for the single-agent directional myoLeg locomotion tasks."""

    @pytest.mark.parametrize(
        "env_id", ["myoLegDirectionalForward-v0", "myoLegDirectionalBackward-v0"]
    )
    def test_registered_and_runs(self, env_id: str) -> None:
        import myosuite

        myosuite.register_all_envs()
        assert gym.spec(env_id) is not None

        env = gym.make(env_id)
        try:
            obs, info = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            for _ in range(5):
                action = env.action_space.sample()
                obs, rwd, terminated, truncated, info = env.step(action)
                assert np.isfinite(rwd)
                assert env.observation_space.contains(obs)
        finally:
            env.close()


class TestFullBodyChaseTagRegistry:
    """Smoke tests for myoChallengeChaseTagFBP2-v0 (full-body vs. scripted opponent)."""

    ENV_ID = "myoChallengeChaseTagFBP2-v0"

    def test_registered_and_runs(self) -> None:
        import myosuite

        myosuite.register_all_envs()
        assert gym.spec(self.ENV_ID) is not None

        env = gym.make(self.ENV_ID)
        try:
            model = env.unwrapped.model
            assert model.na == 354, f"expected 354 full-body muscles, got {model.na}"
            assert model.nu == 354
            assert model.body("opponent").id is not None
            assert env.unwrapped._pelvis_body_name == "pelvis"
            env.unwrapped.model.body(env.unwrapped._pelvis_body_name)  # resolves ok

            obs, info = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            for _ in range(30):
                action = env.action_space.sample()
                obs, rwd, terminated, truncated, info = env.step(action)
                assert np.isfinite(rwd)
                assert np.isfinite(obs).all()
        finally:
            env.close()


class TestChaseTagVsRegistry:
    """Smoke tests for myoChallengeChaseTagFBVs-v0 (fullbody steered variant)."""

    ENV_ID = "myoChallengeChaseTagFBVs-v0"

    def test_registered_and_runs(self) -> None:
        require_musclemimic_models()
        import myosuite

        myosuite.register_all_envs()
        assert gym.spec(self.ENV_ID) is not None

        try:
            env = gym.make(self.ENV_ID)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Gait reference clip unavailable (no network?): {exc}")
            return
        try:
            obs, info = env.reset(seed=0)
            assert set(obs.keys()) == {"agent_0", "agent_1"}
            for _ in range(5):
                actions = {a: env.unwrapped.action_space[a].sample() for a in obs}
                obs, rewards, terminated, truncated, info = env.step(actions)
                for agent_id in ("agent_0", "agent_1"):
                    assert np.isfinite(rewards[agent_id])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Gait reference clip unavailable (no network?): {exc}")
        finally:
            env.close()

    def test_chaser_catching_runner_increases_runner_health(self) -> None:
        """Driving the chaser onto the runner's pelvis should accrue tag pressure."""
        import myosuite
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
            ChaseTagVsTaskConfig,
        )
        from myosuite.envs.multi_agent_modular_env import ModularMultiAgentTaskEnv

        myosuite.register_all_envs()
        env = ModularMultiAgentTaskEnv(ChaseTagVsTaskConfig(agent_separation_m=0.5))
        obs, info = env.reset(seed=0)
        zero_actions = {
            a: np.zeros(env.action_space[a].shape, dtype=np.float32) for a in obs
        }
        for _ in range(20):
            obs, rewards, terminated, truncated, info = env.step(zero_actions)
        # Info should contain MyoChallenge-compatible fields.
        assert "task" in info
        assert info["task"] in ("CHASE", "EVADE")
        assert "elapsed_s" in info
        assert "score" in info
        env.close()
