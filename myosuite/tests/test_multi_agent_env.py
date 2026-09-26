# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for ModularMultiAgentTaskEnv and related infrastructure.

These tests use a minimal two-body MuJoCo model with one actuator per agent
so they run in < 1 s without any MuscleMimic dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.core.multi_agent_config import MultiAgentTaskConfig
from myosuite.core.registry import register_task
from myosuite.envs.multi_agent_modular_env import (
    ModularMultiAgentSingleAgentWrapper,
    ModularMultiAgentTaskEnv,
)

# ---------------------------------------------------------------------------
# Minimal two-agent MJCF
# ---------------------------------------------------------------------------

_MJCF = """
<mujoco model="dummy_two_agent">
  <worldbody>
    <body name="a0_body" pos="0 0.5 0">
      <joint name="a0_j0" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="a1_body" pos="0 -0.5 0">
      <joint name="a1_j0" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="a0_act0" joint="a0_j0" gear="10"/>
    <motor name="a1_act0" joint="a1_j0" gear="10"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Dummy task config
# ---------------------------------------------------------------------------


@dataclass
class DummyMultiAgentConfig(MultiAgentTaskConfig):
    """Minimal two-agent task: two sliding masses, zero-damage, trivial obs."""

    agents: tuple[str, ...] = ("agent_0", "agent_1")
    max_episode_steps: int = 10

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
        model = mujoco.MjModel.from_xml_string(_MJCF)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # meta: maps each agent to its actuator index
        meta = {
            "act_indices": {
                "agent_0": [0],
                "agent_1": [1],
            },
            "jnt_ids": {
                "agent_0": [0],
                "agent_1": [1],
            },
        }
        return model, data, meta

    @property
    def n_substeps(self) -> int:
        return 5

    def obs_dim(self, model, data, meta, agent_id: str) -> int:
        return 2  # [qpos, qvel] for this agent's joint

    def act_indices(self, meta: dict, agent_id: str) -> list[int]:
        return meta["act_indices"][agent_id]

    def get_obs(self, model, data, meta, agent_id: str, health: dict) -> np.ndarray:
        jid = meta["jnt_ids"][agent_id][0]
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        return np.array([data.qpos[qpos_adr], data.qvel[qvel_adr]], dtype=np.float32)

    def compute_damage(self, model, data, meta) -> dict[str, float]:
        return {"agent_0": 0.0, "agent_1": 0.0}

    def compute_reward(
        self, data, meta, agent_id, damage_delivered, health, actions, fell, ko
    ) -> float:
        return 0.0

    def get_info(
        self, data, meta, health, damage_delivered, fell, ko, step_count
    ) -> dict:
        return {"step": step_count}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_env_spaces():
    """Observation and action spaces have correct dtypes and shapes."""
    env = ModularMultiAgentTaskEnv(DummyMultiAgentConfig())
    assert set(env.observation_space.spaces) == {"agent_0", "agent_1"}
    assert set(env.action_space.spaces) == {"agent_0", "agent_1"}
    assert env.observation_space["agent_0"].shape == (2,)
    assert env.observation_space["agent_1"].shape == (2,)
    assert env.action_space["agent_0"].shape == (1,)
    assert env.action_space["agent_1"].shape == (1,)
    env.close()


def test_reset_returns_correct_shapes():
    """reset() returns obs dict with correct shapes."""
    env = ModularMultiAgentTaskEnv(DummyMultiAgentConfig())
    obs, info = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_1"}
    assert obs["agent_0"].shape == (2,)
    assert obs["agent_1"].shape == (2,)
    assert isinstance(info, dict)
    env.close()


def test_step_returns_correct_shapes():
    """step() returns dicts with correct keys and value shapes."""
    env = ModularMultiAgentTaskEnv(DummyMultiAgentConfig())
    env.reset(seed=0)
    actions = {a: env.action_space[a].sample() for a in ("agent_0", "agent_1")}
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert set(obs) == {"agent_0", "agent_1"}
    assert set(rewards) == {"agent_0", "agent_1"}
    assert set(terminated) == {"agent_0", "agent_1"}
    assert set(truncated) == {"agent_0", "agent_1"}
    assert isinstance(info, dict)
    env.close()


def test_truncation_at_max_steps():
    """Episode truncates after max_episode_steps."""
    cfg = DummyMultiAgentConfig(max_episode_steps=3)
    env = ModularMultiAgentTaskEnv(cfg)
    env.reset(seed=0)
    for i in range(3):
        actions = {a: env.action_space[a].sample() for a in env._agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
    assert any(truncated.values()), "Should truncate at step 3"
    env.close()


def test_ko_terminates_episode():
    """Episode terminates when any agent's health reaches ko_threshold."""

    @dataclass
    class HighDamageConfig(DummyMultiAgentConfig):
        @property
        def ko_threshold(self) -> float:
            return 1.0  # very low KO threshold

        def compute_damage(self, model, data, meta) -> dict[str, float]:
            return {"agent_0": 100.0, "agent_1": 100.0}  # massive damage

        @property
        def health_update_scale(self) -> float:
            return 1.0

    env = ModularMultiAgentTaskEnv(HighDamageConfig())
    env.reset(seed=0)
    actions = {a: env.action_space[a].sample() for a in env._agents}
    _, _, terminated, _, _ = env.step(actions)
    assert all(terminated.values()), "Both agents should be terminated by KO"
    env.close()


def test_single_agent_wrapper_reset_and_step():
    """SingleAgentWrapper exposes the correct obs/action space."""
    base = ModularMultiAgentTaskEnv(DummyMultiAgentConfig())
    env = ModularMultiAgentSingleAgentWrapper(base, agent_id="agent_0")

    obs, info = env.reset(seed=0)
    assert obs.shape == (2,), f"Expected (2,), got {obs.shape}"

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (2,)
    assert isinstance(reward, float)
    env.close()


def test_register_task_routes_multi_agent():
    """register_task detects MultiAgentTaskConfig and registers correctly."""
    cfg = DummyMultiAgentConfig()
    env_id = register_task(cfg, env_id="DummyMultiAgent-test-v0")
    assert env_id == "DummyMultiAgent-test-v0"

    import gymnasium as gym

    env = gym.make("DummyMultiAgent-test-v0")
    obs, info = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_1"}
    env.close()
