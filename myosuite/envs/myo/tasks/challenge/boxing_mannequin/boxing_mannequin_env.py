# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Single-agent wrapper factory for boxer-vs-mannequin task."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from myosuite.envs.multi_agent_modular_env import (
    ModularMultiAgentSingleAgentWrapper,
    ModularMultiAgentTaskEnv,
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
    BoxingMannequinConfig,
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_task_config import (
    BoxingMannequinTaskConfig,
)


@dataclass
class _AttackState:
    steps_left: int = 0
    use_right: bool = True
    target_head: bool = True


class ScriptedMannequinPolicy:
    """Random scripted mannequin attacker used for single-agent training."""

    def __init__(
        self,
        env: ModularMultiAgentTaskEnv,
        config: BoxingMannequinTaskConfig,
        rng: np.random.Generator,
    ) -> None:
        self._env = env
        self._config = config
        self._rng = rng
        self._state = _AttackState()
        self._cooldown_steps = 1

    def _sample_steps(self, low_s: float, high_s: float) -> int:
        return max(
            1, int(round(self._rng.uniform(low_s, high_s) / self._config.cfg.ctrl_dt))
        )

    def _yaw_to_face_target(self) -> float:
        data = self._env.unwrapped.data
        meta = self._env.unwrapped._meta
        mannequin_pelvis = data.site_xpos[meta.site_ids["agent_1"]["pelvis_site"]]
        boxer_pelvis = data.site_xpos[meta.site_ids["agent_0"]["pelvis_site"]]
        delta = boxer_pelvis[:2] - mannequin_pelvis[:2]
        return float(np.arctan2(delta[1], delta[0]) - np.pi / 2.0)

    def _target_pos(self) -> np.ndarray:
        data = self._env.unwrapped.data
        meta = self._env.unwrapped._meta
        zone = "head_zone" if self._state.target_head else "body_zone"
        target = data.site_xpos[meta.site_ids["agent_0"][zone]].copy()
        target[0] += self._rng.uniform(
            -self._config.cfg.attack_target_noise_xy_m,
            self._config.cfg.attack_target_noise_xy_m,
        )
        target[1] += self._rng.uniform(
            -self._config.cfg.attack_target_noise_xy_m,
            self._config.cfg.attack_target_noise_xy_m,
        )
        target[2] += self._rng.uniform(
            -self._config.cfg.attack_target_noise_z_m,
            self._config.cfg.attack_target_noise_z_m,
        )
        return target

    def __call__(self, _obs: np.ndarray) -> np.ndarray:
        if self._state.steps_left <= 0:
            if self._cooldown_steps > 0:
                self._cooldown_steps -= 1
            else:
                self._state.use_right = bool(self._rng.integers(0, 2))
                self._state.target_head = bool(self._rng.integers(0, 2))
                self._state.steps_left = self._sample_steps(
                    self._config.cfg.attack_duration_sec_min,
                    self._config.cfg.attack_duration_sec_max,
                )
                self._cooldown_steps = self._sample_steps(
                    self._config.cfg.attack_interval_sec_min,
                    self._config.cfg.attack_interval_sec_max,
                )

        action = np.zeros(self._env.action_space["agent_1"].shape[0], dtype=np.float32)
        yaw = self._yaw_to_face_target()
        action[2] = np.clip(yaw, 0.0, 1.0)

        if self._state.steps_left > 0:
            target = self._target_pos()
            data = self._env.unwrapped.data
            meta = self._env.unwrapped._meta
            pelvis = data.site_xpos[meta.site_ids["agent_1"]["pelvis_site"]]
            local = target - pelvis
            if self._state.use_right:
                action[3:6] = np.clip(
                    [
                        (local[0] + 0.65) / 0.65,
                        local[1] / 0.55,
                        local[2] / 0.45,
                    ],
                    0.0,
                    1.0,
                )
            else:
                action[6:9] = np.clip(
                    [
                        local[0] / 0.65,
                        local[1] / 0.55,
                        local[2] / 0.45,
                    ],
                    0.0,
                    1.0,
                )
            self._state.steps_left -= 1
        else:
            action[3:6] = np.array(
                [
                    1.0,
                    self._config.cfg.mannequin_guard_y / 0.55,
                    self._config.cfg.mannequin_guard_z / 0.45,
                ],
                dtype=np.float32,
            )
            action[6:9] = np.array(
                [
                    0.0,
                    self._config.cfg.mannequin_guard_y / 0.55,
                    self._config.cfg.mannequin_guard_z / 0.45,
                ],
                dtype=np.float32,
            )
        return action


def make_boxing_mannequin_env(
    render_mode: str | None = None,
    task_config: BoxingMannequinTaskConfig | None = None,
) -> gym.Env:
    """Create single-agent boxer-vs-mannequin training environment."""
    cfg = task_config or BoxingMannequinTaskConfig()
    base = ModularMultiAgentTaskEnv(cfg, render_mode=render_mode)
    rng = np.random.default_rng()
    policy = ScriptedMannequinPolicy(base, cfg, rng)
    return ModularMultiAgentSingleAgentWrapper(
        base,
        agent_id="agent_0",
        opponent_policy=policy,
    )


def make_boxing_6targets(render_mode: str | None = None) -> gym.Env:
    """Create six-target pad env (registered as ``myoChallengeBoxingP0-v0``)."""
    cfg = BoxingMannequinTaskConfig(
        cfg=BoxingMannequinConfig(include_boxing_targets_scene=True)
    )
    base = ModularMultiAgentTaskEnv(cfg, render_mode=render_mode)

    def static_opponent_policy(_obs: np.ndarray) -> np.ndarray:
        opp_space = base.action_space["agent_1"]
        return np.zeros(opp_space.shape[0], dtype=np.float32)

    return ModularMultiAgentSingleAgentWrapper(
        base,
        agent_id="agent_0",
        opponent_policy=static_opponent_policy,
    )
