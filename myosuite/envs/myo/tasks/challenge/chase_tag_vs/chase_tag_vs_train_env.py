# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Single-learner training view of ``myoChallengeChaseTagFBVs-v0``.

``myoChallengeChaseTagFBP2-v0`` (537-d, mocap opponent) is not the submission
contract. Submissions play ``myoChallengeChaseTagFBVs-v0``, whose per-agent
observation is 546-d (raw qpos/qvel/act, world-frame pelvis, live opponent
pelvis, role). This env is that contract as a normal Gymnasium single-agent
task: the learner is ``agent_0``, both bodies are in the scene, and
``agent_1`` is driven inside the env.

Observation is assembled with the same :func:`get_agent_obs` the 1v1 env
uses, so a policy trained here exports 546 → 354 with no wrapper.
``myoChallengeChaseTagFBVs-v0`` itself is unchanged.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.multi_agent_modular_env import ModularMultiAgentTaskEnv
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
    ChaseTagVsFullbodySteeredTaskConfig,
)
from myosuite.terms.multiplayer.chase_tag_vs_obs import (
    CHASER_ID,
    FBVS_AGENT_OBS_DIM,
    RUNNER_ID,
    get_agent_obs,
)

LEARNER_ID = CHASER_ID
OPPONENT_ID = RUNNER_ID
_ARENA_HALF = 5.5
_OPP_SPEED = 0.8


class ChaseTagFbvsTrainEnv(gym.Env):
    """Gymnasium env with FBVs ``agent_0`` observation and a scripted opponent.

    Args:
        render_mode: ``"human"`` or ``"rgb_array"``, forwarded to the 1v1 env.
        task_choice: ``"CHASE"``, ``"EVADE"``, or ``"random"`` (FBVs default).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        task_choice: str = "random",
    ) -> None:
        super().__init__()
        self._inner = ModularMultiAgentTaskEnv(
            ChaseTagVsFullbodySteeredTaskConfig(task_choice=task_choice),
            render_mode=render_mode,
        )
        self.render_mode = render_mode
        self._rng = np.random.default_rng()
        self._opp_xy = np.zeros(2, dtype=np.float64)
        self._opp_vel = np.zeros(2, dtype=np.float64)
        self._opp_policy = "hold"

        obs0 = self._learner_obs()
        if obs0.shape != (FBVS_AGENT_OBS_DIM,):
            raise RuntimeError(
                f"FBVs agent_0 obs is {obs0.shape}, expected ({FBVS_AGENT_OBS_DIM},)"
            )
        act_n = len(self._inner._config.act_indices(self._inner._meta, LEARNER_ID))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(FBVS_AGENT_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(act_n, dtype=np.float32),
            high=np.ones(act_n, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def unwrapped_vs(self) -> ModularMultiAgentTaskEnv:
        """The underlying two-agent FBVs env (same model, meta, and terms)."""
        return self._inner

    @property
    def model(self) -> mujoco.MjModel:
        return self._inner.model

    @property
    def data(self) -> mujoco.MjData:
        return self._inner.data

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _obs_dict, info = self._inner.reset(seed=seed, options=options)
        self._reset_opponent()
        self._write_opponent_state()
        mujoco.mj_forward(self._inner.model, self._inner.data)
        return self._learner_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._advance_opponent()
        self._write_opponent_state()
        zero_opp = np.zeros(
            self._inner.action_space[OPPONENT_ID].shape, dtype=np.float32
        )
        _obs_dict, rewards, _terminated, truncated, info = self._inner.step(
            {LEARNER_ID: np.asarray(action, dtype=np.float32), OPPONENT_ID: zero_opp}
        )
        self._write_opponent_state()
        mujoco.mj_forward(self._inner.model, self._inner.data)
        # Scripted opponent must not end the episode by falling.
        learner_fell = bool(info.get("fell", {}).get(LEARNER_ID, False))
        tagged = bool(info.get("tagged", False))
        done = learner_fell or tagged
        trunc = bool(truncated[LEARNER_ID]) and not done
        return self._learner_obs(), float(rewards[LEARNER_ID]), done, trunc, info

    def render(self) -> np.ndarray | None:
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()

    def _learner_obs(self) -> np.ndarray:
        inner = self._inner
        return get_agent_obs(
            inner.model,
            inner.data,
            inner._meta,
            LEARNER_ID,
            inner._config,
            task=inner._config.current_task,
        )

    def _root_xy(self, agent_id: str) -> np.ndarray:
        jid = self._inner._meta.jnt_ids[agent_id][0]
        qadr = int(self._inner.model.jnt_qposadr[jid])
        return self._inner.data.qpos[qadr : qadr + 2].copy()

    def _reset_opponent(self) -> None:
        learner_xy = self._root_xy(LEARNER_ID)
        standing = self._inner._meta.standing_qpos
        jid = self._inner._meta.jnt_ids[OPPONENT_ID][0]
        qadr = int(self._inner.model.jnt_qposadr[jid])
        if standing is not None:
            self._opp_xy = standing[qadr : qadr + 2].copy()
        else:
            self._opp_xy = self._root_xy(OPPONENT_ID)
        # Keep the 1v1 spawn if it is already far enough; otherwise nudge.
        if np.linalg.norm(self._opp_xy - learner_xy) < 2.0:
            angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
            self._opp_xy = learner_xy + 2.5 * np.array([np.cos(angle), np.sin(angle)])
        self._opp_xy = np.clip(self._opp_xy, -_ARENA_HALF, _ARENA_HALF)
        task = self._inner._config.current_task
        # Learner CHASE → opponent wanders or stands; learner EVADE → opponent chases.
        if task == "EVADE":
            self._opp_policy = "chase"
        else:
            self._opp_policy = "wander" if self._rng.random() < 0.5 else "hold"
        self._opp_vel[:] = 0.0

    def _advance_opponent(self) -> None:
        dt = float(self._inner._config.ctrl_dt)
        learner_xy = self._root_xy(LEARNER_ID)
        if self._opp_policy == "hold":
            self._opp_vel[:] = 0.0
        elif self._opp_policy == "wander":
            self._opp_vel += self._rng.normal(0.0, 0.4, size=2)
            speed = float(np.linalg.norm(self._opp_vel))
            if speed > _OPP_SPEED:
                self._opp_vel *= _OPP_SPEED / speed
        else:
            delta = learner_xy - self._opp_xy
            n = float(np.linalg.norm(delta))
            self._opp_vel[:] = (delta / n) * _OPP_SPEED if n > 1e-6 else 0.0
        self._opp_xy = np.clip(
            self._opp_xy + dt * self._opp_vel, -_ARENA_HALF, _ARENA_HALF
        )

    def _write_opponent_state(self) -> None:
        """Pose ``agent_1`` as a standing body at the scripted XY (full-body height)."""
        inner = self._inner
        standing = inner._meta.standing_qpos
        if standing is None:
            return
        model, data, meta = inner.model, inner.data, inner._meta
        for jid in meta.jnt_ids[OPPONENT_ID]:
            qadr = int(model.jnt_qposadr[jid])
            vadr = int(model.jnt_dofadr[jid])
            jtype = int(model.jnt_type[jid])
            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                data.qpos[qadr : qadr + 7] = standing[qadr : qadr + 7]
                data.qpos[qadr : qadr + 2] = self._opp_xy
                data.qvel[vadr : vadr + 6] = 0.0
                data.qvel[vadr : vadr + 2] = self._opp_vel
            else:
                nq = 1
                nv = 1
                data.qpos[qadr : qadr + nq] = standing[qadr : qadr + nq]
                data.qvel[vadr : vadr + nv] = 0.0
        act_idx = meta.act_indices[OPPONENT_ID]
        if data.act is not None and len(data.act) > max(act_idx):
            data.act[act_idx] = 0.05
