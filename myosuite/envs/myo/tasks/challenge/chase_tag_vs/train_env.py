# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""One player of ``myoChallengeChaseTagFBVs-v0``, against an opponent the env drives.

``ChaseTagVsTrainEnv`` is ``myoChallengeChaseTagFBVs-v0`` seen from ``agent_0``:
the same two-body model, reset, role draw, rules, reward and score, and the same
546-d observation an FBVs player (and a MyoChallenge submission) reads. The env
plays ``agent_1`` itself with an opponent policy that reads ``agent_1``'s own
546-d observation of the previous step, so a policy trained here is submitted
as-is (546 -> 354).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
    ChaseTagVsFullbodySteeredTaskConfig,
)
from myosuite.terms.base_obs import qpos_for_joints, qvel_for_joints
from myosuite.terms.multiplayer.chase_tag_vs_obs import (
    get_agent_obs,
    opponent_relative_obs,
    own_pelvis_obs,
    role_obs,
)

LEARNER, OPPONENT = "agent_0", "agent_1"

#: Observation blocks, in FBVs ``get_agent_obs`` order (the mjlab term names).
CHASETAG_VS_OBS_KEYS = ("qpos", "qvel", "act", "pelvis", "opponent_relative", "role")
CHASETAG_VS_OBS_DIM = 546

OpponentPolicy = Callable[[np.ndarray], np.ndarray]


def load_opponent_policy(
    source: OpponentPolicy | str | Path | None, action_dim: int
) -> OpponentPolicy | None:
    """An opponent policy ``obs (546,) -> action (354,)``, or ``None`` (idle).

    Args:
        source: ``None`` (idle opponent: zero excitation), a callable, an ONNX
            file (a MyoChallenge submission) or an RSL-RL checkpoint
            (``model_*.pt`` of ``scripts/train_mjlab.py``, deterministic).
        action_dim: Muscles of one player.

    Returns:
        The policy, or ``None``.

    Raises:
        ValueError: If *source* is a file of another type.
    """
    if source is None or callable(source):
        return source
    path = Path(source)
    if path.suffix == ".onnx":
        import onnxruntime as ort  # noqa: PLC0415

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name

        def onnx_policy(obs: np.ndarray) -> np.ndarray:
            x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            return np.asarray(session.run(None, {name: x})[0]).reshape(-1)

        return onnx_policy
    if path.suffix == ".pt":
        from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: PLC0415

        policy = load_rslrl_policy(path, action_dim)
        return lambda obs: policy.act(np.asarray(obs, dtype=np.float32))
    raise ValueError(f"Opponent policy {path}: expected an .onnx or .pt file.")


def train_task_config(
    task_config: ChaseTagVsFullbodySteeredTaskConfig | None = None,
    frame_skip: int | None = None,
) -> ChaseTagVsFullbodySteeredTaskConfig:
    """The FBVs task config of the train env (both backends).

    Args:
        task_config: FBVs task config (default: the registered FBVs one).
        frame_skip: Physics steps per control step (default: FBVs, 5 x 2 ms).

    Returns:
        The task config, with the control step of *frame_skip*.

    Raises:
        ValueError: If the config drops part of the FBVs observation.
    """
    cfg = task_config or ChaseTagVsFullbodySteeredTaskConfig()
    if frame_skip is not None and frame_skip != cfg.n_substeps:
        cfg = dataclasses.replace(cfg, ctrl_dt=frame_skip * cfg.sim_dt)
    flags = ("observe_own_joints", "observe_own_muscles")
    flags += ("observe_opponent_pelvis", "observe_role")
    if not all(getattr(cfg, f) for f in flags):
        raise ValueError("ChaseTagVsTrainEnv needs the full FBVs observation.")
    return cfg


class ChaseTagVsTrainEnv(MyoGymnasiumEnv):
    """``myoChallengeChaseTagFBVs-v0`` from ``agent_0``, with an in-env opponent.

    Observation: ``agent_0``'s FBVs observation (546). Action: its 354 muscle
    excitations, clipped to ``[0, 1]``. Reward, termination (either player
    falls, or the tag) and ``info["score"]`` are FBVs's for ``agent_0``.

    Args:
        opponent_policy: Policy of ``agent_1`` (see :func:`load_opponent_policy`);
            ``None`` leaves it idle (zero excitation). Swap it with
            :meth:`set_opponent_policy`.
        frame_skip: Physics steps per control step (FBVs: 5 x 2 ms).
        task_config: FBVs task config (default: the registered FBVs one).
        render_mode: ``"human"`` or ``"rgb_array"``.
        **kwargs: Passed to :class:`MyoGymnasiumEnv`.
    """

    def __init__(
        self,
        opponent_policy: OpponentPolicy | str | Path | None = None,
        frame_skip: int | None = None,
        task_config: ChaseTagVsFullbodySteeredTaskConfig | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = train_task_config(task_config, frame_skip)
        super().__init__(frame_skip=cfg.n_substeps, render_mode=render_mode, **kwargs)
        self.task_config = cfg
        self.model, self.data, self.meta = cfg.build_model()
        self._ctrl_dt = cfg.ctrl_dt
        self.obs_keys = list(CHASETAG_VS_OBS_KEYS)
        self._act = {
            a: np.asarray(self.meta.act_indices[a]) for a in (LEARNER, OPPONENT)
        }
        n_act = len(self._act[LEARNER])
        self.action_space = gym.spaces.Box(0.0, 1.0, (n_act,), dtype=np.float32)
        self.observation_space = self._unbounded_obs_space(CHASETAG_VS_OBS_DIM)
        self._opponent = load_opponent_policy(opponent_policy, n_act)
        self._opponent_obs = np.zeros(CHASETAG_VS_OBS_DIM, dtype=np.float32)
        self._action = np.zeros(n_act, dtype=np.float32)
        self._health = {a: 0.0 for a in (LEARNER, OPPONENT)}
        self.steps = 0

    @property
    def dt(self) -> float:
        """Control step in seconds."""
        return self._ctrl_dt

    # ── Opponent ─────────────────────────────────────────────────────────────

    def set_opponent_policy(self, policy: OpponentPolicy | str | Path | None) -> None:
        """Replace the opponent policy (see :func:`load_opponent_policy`)."""
        self._opponent = load_opponent_policy(policy, len(self._act[OPPONENT]))

    @property
    def opponent_obs(self) -> np.ndarray:
        """``agent_1``'s 546-d observation, which its next action is computed from."""
        return self._opponent_obs

    def _opponent_action(self) -> np.ndarray:
        if self._opponent is None:
            return np.zeros(len(self._act[OPPONENT]), dtype=np.float32)
        return np.asarray(self._opponent(self._opponent_obs), dtype=np.float32)

    # ── Episode ──────────────────────────────────────────────────────────────

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """FBVs reset: both players standing 2 m apart, then the role draw."""
        self.data.qpos[:] = self.meta.standing_qpos
        self.data.ctrl[:] = 0.0
        self.task_config.on_reset(self.model, self.data, self.meta, np_random)
        self._action = np.zeros_like(self._action)
        self._health = {a: 0.0 for a in (LEARNER, OPPONENT)}
        self.steps = 0
        return {"task": self.task_config.current_task}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, _ = super().reset(seed=seed, options=options)
        self._opponent_obs = self._agent_obs(OPPONENT)
        return obs, {"task": self.task_config.current_task}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Both players act (the opponent on its last observation), as an FBVs step."""
        del kwargs
        # A copy: the caller may reuse its array (reset() must not touch it).
        self._action = np.array(action, dtype=np.float32)
        opponent = self._opponent_action()
        for agent, act in ((LEARNER, self._action), (OPPONENT, opponent)):
            self.data.ctrl[self._act[agent]] = np.clip(act, 0.0, 1.0).astype(np.float64)
        self._step_physics()
        self.steps += 1

        cfg = self.task_config
        damage = cfg.compute_damage(self.model, self.data, self.meta)
        for agent, opp in ((LEARNER, OPPONENT), (OPPONENT, LEARNER)):
            self._health[agent] = min(
                self._health[agent] + damage[opp] * cfg.health_update_scale,
                cfg.ko_threshold,
            )

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)
        self._opponent_obs = self._agent_obs(OPPONENT)
        obs, reward, terminated, truncated, info = self._finalize_step(
            obs_dict, rwd_dict
        )
        info["task"] = cfg.current_task
        return obs, reward, terminated, truncated, info

    # ── Observation and reward ───────────────────────────────────────────────

    def _agent_obs(self, agent: str) -> np.ndarray:
        """FBVs observation of *agent* in the current state."""
        cfg = self.task_config
        return get_agent_obs(
            self.model, self.data, self.meta, agent, cfg, task=cfg.current_task
        )

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """``agent_0``'s FBVs observation, one entry per block."""
        model, data, meta, cfg = (
            accessor.model,
            accessor.data,
            self.meta,
            self.task_config,
        )
        joints = meta.jnt_ids[LEARNER]
        return {
            "qpos": qpos_for_joints(model, data, joints).astype(np.float32),
            "qvel": qvel_for_joints(model, data, joints).astype(np.float32),
            "act": data.act[self._act[LEARNER]].astype(np.float32),
            "pelvis": own_pelvis_obs(data, meta, LEARNER),
            "opponent_relative": opponent_relative_obs(data, meta, LEARNER, cfg),
            "role": role_obs(LEARNER, cfg.current_task, cfg),
        }

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return self._obs_keys_to_vec(obs_dict)

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """FBVs reward, termination and score of ``agent_0`` in the current state."""
        del obs_dict
        cfg, players = self.task_config, (LEARNER, OPPONENT)
        damage = cfg.compute_damage(self.model, self.data, self.meta)
        fell = {a: cfg.check_fell(self.data, self.meta, a) for a in players}
        ko = {a: self._health[a] >= cfg.ko_threshold for a in players}
        reward = cfg.compute_reward(
            self.data, self.meta, LEARNER, damage, self._health, self._action, fell, ko
        )
        info = cfg.get_info(
            self.data, self.meta, self._health, damage, fell, ko, self.steps
        )
        chaser = cfg.current_task == "CHASE"
        tagged = bool(info["tagged"])
        survived = not tagged and info["elapsed_s"] >= cfg.maxTime
        return {
            "dense": reward,
            "done": any(fell.values()) or any(ko.values()),
            "solved": (tagged and chaser) or (survived and not chaser),
            "score": info["score"],
            "tagged": tagged,
            "chaser": chaser,
            "learner_fell": fell[LEARNER],
            "opponent_fell": fell[OPPONENT],
            "elapsed_s": info["elapsed_s"],
        }

    def player_slices(self, agent: str) -> tuple[slice, slice]:
        """``(qpos, qvel)`` address ranges of *agent* (``agent_0`` / ``agent_1``)."""
        model, joints = self.model, self.meta.jnt_ids[agent]
        q = [int(model.jnt_qposadr[j]) for j in joints]
        v = [int(model.jnt_dofadr[j]) for j in joints]
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        assert int(model.jnt_type[joints[0]]) == free, "the root comes first"
        return slice(min(q), max(q) + 1), slice(min(v), max(v) + 1)
