# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``ChaseTagVsTaskConfig`` — ``MultiAgentTaskConfig`` for the 1v1 chase-tag task.

Rules match MyoChallenge ``chasetag_v0``:
- Binary tag at distance ≤ tag_radius_m ends the episode.
- Evade win: runner survives maxTime seconds.
- Sparse score: ``1 - t/maxTime`` (chase win) or ``t/maxTime`` (evade win).
- Roles (CHASE/EVADE) randomised each episode when ``task_choice="random"``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.multi_agent_config import MultiAgentTaskConfig
from myosuite.terms.multiplayer.chase_tag_vs_reward import CHASER_ID, RUNNER_ID


@dataclass
class ChaseTagVsTaskConfig(MultiAgentTaskConfig):
    """Data-driven configuration for the two-agent 1v1 chase-tag task.

    ``agent_0`` takes the role determined by ``current_task`` each episode:
    - CHASE → ``agent_0`` is chaser, ``agent_1`` is runner.
    - EVADE → ``agent_0`` is runner, ``agent_1`` is chaser.

    When ``task_choice="random"`` (default) the role is sampled at each reset,
    matching the MyoChallenge evaluation protocol.
    """

    # --- Episode / timing (MyoChallenge: 20 s) ---
    maxTime: float = 20.0
    ctrl_dt: float = 0.01
    sim_dt: float = 0.002

    # --- Role ---
    task_choice: str = "random"  # "CHASE" | "EVADE" | "random"
    current_task: str = field(default="CHASE", init=False, repr=False, compare=False)

    # --- Agent placement ---
    agent_separation_m: float = 2.0

    # --- Tag detection (MyoChallenge: 0.5 m) ---
    tag_radius_m: float = 0.5
    ko_health_threshold: float = 1.0  # one tag = instant KO

    # --- Fall detection ---
    fall_pelvis_z_threshold: float = 0.6  # matches single-agent directional env

    # --- Reward weights ---
    r_win_bonus: float = 1.0
    r_loss_penalty: float = -1.0
    r_fall_penalty: float = -5.0
    r_survival_per_step: float = 0.01
    r_act_reg: float = -0.001
    r_chaser_proximity: float = 0.5
    r_runner_proximity: float = -0.5

    # --- Observation flags ---
    observe_own_joints: bool = True
    observe_own_muscles: bool = True
    observe_opponent_pelvis: bool = True
    observe_role: bool = True

    @property
    def max_episode_steps(self) -> int:
        return round(self.maxTime / self.ctrl_dt)

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — model
    # ------------------------------------------------------------------

    def _low_level_config(self):
        """Project this task config onto the model/reward-facing ``ChaseTagVsConfig``.

        ``ChaseTagVsConfig`` is a plain dataclass consumed by the (backend-
        agnostic) model builders and term functions in this package, which
        must not import the ``MultiAgentTaskConfig`` machinery. Its field
        names are a strict subset of this class's own fields (verified by
        ``test_chase_tag_registry.py``'s config-sync test), so the
        projection is done generically here instead of by hand-listing every
        field twice -- avoids the two dataclasses silently drifting apart
        when a field is added to one but not the other.
        """
        import dataclasses

        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
            ChaseTagVsConfig,
        )

        field_names = {f.name for f in dataclasses.fields(ChaseTagVsConfig)}
        return ChaseTagVsConfig(**{name: getattr(self, name) for name in field_names})

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
            build_chase_tag_vs_model,
        )

        return build_chase_tag_vs_model(self._low_level_config())

    def build_spec(self) -> mujoco.MjSpec:
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
            _build_chase_tag_vs_spec,
        )

        return _build_chase_tag_vs_spec(self._low_level_config())

    @property
    def n_substeps(self) -> int:
        return max(1, round(self.ctrl_dt / self.sim_dt))

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — spaces
    # ------------------------------------------------------------------

    def obs_dim(self, model, data, meta, agent_id: str) -> int:
        from myosuite.terms.multiplayer.chase_tag_vs_obs import obs_dim as _obs_dim

        return _obs_dim(model, data, meta, agent_id, self)

    def act_indices(self, meta: Any, agent_id: str) -> list[int]:
        return meta.act_indices[agent_id]

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — episode logic
    # ------------------------------------------------------------------

    def on_reset(self, model=None, data=None, meta=None) -> None:
        """Sample task role for this episode."""
        if self.task_choice == "random":
            self.current_task = random.choice(["CHASE", "EVADE"])
        else:
            self.current_task = self.task_choice

    def get_obs(self, model, data, meta, agent_id: str, health=None) -> np.ndarray:
        from myosuite.terms.multiplayer.chase_tag_vs_obs import get_agent_obs

        return get_agent_obs(model, data, meta, agent_id, self, task=self.current_task)

    def check_fell(self, data, meta, agent_id: str) -> bool:
        from myosuite.terms.multiplayer.chase_tag_vs_reward import is_fallen

        return is_fallen(data, meta, agent_id, self)

    def _actual_chaser_id(self) -> str:
        """Return the agent ID that is chasing this episode."""
        return CHASER_ID if self.current_task == "CHASE" else RUNNER_ID

    def _actual_runner_id(self) -> str:
        """Return the agent ID that is running this episode."""
        return RUNNER_ID if self.current_task == "CHASE" else CHASER_ID

    def compute_damage(self, model, data, meta) -> dict[str, float]:
        """Binary tag: the actual chaser delivers ko_health_threshold to the runner."""
        from myosuite.terms.multiplayer.chase_tag_vs_reward import is_tagged

        tagged = is_tagged(data, meta, self)
        chaser = self._actual_chaser_id()
        runner = self._actual_runner_id()
        # damage_delivered[x] = damage dealt BY agent x to its opponent
        return {
            chaser: self.ko_health_threshold if tagged else 0.0,
            runner: 0.0,
        }

    @property
    def ko_threshold(self) -> float:
        return self.ko_health_threshold

    @property
    def health_update_scale(self) -> float:
        return 1.0  # damage is already scaled; no ctrl_dt multiplier needed

    def compute_reward(
        self,
        data,
        meta,
        agent_id: str,
        damage_delivered: dict[str, float],
        health: dict[str, float],
        actions: np.ndarray,
        fell: dict[str, bool],
        ko: dict[str, bool],
    ) -> float:
        from myosuite.terms.multiplayer.chase_tag_vs_reward import compute_agent_reward

        opp = self.opponent_id(agent_id)
        tagged = ko[self._actual_runner_id()]  # runner KO ↔ tag event
        return compute_agent_reward(
            data=data,
            meta=meta,
            agent_id=agent_id,
            task=self.current_task,
            tagged=tagged,
            self_fell=fell[agent_id],
            opponent_fell=fell[opp],
            actions=actions,
            config=self,
        )

    def get_info(
        self,
        data,
        meta,
        health: dict[str, float],
        damage_delivered: dict[str, float],
        fell: dict[str, bool],
        ko: dict[str, bool],
        step_count: int,
    ) -> dict[str, Any]:
        elapsed = step_count * self.ctrl_dt
        tagged = ko[self._actual_runner_id()]
        runner_survived = (not tagged) and elapsed >= self.maxTime
        won = tagged or runner_survived
        from myosuite.terms.multiplayer.chase_tag_vs_reward import get_score

        score = get_score(elapsed, self.current_task, self) if won else 0.0
        return {
            "task": self.current_task,
            "tagged": tagged,
            "elapsed_s": elapsed,
            "score": score,
            "fell": dict(fell),
        }


@dataclass
class ChaseTagVsFullbodyTaskConfig(ChaseTagVsTaskConfig):
    """Chase-tag on the MuscleMimic full-body model instead of myoLeg-only.

    Identical hyperparameters and reward/obs term functions to
    :class:`ChaseTagVsTaskConfig` — only the model builder differs.
    """

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
            build_chase_tag_vs_fullbody_model,
        )

        return build_chase_tag_vs_fullbody_model(self._low_level_config())

    def build_spec(self) -> mujoco.MjSpec:
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
            _build_chase_tag_vs_fullbody_spec,
        )

        return _build_chase_tag_vs_fullbody_spec(self._low_level_config())


@dataclass
class ChaseTagVsFullbodySteeredTaskConfig(ChaseTagVsFullbodyTaskConfig):
    """Steerable variant: gait style imitation + live opponent steering.

    Drops MuscleMimic's absolute keypoint/root-position tracking and replaces
    it with local joint-angle gait style (how the limbs move) plus a steering
    heading term that dynamically points each agent toward or away from the
    live opponent position each step.
    """

    gait_clip_repo: str = "amathislab/musclemimic-retargeted"
    gait_clip_filename: str = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"
    gait_target_speed: float = 1.2
    w_gait_jpos: float = 0.15
    w_gait_jvel: float = 0.05
    w_steering: float = 0.6

    _gait_qpos_local: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _gait_qvel_local: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _gait_T: int = field(default=0, init=False, repr=False, compare=False)
    _model_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def on_reset(self, model=None, data=None, meta=None) -> None:
        super().on_reset(model=model, data=data, meta=meta)
        if model is not None:
            self._model_cache = model

    def _ensure_gait_clip(self) -> None:
        if self._gait_qpos_local is not None:
            return
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=self.gait_clip_repo,
                filename=self.gait_clip_filename,
                repo_type="dataset",
            )
        )
        npz = np.load(path, allow_pickle=True)
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        qvel = np.asarray(npz["qvel"], dtype=np.float64)
        # Drop root free joint — only local joint-angle style is imitated
        self._gait_qpos_local = qpos[:, 7:]
        self._gait_qvel_local = qvel[:, 6:]
        self._gait_T = int(qpos.shape[0])

    def compute_reward(
        self,
        data,
        meta,
        agent_id: str,
        damage_delivered: dict[str, float],
        health: dict[str, float],
        actions: np.ndarray,
        fell: dict[str, bool],
        ko: dict[str, bool],
    ) -> float:
        base = super().compute_reward(
            data, meta, agent_id, damage_delivered, health, actions, fell, ko
        )
        self._ensure_gait_clip()
        from myosuite.terms.multiplayer.chase_tag_vs_reward import (
            CHASER_ID,
            compute_gait_steering_reward,
        )

        frame = int(round(data.time / self.ctrl_dt)) % self._gait_T
        opp = self.opponent_id(agent_id)
        # Determine whether this agent is the chaser given the current task
        is_chaser = (agent_id == CHASER_ID and self.current_task == "CHASE") or (
            agent_id != CHASER_ID and self.current_task == "EVADE"
        )
        steering_dense, _ = compute_gait_steering_reward(
            model=self._model_cache,
            data=data,
            meta=meta,
            agent_id=agent_id,
            opponent_id=opp,
            gait_qpos_local=self._gait_qpos_local,
            gait_qvel_local=self._gait_qvel_local,
            frame=frame,
            target_speed=self.gait_target_speed,
            pursue=is_chaser,
            w_jpos=self.w_gait_jpos,
            w_jvel=self.w_gait_jvel,
            w_heading=self.w_steering,
        )
        return base + steering_dense
