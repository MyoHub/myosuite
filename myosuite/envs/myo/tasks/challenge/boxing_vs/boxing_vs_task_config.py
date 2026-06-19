# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``BoxingVsTaskConfig`` — ``MultiAgentTaskConfig`` for the boxing VS task.

Wraps all boxing-specific hyperparameters and wires the existing term
functions (``boxing_vs_obs_terms``, ``boxing_vs_reward_terms``) into the
``MultiAgentTaskConfig`` hook interface expected by
``ModularMultiAgentTaskEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.multi_agent_config import MultiAgentTaskConfig
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    _joint_qpos_size,
    _joint_qvel_size,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.core.trajectory_io import load_motion_clip


@dataclass
class BoxingVsTaskConfig(MultiAgentTaskConfig):
    """Data-driven configuration for the two-agent competitive boxing task.

    All physics, hit detection, fall detection, and reward hyperparameters
    live here.  Instantiate with overrides to create custom variants::

        cfg = BoxingVsTaskConfig(
            ko_health_threshold=50.0,
            agent_separation_m=1.2,
        )
        env = ModularMultiAgentTaskEnv(cfg)

    Physics timing
    --------------
    ``n_substeps = ctrl_dt / sim_dt`` (must be an integer).

    Agent placement
    ---------------
    Agent 0 at +Y, default full-body orientation (faces −Y).
    Agent 1 at −Y, rotated 180° around Z to face its opponent.

    Health / damage
    ---------------
    Damage accumulates via ``health[agent] += damage_from_opponent * ctrl_dt``.
    KO at ``ko_health_threshold``; fall when pelvis Z < ``fall_pelvis_z_threshold``.
    """

    # --- Episode ---
    max_episode_steps: int = 500  # 500 × 0.01 s = 5 s per round
    ctrl_dt: float = 0.01
    sim_dt: float = 0.002

    # --- Agent placement ---
    agent_separation_m: float = 1.6

    # --- Model ---
    disable_fingers: bool = True

    # --- Hit detection ---
    hit_threshold_head_m: float = 0.15
    hit_threshold_body_m: float = 0.18
    w_head: float = 2.0
    w_body: float = 1.0

    # --- Health / KO ---
    ko_health_threshold: float = 100.0

    # --- Fall detection ---
    fall_pelvis_z_threshold: float = 0.7

    # --- Reward weights ---
    r_damage_delivered: float = 0.1
    r_damage_received: float = -0.1
    r_ko_bonus: float = 10.0
    r_fall_bonus: float = 5.0
    r_self_fall_penalty: float = -5.0
    r_survival_per_step: float = 0.01
    r_act_reg: float = -0.001

    # --- Observation flags ---
    observe_own_joints: bool = True
    observe_own_muscles: bool = True
    observe_opponent_keypoints: bool = True
    observe_opponent_velocities: bool = True
    observe_opponent_joint_subset: bool = True
    observe_health: bool = True
    append_fullbody_obs: bool = False
    fullbody_motion_path: str = (
        "Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz"
    )
    fullbody_goal_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_step_lookahead": 5,
            "n_step_stride": 20,
            "enable_motion_phase": True,
            "use_concise_lookahead": True,
            "sites_for_mimic": list(FULLBODY_BODY2SITES_FOR_MIMIC.values()),
        }
    )
    _fullbody_model: mujoco.MjModel | None = field(init=False, default=None, repr=False)
    _fullbody_data: mujoco.MjData | None = field(init=False, default=None, repr=False)
    _fullbody_clip: Any | None = field(init=False, default=None, repr=False)
    _fullbody_adapter: FullbodyObsAdapter | None = field(
        init=False, default=None, repr=False
    )
    _agent_qpos_idx: dict[str, np.ndarray] = field(
        init=False, default_factory=dict, repr=False
    )
    _agent_qvel_idx: dict[str, np.ndarray] = field(
        init=False, default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — model
    # ------------------------------------------------------------------

    def _low_level_config(self):
        """Build the low-level :class:`BoxingVsConfig` from this task config."""
        from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import (
            BoxingVsConfig,
        )

        return BoxingVsConfig(
            max_episode_steps=self.max_episode_steps,
            ctrl_dt=self.ctrl_dt,
            sim_dt=self.sim_dt,
            agent_separation_m=self.agent_separation_m,
            disable_fingers=self.disable_fingers,
            hit_threshold_head_m=self.hit_threshold_head_m,
            hit_threshold_body_m=self.hit_threshold_body_m,
            w_head=self.w_head,
            w_body=self.w_body,
            ko_health_threshold=self.ko_health_threshold,
            fall_pelvis_z_threshold=self.fall_pelvis_z_threshold,
        )

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
        """Build the two-agent boxing MuJoCo model.

        Returns:
            Tuple of ``(model, data, BoxingVsModelMeta)``.
        """
        from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
            build_boxing_vs_model,
        )

        return build_boxing_vs_model(self._low_level_config())

    def build_spec(self) -> mujoco.MjSpec:
        """Build the combined boxing MjSpec (no compilation).

        Returns:
            :class:`mujoco.MjSpec` for XML export.
        """
        from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
            _build_boxing_vs_spec,
        )

        return _build_boxing_vs_spec(self._low_level_config())

    @property
    def n_substeps(self) -> int:
        """Number of MuJoCo substeps per control step."""
        return max(1, round(self.ctrl_dt / self.sim_dt))

    def _ensure_fullbody_bridge(self, model: mujoco.MjModel, meta: Any) -> None:
        if not self.append_fullbody_obs:
            return
        if self._fullbody_model is None:
            cfg = default_mimic_fullbody_config()
            cfg.disable_fingers = self.disable_fingers
            cfg.sim_dt = self.sim_dt
            fb_model, _fb_spec, _fb_xml = compile_mimic_fullbody_mjmodel(cfg)
            self._fullbody_model = fb_model
            self._fullbody_data = mujoco.MjData(fb_model)
            mujoco.mj_resetDataKeyframe(fb_model, self._fullbody_data, 0)
            mujoco.mj_forward(fb_model, self._fullbody_data)
            self._fullbody_clip = load_motion_clip(
                Path(self.fullbody_motion_path),
                expected_nq=fb_model.nq,
                expected_nv=fb_model.nv,
            )
            self._fullbody_adapter = FullbodyObsAdapter(
                fb_model,
                self._fullbody_clip,
                self.fullbody_goal_params,
            )
        for agent_id in self.agents:
            if agent_id in self._agent_qpos_idx and agent_id in self._agent_qvel_idx:
                continue
            qpos_idx: list[int] = []
            qvel_idx: list[int] = []
            for jid in meta.jnt_ids[agent_id]:
                jtype = int(model.jnt_type[jid])
                qadr = int(model.jnt_qposadr[jid])
                vadr = int(model.jnt_dofadr[jid])
                qsize = _joint_qpos_size(jtype)
                vsize = _joint_qvel_size(jtype)
                qpos_idx.extend(range(qadr, qadr + qsize))
                qvel_idx.extend(range(vadr, vadr + vsize))
            self._agent_qpos_idx[agent_id] = np.asarray(qpos_idx, dtype=np.int32)
            self._agent_qvel_idx[agent_id] = np.asarray(qvel_idx, dtype=np.int32)

    def _fullbody_obs_for_agent(
        self,
        data: mujoco.MjData,
        meta: Any,
        agent_id: str,
    ) -> np.ndarray:
        if (
            self._fullbody_model is None
            or self._fullbody_data is None
            or self._fullbody_adapter is None
            or self._fullbody_clip is None
        ):
            raise RuntimeError("Fullbody bridge not initialized.")
        if agent_id not in self._agent_qpos_idx or agent_id not in self._agent_qvel_idx:
            raise RuntimeError(f"Missing fullbody indices for {agent_id}.")

        self._fullbody_data.qpos[:] = data.qpos[self._agent_qpos_idx[agent_id]]
        self._fullbody_data.qvel[:] = data.qvel[self._agent_qvel_idx[agent_id]]
        if self._fullbody_data.act is not None and len(self._fullbody_data.act) > 0:
            self._fullbody_data.act[:] = data.act[meta.act_indices[agent_id]]
        self._fullbody_data.ctrl[:] = data.ctrl[meta.act_indices[agent_id]]
        mujoco.mj_forward(self._fullbody_model, self._fullbody_data)
        frame_idx = int(round(float(data.time) / float(self.ctrl_dt)))
        frame_idx = frame_idx % max(1, int(self._fullbody_clip.qpos.shape[0]))
        return self._fullbody_adapter.build(self._fullbody_data, frame_idx)

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — spaces
    # ------------------------------------------------------------------

    def obs_dim(self, model, data, meta, agent_id: str) -> int:
        """Return observation dimension for *agent_id* via a dry run."""
        from myosuite.terms.multiplayer.boxing_vs_obs import obs_dim as _obs_dim

        dim = _obs_dim(model, data, meta, agent_id, {a: 0.0 for a in self.agents}, self)
        if self.append_fullbody_obs:
            self._ensure_fullbody_bridge(model, meta)
            if self._fullbody_data is None or self._fullbody_adapter is None:
                raise RuntimeError("Fullbody adapter initialization failed.")
            dim += int(self._fullbody_adapter.build(self._fullbody_data, 0).shape[0])
        return dim

    def act_indices(self, meta: Any, agent_id: str) -> list[int]:
        """Return actuator indices for *agent_id*."""
        return meta.act_indices[agent_id]

    # ------------------------------------------------------------------
    # MultiAgentTaskConfig — episode logic
    # ------------------------------------------------------------------

    def get_obs(
        self, model, data, meta, agent_id: str, health: dict[str, float]
    ) -> np.ndarray:
        """Compute observation vector for *agent_id*."""
        from myosuite.terms.multiplayer.boxing_vs_obs import get_agent_obs

        obs = get_agent_obs(model, data, meta, agent_id, health, self)
        if self.append_fullbody_obs:
            self._ensure_fullbody_bridge(model, meta)
            full = self._fullbody_obs_for_agent(data, meta, agent_id)
            obs = np.concatenate([obs, full.astype(np.float32)], axis=0)
        return obs

    def compute_damage(self, model, data, meta) -> dict[str, float]:
        """Compute damage delivered by each agent this step."""
        from myosuite.terms.multiplayer.boxing_vs_reward import compute_total_damage

        return {a: compute_total_damage(data, meta, a, self) for a in self.agents}

    def check_fell(self, data, meta, agent_id: str) -> bool:
        """Return True if *agent_id*'s pelvis dropped below the fall threshold."""
        from myosuite.terms.multiplayer.boxing_vs_reward import is_fallen

        return is_fallen(data, meta, agent_id, self)

    @property
    def ko_threshold(self) -> float:
        """KO health threshold (alias for ``ko_health_threshold``)."""
        return self.ko_health_threshold

    @property
    def health_update_scale(self) -> float:
        """Scale factor applied to damage when updating health (= ctrl_dt)."""
        return self.ctrl_dt

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
        """Compute scalar reward for *agent_id*."""
        from myosuite.terms.multiplayer.boxing_vs_reward import compute_agent_reward

        opp = self.opponent_id(agent_id)
        return compute_agent_reward(
            data=data,
            meta=meta,
            agent_id=agent_id,
            delivered_damage=damage_delivered[agent_id],
            received_damage=damage_delivered[opp],
            self_fell=fell[agent_id],
            opponent_fell=fell[opp],
            ko_happened=ko[opp],
            own_health=health[agent_id],
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
        """Build the step info dict."""
        return self.default_step_info(health, damage_delivered, fell, ko, step_count)
