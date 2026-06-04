# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for boxer-vs-mannequin training environment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.multi_agent_config import MultiAgentTaskConfig
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
    BoxingMannequinConfig,
)
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
from myosuite.integrations.musclemimic.trajectory_io import load_motion_clip

_DEFAULT_CFG = BoxingMannequinConfig()
# pylint: disable=no-member


@dataclass
class BoxingMannequinTaskConfig(MultiAgentTaskConfig):
    """Multi-agent task config where agent_1 is a scripted mannequin."""

    max_episode_steps: int = _DEFAULT_CFG.max_episode_steps
    ctrl_dt: float = _DEFAULT_CFG.ctrl_dt
    sim_dt: float = _DEFAULT_CFG.sim_dt
    agent_separation_m: float = _DEFAULT_CFG.agent_separation_m
    disable_fingers: bool = _DEFAULT_CFG.disable_fingers

    hit_threshold_head_m: float = _DEFAULT_CFG.hit_threshold_head_m
    hit_threshold_body_m: float = _DEFAULT_CFG.hit_threshold_body_m
    hit_threshold_waist_m: float = _DEFAULT_CFG.hit_threshold_waist_m
    w_head: float = _DEFAULT_CFG.w_head
    w_body: float = _DEFAULT_CFG.w_body
    w_waist: float = _DEFAULT_CFG.w_waist

    ko_health_threshold: float = _DEFAULT_CFG.ko_health_threshold
    fall_pelvis_z_threshold: float = _DEFAULT_CFG.fall_pelvis_z_threshold

    r_damage_delivered: float = _DEFAULT_CFG.r_damage_delivered
    r_damage_received: float = _DEFAULT_CFG.r_damage_received
    r_ko_bonus: float = _DEFAULT_CFG.r_ko_bonus
    r_fall_bonus: float = _DEFAULT_CFG.r_fall_bonus
    r_self_fall_penalty: float = _DEFAULT_CFG.r_self_fall_penalty
    r_survival_per_step: float = _DEFAULT_CFG.r_survival_per_step
    r_act_reg: float = _DEFAULT_CFG.r_act_reg

    mannequin_guard_y: float = _DEFAULT_CFG.mannequin_guard_y
    mannequin_guard_z: float = _DEFAULT_CFG.mannequin_guard_z
    attack_interval_sec_min: float = _DEFAULT_CFG.attack_interval_sec_min
    attack_interval_sec_max: float = _DEFAULT_CFG.attack_interval_sec_max
    attack_duration_sec_min: float = _DEFAULT_CFG.attack_duration_sec_min
    attack_duration_sec_max: float = _DEFAULT_CFG.attack_duration_sec_max
    attack_target_noise_xy_m: float = _DEFAULT_CFG.attack_target_noise_xy_m
    attack_target_noise_z_m: float = _DEFAULT_CFG.attack_target_noise_z_m
    target_hit_normal_force_threshold_n: float = (
        _DEFAULT_CFG.target_hit_normal_force_threshold_n
    )
    include_boxing_targets_scene: bool = _DEFAULT_CFG.include_boxing_targets_scene
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
    _agent0_qpos_idx: np.ndarray | None = field(init=False, default=None, repr=False)
    _agent0_qvel_idx: np.ndarray | None = field(init=False, default=None, repr=False)
    _pad_hits_total: float = field(init=False, default=0.0, repr=False)
    _last_step_count: int = field(init=False, default=-1, repr=False)

    def _low_level_config(self) -> BoxingMannequinConfig:
        cfg_kwargs = {
            field.name: getattr(self, field.name)
            for field in fields(BoxingMannequinConfig)
        }
        return BoxingMannequinConfig(**cfg_kwargs)

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
        from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
            build_boxing_mannequin_model,
        )

        return build_boxing_mannequin_model(self._low_level_config())

    def build_spec(self) -> mujoco.MjSpec:
        from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
            _build_boxing_mannequin_spec,
        )

        return _build_boxing_mannequin_spec(self._low_level_config())

    @property
    def n_substeps(self) -> int:
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
        if self._agent0_qpos_idx is None or self._agent0_qvel_idx is None:
            qpos_idx: list[int] = []
            qvel_idx: list[int] = []
            for jid in meta.jnt_ids["agent_0"]:
                jtype = int(model.jnt_type[jid])
                qadr = int(model.jnt_qposadr[jid])
                vadr = int(model.jnt_dofadr[jid])
                qsize = _joint_qpos_size(jtype)
                vsize = _joint_qvel_size(jtype)
                qpos_idx.extend(range(qadr, qadr + qsize))
                qvel_idx.extend(range(vadr, vadr + vsize))
            self._agent0_qpos_idx = np.asarray(qpos_idx, dtype=np.int32)
            self._agent0_qvel_idx = np.asarray(qvel_idx, dtype=np.int32)

    def _fullbody_obs_for_agent0(self, data: mujoco.MjData, meta: Any) -> np.ndarray:
        if (
            self._fullbody_model is None
            or self._fullbody_data is None
            or self._fullbody_adapter is None
            or self._fullbody_clip is None
            or self._agent0_qpos_idx is None
            or self._agent0_qvel_idx is None
        ):
            raise RuntimeError("Fullbody bridge not initialized.")
        self._fullbody_data.qpos[:] = data.qpos[self._agent0_qpos_idx]
        self._fullbody_data.qvel[:] = data.qvel[self._agent0_qvel_idx]
        if self._fullbody_data.act is not None and len(self._fullbody_data.act) > 0:
            self._fullbody_data.act[:] = data.act[meta.act_indices["agent_0"]]
        self._fullbody_data.ctrl[:] = data.ctrl[meta.act_indices["agent_0"]]
        mujoco.mj_forward(self._fullbody_model, self._fullbody_data)
        frame_idx = int(round(float(data.time) / float(self.ctrl_dt)))
        frame_idx = frame_idx % max(1, int(self._fullbody_clip.qpos.shape[0]))
        return self._fullbody_adapter.build(self._fullbody_data, frame_idx)

    def obs_dim(self, model, data, meta, agent_id: str) -> int:
        from myosuite.terms.multiplayer.boxing_mannequin_obs import obs_dim

        dim = obs_dim(
            model,
            data,
            meta,
            agent_id,
            {a: 0.0 for a in self.agents},
            self.ko_health_threshold,
            self.target_hit_normal_force_threshold_n,
        )
        if self.append_fullbody_obs and agent_id == "agent_0":
            self._ensure_fullbody_bridge(model, meta)
            if self._fullbody_adapter is None or self._fullbody_data is None:
                raise RuntimeError("Fullbody adapter initialization failed.")
            dim += int(self._fullbody_adapter.build(self._fullbody_data, 0).shape[0])
        return dim

    def act_indices(self, meta: Any, agent_id: str) -> list[int]:
        return meta.act_indices[agent_id]

    def get_obs(
        self, model, data, meta, agent_id: str, health: dict[str, float]
    ) -> np.ndarray:
        from myosuite.terms.multiplayer.boxing_mannequin_obs import get_agent_obs

        obs = get_agent_obs(
            model,
            data,
            meta,
            agent_id,
            health,
            self.ko_health_threshold,
            self.target_hit_normal_force_threshold_n,
        )
        if self.append_fullbody_obs and agent_id == "agent_0":
            self._ensure_fullbody_bridge(model, meta)
            full = self._fullbody_obs_for_agent0(data, meta)
            obs = np.concatenate([obs, full.astype(np.float32)], axis=0)
        return obs

    def compute_damage(self, model, data, meta) -> dict[str, float]:
        from myosuite.terms.multiplayer.boxing_mannequin_reward import (
            compute_pad_damage,
            compute_total_damage,
        )

        cfg = self._low_level_config()
        if self.include_boxing_targets_scene:
            return {
                "agent_0": compute_pad_damage(
                    model,
                    data,
                    meta,
                    "agent_0",
                    hit_force_threshold_n=self.target_hit_normal_force_threshold_n,
                ),
                "agent_1": 0.0,
            }
        return {
            "agent_0": compute_total_damage(data, meta, "agent_0", "agent_1", cfg),
            "agent_1": compute_total_damage(data, meta, "agent_1", "agent_0", cfg),
        }

    def check_fell(self, data, meta, agent_id: str) -> bool:
        from myosuite.terms.multiplayer.boxing_mannequin_reward import is_fallen

        return is_fallen(data, meta, agent_id, self._low_level_config())

    @property
    def ko_threshold(self) -> float:
        return self.ko_health_threshold

    @property
    def health_update_scale(self) -> float:
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
        from myosuite.terms.multiplayer.boxing_mannequin_reward import (
            compute_agent_reward,
        )

        cfg = self._low_level_config()
        opp = self.opponent_id(agent_id)
        if agent_id == "agent_1":
            return 0.0
        if self.include_boxing_targets_scene:
            # P0 is a static-pad objective, so KO/fall terms are disabled.
            cfg = replace(
                cfg,
                r_damage_received=0.0,
                r_ko_bonus=0.0,
                r_fall_bonus=0.0,
                r_self_fall_penalty=0.0,
            )
        return compute_agent_reward(
            delivered_damage=damage_delivered[agent_id],
            received_damage=damage_delivered[opp],
            self_fell=fell[agent_id],
            opponent_fell=fell[opp],
            ko_happened=ko[opp],
            actions=actions,
            config=cfg,
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
        info = self.default_step_info(health, damage_delivered, fell, ko, step_count)
        if not self.include_boxing_targets_scene:
            return info

        if step_count <= self._last_step_count:
            self._pad_hits_total = 0.0
        self._last_step_count = step_count
        pad_hits_step = float(max(0.0, damage_delivered.get("agent_0", 0.0)))
        self._pad_hits_total += pad_hits_step
        info.update(
            {
                "pad_hits_step/agent_0": pad_hits_step,
                "pad_hits_total/agent_0": float(self._pad_hits_total),
            }
        )
        return info
