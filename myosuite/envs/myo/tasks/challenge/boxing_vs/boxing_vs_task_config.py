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

import logging
from dataclasses import dataclass, field, fields
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

logger = logging.getLogger(__name__)

# Repository root (…/myosuite-capstone), used to resolve relative clip paths.
_REPO_ROOT = Path(__file__).resolve().parents[6]


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

    # --- Reward v1 options (mirror BoxingVsConfig; defaults = legacy) ---
    cross_agent_collisions: bool = False
    glove_collision_radius_m: float = 0.05
    glove_solref: tuple[float, float] = (0.02, 1.0)
    massless_visual_gear: bool = False
    hit_detection: str = "proximity"
    hit_min_speed_mps: float = 1.5
    hit_full_speed_mps: float = 4.5
    hit_cooldown_s: float = 0.15
    reward_mode: str = "legacy"
    fall_reward_mode: str = "legacy"
    knockdown_window_s: float = 1.0
    r_double_fall_penalty: float = -5.0
    r_hit: float = 1.0
    shaping_alpha: float = 1.0
    r_alive: float = 0.01
    r_reach: float = 1.0
    r_facing: float = 0.01
    r_effort: float = -0.01
    reset_pose: str = "default"
    reset_clip_path: str = (
        "boxing_motions/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/"
        "punchboxing_push_poses.npz"
    )
    reset_clip_frame: int = 0
    reset_noise_separation_m: float = 0.0
    reset_noise_lateral_m: float = 0.0
    reset_noise_yaw_deg: float = 0.0
    reset_noise_joint_rad: float = 0.0
    dummy_opponent: str | None = None

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
    # Per-episode reward-v1 state (reset in on_reset).
    _rv1: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    # Model-dependent caches (agent index arrays, reset clip frame).
    _rv1_cache: dict[str, Any] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.hit_detection not in ("proximity", "contact"):
            raise ValueError(
                f"hit_detection must be 'proximity' or 'contact', got {self.hit_detection!r}"
            )
        if self.hit_detection == "contact" and not self.cross_agent_collisions:
            raise ValueError(
                "hit_detection='contact' requires cross_agent_collisions=True"
            )
        if self.reward_mode not in ("legacy", "v1"):
            raise ValueError(
                f"reward_mode must be 'legacy' or 'v1', got {self.reward_mode!r}"
            )
        if self.fall_reward_mode not in ("legacy", "attributed"):
            raise ValueError(
                f"fall_reward_mode must be 'legacy' or 'attributed', got {self.fall_reward_mode!r}"
            )
        if self.reset_pose not in ("default", "clip"):
            raise ValueError(
                f"reset_pose must be 'default' or 'clip', got {self.reset_pose!r}"
            )
        if self.dummy_opponent not in (None, *self.agents):
            raise ValueError(f"dummy_opponent must be None or one of {self.agents}")
        if self.reward_mode == "v1" and self.hit_detection == "proximity":
            logger.warning(
                "reward_mode='v1' is tuned for hit_detection='contact'; with proximity "
                "hits, r_hit pays every step the fist is near a zone."
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
            **{f.name: getattr(self, f.name) for f in fields(BoxingVsConfig)}
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

    def on_reset(self, model, data, meta, np_random) -> None:
        """Start pose / noise / dummy hold, then reset the reward-v1 state.

        With default options nothing is written to ``data`` and no random
        number is drawn, so legacy rollouts are unchanged.
        """
        changed = False
        if self.reset_pose == "clip":
            self._apply_clip_pose(model, data, meta)
            changed = True
        changed |= self._apply_reset_noise(model, data, meta, np_random)
        if changed:
            mujoco.mj_forward(model, data)

        hold = None
        if self.dummy_opponent is not None:
            qidx, vidx, _ = self._agent_indices(model, meta, self.dummy_opponent)
            data.qvel[vidx] = 0.0
            hold = np.array(data.qpos[qidx], copy=True)
            mujoco.mj_forward(model, data)

        from myosuite.terms.multiplayer.boxing_vs_reward_v1 import (
            HitTracker,
            reach_potential,
            reward_term_names,
            snapshot_kinematics,
        )

        tracker = HitTracker()
        if self.hit_detection == "contact":
            tracker.prev = snapshot_kinematics(data, self._glove_geoms(meta))
        self._rv1 = {
            "step": 0,
            "dummy_hold": hold,
            "tracker": tracker,
            "events": [],
            "last_hit_step": {a: None for a in self.agents},
            "reach_phi": {a: reach_potential(data, meta, a) for a in self.agents},
            "reach_progress": {a: 0.0 for a in self.agents},
            "facing": {a: 0.0 for a in self.agents},
            "terms": {
                a: {k: 0.0 for k in reward_term_names(self)} for a in self.agents
            },
        }

    def pre_physics_step(self, model, data, meta) -> None:
        """Hold the dummy opponent (if any) in its reset pose, muscles off."""
        if self.dummy_opponent is None:
            return
        hold = self._rv1.get("dummy_hold")
        if hold is None:
            return
        qidx, vidx, act_idx = self._agent_indices(model, meta, self.dummy_opponent)
        data.qpos[qidx] = hold
        data.qvel[vidx] = 0.0
        data.ctrl[act_idx] = 0.0
        actadr = model.actuator_actadr[act_idx]
        actadr = actadr[actadr >= 0]
        if actadr.size:
            data.act[actadr] = 0.0

    def compute_damage(self, model, data, meta) -> dict[str, float]:
        """Compute damage delivered by each agent this step.

        Called exactly once per control step, before the rewards; also
        updates the per-step reward-v1 signals.
        """
        from myosuite.terms.multiplayer.boxing_vs_reward import compute_total_damage

        if not self._rv1:  # step() without reset(): initialise lazily
            self.on_reset(model, data, meta, np.random.default_rng(0))
        st = self._rv1
        st["step"] += 1

        if self.hit_detection == "contact":
            from myosuite.terms.multiplayer.boxing_vs_reward_v1 import (
                collect_glove_contacts,
                snapshot_kinematics,
            )

            contacts = collect_glove_contacts(model, data, meta)
            events = st["tracker"].update(contacts, model, st["step"], self)
            st["tracker"].prev = snapshot_kinematics(data, self._glove_geoms(meta))
            damage = {
                a: float(sum(e.damage for e in events if e.attacker == a))
                for a in self.agents
            }
        else:
            events = []
            damage = {a: compute_total_damage(data, meta, a, self) for a in self.agents}
        st["events"] = events
        for a in self.agents:
            if damage[a] > 0.0:
                st["last_hit_step"][a] = st["step"]

        if self.reward_mode == "v1":
            from myosuite.terms.multiplayer.boxing_vs_reward_v1 import (
                facing_cosine,
                reach_potential,
            )

            for a in self.agents:
                phi = reach_potential(data, meta, a)
                # Clip so a teleport (reset, dummy hold) cannot pay much.
                st["reach_progress"][a] = float(
                    np.clip(phi - st["reach_phi"][a], -0.1, 0.1)
                )
                st["reach_phi"][a] = phi
                st["facing"][a] = facing_cosine(data, meta, a)
        return damage

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
        """Compute scalar reward for *agent_id* (and store its breakdown)."""
        from myosuite.terms.multiplayer.boxing_vs_reward import compute_agent_reward
        from myosuite.terms.multiplayer.boxing_vs_reward_v1 import (
            attributed_fall_term,
            legacy_reward_terms,
            v1_reward_terms,
        )

        opp = self.opponent_id(agent_id)
        dealt, received = damage_delivered[agent_id], damage_delivered[opp]
        st = self._rv1
        hit_recently = self._hit_recently(agent_id)

        if self.reward_mode == "v1":
            terms = v1_reward_terms(
                dealt=dealt,
                received=received,
                self_fell=fell[agent_id],
                opponent_fell=fell[opp],
                opponent_ko=ko[opp],
                hit_opponent_recently=hit_recently,
                reach_progress=st["reach_progress"][agent_id],
                facing=st["facing"][agent_id],
                actions=actions,
                config=self,
            )
            reward = float(sum(terms.values()))
        else:
            attributed = self.fall_reward_mode == "attributed"
            reward = compute_agent_reward(
                data=data,
                meta=meta,
                agent_id=agent_id,
                delivered_damage=dealt,
                received_damage=received,
                self_fell=False if attributed else fell[agent_id],
                opponent_fell=False if attributed else fell[opp],
                ko_happened=ko[opp],
                own_health=health[agent_id],
                actions=actions,
                config=self,
            )
            terms = legacy_reward_terms(
                delivered_damage=dealt,
                received_damage=received,
                actions=actions,
                self_fell=fell[agent_id],
                opponent_fell=fell[opp],
                ko_happened=ko[opp],
                config=self,
            )
            if attributed:
                fall = attributed_fall_term(
                    self_fell=fell[agent_id],
                    opponent_fell=fell[opp],
                    hit_opponent_recently=hit_recently,
                    r_fall_bonus=self.r_fall_bonus,
                    r_self_fall_penalty=self.r_self_fall_penalty,
                    r_double_fall_penalty=self.r_double_fall_penalty,
                )
                reward += fall
                del terms["fall_bonus"], terms["self_fall"]
                terms["fall"] = fall
        if st:
            st["terms"][agent_id] = terms
        return reward

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
        """Build the step info dict.

        Adds ``reward/<term>/<agent>`` (per-term reward) and per-step hit
        statistics ``hits_head``, ``hits_body``, ``blocked``, ``hits_r``,
        ``hits_l``, ``impact_speed_max`` (``/<agent>``, agent as attacker).
        """
        from myosuite.terms.multiplayer.boxing_vs_reward_v1 import event_counts

        info = self.default_step_info(health, damage_delivered, fell, ko, step_count)
        st = self._rv1
        if not st:
            return info
        for a in self.agents:
            for name, value in st["terms"][a].items():
                info[f"reward/{name}/{a}"] = float(value)
            for name, value in event_counts(st["events"], a).items():
                info[f"{name}/{a}"] = value
        return info

    # ------------------------------------------------------------------
    # Reward-v1 helpers
    # ------------------------------------------------------------------

    def _hit_recently(self, agent_id: str) -> bool:
        """True if *agent_id* damaged its opponent within the knockdown window."""
        st = self._rv1
        last = st.get("last_hit_step", {}).get(agent_id) if st else None
        if last is None:
            return False
        return (st["step"] - last) * self.ctrl_dt <= self.knockdown_window_s

    def _glove_geoms(self, meta) -> list[int]:
        return [g for gloves in meta.glove_geom_ids.values() for g in gloves.values()]

    def _agent_indices(
        self, model, meta, agent_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(qpos_idx, qvel_idx, actuator_idx)`` of *agent_id* (cached)."""
        key = ("idx", agent_id)
        if key not in self._rv1_cache:
            qpos_idx: list[int] = []
            qvel_idx: list[int] = []
            for jid in meta.jnt_ids[agent_id]:
                jtype = int(model.jnt_type[jid])
                qadr = int(model.jnt_qposadr[jid])
                vadr = int(model.jnt_dofadr[jid])
                qpos_idx.extend(range(qadr, qadr + _joint_qpos_size(jtype)))
                qvel_idx.extend(range(vadr, vadr + _joint_qvel_size(jtype)))
            self._rv1_cache[key] = (
                np.asarray(qpos_idx, dtype=np.int64),
                np.asarray(qvel_idx, dtype=np.int64),
                np.asarray(meta.act_indices[agent_id], dtype=np.int64),
            )
        return self._rv1_cache[key]

    def _root_qpos_adr(self, model, meta, agent_id: str) -> int:
        for jid in meta.jnt_ids[agent_id]:
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[jid])
        raise RuntimeError(f"No free root joint for {agent_id}")

    def _resolve_reset_clip(self) -> Path:
        from myosuite.core.trajectory_io import resolve_motion_path

        raw = Path(self.reset_clip_path).expanduser()
        if raw.is_file():
            return raw.resolve()
        if (_REPO_ROOT / raw).is_file():
            return (_REPO_ROOT / raw).resolve()
        return resolve_motion_path(self.reset_clip_path)

    def _clip_pose_maps(self, model, meta):
        """Reset clip frame + index maps from the clip (full-body model order)
        to each agent's joints in the combined model (by joint name)."""
        if "clip" in self._rv1_cache:
            return self._rv1_cache["clip"]
        from myosuite.core.trajectory_io import expand_motion_clip_to_model

        cfg = default_mimic_fullbody_config()
        cfg.disable_fingers = self.disable_fingers
        cfg.sim_dt = self.sim_dt
        fb_model, _spec, _xml = compile_mimic_fullbody_mjmodel(cfg)
        clip = load_motion_clip(
            self._resolve_reset_clip(),
            expected_nq=fb_model.nq,
            expected_nv=fb_model.nv,
        )
        clip = expand_motion_clip_to_model(clip, fb_model)
        n_frames = int(clip.qpos.shape[0])
        if not 0 <= self.reset_clip_frame < n_frames:
            raise ValueError(
                f"reset_clip_frame={self.reset_clip_frame} outside clip ({n_frames} frames)"
            )
        frame = np.asarray(clip.qpos[self.reset_clip_frame], dtype=np.float64)
        fb_root = None
        maps = {}
        for agent_id, prefix in (("agent_0", "a0_"), ("agent_1", "a1_")):
            dst: list[int] = []
            src: list[int] = []
            for jid in meta.jnt_ids[agent_id]:
                jtype = int(model.jnt_type[jid])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)[
                    len(prefix) :
                ]
                fb_jid = mujoco.mj_name2id(fb_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if fb_jid < 0:
                    raise RuntimeError(f"Joint {name!r} missing in the full-body model")
                fb_adr = int(fb_model.jnt_qposadr[fb_jid])
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    fb_root = fb_adr
                    continue
                width = _joint_qpos_size(jtype)
                adr = int(model.jnt_qposadr[jid])
                dst.extend(range(adr, adr + width))
                src.extend(range(fb_adr, fb_adr + width))
            maps[agent_id] = (np.asarray(dst), np.asarray(src))
        if fb_root is None:
            raise RuntimeError("Full-body clip has no free root joint")
        self._rv1_cache["clip"] = (frame, fb_root, maps)
        return self._rv1_cache["clip"]

    def _apply_clip_pose(self, model, data, meta) -> None:
        """Joint angles + root height/tilt from the clip frame; each agent
        keeps its placement (root XY and facing direction)."""
        frame, fb_root, maps = self._clip_pose_maps(model, meta)
        for agent_id in self.agents:
            adr = self._root_qpos_adr(model, meta, agent_id)
            place_quat = np.array(data.qpos[adr + 3 : adr + 7], copy=True)
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, place_quat, frame[fb_root + 3 : fb_root + 7])
            data.qpos[adr + 2] = frame[fb_root + 2]
            data.qpos[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
            dst, src = maps[agent_id]
            data.qpos[dst] = frame[src]
            _, vidx, _ = self._agent_indices(model, meta, agent_id)
            data.qvel[vidx] = 0.0

    def _apply_reset_noise(self, model, data, meta, rng) -> bool:
        """Randomise start distance, lateral offset, yaw and joint angles.

        Draws random numbers only for non-zero noise settings.
        """
        changed = False
        adrs = {a: self._root_qpos_adr(model, meta, a) for a in self.agents}
        if self.reset_noise_separation_m > 0.0:
            ds = float(rng.uniform(-1.0, 1.0)) * self.reset_noise_separation_m
            data.qpos[adrs["agent_0"] + 1] += ds / 2.0  # agent 0 sits at +Y
            data.qpos[adrs["agent_1"] + 1] -= ds / 2.0
            changed = True
        for a in self.agents:
            adr = adrs[a]
            if self.reset_noise_lateral_m > 0.0:
                data.qpos[adr] += (
                    float(rng.uniform(-1.0, 1.0)) * self.reset_noise_lateral_m
                )
                changed = True
            if self.reset_noise_yaw_deg > 0.0:
                yaw = np.deg2rad(
                    float(rng.uniform(-1.0, 1.0)) * self.reset_noise_yaw_deg
                )
                q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, q_yaw, np.array(data.qpos[adr + 3 : adr + 7]))
                data.qpos[adr + 3 : adr + 7] = quat
                changed = True
            if self.reset_noise_joint_rad > 0.0:
                for jid in meta.jnt_ids[a]:
                    jtype = int(model.jnt_type[jid])
                    if jtype not in (
                        int(mujoco.mjtJoint.mjJNT_HINGE),
                        int(mujoco.mjtJoint.mjJNT_SLIDE),
                    ):
                        continue
                    qadr = int(model.jnt_qposadr[jid])
                    q = (
                        data.qpos[qadr]
                        + float(rng.uniform(-1.0, 1.0)) * self.reset_noise_joint_rad
                    )
                    if model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        q = float(np.clip(q, lo, hi))
                    data.qpos[qadr] = q
                changed = True
        return changed
