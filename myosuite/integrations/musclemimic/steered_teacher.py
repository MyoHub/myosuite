# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Role-aware mm-10m-2 player for ChaseTag FBVs 1v1.

Composes existing helpers: :func:`load_teacher_runner`,
:meth:`FullbodyObsAdapter.with_clip`, :func:`write_rotated_clip_npz`,
:class:`AgentShadowMap`, and :func:`aim_heading_xy`. Does not change the
scored FBVs observation (546-d).

``amathislab/mm-10m-2`` was trained on musclemimic_models 1.0.5. Set
``MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5`` before ``make_env`` so the
ChaseTag body matches the teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.trajectory_io import MotionClip, load_motion_clip
from myosuite.core.trajectory_yaw import (
    clip_natural_yaw,
    pingpong_frame,
    write_rotated_clip_npz,
)
from myosuite.integrations.musclemimic.bc_directional_collector import (
    load_teacher_runner,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import LocalPolicyRunner
from myosuite.integrations.musclemimic.two_agent_scene import (
    AgentShadowMap,
    build_agent_shadow_map,
    copy_agent_to_shadow,
    write_clip_pose,
)
from myosuite.terms.multiplayer.chase_tag_vs_obs import agent_is_chaser
from myosuite.terms.multiplayer.heading import (
    COMPASS_LABELS,
    COMPASS_STEP,
    aim_heading_xy,
    body_yaw,
    clamp_arena_xy,
    compass_index,
    compass_label,
    heading_yaw,
    role_heading_xy,
    step_compass,
    wall_repulsion,
)

AGENTS: tuple[str, str] = ("agent_0", "agent_1")
PREFIX: dict[str, str] = {"agent_0": "a0_", "agent_1": "a1_"}
OPPONENT: dict[str, str] = {"agent_0": "agent_1", "agent_1": "agent_0"}
DEFAULT_WALK = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _clip_qvel(clip: MotionClip, frame: int) -> np.ndarray:
    if clip.qvel is None:
        raise ValueError("steered teacher requires a clip with qvel")
    return clip.qvel[frame]


def ensure_compass_clips(
    src: Path,
    cache_dir: Path,
    *,
    expected_nq: int,
    expected_nv: int,
) -> dict[str, MotionClip]:
    """Yaw-rotate *src* onto the 8 compass headings (cached on disk)."""
    src = Path(src)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = load_motion_clip(src, expected_nq=expected_nq, expected_nv=expected_nv)
    natural = clip_natural_yaw(base.qpos, base.qvel)
    clips: dict[str, MotionClip] = {}
    for i, lab in enumerate(COMPASS_LABELS):
        dst = cache_dir / f"{src.stem}_{lab}.npz"
        if not dst.is_file():
            write_rotated_clip_npz(src, dst, i * COMPASS_STEP - natural)
        clips[lab] = load_motion_clip(
            dst, expected_nq=expected_nq, expected_nv=expected_nv
        )
    return clips


@dataclass
class SteeredTeacher:
    """One agent: clip-conditioned teacher that can switch 8-bin heading."""

    runner: LocalPolicyRunner
    clips: dict[str, MotionClip]
    shadow_model: mujoco.MjModel
    shadow_data: mujoco.MjData
    heading: str = "E"
    t: int = 0
    pingpong: bool = True

    def __post_init__(self) -> None:
        base = self.runner.obs_adapter
        if base is None:
            raise ValueError("SteeredTeacher requires a FullbodyObsAdapter")
        self.adapters = {lab: base.with_clip(self.clips[lab]) for lab in COMPASS_LABELS}
        self.runner.obs_adapter = self.adapters[self.heading]

    @classmethod
    def create(
        cls,
        teacher_root: Path,
        shadow_model: mujoco.MjModel,
        clips: dict[str, MotionClip],
        *,
        seed: int = 0,
        heading: str = "E",
    ) -> SteeredTeacher:
        runner = load_teacher_runner(
            teacher_root, shadow_model, clips[heading], seed=seed
        )
        return cls(
            runner=runner,
            clips=clips,
            shadow_model=shadow_model,
            shadow_data=mujoco.MjData(shadow_model),
            heading=heading,
        )

    @property
    def clip(self) -> MotionClip:
        return self.clips[self.heading]

    @property
    def idx(self) -> int:
        return COMPASS_LABELS.index(self.heading)

    def reset(self, heading: str) -> None:
        self.t = 0
        self.heading = heading
        self.runner.obs_adapter = self.adapters[heading]
        self.runner.reset()

    def set_heading(self, heading: str) -> bool:
        """Switch the goal clip. Does not teleport the body or reset history."""
        if heading == self.heading:
            return False
        self.heading = heading
        self.runner.obs_adapter = self.adapters[heading]
        return True

    def goal_frame(self) -> int:
        n = int(self.clip.qpos.shape[0])
        return pingpong_frame(self.t, n) if self.pingpong else self.t % n

    def action(self, fbvs_data: mujoco.MjData, amap: AgentShadowMap) -> np.ndarray:
        copy_agent_to_shadow(fbvs_data, self.shadow_data, amap)
        mujoco.mj_forward(self.shadow_model, self.shadow_data)
        act = self.runner.action_for(
            self.shadow_data,
            self.clip,
            self.goal_frame(),
            history_idx=self.t,
        )
        self.t += 1
        return np.clip(act, 0.0, 1.0).astype(np.float32)


@dataclass
class ChaseTagSteeredPair:
    """Two :class:`SteeredTeacher`s: they look at each other; chaser walks
    toward the opponent, runner walks away.

    ``track_alpha`` blends each agent toward the current clip frame before
    the teacher acts. ``1`` is kinematic playback; ``0`` is teacher-only
    (falls on some headings). A value in between keeps the gait continuous
    instead of snapping joints every few hundred milliseconds.
    """

    walkers: dict[str, SteeredTeacher]
    maps: dict[str, AgentShadowMap]
    track_alpha: float = 0.45

    @classmethod
    def create(
        cls,
        teacher_root: Path,
        fbvs_model: mujoco.MjModel,
        meta: Any,
        clips: dict[str, MotionClip],
        shadow_model: mujoco.MjModel,
        *,
        track_alpha: float = 0.45,
    ) -> ChaseTagSteeredPair:
        maps = {
            agent: build_agent_shadow_map(
                fbvs_model,
                shadow_model,
                PREFIX[agent],
                meta.act_indices[agent],
            )
            for agent in AGENTS
        }
        walkers = {
            agent: SteeredTeacher.create(teacher_root, shadow_model, clips, seed=i)
            for i, agent in enumerate(AGENTS)
        }
        return cls(walkers=walkers, maps=maps, track_alpha=float(track_alpha))

    @classmethod
    def load(
        cls,
        fbvs_env: Any,
        *,
        teacher_root: Path | None = None,
        clips_dir: Path | None = None,
        track_alpha: float = 0.45,
    ) -> ChaseTagSteeredPair:
        """Download ``mm-10m-2`` and compass clips, then :meth:`create`."""
        from huggingface_hub import hf_hub_download, snapshot_download

        from myosuite.integrations.musclemimic.fullbody_model import (
            compile_mimic_fullbody_mjmodel,
            default_mimic_fullbody_config,
        )
        from myosuite.integrations.musclemimic.model_versions import (
            checkpoint_models_version,
        )

        raw = fbvs_env.unwrapped
        root = Path(teacher_root or snapshot_download(repo_id="amathislab/mm-10m-2"))
        cfg = default_mimic_fullbody_config()
        cfg.model_version = checkpoint_models_version("hf://amathislab/mm-10m-2")
        shadow, _, _ = compile_mimic_fullbody_mjmodel(cfg)
        src = Path(
            hf_hub_download(
                repo_id="amathislab/musclemimic-retargeted",
                filename=DEFAULT_WALK,
                repo_type="dataset",
            )
        )
        cache = Path(
            clips_dir or (Path.home() / ".cache" / "myosuite" / "compass_walk")
        )
        clips = ensure_compass_clips(
            src, cache, expected_nq=shadow.nq, expected_nv=shadow.nv
        )
        return cls.create(
            root, raw.model, raw._meta, clips, shadow, track_alpha=track_alpha
        )

    def reset(
        self, model: mujoco.MjModel, data: mujoco.MjData, meta: Any, task: str
    ) -> None:
        for agent, walker in self.walkers.items():
            face, travel = self._aim(data, meta, agent, task)
            walker.reset(compass_label(compass_index(heading_yaw(travel))))
            self._track_pose(data, agent, walker, face, travel, alpha=1.0)
        mujoco.mj_forward(model, data)

    def actions(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
        task: str,
        step: int,
    ) -> dict[str, np.ndarray]:
        for agent, walker in self.walkers.items():
            face, travel = self._aim(data, meta, agent, task)
            want = compass_index(heading_yaw(travel))
            walker.set_heading(compass_label(step_compass(walker.idx, want)))
            if self.track_alpha > 0.0:
                self._track_pose(
                    data,
                    agent,
                    walker,
                    face,
                    travel,
                    alpha=self.track_alpha,
                    act=None,
                )
        if self.track_alpha > 0.0:
            mujoco.mj_forward(model, data)
        return {
            agent: walker.action(data, self.maps[agent])
            for agent, walker in self.walkers.items()
        }

    def _aim(
        self, data: mujoco.MjData, meta: Any, agent: str, task: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look at the opponent; walk toward (chaser) or away (runner).

        Near a fence, look along the travel heading so a wall turn is not a
        sidestep through the panel.
        """
        me = data.site_xpos[meta.site_ids[agent]["pelvis_site"]][:2]
        opp = data.site_xpos[meta.site_ids[OPPONENT[agent]]["pelvis_site"]][:2]
        face = role_heading_xy(me, opp, pursue=True)
        travel = aim_heading_xy(me, opp, pursue=agent_is_chaser(agent, task))
        if float(np.linalg.norm(wall_repulsion(me))) > 1e-6:
            face = travel
        return face, travel

    def _track_pose(
        self,
        data: mujoco.MjData,
        agent: str,
        walker: SteeredTeacher,
        face: np.ndarray,
        travel: np.ndarray,
        *,
        alpha: float,
        act: float | None = 0.05,
    ) -> None:
        frame = 0 if alpha >= 1.0 else walker.goal_frame()
        write_clip_pose(
            data,
            self.maps[agent],
            walker.clip.qpos[frame],
            _clip_qvel(walker.clip, frame),
            keep_xy=True,
            face_yaw=body_yaw(face),
            travel_yaw=heading_yaw(travel),
            act=act,
            alpha=alpha,
        )
        fa = self.maps[agent].qpos_pairs[0][0]
        data.qpos[fa : fa + 2] = clamp_arena_xy(data.qpos[fa : fa + 2])
