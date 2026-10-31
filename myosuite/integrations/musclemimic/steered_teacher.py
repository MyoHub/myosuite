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
    compass_index,
    compass_label,
    heading_yaw,
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
    """Two :class:`SteeredTeacher`s: chaser toward opponent, runner away.

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

    def reset(
        self, model: mujoco.MjModel, data: mujoco.MjData, meta: Any, task: str
    ) -> None:
        for agent, walker in self.walkers.items():
            lab = self._desired_label(data, meta, agent, task)
            walker.reset(lab)
            write_clip_pose(
                data,
                self.maps[agent],
                walker.clip.qpos[0],
                _clip_qvel(walker.clip, 0),
                keep_xy=True,
                keep_yaw=False,
            )
        mujoco.mj_forward(model, data)

    def actions(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        meta: Any,
        task: str,
        step: int,
    ) -> dict[str, np.ndarray]:
        if self.track_alpha > 0.0:
            for agent, walker in self.walkers.items():
                frame = walker.goal_frame()
                write_clip_pose(
                    data,
                    self.maps[agent],
                    walker.clip.qpos[frame],
                    _clip_qvel(walker.clip, frame),
                    keep_xy=True,
                    keep_yaw=False,
                    act=None,
                    alpha=self.track_alpha,
                )
            mujoco.mj_forward(model, data)
        return {
            agent: walker.action(data, self.maps[agent])
            for agent, walker in self.walkers.items()
        }

    def _desired_label(
        self, data: mujoco.MjData, meta: Any, agent: str, task: str
    ) -> str:
        me = data.site_xpos[meta.site_ids[agent]["pelvis_site"]][:2]
        opp = data.site_xpos[meta.site_ids[OPPONENT[agent]]["pelvis_site"]][:2]
        v = aim_heading_xy(me, opp, pursue=agent_is_chaser(agent, task))
        return compass_label(compass_index(heading_yaw(v)))
