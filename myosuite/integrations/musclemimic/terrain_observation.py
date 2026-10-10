# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TERRA's policy observation, rebuilt from MuJoCo data and a reference motion.

The physical state is built by MuscleMimic's shared FullbodyStateAdapter.
Only the terrain heightmap and checkpoint-specific reference goal are added here.
The pinned TERRA-4B release uses egocentric root state and full-body tracking errors;
the earlier compact TerraGoal layout is retained for the upstream replay fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from myosuite.integrations.musclemimic.fullbody_model import (
    TERRAIN_GROUP,
    FULLBODY_BODY2SITES_FOR_MIMIC,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import FullbodyStateAdapter
from myosuite.physics.quat_math import quat2yaw
from myosuite.core.trajectory_io import MotionClip

MIMIC_SITES = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())
TOUCH_SENSORS = ("r_foot", "r_toes", "l_foot", "l_toes")
ANKLE_SITES = ("left_ankle_mimic", "right_ankle_mimic")


@dataclass(frozen=True)
class TerrainObsCfg:
    """Observation settings of a TERRA checkpoint (``experiment.env_params``).

    Attributes:
        sites: ``goal_params.sites_for_mimic`` (the first is the reference site).
        grid: Heightmap ``(rows, cols)``.
        resolution: Heightmap spacing in metres.
        forward_offset: Forward shift of the grid in metres.
        grid_body: Body the grid is centred on.
        future_stride: Frames between future cues.
        future_horizon: Last future frame.
    """

    sites: tuple[str, ...] = MIMIC_SITES
    grid: tuple[int, int] = (11, 11)
    resolution: float = 0.1
    forward_offset: float = 0.0
    grid_body: str = "pelvis"
    future_stride: int = 10
    future_horizon: int = 100
    tracking: bool = False
    lookahead_steps: tuple[int, ...] = (1, 20, 40, 60, 80)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TerrainObsCfg:
        """Settings saved in a checkpoint's ``config`` (raises on an unsupported layout)."""
        env = config["experiment"]["env_params"]
        goal = env["goal_params"]
        tracking = env.get("goal_type") == "TerraFullBodyTrackingGoal"
        expected = {
            "goal_type": "TerraFullBodyTrackingGoal" if tracking else "TerraGoal",
            "use_egocentric_root_observations": tracking,
            "enable_global_root_position_observation": False,
            "preserve_trajectory_root_xy": True,
            "enable_heightmap_observations": True,
            "enable_touch_sensor_observations": True,
            "enable_joint_pos_observations": True,
            "enable_joint_vel_observations": True,
            "enable_muscle_length_observations": False,
            "enable_muscle_velocity_observations": False,
            "enable_muscle_force_observations": False,
            "enable_muscle_excitation_observations": True,
            "enable_muscle_activation_observations": True,
        }
        wrong = {k: env.get(k) for k, v in expected.items() if env.get(k, v) != v}
        if not goal.get("enable_future_reference_observations", False):
            wrong["enable_future_reference_observations"] = False
        exp = config["experiment"]
        for key in ("actor_obs_group", "use_moe"):
            if exp.get(key):
                wrong[key] = exp[key]
        if wrong or int(exp.get("len_obs_history", 1) or 1) != 1:
            raise ValueError(f"Unsupported TERRA observation settings: {wrong}")
        steps = tuple(goal.get("lookahead_steps", (1, 20, 40, 60, 80)))
        if tracking and (
            not steps
            or steps[0] != 1
            or any(a >= b for a, b in zip(steps, steps[1:]))
            or goal.get("include_support_intent", False)
        ):
            raise ValueError("Unsupported tracking lookahead settings")
        return cls(
            tracking=tracking,
            lookahead_steps=steps,
            sites=tuple(goal["sites_for_mimic"]),
            grid=(int(env["heightmap_grid_rows"]), int(env["heightmap_grid_cols"])),
            resolution=float(env["heightmap_grid_resolution"]),
            forward_offset=float(env.get("heightmap_grid_forward_offset", 0.0)),
            grid_body=str(env.get("heightmap_body_name") or "pelvis"),
            future_stride=int(goal["future_reference_stride"]),
            future_horizon=int(goal["future_reference_horizon"]),
        )


class TerrainHeights:
    """Terrain height under world ``(x, y)`` points: rays onto :data:`TERRAIN_GROUP` geoms.

    The floor (``z = 0``) is where nothing is hit, as TERRA's box terrain.
    """

    def __init__(self, model: mujoco.MjModel, top: float = 50.0) -> None:
        self._model, self._top = model, top
        self._group = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
        self._group[TERRAIN_GROUP] = 1

    def __call__(self, data: mujoco.MjData, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        down, geomid = np.array([0.0, 0.0, -1.0]), np.zeros(1, dtype=np.int32)
        heights = np.zeros(len(xy))
        for i, (x, y) in enumerate(xy):
            dist = mujoco.mj_ray(
                self._model,
                data,
                np.array([x, y, self._top]),
                down,
                self._group,
                1,
                -1,
                geomid,
            )
            heights[i] = self._top - dist if geomid[0] >= 0 else 0.0
        return heights


class TerrainObservation:
    """Builds the TERRA policy observation for a model with the TERRA actor."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: TerrainObsCfg | None = None,
        reference: MotionClip | None = None,
    ) -> None:
        self.reference = reference
        self.cfg = cfg or TerrainObsCfg()
        self._model = model
        self._heights = TerrainHeights(model)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root != 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                "The TERRA actor's first joint must be the free joint 'root'"
            )
        self._sites = np.array([self._site(n) for n in self.cfg.sites])
        self._ref_ankles = [self.cfg.sites.index(n) for n in ANKLE_SITES]
        self._state = FullbodyStateAdapter(
            model,
            {
                "enable_muscle_length_observations": False,
                "enable_muscle_velocity_observations": False,
                "enable_muscle_force_observations": False,
            },
            egocentric_root=self.cfg.tracking,
        )
        self._grid_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.grid_body
        )
        rows, cols = self.cfg.grid
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        res = self.cfg.resolution
        self._grid = np.column_stack(
            [
                ((i - (rows - 1) / 2) * res + self.cfg.forward_offset).ravel(),
                ((j - (cols - 1) / 2) * res).ravel(),
            ]
        )
        self._offsets = np.arange(
            self.cfg.future_stride, self.cfg.future_horizon + 1, self.cfg.future_stride
        )

    def _site(self, name: str) -> int:
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site {name!r} is missing from the TERRA actor")
        return sid

    def heightmap(self, data: mujoco.MjData) -> np.ndarray:
        """Terrain heights on the yaw-aligned grid, relative to the grid body."""
        x, y, z = data.xpos[self._grid_body]
        yaw = quat2yaw(data.qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        g = self._grid
        world = np.column_stack(
            [x + c * g[:, 0] - s * g[:, 1], y + s * g[:, 0] + c * g[:, 1]]
        )
        return self._heights(data, world) - z

    def goal(self, data: mujoco.MjData, ref: MotionClip, frame: int) -> np.ndarray:
        """TerraGoal of reference *frame*."""
        if self.cfg.tracking:
            return self._tracking_goal(data, ref, frame)
        frame = int(np.clip(frame, 0, ref.num_frames - 1))
        sim_sites = data.site_xpos[self._sites]
        ref_sites = ref.site_xpos[frame]
        sim_rpos = sim_sites[1:] - sim_sites[0]
        ref_rpos = ref_sites[1:] - ref_sites[0]
        future = np.clip(frame + self._offsets, 0, ref.num_frames - 1)
        root_now = ref.qpos[frame, 2]
        cues = [
            np.concatenate(
                [
                    [ref.qpos[f, 2] - root_now],
                    (ref.site_xpos[f, self._ref_ankles] - ref.qpos[f, :3]).ravel(),
                ]
            )
            for f in future
        ]
        return np.concatenate(
            [
                ref.qpos[frame, 2:],
                ref.qvel[frame],
                ref.qpos[frame, :2] - data.qpos[:2],
                (ref_rpos - sim_rpos).ravel(),
                np.concatenate(cues),
            ]
        )

    @staticmethod
    def _rotation(quat: np.ndarray) -> np.ndarray:
        return Rotation.from_quat(np.roll(quat, -1)).as_matrix()

    @classmethod
    def _heading(cls, quat: np.ndarray) -> np.ndarray:
        r = cls._rotation(quat)
        yaw = np.arctan2(r[1, 0] - r[0, 1], r[0, 0] + r[1, 1])
        return Rotation.from_rotvec([0, 0, yaw]).as_matrix()

    @staticmethod
    def _velocity(
        qvel: np.ndarray, rotation: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([heading.T @ qvel[:3], heading.T @ rotation @ qvel[3:6]])

    @staticmethod
    def _rotation_error(
        current: np.ndarray, target: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
        error = target @ np.swapaxes(current, -1, -2)
        return Rotation.from_matrix(heading.T @ error @ heading).as_rotvec()

    def _tracking_goal(
        self, data: mujoco.MjData, ref: MotionClip, frame: int
    ) -> np.ndarray:
        if ref.site_xmat is None or ref.site_velocity is None:
            raise ValueError(
                "Tracking observations need MotionClip.from_states kinematics"
            )
        heading = self._heading(data.qpos[3:7])
        rotation = self._rotation(data.qpos[3:7])
        current_velocity = self._velocity(data.qvel, rotation, heading)
        current_sites = data.site_xpos[self._sites]
        site_rotation = data.site_xmat[self._sites].reshape(-1, 3, 3)
        velocity = np.zeros((len(self._sites), 6))
        for i, sid in enumerate(self._sites):
            mujoco.mj_objectVelocity(
                self._model, data, mujoco.mjtObj.mjOBJ_SITE, int(sid), velocity[i], 0
            )
        components = []
        for i, offset in enumerate(self.cfg.lookahead_steps):
            requested = frame + offset
            f = int(np.clip(requested, 0, ref.num_frames - 1))
            target_rotation = self._rotation(ref.qpos[f, 3:7])
            root = [
                heading.T @ (ref.qpos[f, :3] - data.qpos[:3]),
                self._rotation_error(rotation, target_rotation, heading),
                self._velocity(ref.qvel[f], target_rotation, heading),
            ]
            relative = ref.site_xpos[f, 1:] - ref.site_xpos[f, 0]
            if i == 0:
                root[-1] -= current_velocity
                position_error = relative - (current_sites[1:] - current_sites[0])
                velocity_error = (
                    ref.site_velocity[f, 1:] - ref.site_velocity[f, 0]
                ) - (velocity[1:] - velocity[0])
                components.extend(
                    root
                    + [
                        (position_error @ heading).ravel(),
                        self._rotation_error(
                            site_rotation[1:], ref.site_xmat[f, 1:], heading
                        ).ravel(),
                        np.concatenate(
                            [
                                velocity_error[:, :3] @ heading,
                                velocity_error[:, 3:] @ heading,
                            ],
                            axis=-1,
                        ).ravel(),
                    ]
                )
            else:
                components.extend(
                    root
                    + [
                        (relative @ heading).ravel(),
                        np.array(
                            [
                                offset * (1 / ref.frequency_hz),
                                requested < ref.num_frames,
                            ]
                        ),
                    ]
                )
        return np.concatenate(components)

    def build(self, data: mujoco.MjData, frame_idx: int) -> np.ndarray:
        if self.reference is None:
            raise ValueError("A reference clip is required for policy inference")
        return self(data, self.reference, frame_idx)

    def __call__(self, data: mujoco.MjData, ref: MotionClip, frame: int) -> np.ndarray:
        """Flat float32 observation of *data* tracking reference *frame*."""
        return np.concatenate(
            [
                self._state.build_state(data),
                self.heightmap(data),
                self.goal(data, ref, frame),
            ]
        ).astype(np.float32)
