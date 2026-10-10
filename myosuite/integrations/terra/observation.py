# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TERRA's policy observation, rebuilt from MuJoCo data and a reference motion.

Layout (``terra.rl.environment._TerraObservationLayout`` + ``terra.rl.observations
.TerraGoal``, with the TERRA-4B settings):

1. root ``z`` and quaternion, then every other joint position;
2. root velocity (6), then every other joint velocity;
3. per muscle: excitation (``ctrl``) then activation (``act``);
4. foot touch sensors ``r_foot, r_toes, l_foot, l_toes``;
5. terrain heights on an 11 x 11 yaw-aligned grid around the pelvis, minus its height;
6. goal: reference ``qpos`` (without root ``x, y``) and ``qvel``, reference minus
   simulated root ``x, y``, reference minus simulated mimic-site offsets from the
   pelvis site, and every 10 frames up to 100 ahead the reference root-height change
   and root-relative ankle positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from myosuite.integrations.terra.actor import TERRAIN_GROUP
from myosuite.integrations.terra.reference import ReferenceMotion

MIMIC_SITES = (
    "pelvis_mimic",
    "upper_body_mimic",
    "head_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
)
TOUCH_SENSORS = ("r_foot", "r_toes", "l_foot", "l_toes")
ANKLE_SITES = ("left_ankle_mimic", "right_ankle_mimic")


@dataclass(frozen=True)
class TerraObsCfg:
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TerraObsCfg:
        """Settings saved in a checkpoint's ``config`` (raises on an unsupported layout)."""
        env = config["experiment"]["env_params"]
        goal = env["goal_params"]
        expected = {
            "goal_type": "TerraGoal",
            "use_egocentric_root_observations": False,
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
        return cls(
            sites=tuple(goal["sites_for_mimic"]),
            grid=(int(env["heightmap_grid_rows"]), int(env["heightmap_grid_cols"])),
            resolution=float(env["heightmap_grid_resolution"]),
            forward_offset=float(env.get("heightmap_grid_forward_offset", 0.0)),
            grid_body=str(env.get("heightmap_body_name") or "pelvis"),
            future_stride=int(goal["future_reference_stride"]),
            future_horizon=int(goal["future_reference_horizon"]),
        )


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


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


class TerraObservation:
    """Builds the TERRA policy observation for a model with the TERRA actor."""

    def __init__(self, model: mujoco.MjModel, cfg: TerraObsCfg | None = None) -> None:
        self.cfg = cfg or TerraObsCfg()
        self._model = model
        self._heights = TerrainHeights(model)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root != 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                "The TERRA actor's first joint must be the free joint 'root'"
            )
        self._sites = np.array([self._site(n) for n in self.cfg.sites])
        self._ankles = np.array([self._site(n) for n in ANKLE_SITES])
        touch = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in TOUCH_SENSORS
        ]
        if min(touch) < 0:
            raise ValueError(f"Touch sensors {TOUCH_SENSORS} are missing")
        self._touch = model.sensor_adr[touch]
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
        yaw = _yaw(data.qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        g = self._grid
        world = np.column_stack(
            [x + c * g[:, 0] - s * g[:, 1], y + s * g[:, 0] + c * g[:, 1]]
        )
        return self._heights(data, world) - z

    def goal(self, data: mujoco.MjData, ref: ReferenceMotion, frame: int) -> np.ndarray:
        """TerraGoal of reference *frame*."""
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
                    (ref.ankle_xpos[f] - ref.qpos[f, :3]).ravel(),
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

    def __call__(
        self, data: mujoco.MjData, ref: ReferenceMotion, frame: int
    ) -> np.ndarray:
        """Flat float32 observation of *data* tracking reference *frame*."""
        muscle = np.column_stack([data.ctrl, data.act]).ravel()
        obs = np.concatenate(
            [
                data.qpos[2:],
                data.qvel,
                muscle,
                data.sensordata[self._touch],
                self.heightmap(data),
                self.goal(data, ref, frame),
            ]
        )
        return obs.astype(np.float32)
