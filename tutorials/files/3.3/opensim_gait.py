# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Map OpenSim's gait2392 treadmill walk (``files/3.5/subject01_walk1_ik.mot``) onto MyoLeg.

Shared by tutorials 3.3 (inverse dynamics, static optimization) and 3.4 (computed
muscle control).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

# OpenSim ground axes (X forward, Y up, Z right) in MuJoCo world coordinates: MyoLeg
# walks along -Y with Z up, and its pelvis frame at the default pose is OpenSim's.
_OPENSIM_TO_WORLD = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def read_mot(path: str | Path) -> pd.DataFrame:
    """Read an OpenSim ``.mot`` / ``.sto`` table (header up to ``endheader``)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    skip = next(i for i, line in enumerate(lines) if line.strip() == "endheader") + 1
    return pd.read_csv(path, sep="\t", skiprows=skip)


def _foot_floor_distance(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """``(T, 2)`` distance [m] of the right and left foot collision geoms to the floor."""
    floor = model.geom("floor").id
    feet = [
        [
            g
            for g in range(model.ngeom)
            if model.geom_contype[g]
            and model.body(model.geom_bodyid[g]).name in (f"calcn_{s}", f"toes_{s}")
        ]
        for s in ("r", "l")
    ]
    data, dist = mujoco.MjData(model), np.empty((len(qpos), 2))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for k, geoms in enumerate(feet):
            dist[t, k] = min(
                mujoco.mj_geomDistance(model, data, floor, g, 1.0, None) for g in geoms
            )
    return dist


def treadmill_walk_to_qpos(
    model: mujoco.MjModel,
    mot: pd.DataFrame,
    rate: float = 100.0,
    penetration: float = 0.01,
) -> tuple[np.ndarray, float]:
    """MyoLeg coordinates of an OpenSim gait2392 walk, resampled and made overground.

    Pelvis angles (body-fixed Z-X-Y: tilt, list, rotation) and translations become
    the free root, the leg angles map by name (MyoLeg's knee flexes positive) and the
    coupled knee joints follow their polynomials. The
    trial was recorded on a treadmill: adding the belt speed, estimated from the
    stance foot, gives the same dynamics overground. The model then moves down
    until its lowest foot point is *penetration* into the floor.

    Args:
        model: MyoLeg model (``myoLegWalk-v0``).
        mot: Table from :func:`read_mot`, angles in degrees.
        rate: Output frame rate [Hz]; a whole number of physics steps per frame.
        penetration: Depth of the lowest foot point below the floor [m].

    Returns:
        ``(qpos, belt_speed)``: ``(T, nq)`` positions at *rate* and the belt speed [m/s].
    """
    t = mot["time"].to_numpy() - mot["time"].iloc[0]
    tt = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate)

    def col(name: str) -> np.ndarray:
        return CubicSpline(t, mot[name].to_numpy())(tt)

    g = _OPENSIM_TO_WORLD
    qpos = np.tile(model.qpos0, (len(tt), 1))
    pelvis = Rotation.from_euler(
        "ZXY", np.deg2rad(np.c_[col("pelvis_tilt"), col("pelvis_list"), col("pelvis_rotation")])
    ).as_matrix()
    root0 = Rotation.from_quat(model.qpos0[3:7], scalar_first=True).as_matrix()
    qpos[:, 3:7] = Rotation.from_matrix(g @ pelvis @ g.T @ root0).as_quat(scalar_first=True)
    qpos[:, :3] = np.c_[col("pelvis_tx"), col("pelvis_ty"), col("pelvis_tz")] @ g.T
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in mot.columns:
            sign = -1.0 if name.startswith("knee_angle") else 1.0
            qpos[:, model.jnt_qposadr[j]] = sign * np.deg2rad(col(name))
    project_joint_couplings(JointCouplings.from_model(model), qpos)  # knee couplings

    # Belt speed: the right heel moves backwards at it while the foot is on the belt.
    dist = _foot_floor_distance(model, qpos)
    data, heel = mujoco.MjData(model), np.empty((len(qpos), 3))
    for k, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        heel[k] = data.xpos[model.body("calcn_r").id]
    stance = dist[:, 0] < dist[:, 0].min() + 0.015
    belt = -np.median(np.gradient(heel, 1.0 / rate, axis=0)[stance, :2], axis=0)
    qpos[:, :2] += tt[:, None] * belt - qpos[0, :2]
    qpos[:, 2] -= dist.min() + penetration
    return qpos, float(np.linalg.norm(belt))


def render_walk(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    activation: np.ndarray | None = None,
    every: int = 3,
    size: tuple[int, int] = (480, 360),
) -> np.ndarray:
    """Frames of a side view that follows the pelvis; muscles redden with *activation*."""
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    data, cam = mujoco.MjData(model), mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.6, 0.0, -10.0
    frames = []
    for t in range(0, len(qpos), every):
        data.qpos[:] = qpos[t]
        if activation is not None:
            data.act[:] = activation[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    return np.asarray(frames)
