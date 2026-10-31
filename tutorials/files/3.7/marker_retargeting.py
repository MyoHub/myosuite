"""Two-link arm with closed-form joint angles, for tutorial 3.7 (marker retargeting).

The markers come from the closed-form forward kinematics, so the joint angles
that retarget recovers can be compared with the truth.

Headless check:
    python tutorials/files/3.7/marker_retargeting.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.core.retargeting import retarget

FPS = 100.0
UPPER, FOREARM = 0.3, 0.25  # segment lengths in metres
SHOULDER = np.array([0.0, 0.0, 1.0])  # fixed base

# Hinges about y: the arm swings in the x-z plane. Sites carry the marker names.
XML = f"""
<mujoco model="two-link arm">
  <compiler angle="radian"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".95 .95 1" rgb2=".6 .7 .85"
             width="32" height="32"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2.5" dir="0 .4 -1"/>
    <geom type="plane" size="1 1 .01" rgba=".85 .85 .85 1"/>
    <body name="upper" pos="{SHOULDER[0]} {SHOULDER[1]} {SHOULDER[2]}">
      <joint name="shoulder" axis="0 1 0" range="-2 2"/>
      <geom type="capsule" fromto="0 0 0 {UPPER} 0 0" size=".02" rgba=".8 .6 .4 1"/>
      <site name="elbow" pos="{UPPER} 0 0" size=".03" rgba="0 .4 1 1"/>
      <body name="forearm" pos="{UPPER} 0 0">
        <joint name="elbow" axis="0 1 0" range="-2.5 0"/>
        <geom type="capsule" fromto="0 0 0 {FOREARM} 0 0" size=".018" rgba=".8 .6 .4 1"/>
        <site name="wrist" pos="{FOREARM} 0 0" size=".03" rgba="0 .4 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def build_model() -> mujoco.MjModel:
    """Compile the two-link arm."""
    return mujoco.MjModel.from_xml_string(XML)


def closed_form_angles(seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Time stamps ``(T,)`` and shoulder and elbow angles ``(T, 2)`` in radians."""
    t = np.arange(int(seconds * FPS)) / FPS
    shoulder = -0.3 + 0.6 * np.sin(np.pi * t)
    elbow = -1.0 - 0.7 * np.sin(2 * np.pi * t)
    return t, np.stack([shoulder, elbow], axis=1)


def markers_from_angles(angles: np.ndarray) -> dict[str, np.ndarray]:
    """Elbow and wrist marker trajectories ``(T, 3)`` in metres, in closed form.

    A hinge about +y turns the x axis to ``(cos q, 0, -sin q)``.
    """
    q1, q12 = angles[:, 0], angles[:, 0] + angles[:, 1]
    zero = np.zeros_like(q1)
    elbow = SHOULDER + UPPER * np.stack([np.cos(q1), zero, -np.sin(q1)], axis=1)
    wrist = elbow + FOREARM * np.stack([np.cos(q12), zero, -np.sin(q12)], axis=1)
    return {"elbow": elbow, "wrist": wrist}


def replay(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Site trajectories ``(T, 3)`` of a replayed ``clip.qpos``."""
    data = mujoco.MjData(model)
    sites = {model.site(i).name: np.zeros((len(qpos), 3)) for i in range(model.nsite)}
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for name, track in sites.items():
            track[frame] = data.site(name).xpos
    return sites


def render(model: mujoco.MjModel, qpos: np.ndarray, every: int = 2) -> np.ndarray:
    """Frames ``(N, 480, 640, 3)`` of a replayed ``clip.qpos``, seen from the side."""
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = SHOULDER + [0.25, 0.0, -0.15]
    camera.distance, camera.azimuth, camera.elevation = 1.4, 90.0, -10.0
    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for pose in qpos[::every]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    return np.asarray(frames)


if __name__ == "__main__":
    model = build_model()
    _, angles = closed_form_angles()
    markers = markers_from_angles(angles)
    clip = retarget(model, markers, fps=FPS)
    replayed = replay(model, clip.qpos)
    angle_error = np.abs(clip.qpos - angles).max()
    replay_error = max(np.abs(replayed[n] - markers[n]).max() for n in markers)
    print(
        f"retarget ok: {len(clip.qpos)} frames, max angle error {angle_error:.1e} rad, "
        f"max replayed marker error {replay_error:.1e} m"
    )
