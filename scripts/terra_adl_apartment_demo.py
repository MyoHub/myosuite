#!/usr/bin/env python3
"""Run TERRA-4B through a furnished split-level ADL apartment and render it."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from myosuite import make_env
from myosuite.envs.waypoint import WaypointTaskCfg
from myosuite.integrations.musclemimic import (
    GaitParameters,
    MIMIC_SITES,
    TerrainHeights,
    build_terrain_fullbody_spec,
    download_terrain_checkpoint,
    load_terrain_policy,
    plan_waypoint_reference,
)
from myosuite.integrations.musclemimic.adl_apartment import adl_apartment_scene, adl_route
from myosuite.terms.waypoint import WaypointRewardCfg
from myosuite.viz.bone_ghost_viz import BoneGhostBodyViz


def main() -> None:
    seed = int(os.environ.get("TERRA_SEED", "3"))
    stochastic = os.environ.get("TERRA_STOCHASTIC", "1") != "0"
    out = Path(os.environ.get("TERRA_ADL_OUT", "terra_adl_outputs"))
    out.mkdir(parents=True, exist_ok=True)

    route = adl_route()
    model = build_terrain_fullbody_spec(adl_apartment_scene).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    heights = TerrainHeights(model, top=12.0)

    initial_qpos = model.qpos0.copy()
    initial_qpos[:2] = route[0]
    direction = route[1] - route[0]
    spawn_yaw = float(np.arctan2(direction[0], -direction[1]))
    initial_qpos[3:7] = [np.cos(spawn_yaw / 2), 0.0, 0.0, np.sin(spawn_yaw / 2)]
    reference, planner = plan_waypoint_reference(
        model,
        route,
        lambda xy: float(heights(data, [xy])[0]),
        initial_qpos,
        sites=MIMIC_SITES,
        parameters=GaitParameters(speed=0.10, cycle_s=1.2, clearance=0.12),
    )

    checkpoint = (
        Path(os.environ["TERRA_CHECKPOINT"])
        if "TERRA_CHECKPOINT" in os.environ
        else download_terrain_checkpoint()
    )
    policy = load_terrain_policy(
        checkpoint, model, reference, stochastic=stochastic, seed=seed
    )
    task = WaypointTaskCfg(
        site_name="pelvis_mimic",
        waypoints=tuple(map(tuple, route[1:])),
        arrival_radius=0.55,
        min_site_height=0.30,
        reward=WaypointRewardCfg(),
    )
    env = make_env(
        "myoFullBodyWaypoint-v0",
        model=model,
        task=task,
        initial_qpos=initial_qpos,
        initial_qvel=np.zeros(model.nv),
        frame_skip=5,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    base = env.unwrapped

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    option.geomgroup[3] = 1
    option.sitegroup[:] = 0
    option.tendongroup[:] = 1
    ghost = BoneGhostBodyViz.from_motion_clip(
        reference, model, bone_meshes=True, skeleton_rgba=(0.20, 0.70, 1.0, 0.32)
    )

    frames: list[np.ndarray] = []
    render_stride = 10
    info: dict = {"next_waypoint": 0, "solved": False, "failed": False}
    min_pelvis_height = float("inf")
    with mujoco.Renderer(model, 540, 960, max_geom=model.ngeom + 512) as renderer:
        for step in range(reference.num_frames - 1):
            _, _, terminated, truncated, info = base.step(
                policy.action_for(base.data, reference, step)
            )
            pelvis_height = float(base.data.site("pelvis_mimic").xpos[2])
            min_pelvis_height = min(min_pelvis_height, pelvis_height)
            if step % render_stride == 0:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = base.data.qpos[:3]
                camera.lookat[2] += 0.35
                camera.distance = 3.2
                camera.azimuth = 150
                camera.elevation = -12
                renderer.update_scene(base.data, camera=camera, scene_option=option)
                ghost.draw(min(step + 80, reference.num_frames - 1), renderer.scene)
                frames.append(renderer.render().copy())
            if terminated or truncated:
                break

    final_root = base.data.qpos[:3].copy()
    env.close()
    video = out / "terra_adl_apartment.mp4"
    preview = out / "terra_adl_apartment_preview.jpg"
    imageio.mimsave(video, frames, fps=10, macro_block_size=1)
    imageio.imwrite(preview, frames[len(frames) // 2], quality=92)
    result = {
        "seed": seed,
        "stochastic": stochastic,
        "reference_frames": reference.num_frames,
        "reference_duration_s": planner["duration_s"],
        "executed_steps": step + 1,
        "waypoints_reached": int(info["next_waypoint"]),
        "waypoints_total": len(route) - 1,
        "solved": bool(info["solved"]),
        "failed": bool(info["failed"]),
        "min_pelvis_height": min_pelvis_height,
        "final_root_xyz": final_root.tolist(),
        "video": str(video),
        "preview": str(preview),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
