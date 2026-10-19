#!/usr/bin/env python3
"""Run the tutorial TERRA-4B pipeline through the furnished ADL apartment."""
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
    TerrainController,
    TerrainHeights,
    build_terrain_fullbody_spec,
    download_terrain_checkpoint,
    load_terrain_policy,
    plan_waypoint_reference,
)
from myosuite.integrations.musclemimic.adl_apartment import adl_apartment_scene, adl_route
from myosuite.terms.waypoint import WaypointRewardCfg
from myosuite.viz.ghost_body_viz import GhostBodyViz


def main() -> None:
    seed = int(os.environ.get("TERRA_SEED", "0"))
    stochastic = os.environ.get("TERRA_STOCHASTIC", "1") != "0"
    out = Path(os.environ.get("TERRA_ADL_OUT", "terra_adl_outputs"))
    out.mkdir(parents=True, exist_ok=True)

    route = adl_route()
    model = build_terrain_fullbody_spec(adl_apartment_scene).compile()
    probe = mujoco.MjData(model)
    mujoco.mj_forward(model, probe)
    heights = TerrainHeights(model, top=12.0)
    height = lambda xy: float(heights(probe, [xy])[0])

    q0 = model.qpos0.copy()
    q0[:2] = route[0]
    gait = GaitParameters(speed=0.25, cycle_s=1.2, clearance=0.12)
    reference, planner = plan_waypoint_reference(
        model, route, height, q0, sites=MIMIC_SITES, parameters=gait
    )

    checkpoint = Path(os.environ["TERRA_CHECKPOINT"]) if "TERRA_CHECKPOINT" in os.environ else download_terrain_checkpoint()
    policy = load_terrain_policy(checkpoint, model, reference, stochastic=stochastic, seed=seed)
    controller = TerrainController(policy, model, reference, frame_skip=5)
    controller.reset()

    task = WaypointTaskCfg(
        site_name="pelvis_mimic",
        waypoints=tuple(map(tuple, route[1:])),
        arrival_radius=0.65,
        min_site_height=0.18,
        reward=WaypointRewardCfg(),
    )
    env = make_env(
        "myoFullBodyWaypoint-v0", model=model, task=task, initial_qpos=q0,
        initial_qvel=np.zeros(model.nv), frame_skip=5, render_mode="rgb_array",
    )
    env.reset(seed=seed)
    base = env.unwrapped

    # Rendering is intentionally independent of locomotion. Group 0 contains the
    # actor's collision/wrapping surfaces, so hide it and show the anatomical
    # meshes (group 1), traversable apartment (2) and furniture (3).
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1] = option.geomgroup[2] = option.geomgroup[3] = 1
    option.sitegroup[:] = 0
    option.tendongroup[:] = 1
    # A simple capsule/joint reference is more legible than translucent duplicate
    # anatomy and avoids mesh-ID/render-order ambiguity.
    ghost = GhostBodyViz.from_motion_clip(
        reference, model, bone_meshes=False,
        skeleton_rgba=(0.95, 0.95, 0.95, 0.55),
        joint_rgba=(1.0, 1.0, 1.0, 0.72),
        capsule_radius=0.012, joint_radius=0.017,
    )

    frames = []
    info = {"next_waypoint": 0, "solved": False, "failed": False}
    min_h = float("inf")
    max_steps = min(reference.num_frames + 300, 20000)
    with mujoco.Renderer(model, 360, 640, max_geom=model.ngeom + 1024) as renderer:
        for step in range(max_steps):
            action = controller(base.data)
            _, _, terminated, truncated, info = base.step(action)
            min_h = min(min_h, float(base.data.site("pelvis_mimic").xpos[2]))

            if step % 10 == 0:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = base.data.qpos[:3]
                cam.lookat[2] += 0.35
                cam.distance = 3.2
                cam.azimuth = 150
                cam.elevation = -12
                renderer.update_scene(base.data, camera=cam, scene_option=option)
                ghost.draw(min(controller._frame + 50, reference.num_frames - 1), renderer.scene)
                frames.append(renderer.render().copy())

            if info["solved"] or terminated or truncated:
                break

    final = base.data.qpos[:3].copy()
    env.close()
    video = out / "terra_adl_apartment.mp4"
    preview = out / "terra_adl_apartment_preview.jpg"
    imageio.mimsave(video, frames, fps=10, macro_block_size=1)
    imageio.imwrite(preview, frames[len(frames) // 2], quality=92)
    result = {
        "seed": seed, "controller": "terra_tutorial_continuous_reference",
        "stochastic": stochastic, "reference_frames": reference.num_frames,
        "reference_duration_s": planner["duration_s"], "executed_steps": step + 1,
        "waypoints_reached": int(info["next_waypoint"]), "waypoints_total": len(route) - 1,
        "solved": bool(info["solved"]), "failed": bool(info["failed"]),
        "min_pelvis_height": min_h, "final_root_xyz": final.tolist(),
        "video": str(video), "preview": str(preview),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
