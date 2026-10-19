#!/usr/bin/env python3
"""Render the complete planned MyoFullBody ADL reference in the apartment."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

from myosuite.integrations.musclemimic import (
    GaitParameters,
    MIMIC_SITES,
    TerrainHeights,
    build_terrain_fullbody_spec,
    plan_waypoint_reference,
)
from myosuite.integrations.musclemimic.adl_apartment import adl_apartment_scene, adl_route


def main() -> None:
    out = Path(os.environ.get("TERRA_ADL_OUT", "terra_adl_outputs"))
    out.mkdir(parents=True, exist_ok=True)
    route = adl_route()
    model = build_terrain_fullbody_spec(adl_apartment_scene).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    heights = TerrainHeights(model, top=12.0)
    qpos0 = model.qpos0.copy()
    qpos0[:2] = route[0]
    reference, planner = plan_waypoint_reference(
        model,
        route,
        lambda xy: float(heights(data, [xy])[0]),
        qpos0,
        sites=MIMIC_SITES,
        parameters=GaitParameters(),
    )

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    option.geomgroup[3] = 1
    option.sitegroup[:] = 0
    option.tendongroup[:] = 1

    frames = []
    stride = 10
    with mujoco.Renderer(model, 540, 960) as renderer:
        for step in range(0, reference.num_frames, stride):
            data.qpos[:] = reference.qpos[step]
            if reference.qvel is not None:
                data.qvel[:] = reference.qvel[step]
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] += 0.35
            camera.distance = 3.2
            camera.azimuth = 150
            camera.elevation = -12
            renderer.update_scene(data, camera=camera, scene_option=option)
            frames.append(renderer.render().copy())

    video = out / "myofullbody_adl_reference.mp4"
    preview = out / "myofullbody_adl_reference_preview.jpg"
    imageio.mimsave(video, frames, fps=10, macro_block_size=1)
    imageio.imwrite(preview, frames[len(frames) // 2], quality=94)
    result = {
        "kind": "planned_reference_not_policy_rollout",
        "frames": reference.num_frames,
        "duration_s": planner["duration_s"],
        "route_points": len(route),
        "video": str(video),
        "preview": str(preview),
    }
    (out / "reference_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
