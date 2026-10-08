#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the repository root.
"""Export a CPU policy rollout to USD and render a Blender studio video."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def export_rollout(args: argparse.Namespace) -> dict[str, Any]:
    """Export one episode, preserving its control-step timing and task geometry."""
    import mujoco
    import numpy as np
    from mujoco.usd import objects, shapes
    from mujoco.usd.exporter import USDExporter
    from pxr import Tf, Usd, UsdGeom

    from myosuite import make_env
    from myosuite.utils.checkpoint_utils import (
        find_vec_normalize,
        load_sb3_model,
        load_vec_normalize,
        sb3_policy,
    )

    class LightTendon(objects.USDTendon):
        def generate_primitive_mesh(self) -> dict[str, Any]:
            parts = {}
            for config in self.mesh_config:
                name, mesh = shapes.mesh_factory(config, None, resolution=12)
                mesh.translate(-mesh.get_center())
                parts[name] = mesh
            return parts

    class NamedUSDExporter(USDExporter):
        def _get_geom_name(self, geom: Any) -> str:
            return Tf.MakeValidIdentifier(super()._get_geom_name(geom))

        def _load_geom(self, geom: Any) -> None:
            if geom.objtype != mujoco.mjtObj.mjOBJ_TENDON:
                super()._load_geom(geom)
                return
            name = self._get_geom_name(geom)
            config = shapes.mesh_config_generator(
                name, geom.type, np.ones(3), decouple=True
            )
            self.geom_refs[name] = LightTendon(
                config, self.stage, self.model, geom, name, rgba=geom.rgba
            )
            self.geom_names.add(name)

    env = make_env(args.env)
    try:
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        model, data = env.unwrapped.model, env.unwrapped.data
        if model.nskin or model.nflex:
            raise ValueError(
                "Skin/flex export is outside this minimal renderer's scope."
            )
        if args.checkpoint:
            checkpoint = args.checkpoint.absolute()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if checkpoint.suffix == ".zip":
                policy_model = load_sb3_model(checkpoint)
                if (
                    policy_model.observation_space.shape != env.observation_space.shape
                    or policy_model.action_space.shape != env.action_space.shape
                ):
                    raise ValueError(
                        "Checkpoint observation/action shapes do not match the environment."
                    )
                stats = find_vec_normalize(checkpoint)
                policy = sb3_policy(
                    policy_model, load_vec_normalize(stats) if stats else None
                )
            elif checkpoint.suffix == ".pt":
                from myosuite.utils.rslrl_policy import load_rslrl_policy

                actor = load_rslrl_policy(checkpoint, env.action_space.shape[0])

                def policy(observation: np.ndarray) -> np.ndarray:
                    return actor.act(np.atleast_2d(observation))[0]
            else:
                raise ValueError("Use an SB3 .zip or RSL-RL .pt checkpoint.")
        else:

            def policy(observation: np.ndarray) -> np.ndarray:
                return env.action_space.sample()

        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        exporter = NamedUSDExporter(
            model,
            width=1,
            height=1,
            output_directory="usd",
            output_directory_root=str(out),
            verbose=False,
        )
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[:3] = 1
        option.sitegroup[:] = 0
        option.tendongroup[:] = 1
        times, positions, rotations, qposes = [], [], [], []
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        start = float(data.time)
        done = False
        while True:
            mujoco.mj_forward(model, data)
            exporter.update_scene(data, scene_option=option)
            times.append(float(data.time) - start)
            positions.append(data.geom_xpos.copy())
            rotations.append(data.geom_xmat.copy())
            qposes.append(data.qpos.copy())
            for geom in exporter.scene.geoms[: exporter.scene.ngeom]:
                if geom.type != mujoco.mjtGeom.mjGEOM_PLANE and not (
                    geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and model.geom_bodyid[geom.objid] == 0
                ):
                    radius = float(np.linalg.norm(geom.size))
                    low = np.minimum(low, geom.pos - radius)
                    high = np.maximum(high, geom.pos + radius)
            if done or times[-1] >= args.seconds - 1e-9:
                break
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
        if len(times) < 2:
            raise ValueError("The episode contains no motion samples.")
        dt = float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times), dt):
            raise ValueError("Variable-step episodes are not supported.")
        UsdGeom.SetStageMetersPerUnit(exporter.stage, 1.0)
        exporter.stage.SetTimeCodesPerSecond(1 / dt)
        exporter.stage.SetFramesPerSecond(1 / dt)
        exporter.save_scene("usdc")
        usd = out / "usd" / "frames" / f"frame_{exporter.frame_count}.usdc"
        stage = Usd.Stage.Open(str(usd))
        stage.SetEndTimeCode(len(times) - 1)
        stage.GetRootLayer().Save()
        np.savez_compressed(
            out / "reference.npz",
            time=times,
            geom_xpos=positions,
            geom_xmat=rotations,
            qpos=qposes,
        )
        meta = {
            "env": args.env,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "motion_source": "checkpoint" if args.checkpoint else "random",
            "usd": str(usd),
            "dt": dt,
            "duration": times[-1],
            "samples": len(times),
            "fps": args.fps,
            "resolution": args.resolution,
            "preview": args.preview,
            "bounds": [low.tolist(), high.tolist()],
            "tendon_objects": sorted(
                name for name in exporter.geom_names if "_tendon" in name
            ),
            "mujoco_version": mujoco.__version__,
            "plane_objects": [
                exporter._get_geom_name(g)
                for g in exporter.scene.geoms[: exporter.scene.ngeom]
                if g.type == mujoco.mjtGeom.mjGEOM_PLANE
                or (
                    g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and model.geom_bodyid[g.objid] == 0
                )
            ],
        }
        frames = max(1, math.ceil(times[-1] * args.fps - 1e-9))
        meta["visibility"] = {
            prim.GetName(): [
                UsdGeom.Imageable(prim).ComputeVisibility(f / args.fps / dt)
                != "invisible"
                for f in range(frames)
            ]
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Xform"
            and prim.GetParent().GetName() == "World"
            and prim.GetName().startswith("Mesh_Xform_")
        }
        (out / "render.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    finally:
        env.close()


def build_blender_scene(output: Path) -> None:
    """Import the exported animation and render a fixed studio composition."""
    import bpy
    from mathutils import Vector

    meta = json.loads((output / "render.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = meta["fps"]
    bpy.ops.wm.usd_import(
        filepath=str(output / "usd" / "frames" / Path(meta["usd"]).name),
        import_cameras=False,
        import_lights=False,
        set_frame_range=False,
    )
    for cache in bpy.data.cache_files:
        cache.override_frame = True
        cache.driver_add(
            "frame"
        ).driver.expression = f"frame / {meta['fps'] * meta['dt']}"
    for root_name, visible in meta.get("visibility", {}).items():
        root = bpy.data.objects.get(root_name)
        if root is not None:
            for obj in [root, *root.children_recursive]:
                if obj.type != "MESH":
                    continue
                for frame, shown in enumerate(visible):
                    obj.hide_render = not shown
                    obj.keyframe_insert("hide_render", frame=frame)
    scene.frame_start = 0
    scene.frame_end = max(0, math.ceil(meta["duration"] * meta["fps"] - 1e-9) - 1)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16 if meta["preview"] else 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = meta["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    bpy.context.preferences.filepaths.save_version = 0
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.6
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.22,
        0.24,
        0.28,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35

    def material(name: str, color: tuple[float, ...], roughness: float) -> Any:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = roughness
        return mat

    tendon = material("Muscle paths", (0.42, 0.025, 0.04, 1), 0.4)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            if "_tendon" in obj.name:
                obj.data.materials.clear()
                obj.data.materials.append(tendon)
    low, high = (Vector(v) for v in meta["bounds"])
    center = (low + high) / 2
    span = max((high - low).length, 0.3)
    aspect = scene.render.resolution_x / scene.render.resolution_y
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Studio camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 1.15 / min(1, aspect)
    camera.location = center + Vector((1.3, -1.7, 1.0)).normalized() * span * 3
    camera.rotation_euler = (
        (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.clip_end = max(100, span * 20)
    scene.camera = camera
    anatomy = [
        o
        for o in scene.objects
        if o.type == "MESH"
        and not any(name in o.name for name in meta.get("plane_objects", []))
    ]
    view_points = []
    rotation = camera.rotation_euler.to_matrix()
    for frame in sorted({scene.frame_start, scene.frame_end // 2, scene.frame_end}):
        scene.frame_set(frame)
        view_points.extend(
            rotation.transposed() @ (o.matrix_world @ Vector(corner))
            for o in anatomy
            if not o.hide_render
            for corner in o.bound_box
        )
    if view_points:
        vlow = Vector(tuple(min(p[i] for p in view_points) for i in range(3)))
        vhigh = Vector(tuple(max(p[i] for p in view_points) for i in range(3)))
        target = rotation @ ((vlow + vhigh) / 2)
        camera.location += target - center
        camera.data.ortho_scale = (
            max(vhigh.y - vlow.y, (vhigh.x - vlow.x) / aspect) * 1.2
        )
    for name, direction, power in [
        ("Key", (1, -1, 2), 500),
        ("Fill", (-1, -0.5, 1), 200),
        ("Rim", (0, 1, 1.5), 400),
    ]:
        bpy.ops.object.light_add(
            type="AREA", location=center + Vector(direction) * span
        )
        light = bpy.context.object
        light.name = name
        light.data.energy = power * span**2
        light.data.shape = "DISK"
        light.data.size = span * 1.2
        light.rotation_euler = (
            (center - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    bpy.ops.mesh.primitive_plane_add(
        size=span * 200, location=(center.x, center.y, low.z - span * 0.01)
    )
    bpy.context.object.name = "Studio floor"
    bpy.context.object.data.materials.append(
        material("Floor", (0.12, 0.14, 0.17, 1), 0.85)
    )
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(output / "frames" / "frame_")
    for cache in bpy.data.cache_files:
        cache.filepath = bpy.path.relpath(cache.filepath, start=str(output))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), compress=True)
    if meta["preview"]:
        scene.render.filepath = str(output / "preview.png")
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


def main() -> None:
    """Export in ordinary Python; construct/render in a separate Blender process."""
    if "--blender-stage" in sys.argv:
        build_blender_scene(Path(sys.argv[sys.argv.index("--blender-stage") + 1]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--random", action="store_true", help="Explicit random-action transport demo."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.fps <= 0
        or min(args.resolution) <= 0
    ):
        parser.error("Seconds, FPS and resolution must be positive.")
    if (args.output / "render.json").exists():
        parser.error("Choose a new output directory to avoid mixing runs.")
    meta = export_rollout(args)
    print(
        f"Exported {meta['samples']} samples, {meta['duration']:.3f}s to {meta['usd']}",
        flush=True,
    )
    if args.export_only:
        return
    subprocess.run(
        [
            args.blender,
            "--background",
            "--threads",
            "4",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--blender-stage",
            str(args.output.resolve()),
        ],
        check=True,
    )
    if not args.preview:
        import imageio.v3 as iio

        from myosuite.utils.video_io import write_video

        frames = sorted((args.output / "frames").glob("frame_*.png"))
        write_video(
            args.output / "video.mp4",
            [iio.imread(f)[..., :3] for f in frames],
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
