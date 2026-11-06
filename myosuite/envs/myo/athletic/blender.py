# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Athletic rollout rendered in Blender Cycles with a studio light rig.

MuJoCo's headlight is one camera lamp on a gray world, so the body reads flat.
This rig keeps the world dark and models the figure with five area lights: a
large warm key, a larger cool fill, a narrow rim, a floor bounce, and a hair
light. The rig is parented to the pelvis, so a sprint stays lit as it moves.

The module imports only the standard library at import time. Blender runs
:func:`build_blender_scene` with its own ``bpy``. The host records the rollout
and launches ``blender --background``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

LightingName = Literal["studio"]
DEFAULT_LIGHTING: LightingName = "studio"


@dataclass(frozen=True)
class AreaLight:
    """One studio lamp, placed in camera right / camera forward / world up.

    ``offset`` is in subject spans (1 = the body diagonal). Negative forward
    sits the lamp on the camera side of the body. ``power`` is watts per
    square metre before it is scaled by ``span²``. ``size`` is the lamp
    diameter, again in spans: a large key is soft, a small rim stays crisp.
    """

    name: str
    offset: tuple[float, float, float]
    power: float
    color: tuple[float, float, float]
    size: float
    shape: Literal["RECTANGLE", "DISK"] = "RECTANGLE"


# Dark world, soft directional key. A gray world at strength 0.25 (or a
# headlight) fills every shadow and the muscles stop reading.
STUDIO_LIGHTS: tuple[AreaLight, ...] = (
    AreaLight("Key", (-1.15, -1.45, 1.55), 85.0, (1.0, 0.93, 0.84), 1.8),
    AreaLight("Fill", (1.6, -1.2, 0.45), 20.0, (0.75, 0.84, 1.0), 2.4),
    AreaLight("Rim", (0.9, 1.65, 1.35), 120.0, (1.0, 0.96, 0.9), 0.32, "DISK"),
    AreaLight("Bounce", (0.15, -0.85, -0.55), 14.0, (1.0, 0.88, 0.74), 3.0),
    AreaLight("Hair", (-0.15, 0.35, 2.15), 42.0, (0.94, 0.96, 1.0), 0.4, "DISK"),
)
WORLD_COLOR = (0.012, 0.014, 0.018, 1.0)
WORLD_STRENGTH = 0.035
EXPOSURE = 0.2
SAMPLES = 128


def lighting_spec(name: LightingName = DEFAULT_LIGHTING) -> dict[str, Any]:
    """JSON-ready description of a named rig. Only ``studio`` exists."""
    if name != "studio":
        raise ValueError(f"unknown lighting {name!r}; expected 'studio'")
    return {
        "name": name,
        "lights": [asdict(light) for light in STUDIO_LIGHTS],
        "world_color": WORLD_COLOR,
        "world_strength": WORLD_STRENGTH,
        "exposure": EXPOSURE,
        "view_transform": "AgX",
        "look": "Medium High Contrast",
    }


# Athletic oval: rubber ring, white lane lines, grass infield. Same colours as the
# MuJoCo sprint track. Compact enough that a still frame shows the whole ring
# and the runner standing in the inside lane.
RING_LANE_WIDTH = 1.22
RING_LANES = 8
RING_INNER_RADIUS = 6.5
RING_STRAIGHT = 12.0
RING_LINE_WIDTH = 0.05
RUBBER_COLOR = (0.62, 0.24, 0.15, 1.0)
LINE_COLOR = (0.95, 0.95, 0.95, 1.0)
GRASS_COLOR = (0.22, 0.34, 0.18, 1.0)


def running_ring_paths(
    pelvis: tuple[float, float, float],
    up_axis: str = "z",
) -> dict[str, Any]:
    """World-space polylines for the oval, one per lane boundary.

    The runner stands in the inside lane of the home straight. Fullbody runs
    toward −X (Z up); the sprinter runs toward +X (Y up). The infield sits on
    the far side of that straight.
    """
    half = RING_STRAIGHT / 2
    runner_local = (half - 1.5, -(RING_INNER_RADIUS + 0.5 * RING_LANE_WIDTH))
    sign = 1.0 if up_axis == "z" else -1.0
    px, py, pz = pelvis
    side0 = py if up_axis == "z" else pz

    def world(x: float, y: float, z: float) -> tuple[float, float, float]:
        wx = px + sign * (x - runner_local[0])
        ws = side0 + (y - runner_local[1])
        if up_axis == "z":
            return (wx, ws, z)
        return (wx, z, ws)

    def samples(radius: float) -> list[tuple[float, float]]:
        n_straight, n_bend = 28, 40
        pts: list[tuple[float, float]] = []
        for i in range(n_straight):
            t = i / n_straight
            pts.append((half - t * RING_STRAIGHT, -radius))
        for i in range(n_bend):
            ang = -math.pi / 2 - (i / n_bend) * math.pi
            pts.append((-half + radius * math.cos(ang), radius * math.sin(ang)))
        for i in range(n_straight):
            t = i / n_straight
            pts.append((-half + t * RING_STRAIGHT, radius))
        for i in range(n_bend):
            ang = math.pi / 2 - (i / n_bend) * math.pi
            pts.append((half + radius * math.cos(ang), radius * math.sin(ang)))
        return pts

    def outward(pts: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
        framed = []
        n = len(pts)
        for i, (x, y) in enumerate(pts):
            x0, y0 = pts[(i - 1) % n]
            x2, y2 = pts[(i + 1) % n]
            tx, ty = x2 - x0, y2 - y0
            nx, ny = -ty, tx
            if nx * x + ny * y < 0:
                nx, ny = -nx, -ny
            length = math.hypot(nx, ny) or 1.0
            framed.append((x, y, nx / length, ny / length))
        return framed

    lines = []
    for k in range(RING_LANES + 1):
        radius = RING_INNER_RADIUS + k * RING_LANE_WIDTH
        framed = outward(samples(radius))
        left, right = [], []
        for x, y, nx, ny in framed:
            left.append(world(x - nx * RING_LINE_WIDTH / 2, y - ny * RING_LINE_WIDTH / 2, 0.012))
            right.append(world(x + nx * RING_LINE_WIDTH / 2, y + ny * RING_LINE_WIDTH / 2, 0.012))
        lines.append((left, right))
    inner = [world(x, y, 0.004) for x, y in samples(RING_INNER_RADIUS)]
    outer = [world(x, y, 0.004) for x, y in samples(RING_INNER_RADIUS + RING_LANES * RING_LANE_WIDTH)]
    center = world(0.0, 0.0, 1.1)
    return {
        "lines": lines,
        "rubber": (inner, outer),
        "center": center,
        "n_lanes": RING_LANES,
        "ground": 0.0,
    }


def _camera_offsets(up_axis: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Three-quarter camera and look offsets in the plant's world frame."""
    if up_axis == "y":
        # Sprinter faces +X. Sit in front and to the side, above the pelvis.
        return (2.2, 1.6, 5.6), (0.4, 0.2, 0.0)
    # Fullbody faces −X, Z up. Far enough for a 50 mm full-body frame.
    return (-2.0, -5.6, 1.5), (-0.2, 0.0, 0.15)


def _safe_name(name: str, gid: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return cleaned or f"geom_{gid}"


def _primitive(kind: str, size: Any) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Low-poly geom in its local frame. ``size`` is MuJoCo's geom size."""
    import numpy as np

    def sphere(radius: float, n: int = 10) -> tuple[list, list]:
        verts = []
        for i in range(n + 1):
            phi = math.pi * i / n
            for j in range(n):
                theta = 2 * math.pi * j / n
                verts.append(
                    (
                        radius * math.sin(phi) * math.cos(theta),
                        radius * math.sin(phi) * math.sin(theta),
                        radius * math.cos(phi),
                    )
                )
        faces = []
        for i in range(n):
            for j in range(n):
                a = i * n + j
                b = a + n
                c = i * n + (j + 1) % n
                d = b + 1 if (j + 1) % n else b - n + 1
                faces.append((a, c, b))
                faces.append((c, d, b))
        return verts, faces

    radius = float(size[0])
    if kind == "sphere":
        return sphere(radius)
    if kind == "box":
        x, y, z = (float(v) for v in size[:3])
        corners = [(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        # Faces as two triangles, outward-ish.
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        faces = [(*q[:3],) for q in quads] + [(q[0], q[2], q[3]) for q in quads]
        return corners, faces
    half = float(size[1]) if len(size) > 1 else radius
    # Capsule / cylinder along local z.
    n = 12
    rings = []
    if kind == "capsule":
        for i in range(6):
            phi = math.pi * i / 5
            z = math.cos(phi) * radius + (half if i < 3 else -half)
            r = abs(math.sin(phi) * radius)
            rings.append((z, r))
    else:
        rings = [(half, radius), (-half, radius)]
    verts = []
    for z, r in rings:
        for j in range(n):
            theta = 2 * math.pi * j / n
            verts.append((r * math.cos(theta), r * math.sin(theta), z))
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            c = a + n
            d = b + n
            faces.append((a, b, c))
            faces.append((b, d, c))
    return verts, faces


def _geom_surface(model: Any, gid: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    """Mesh or primitive for one geom, or ``None`` when the body already has a mesh."""
    import mujoco
    import numpy as np

    kind = int(model.geom_type[gid])
    if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            return None
        v0 = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        f0 = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        if vn == 0 or fn == 0:
            return None
        verts = [(float(x), float(y), float(z)) for x, y, z in np.asarray(model.mesh_vert[v0 : v0 + vn])]
        faces = [tuple(int(i) for i in face) for face in np.asarray(model.mesh_face[f0 : f0 + fn])]
        return verts, faces
    names = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    }
    label = names.get(kind)
    if label is None:
        return None
    return _primitive(label, np.asarray(model.geom_size[gid]))


def _bodies_with_mesh(model: Any) -> set[int]:
    import mujoco

    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    return {int(model.geom_bodyid[g]) for g in range(model.ngeom) if int(model.geom_type[g]) == mesh}


def write_stage(
    model: Any,
    snapshots: list[tuple[Any, Any, Any]],
    output: Path,
    *,
    up_axis: str,
    fps: int,
    samples: int = SAMPLES,
    preview: bool = False,
    resolution: tuple[int, int] = (1280, 720),
    lighting: LightingName = DEFAULT_LIGHTING,
) -> dict[str, Any]:
    """Write one OBJ plus per-frame geom matrices for the Blender stage.

    Each snapshot is ``(geom_xpos, geom_xmat, pelvis_xyz)`` after ``mj_forward``.
    Mesh geoms win over collision primitives on the same body.
    """
    import mujoco
    import numpy as np

    if not snapshots:
        raise ValueError("A Blender stage needs at least one posed frame.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "render.json").exists():
        raise FileExistsError(f"{output} already has a render. Choose a new directory.")

    mesh_bodies = _bodies_with_mesh(model)
    mesh_kind = int(mujoco.mjtGeom.mjGEOM_MESH)
    chosen: list[tuple[int, str, list, list]] = []
    for gid in range(int(model.ngeom)):
        # Groups 0–2 are the visible skeleton. Group 3+ holds muscle-wrap
        # spheres and collision capsules, which are not part of the render.
        if int(model.geom_group[gid]) > 2:
            continue
        kind = int(model.geom_type[gid])
        body = int(model.geom_bodyid[gid])
        if kind != mesh_kind and body in mesh_bodies:
            continue
        surface = _geom_surface(model, gid)
        if surface is None:
            continue
        verts, faces = surface
        chosen.append((gid, _safe_name(model.geom(gid).name, gid), verts, faces))

    lines = []
    vertex_base = 1
    for _gid, name, verts, faces in chosen:
        lines.append(f"o {name}")
        for x, y, z in verts:
            lines.append(f"v {x:.6g} {y:.6g} {z:.6g}")
        for a, b, c in faces:
            lines.append(f"f {a + vertex_base} {b + vertex_base} {c + vertex_base}")
        vertex_base += len(verts)
    (output / "geoms.obj").write_text("\n".join(lines) + "\n")

    frames = []
    pelvis = []
    for xpos, xmat, root in snapshots:
        mats = []
        for gid, _name, _verts, _faces in chosen:
            # mesh_vert is already in the geom frame. geom_size on a compiled
            # mesh is the half-extent, not a second scale.
            rotation = np.asarray(xmat[gid], dtype=float).reshape(3, 3)
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = np.asarray(xpos[gid], dtype=float)
            mats.append(matrix.reshape(-1).tolist())
        frames.append(mats)
        pelvis.append([float(v) for v in np.asarray(root).reshape(3)])

    camera, look = _camera_offsets(up_axis)
    meta = {
        "geoms": [name for _gid, name, _verts, _faces in chosen],
        "frames": frames,
        "pelvis": pelvis,
        "fps": int(fps),
        "samples": int(samples),
        "preview": bool(preview),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "camera_offset": list(camera),
        "look_offset": list(look),
        "lighting": lighting_spec(lighting),
        "track": {
            "kind": "ring",
            "n_lanes": RING_LANES,
            "lane_width": RING_LANE_WIDTH,
            "inner_radius": RING_INNER_RADIUS,
            "straight": RING_STRAIGHT,
            "up_axis": up_axis,
        },
    }
    (output / "render.json").write_text(json.dumps(meta))
    return meta


def render_athletic(
    task: str,
    output: str | Path,
    *,
    model: str | None = None,
    tendons: str = "rigid",
    plant: str | None = None,
    lighting: LightingName = DEFAULT_LIGHTING,
    seconds: float = 2.0,
    fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    samples: int = SAMPLES,
    preview: bool = False,
    seed: int = 0,
    blender: str = "blender",
    export_only: bool = False,
) -> Path:
    """Roll an athletic episode and render it with the studio rig.

    ``model`` defaults to ``myofullbody`` and ``tendons`` to ``rigid``, the
    same defaults as :func:`~myosuite.envs.myo.athletic.catalog.make_athletic`.
    ``preview`` renders the reset pose only. Returns the output directory.
    """
    from myosuite.envs.myo.athletic.catalog import athletic_bundle
    from myosuite.envs.myo.athletic.specs import ensure_tendons_applied

    if seconds <= 0 or fps <= 0 or min(resolution) <= 0 or samples <= 0:
        raise ValueError("seconds, fps, resolution and samples must be positive")
    bundle = athletic_bundle(task, model, tendons, plant=plant)  # type: ignore[arg-type]
    ensure_tendons_applied(bundle)
    lighting_spec(lighting)

    import numpy as np

    from myosuite.envs.myo.athletic.factory import (
        AthleticModularTaskEnv,
        task_config_from_bundle,
    )

    out = Path(output)
    env = AthleticModularTaskEnv(task_config_from_bundle(bundle))
    try:
        env.reset(seed=seed)
        ctrl_dt = float(bundle.athletic.ctrl_dt)
        steps_per_frame = max(1, int(round((1.0 / fps) / ctrl_dt)))
        n_frames = 1 if preview else max(1, int(math.ceil(seconds * fps)))
        snapshots = []

        def snap() -> None:
            snapshots.append(
                (
                    np.asarray(env.data.geom_xpos).copy(),
                    np.asarray(env.data.geom_xmat).copy(),
                    np.asarray(env.data.qpos[:3]).copy(),
                )
            )

        snap()
        for _ in range(n_frames - 1):
            for _ in range(steps_per_frame):
                env.step(env.action_space.sample())
            snap()
        write_stage(
            env.model,
            snapshots,
            out,
            up_axis=bundle.plant.up_axis,
            fps=fps,
            samples=samples,
            preview=preview or n_frames == 1,
            resolution=resolution,
            lighting=lighting,
        )
    finally:
        env.close()

    if export_only:
        return out
    _run_blender(out, blender)
    return out


def _run_blender(output: Path, blender: str) -> None:
    executable = shutil.which(blender)
    if executable is None and Path(blender).is_file():
        executable = blender
    if executable is None:
        raise FileNotFoundError(
            f"Blender executable {blender!r} was not found. The stage is in {output} "
            "(geoms.obj, render.json); rerun when blender is on PATH."
        )
    command = [
        executable,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--stage",
        str(output.resolve()),
    ]
    subprocess.run(command, check=True)


def build_blender_scene(output: Path) -> None:
    """Import the stage and render it. Runs inside Blender."""
    import bpy
    from mathutils import Vector

    meta = json.loads((output / "render.json").read_text())
    rig = meta["lighting"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(meta["samples"])
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "use_light_tree"):
        scene.cycles.use_light_tree = True
    _prefer_cycles_gpu(bpy)
    scene.render.fps = int(meta["fps"])
    scene.render.resolution_x, scene.render.resolution_y = meta["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = rig.get("view_transform", "AgX")
    scene.view_settings.exposure = float(rig["exposure"])
    for look in (f"AgX - {rig['look']}", rig["look"], "AgX - Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue

    try:
        bpy.ops.wm.obj_import(filepath=str(output / "geoms.obj"))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(output / "geoms.obj"))

    bone = _material(bpy, "Bone", (0.82, 0.74, 0.62, 1), 0.42, subsurface=0.18, coat=0.12)
    muscle = _material(
        bpy, "Muscle", (0.55, 0.14, 0.12, 1), 0.32, subsurface=0.35, coat=0.35
    )
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        mat = muscle if "muscle" in obj.name.lower() or "tendon" in obj.name.lower() else bone
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.use_smooth = True

    names = meta["geoms"]
    # Pose the reset frame before measuring the body. Imported vertices sit in
    # geom-local space, so lights sized from that pile miss the skeleton.
    for name, flat in zip(names, meta["frames"][0]):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.matrix_world = _matrix(bpy, flat)
    bpy.context.view_layer.update()

    subject = bpy.data.objects.new("Subject", None)
    scene.collection.objects.link(subject)
    subject.location = Vector(meta["pelvis"][0])
    look = bpy.data.objects.new("Look", None)
    scene.collection.objects.link(look)
    look.parent = subject
    look.location = Vector(meta["look_offset"])

    camera_data = bpy.data.cameras.new("Studio")
    camera_data.lens = 50
    camera_data.sensor_width = 36
    camera_data.clip_end = 200
    camera = bpy.data.objects.new("Studio camera", camera_data)
    scene.collection.objects.link(camera)
    camera.parent = subject
    camera.location = Vector(meta["camera_offset"])
    track = camera.constraints.new("TRACK_TO")
    track.target = look
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = camera
    if meta.get("track", {}).get("kind") == "ring":
        _add_running_ring(bpy, meta)
        _frame_running_ring(bpy, camera, meta)

    # Lamp positions are in the camera frame of the reset pose, then parented
    # to the pelvis so the same modelling follows the runner.
    bpy.context.view_layer.update()
    forward = (look.matrix_world.translation - camera.matrix_world.translation).normalized()
    right = forward.cross(Vector((0, 0, 1)))
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    right.normalize()
    low, high = _bounds([o for o in bpy.data.objects if o.type == "MESH"])
    span = max((high - low).length, 0.4)
    for spec in rig["lights"]:
        offset = Vector(spec["offset"])
        location = right * offset.x + forward * offset.y + Vector((0, 0, offset.z))
        location *= span
        bpy.ops.object.light_add(type="AREA", location=location)
        lamp = bpy.context.object
        lamp.name = spec["name"]
        lamp.data.energy = float(spec["power"]) * span * span
        lamp.data.color = spec["color"]
        lamp.data.shape = spec["shape"]
        lamp.data.size = float(spec["size"]) * span
        if spec["shape"] == "RECTANGLE":
            lamp.data.size_y = lamp.data.size * 0.72
        aim = lamp.constraints.new("TRACK_TO")
        aim.target = look
        aim.track_axis = "TRACK_NEGATIVE_Z"
        aim.up_axis = "UP_Y"
        lamp.parent = subject
        lamp.matrix_parent_inverse = subject.matrix_world.inverted()

    world = bpy.data.worlds.new("Studio")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    if meta.get("track", {}).get("kind") == "ring":
        background.inputs[0].default_value = (0.56, 0.64, 0.74, 1.0)
        background.inputs[1].default_value = 0.45
    else:
        background.inputs[0].default_value = rig["world_color"]
        background.inputs[1].default_value = float(rig["world_strength"])

    if meta.get("track", {}).get("kind") != "ring":
        _cyclorama(bpy, low, high, forward)
    scene.frame_start = 0
    scene.frame_end = 0 if meta["preview"] else max(0, len(meta["frames"]) - 1)
    for frame, (mats, pelvis) in enumerate(zip(meta["frames"], meta["pelvis"])):
        scene.frame_set(frame)
        subject.location = Vector(pelvis)
        subject.keyframe_insert("location", frame=frame)
        for name, flat in zip(names, mats):
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            obj.matrix_world = _matrix(bpy, flat)
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_euler", frame=frame)
            obj.keyframe_insert("scale", frame=frame)

    scene.frame_set(0)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), compress=True)
    if meta["preview"]:
        scene.render.filepath = str(output / "preview.png")
        bpy.ops.render.render(write_still=True)
    else:
        scene.render.filepath = str(output / "frames" / "frame_")
        bpy.ops.render.render(animation=True)


def _prefer_cycles_gpu(bpy: Any) -> None:
    """Use Metal or CUDA when Blender can see a GPU. CPU remains the fallback."""
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = device_type
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        if not gpus:
            continue
        for device in prefs.devices:
            device.use = device.type != "CPU"
        scene.cycles.device = "GPU"
        return
    scene.cycles.device = "CPU"


def _material(bpy: Any, name: str, color: tuple, roughness: float, *, subsurface: float, coat: float) -> Any:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    for key, value in (
        ("Subsurface Weight", subsurface),
        ("Subsurface", subsurface),
        ("Coat Weight", coat),
        ("Coat", coat),
    ):
        socket = shader.inputs.get(key)
        if socket is not None:
            socket.default_value = value
    return mat


def _matrix(bpy: Any, flat: list[float]) -> Any:
    from mathutils import Matrix

    return Matrix(tuple(tuple(flat[row * 4 : (row + 1) * 4]) for row in range(4)))


def _bounds(objects: list[Any]) -> tuple[Any, Any]:
    from mathutils import Vector

    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    return low, high


def _add_running_ring(bpy: Any, meta: dict[str, Any]) -> None:
    """Rubber oval, a white line on every lane boundary, grass under it."""
    ring = running_ring_paths(tuple(meta["pelvis"][0]), meta["track"]["up_axis"])
    rubber = _material(bpy, "Rubber", RUBBER_COLOR, 0.55, subsurface=0.0, coat=0.04)
    line_mat = _material(bpy, "Lane line", LINE_COLOR, 0.35, subsurface=0.0, coat=0.0)
    grass = _material(bpy, "Grass", GRASS_COLOR, 0.9, subsurface=0.0, coat=0.0)
    _ribbon(bpy, "Track", ring["rubber"][0], ring["rubber"][1], rubber)
    for index, (left, right) in enumerate(ring["lines"]):
        _ribbon(bpy, f"Lane_{index}", left, right, line_mat)
    _grass_plane(bpy, ring, grass)
    sun_data = bpy.data.lights.new("Track sun", "SUN")
    sun_data.energy = 3.2
    sun_data.color = (1.0, 0.94, 0.86)
    sun = bpy.data.objects.new("Track sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(-35))


def _frame_running_ring(bpy: Any, camera: Any, meta: dict[str, Any]) -> None:
    """Pull back so the oval and its lane lines fill the frame."""
    from mathutils import Vector

    center = Vector(running_ring_paths(tuple(meta["pelvis"][0]), meta["track"]["up_axis"])["center"])
    aim = bpy.data.objects.new("Ring center", None)
    bpy.context.scene.collection.objects.link(aim)
    aim.location = center
    for constraint in camera.constraints:
        if constraint.type == "TRACK_TO":
            constraint.target = aim
    up = meta["track"]["up_axis"]
    if up == "z":
        camera_pos = center + Vector((-9.0, -24.0, 14.0))
    else:
        camera_pos = center + Vector((-9.0, 14.0, -24.0))
    camera.parent = None
    camera.location = camera_pos
    camera.data.lens = 28
    camera.data.clip_end = 400
    bpy.context.view_layer.update()


def _ribbon(bpy: Any, name: str, left: list[tuple[float, float, float]], right: list[tuple[float, float, float]], mat: Any) -> None:
    n = len(left)
    verts = list(left) + list(right)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)


def _grass_plane(bpy: Any, ring: dict[str, Any], mat: Any) -> None:
    pts = ring["rubber"][1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    pad = 18.0
    z = min(zs) - 0.01
    if abs(max(ys) - min(ys)) > abs(max(zs) - min(zs)):
        # Z-up: the oval spreads in x,y.
        verts = [
            (min(xs) - pad, min(ys) - pad, z),
            (max(xs) + pad, min(ys) - pad, z),
            (max(xs) + pad, max(ys) + pad, z),
            (min(xs) - pad, max(ys) + pad, z),
        ]
    else:
        verts = [
            (min(xs) - pad, z, min(zs) - pad),
            (max(xs) + pad, z, min(zs) - pad),
            (max(xs) + pad, z, max(zs) + pad),
            (min(xs) - pad, z, max(zs) + pad),
        ]
    mesh = bpy.data.meshes.new("Grass")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Grass", mesh)
    bpy.context.scene.collection.objects.link(obj)


def _cyclorama(bpy: Any, low: Any, high: Any, forward: Any) -> None:
    """Floor that bends up behind the subject, so the bounce light has a surface."""
    from mathutils import Vector

    center = (low + high) / 2
    span = max((high - low).length, 0.4)
    toward = Vector((-forward.x, -forward.y, 0)).normalized()
    side = Vector((0, 0, 1)).cross(toward).normalized()
    radius = 1.4 * span
    wall = 14 * span
    profile = [(6 * span, 0.0), (-1.4 * span + radius, 0.0)]
    profile += [
        (
            -1.4 * span + radius - radius * math.sin(t),
            radius - radius * math.cos(t),
        )
        for t in [i / 24 * math.pi / 2 for i in range(1, 25)]
    ]
    profile.append((profile[-1][0], wall))
    width = 30 * span
    origin = Vector((center.x, center.y, low.z - 0.01))
    verts = [
        origin + side * x + toward * y + Vector((0, 0, z))
        for y, z in profile
        for x in (-width / 2, width / 2)
    ]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(profile) - 1)]
    mesh = bpy.data.meshes.new("Cyclorama")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mat = _material(bpy, "Backdrop", (0.045, 0.046, 0.05, 1), 0.85, subsurface=0.0, coat=0.0)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Cyclorama", mesh)
    bpy.context.scene.collection.objects.link(obj)


def main(argv: list[str] | None = None) -> None:
    """Blender entry. ``blender --python this_file -- --stage DIR``."""
    argv = sys.argv[1:] if argv is None else argv
    if "--stage" not in argv:
        raise SystemExit("usage: blender --background --python blender.py -- --stage DIR")
    build_blender_scene(Path(argv[argv.index("--stage") + 1]))


if __name__ == "__main__":
    main()
