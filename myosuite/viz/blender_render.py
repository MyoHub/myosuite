# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render a CPU rollout in Blender: animated USD export and a Cycles studio scene.

:func:`export_rollout` runs one episode in ordinary Python and writes the USD
animation (with volumetric muscles, see :mod:`myosuite.viz.muscle_tubes`).
:func:`build_blender_scene` runs inside Blender, which executes this file by
path, so module-level imports are limited to the standard library.
``scripts/render_blender.py`` is the command-line entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Anatomical muscle meshes (BodyParts3D, CC BY 4.0) on the Hugging Face Hub, posed
# on myoMimicFullbody-v0's rest pose; see viz/assets/CREDITS.md.
MUSCLE_MESH_REPO = "myohub/myosuite-assets"
MUSCLE_MESHES = {
    "atlas": "muscles/fullbody_muscles_light.glb",
    "atlas-hd": "muscles/fullbody_muscles.glb",
}


@dataclass
class RenderConfig:
    """Settings of one render.

    Attributes:
        env: Registered env id.
        output: New output directory.
        checkpoint: SB3 ``.zip`` or RSL-RL ``.pt``/run directory; ``None`` drives
            the env with random actions.
        seed: Episode seed.
        seconds: Clip length; recording also stops when the episode ends.
        fps: Video frame rate.
        resolution: ``(width, height)`` in pixels.
        preview: Render only the first frame.
        muscles: ``"volumetric"`` muscle bellies or MuJoCo's thin ``"paths"``.
        muscle_color: ``"activation"`` tints each muscle by its activation (on
            paths with MuJoCo's dark-to-red viewer colours), ``"uniform"``
            keeps one anatomical red, e.g. for still renders.
        muscle_scale: Multiplier of all muscle radii.
        scene: ``"studio"`` backdrop, or ``"mujoco"`` to keep the task's floor.
        skin: ``.skn`` body skin, or ``"fullbody"`` for the bundled one; ``None``
            renders no skin.
        skin_style: ``"translucent"`` skin over muscles and bones, or ``"opaque"``
            skin that hides them.
        skin_alpha: Opacity of the translucent skin.
        skin_texture: Colour image for the skin, in the skin's UV layout (the
            bundled skin uses MakeHuman's); ``None`` keeps the plain skin tone.
        skin_inflate: Offset of the skin along its normals (m).
        muscle_mesh: Anatomical muscle meshes drawn in place of the tubes and
            deformed by the bones: ``"atlas"`` (light) or ``"atlas-hd"``
            (BodyParts3D, downloaded from Hugging Face on first use, full body
            only), or a ``.glb`` in the model's rest pose; ``None`` keeps the tubes.
        samples: Cycles samples per frame.
        blender: Blender executable.
    """

    env: str
    output: Path
    checkpoint: Path | None = None
    seed: int = 0
    seconds: float = 3.0
    fps: int = 30
    resolution: tuple[int, int] = (640, 640)
    preview: bool = False
    muscles: str = "volumetric"
    muscle_color: str = "activation"
    muscle_scale: float = 1.0
    scene: str = "studio"
    skin: Path | str | None = None
    skin_style: str = "translucent"
    skin_alpha: float = 0.3
    skin_texture: Path | None = None
    skin_inflate: float = 0.0
    muscle_mesh: str | None = None
    samples: int = 64
    blender: str = "blender"

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.seconds)
            and self.seconds > 0
            and self.fps > 0
            and min(self.resolution) > 0
            and self.samples > 0
            and self.muscle_scale > 0
        ):
            raise ValueError(
                "Seconds, FPS, resolution, samples and scale must be positive."
            )
        if self.muscles not in ("volumetric", "paths"):
            raise ValueError(f"Unknown muscles mode: {self.muscles}")
        if self.muscle_color not in ("activation", "uniform"):
            raise ValueError(f"Unknown muscle colour: {self.muscle_color}")
        if self.scene not in ("studio", "mujoco"):
            raise ValueError(f"Unknown scene: {self.scene}")
        if self.skin_style not in ("translucent", "opaque"):
            raise ValueError(f"Unknown skin style: {self.skin_style}")
        if not 0 < self.skin_alpha <= 1:
            raise ValueError("Skin alpha must be in (0, 1].")
        if self.skin_texture is not None:
            if self.skin is None:
                raise ValueError("A skin texture needs a skin.")
            if not Path(self.skin_texture).is_file():
                raise ValueError(f"Skin texture not found: {self.skin_texture}")
        if self.muscle_mesh in MUSCLE_MESHES:
            if not self.env.startswith("myoMimicFullbody"):
                raise ValueError(
                    f"--muscle-mesh {self.muscle_mesh} fits myoMimicFullbody envs only."
                )
        elif self.muscle_mesh is not None and not Path(self.muscle_mesh).is_file():
            raise ValueError(f"Muscle mesh not found: {self.muscle_mesh}")


def export_rollout(config: RenderConfig) -> dict[str, Any]:
    """Export one episode to animated USD, keeping its control-step timing.

    Args:
        config: Render settings.

    Returns:
        The metadata also written to ``render.json`` for the Blender stage.

    Raises:
        ValueError: Unsupported model (skins, flexes), variable step, or an
            empty episode. A checkpoint that does not fit the env raises too.
    """
    import mujoco
    import numpy as np
    from mujoco.usd import objects, shapes
    from mujoco.usd.exporter import USDExporter
    from pxr import Tf, Usd, UsdGeom

    from myosuite import make_env
    from myosuite.utils.checkpoint_utils import load_policy
    from myosuite.viz.muscle_tubes import muscle_actuators
    from myosuite.viz.skin import SkinPose, load_skn

    class LightTendon(objects.USDTendon):
        def generate_primitive_mesh(self) -> dict[str, Any]:
            parts = {}
            for mesh_config in self.mesh_config:
                name, mesh = shapes.mesh_factory(mesh_config, None, resolution=12)
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
            mesh_config = shapes.mesh_config_generator(
                name, geom.type, np.ones(3), decouple=True
            )
            self.geom_refs[name] = LightTendon(
                mesh_config, self.stage, self.model, geom, name, rgba=geom.rgba
            )
            self.geom_names.add(name)

    env = make_env(config.env)
    try:
        obs, _ = env.reset(seed=config.seed)
        env.action_space.seed(config.seed)
        model, data = env.unwrapped.model, env.unwrapped.data
        if model.nskin or model.nflex:
            raise ValueError(
                "Skin/flex export is outside this minimal renderer's scope."
            )
        if config.checkpoint:
            # Strict: an unusable checkpoint raises instead of rendering random motion.
            policy = load_policy(env, config.checkpoint.absolute(), strict=True)
        else:

            def policy(observation: np.ndarray) -> np.ndarray:
                return env.action_space.sample()

        skin = (
            SkinPose.bind(load_skn(config.skin), model, config.skin_inflate)
            if config.skin is not None
            else None
        )
        out = config.output.resolve()
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
        muscles = muscle_actuators(model)
        muscle_tendons = model.actuator_trnid[muscles, 0] if len(muscles) else []
        times, positions, rotations, qposes, paths, activations = [], [], [], [], [], []
        skin_frames = []
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
            # wrap_xpos holds two points per row; ten_wrapadr indexes the points.
            if config.muscles == "volumetric":
                wrap_points = data.wrap_xpos.reshape(-1, 3)
                paths.append(
                    [
                        wrap_points[a : a + n].copy()
                        for a, n in zip(
                            data.ten_wrapadr[muscle_tendons],
                            data.ten_wrapnum[muscle_tendons],
                        )
                    ]
                )
            activations.append(
                data.act[model.actuator_actadr[muscles]].copy() if len(muscles) else []
            )
            if skin is not None:
                skin_frames.append(skin.vertices(data))
                low = np.minimum(low, skin_frames[-1].min(0))
                high = np.maximum(high, skin_frames[-1].max(0))
            for geom in exporter.scene.geoms[: exporter.scene.ngeom]:
                if geom.type != mujoco.mjtGeom.mjGEOM_PLANE and not (
                    geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and model.geom_bodyid[geom.objid] == 0
                ):
                    radius = float(np.linalg.norm(geom.size))
                    low = np.minimum(low, geom.pos - radius)
                    high = np.maximum(high, geom.pos + radius)
            if done or times[-1] >= config.seconds - 1e-9:
                break
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
        if len(times) < 2:
            raise ValueError("The episode contains no motion samples.")
        scene_geoms = exporter.scene.geoms[: exporter.scene.ngeom]

        def _is_static(g: Any) -> bool:
            return g.type == mujoco.mjtGeom.mjGEOM_PLANE or (
                g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                and model.geom_bodyid[g.objid] == 0
            )

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
        muscle_names = []
        if len(muscles) and config.muscles == "volumetric":
            muscle_names = _write_muscle_tubes(
                stage, model, muscles, paths, config.muscle_scale
            )
        skin_names = (
            _write_skin(stage, skin.skin, np.asarray(skin_frames))
            if skin is not None
            else []
        )
        if len(muscles):
            np.savez_compressed(
                out / "muscles.npz", activation=np.asarray(activations, np.float32)
            )
        stage.GetRootLayer().Save()
        np.savez_compressed(
            out / "reference.npz",
            time=times,
            geom_xpos=positions,
            geom_xmat=rotations,
            qpos=qposes,
        )
        anatomy = _anatomy_bodies(model)
        bone_geoms = {
            exporter._get_geom_name(g): int(g.objid)
            for g in scene_geoms
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
            and g.type == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_bodyid[g.objid] in anatomy
        }
        meta = {
            "env": config.env,
            "seed": config.seed,
            "checkpoint": str(config.checkpoint) if config.checkpoint else None,
            "motion_source": "checkpoint" if config.checkpoint else "random",
            "usd": str(usd),
            "dt": dt,
            "duration": times[-1],
            "samples": len(times),
            "fps": config.fps,
            "resolution": config.resolution,
            "preview": config.preview,
            "bounds": [low.tolist(), high.tolist()],
            "tendon_objects": sorted(
                name for name in exporter.geom_names if "_tendon" in name
            ),
            "mujoco_version": mujoco.__version__,
            "scene": config.scene,
            "samples_per_frame": config.samples,
            "muscles": config.muscles,
            "skin_objects": skin_names,
            "skin_style": config.skin_style,
            "skin_alpha": config.skin_alpha,
            "skin_texture": _copy_skin_texture(config, out),
            "muscle_color": config.muscle_color,
            # Blender object of each muscle: its tube, or its MuJoCo path segments.
            "muscle_objects": muscle_names
            or [f"_id{t}_tendon" for t in muscle_tendons],
            # Volumetric muscles replace the exporter's thin tendon segments.
            "replaced_tendons": [f"_id{t}_tendon" for t in muscle_tendons]
            if muscle_names
            else [],
            "bone_objects": list(bone_geoms),
            "muscle_mesh": _copy_muscle_mesh(config, out),
            # Bone poses at qpos0, the rest pose the muscle meshes are modelled in.
            "bone_rest": _rest_poses(model, bone_geoms) if config.muscle_mesh else {},
            "plane_objects": [
                exporter._get_geom_name(g)
                for g in scene_geoms
                if g.type == mujoco.mjtGeom.mjGEOM_PLANE
            ],
            "floor_height": max(
                (
                    float(g.pos[2])
                    for g in scene_geoms
                    if g.type == mujoco.mjtGeom.mjGEOM_PLANE
                ),
                default=None,
            ),
            # Static world geometry: kept in the render, left out of camera fitting.
            "static_objects": [
                exporter._get_geom_name(g) for g in scene_geoms if _is_static(g)
            ],
            # Visual-only world geometry (room shells, wall props): hidden, since the
            # studio replaces it. Collidable task geometry (goals, fences) stays.
            "scenery_objects": [
                exporter._get_geom_name(g)
                for g in scene_geoms
                if _is_static(g)
                and g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                and g.type != mujoco.mjtGeom.mjGEOM_PLANE
                and model.geom_contype[g.objid] == 0
                and model.geom_conaffinity[g.objid] == 0
            ],
        }
        frames = max(1, math.ceil(times[-1] * config.fps - 1e-9))
        visibility = {
            prim.GetName(): [
                UsdGeom.Imageable(prim).ComputeVisibility(f / config.fps / dt)
                != "invisible"
                for f in range(frames)
            ]
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Xform"
            and prim.GetParent().GetName() == "World"
            and prim.GetName().startswith("Mesh_Xform_")
        }
        # Always-visible prims need no keyframes.
        meta["visibility"] = {k: v for k, v in visibility.items() if not all(v)}
        (out / "render.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    finally:
        env.close()


def _anatomy_bodies(model: Any) -> set[int]:
    """Bodies in any kinematic tree that holds muscle-path sites.

    Their meshes are bones; free task objects (dice, balls, paddles) form trees
    of their own and keep their colours.
    """
    import mujoco

    def root(body: int) -> int:
        while model.body_parentid[body] != 0:
            body = int(model.body_parentid[body])
        return body

    roots = {
        root(int(model.site_bodyid[model.wrap_objid[w]]))
        for w in range(model.nwrap)
        if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE
    }
    return {b for b in range(1, model.nbody) if root(b) in roots}


def _write_muscle_tubes(
    stage: Any, model: Any, muscles: Any, paths: list, scale: float
) -> list[str]:
    """Add one animated, volume-preserving tube mesh per muscle to *stage*."""
    import numpy as np
    from pxr import Gf, Tf, UsdGeom, Vt

    from myosuite.viz.muscle_tubes import (
        belly_radius,
        belly_volume,
        muscle_shape,
        radius_profile,
        resample_path,
        tube_mesh,
    )

    resampled = [[resample_path(p) for p in frame] for frame in paths]
    centres = np.array([[c for c, _ in frame] for frame in resampled])  # (T, M, N, 3)
    lengths = np.array([[n for _, n in frame] for frame in resampled])  # (T, M)
    # Shape at the first frame; the volume then stays constant, so a shortening
    # muscle thickens.
    peak, fraction = muscle_shape(model, lengths[0])
    profile = radius_profile(fraction)  # (M, N)
    volume = belly_volume(peak, lengths[0], profile)
    peak = belly_radius(volume, lengths, profile) * scale  # (T, M)
    points, counts, indices = tube_mesh(centres, peak[..., None] * profile)
    UsdGeom.Xform.Define(stage, "/World/Muscles")
    names = []
    for m, actuator in enumerate(muscles):
        name = Tf.MakeValidIdentifier(f"Muscle_{model.actuator(actuator).name}")
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Muscles/{name}")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.08, 0.08)])
        attr = mesh.CreatePointsAttr()
        for frame in range(points.shape[0]):
            attr.Set(
                Vt.Vec3fArray.FromNumpy(points[frame, m].astype(np.float32)), frame
            )
        names.append(name)
    return names


def _resolve_muscle_mesh(spec: str) -> Path:
    """Local path of a muscle mesh, downloading ``"atlas"``/``"atlas-hd"`` if needed.

    Raises:
        RuntimeError: The Hugging Face download failed.
    """
    if spec not in MUSCLE_MESHES:
        return Path(spec)
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=MUSCLE_MESH_REPO,
                filename=MUSCLE_MESHES[spec],
                repo_type="dataset",
            )
        )
    except Exception as err:
        raise RuntimeError(
            f"Could not download {MUSCLE_MESHES[spec]} from the Hugging Face dataset "
            f"{MUSCLE_MESH_REPO}: {err}"
        ) from err


def _copy_muscle_mesh(config: RenderConfig, out: Path) -> str | None:
    """Put the muscle mesh next to the scene; returns its file name there."""
    if config.muscle_mesh is None:
        return None
    shutil.copyfile(_resolve_muscle_mesh(config.muscle_mesh), out / "muscle_mesh.glb")
    return "muscle_mesh.glb"


def _rest_poses(model: Any, geoms: dict[str, int]) -> dict[str, list[float]]:
    """Row-major 4x4 world matrix of each named geom at ``qpos0``."""
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    poses = {}
    for name, g in geoms.items():
        pose = np.eye(4)
        pose[:3, :3] = data.geom_xmat[g].reshape(3, 3)
        pose[:3, 3] = data.geom_xpos[g]
        poses[name] = pose.ravel().tolist()
    return poses


_LEG = re.compile(r"femur|tibia|fibula|patella|talus|foot|calcn|toes")
_ARM = re.compile(
    r"humer|ulna|radius|lunate|scaphoid|pisiform|triquetrum|capitate|trapez"
    r"|hamate|mc|ph|thumb|clavicle|scapula"
)


def _limb(name: str) -> str:
    """Body region of a bone object: ``"leg_r"``, ``"arm_l"``, ... or ``"trunk"``."""
    side = "_l" if re.search(r"_l(_|$)", name) else "_r"
    if _LEG.search(name):
        return "leg" + side
    return "arm" + side if _ARM.search(name) else "trunk"


def _muscle_weights(
    points: Any,
    edges: Any,
    bone_points: Any,
    bone_owner: Any,
    bone_limb: list[str],
    k: int = 16,
) -> tuple[Any, Any, Any]:
    """Skinning weights of mesh vertices to the bones (numpy only, runs in Blender).

    Each vertex weighs its ``k`` nearest bone points by inverse squared distance.
    A connected piece of the mesh then keeps its main region (a limb side or the
    trunk), plus the trunk or a limb holding at least 15% of its weight, but never a
    second limb: a hand muscle hanging by the thigh never follows the femur, while
    a pectoral follows both the thorax and the humerus.

    Args:
        points: ``(V, 3)`` mesh vertices in the rest pose.
        edges: ``(E, 2)`` vertex index pairs.
        bone_points: ``(P, 3)`` bone surface samples in the rest pose.
        bone_owner: ``(P,)`` bone index of each sample.
        bone_limb: Region of each bone, see :func:`_limb`.
        k: Nearest bone samples per vertex.

    Returns:
        ``(vertex, bone, weight)`` arrays; weights of each vertex sum to one.
    """
    import numpy as np

    points = np.asarray(points, np.float64)
    bone_points = np.asarray(bone_points, np.float64)
    nbone = len(bone_limb)
    k = min(k, len(bone_points))
    near, dist = [], []
    squared = (bone_points**2).sum(1)
    for start in range(0, len(points), 512):
        chunk = points[start : start + 512]
        d2 = (chunk**2).sum(1)[:, None] + squared - 2 * chunk @ bone_points.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        near.append(idx)
        dist.append(np.sqrt(np.maximum(np.take_along_axis(d2, idx, 1), 0)))
    near, dist = np.concatenate(near), np.concatenate(dist)
    vertex = np.repeat(np.arange(len(points)), k)
    bone = np.asarray(bone_owner)[near].ravel()
    raw = (1.0 / np.maximum(dist, 1e-3) ** 2).ravel()
    key, inverse = np.unique(vertex * nbone + bone, return_inverse=True)
    weight = np.bincount(inverse, raw)
    vertex, bone = key // nbone, key % nbone
    weight /= np.bincount(vertex, weight, len(points))[vertex]
    # Connected pieces by union-find over the edges.
    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.asarray(edges, int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    piece = np.unique([find(a) for a in range(len(points))], return_inverse=True)[1]
    regions = sorted(set(bone_limb))
    region = np.array([regions.index(r) for r in bone_limb])[bone]
    share = np.zeros((piece.max() + 1, len(regions)))
    np.add.at(share, (piece[vertex], region), weight)
    share /= share.sum(1, keepdims=True)
    main = share.argmax(1)
    trunk = np.array([r == "trunk" for r in regions])
    # The main region, plus the trunk for limb pieces (hip, shoulder) or limbs for
    # trunk pieces (pectoral); never a second limb.
    allowed = (share >= 0.15) & (trunk | trunk[main][:, None])
    allowed[np.arange(len(main)), main] = True
    weight = weight * allowed[piece[vertex], region]
    weight[weight < 0.03] = 0
    total = np.bincount(vertex, weight, len(points))
    # A vertex left without weight follows its piece's main bone.
    by_bone = np.zeros((piece.max() + 1, nbone))
    np.add.at(by_bone, (piece[vertex], bone), weight)
    empty = np.flatnonzero(total == 0)
    vertex = np.concatenate([vertex, empty])
    bone = np.concatenate([bone, by_bone[piece[empty]].argmax(1)])
    weight = np.concatenate([weight, np.ones(len(empty))])
    keep = weight > 0
    vertex, bone, weight = vertex[keep], bone[keep], weight[keep]
    weight /= np.bincount(vertex, weight, len(points))[vertex]
    return vertex, bone, weight


def _add_muscle_mesh(
    output: Path, meta: dict[str, Any], scene: Any, material: Any
) -> None:
    """Import the muscle meshes and let the bone objects deform them (runs in Blender)."""
    import bpy
    import numpy as np
    from mathutils import Matrix

    bones, bone_points, bone_owner = [], [], []
    for name, pose in meta["bone_rest"].items():
        obj = bpy.data.objects.get(f"Mesh_{name}")
        if obj is None:
            continue
        rest = np.asarray(pose).reshape(4, 4)
        local = np.zeros(3 * len(obj.data.vertices))
        obj.data.vertices.foreach_get("co", local)
        local = local.reshape(-1, 3)[:: max(1, len(obj.data.vertices) // 400)]
        bone_points.append(local @ rest[:3, :3].T + rest[:3, 3])
        bone_owner.append(np.full(len(local), len(bones)))
        bones.append((name, obj, Matrix(rest.tolist())))
    rig = bpy.data.objects.new("MuscleRig", bpy.data.armatures.new("MuscleRig"))
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, _, rest in bones:
        edit = rig.data.edit_bones.new(name)
        edit.tail = (0, 0.02, 0)
        edit.matrix = rest
    bpy.ops.object.mode_set(mode="OBJECT")
    for name, obj, _ in bones:
        rig.pose.bones[name].constraints.new("COPY_TRANSFORMS").target = obj
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(output / meta["muscle_mesh"]))
    imported = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    named = [o for o in meshes if o.name.startswith("Muscle")]
    for obj in imported:
        if obj not in (named or meshes):
            bpy.data.objects.remove(obj)
    bone_points, bone_owner = np.concatenate(bone_points), np.concatenate(bone_owner)
    limbs = [_limb(name) for name, _, _ in bones]
    for obj in named or meshes:
        mesh = obj.data
        points = np.zeros(3 * len(mesh.vertices))
        mesh.vertices.foreach_get("co", points)
        world = np.array(obj.matrix_world)
        points = points.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
        edges = np.zeros(2 * len(mesh.edges), int)
        mesh.edges.foreach_get("vertices", edges)
        vertex, bone, weight = _muscle_weights(
            points, edges.reshape(-1, 2), bone_points, bone_owner, limbs
        )
        for b in np.unique(bone):
            group = obj.vertex_groups.new(name=bones[b][0])
            for v, w in zip(vertex[bone == b], weight[bone == b]):
                group.add([int(v)], float(w), "REPLACE")
        obj.modifiers.new("Bones", "ARMATURE").object = rig
        mesh.materials.clear()
        mesh.materials.append(material)
        obj.color = MUSCLE_UNIFORM
        for polygon in mesh.polygons:
            polygon.use_smooth = True


def _copy_skin_texture(config: RenderConfig, out: Path) -> str | None:
    """Put the skin texture next to the scene; returns its file name there."""
    if config.skin_texture is None:
        return None
    source = Path(config.skin_texture)
    name = f"skin_texture{source.suffix.lower()}"
    shutil.copyfile(source, out / name)
    return name


def _write_skin(stage: Any, skin: Any, frames: Any) -> list[str]:
    """Add the posed skin to *stage* as one animated mesh with UVs."""
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, Vt

    UsdGeom.Xform.Define(stage, "/World/Skin")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Skin/Skin")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(skin.face)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(skin.face.ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*SKIN_COLOR[:3])])
    if len(skin.texcoord):
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(
            # .skn stores V down the image, USD (and Blender) up.
            Vt.Vec2fArray.FromNumpy(
                (skin.texcoord * [1, -1] + [0, 1]).astype(np.float32)
            )
        )
    attr = mesh.CreatePointsAttr()
    for frame, points in enumerate(frames):
        attr.Set(Vt.Vec3fArray.FromNumpy(points), frame)
    return ["Skin"]


# Warm key, cool fill and a rim light that separates the figure from the backdrop:
# (name, offset from the subject in camera right/forward/up spans, W/m^2, colour).
STUDIO_LIGHTS = [
    ("Key", (-1.1, -1.0, 1.4), 60, (1.0, 0.92, 0.82)),
    ("Fill", (1.4, -1.2, 0.3), 14, (0.82, 0.9, 1.0)),
    ("Rim", (0.5, 1.4, 1.2), 70, (1.0, 1.0, 1.0)),
]
EXPOSURE = 0.0
BACKDROP = (0.16, 0.16, 0.17, 1.0)
# Darker studio behind a pale skin, so its silhouette and the anatomy stand out.
SKIN_BACKDROP = (0.006, 0.0065, 0.008, 1.0)
MUSCLE_RELAXED = (0.50, 0.16, 0.15, 1.0)
MUSCLE_ACTIVE = (0.62, 0.012, 0.02, 1.0)
MUSCLE_UNIFORM = (0.42, 0.11, 0.10, 1.0)
# MyoSuite's MuJoCo viewer colouring: activation ** 0.25 from near black to red.
PATH_RELAXED = (0.05, 0.05, 0.05, 1.0)
PATH_ACTIVE = (0.95, 0.3, 0.3, 1.0)
SKIN_COLOR = (0.72, 0.62, 0.58, 1.0)


def _muscle_colour(activation: float, paths: bool) -> tuple[float, ...]:
    if paths:
        a = activation**0.25
        relaxed, active = PATH_RELAXED, PATH_ACTIVE
    else:
        a, relaxed, active = activation, MUSCLE_RELAXED, MUSCLE_ACTIVE
    return tuple(r + (q - r) * a for r, q in zip(relaxed, active))


def _animate_activation(output: Path, meta: dict[str, Any], scene: Any) -> None:
    """Keyframe each muscle's colour from its activation."""
    import bpy
    import numpy as np

    path = output / "muscles.npz"
    if meta.get("muscle_color", "activation") != "activation" or not path.exists():
        return
    paths = meta.get("muscles") == "paths"
    activation = np.load(path)["activation"]  # (samples, muscles)
    times = np.arange(len(activation)) * meta["dt"]
    for m, key in enumerate(meta["muscle_objects"]):
        objects = [o for o in bpy.data.objects if o.type == "MESH" and key in o.name]
        keyed = None
        for frame in range(scene.frame_start, scene.frame_end + 1):
            a = float(np.interp(frame / meta["fps"], times, activation[:, m]))
            if keyed is not None and abs(a - keyed) < 0.03 and frame != scene.frame_end:
                continue
            for obj in objects:
                obj.color = _muscle_colour(a, paths)
                obj.keyframe_insert("color", frame=frame)
            keyed = a


def _add_cyclorama(
    center: Any, forward: Any, floor_z: float, span: float, mat: Any
) -> None:
    """Seamless studio backdrop: floor curving up into a wall behind the subject."""
    import math as _math

    import bpy
    from mathutils import Vector

    toward = Vector((-forward.x, -forward.y, 0)).normalized()
    side = Vector((0, 0, 1)).cross(toward)
    back, radius, height, front, width = (
        1.6 * span,
        span,
        30 * span,
        30 * span,
        80 * span,
    )
    profile = [(front, 0.0), (-back + radius, 0.0)]
    profile += [
        (-back + radius - radius * _math.sin(t), radius - radius * _math.cos(t))
        for t in [i / 12 * _math.pi / 2 for i in range(1, 13)]
    ]
    profile.append((-back, height))
    origin = Vector((center.x, center.y, floor_z))
    verts = [
        origin + side * x + toward * y + Vector((0, 0, z))
        for y, z in profile
        for x in (-width / 2, width / 2)
    ]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(profile) - 1)]
    mesh = bpy.data.meshes.new("Cyclorama")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("Cyclorama", mesh)
    bpy.context.scene.collection.objects.link(obj)


def build_blender_scene(output: Path) -> None:
    """Build the studio scene from an export and render it (runs inside Blender).

    Args:
        output: Directory written by :func:`export_rollout`; receives
            ``scene.blend`` and ``preview.png`` or ``frames/``.
    """
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
                    if frame == 0 or shown != visible[frame - 1]:
                        obj.hide_render = not shown
                        obj.keyframe_insert("hide_render", frame=frame)
    scene.frame_start = 0
    scene.frame_end = max(0, math.ceil(meta["duration"] * meta["fps"] - 1e-9) - 1)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = meta.get("samples_per_frame", 64)
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = meta["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    bpy.context.preferences.filepaths.save_version = 0
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = EXPOSURE
    # High contrast makes the plain skin's silhouette read; it would oversaturate a texture.
    plain_skin = meta.get("skin_objects") and not meta.get("skin_texture")
    contrast = "High" if plain_skin else "Medium High"
    for look in (f"AgX - {contrast} Contrast", f"{contrast} Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.42, 0.43, 0.45, 1)
    background.inputs[1].default_value = 0.25

    def material(
        name: str, color: tuple[float, ...], roughness: float, **inputs: Any
    ) -> Any:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = roughness
        for key, value in inputs.items():
            socket = shader.inputs.get(key.replace("_", " "))
            if socket is not None:  # input names differ across Blender versions
                socket.default_value = value
        return mat

    # Waxy ivory bone and glossy, translucent muscle, as in anatomical renders.
    bone = material(
        "Bone",
        (0.80, 0.73, 0.60, 1),
        0.45,
        Subsurface_Weight=0.2,
        Subsurface_Radius=(1.0, 0.7, 0.45),
        Subsurface_Scale=0.004,
        Coat_Weight=0.15,
    )
    muscle = material(
        "Muscle",
        MUSCLE_RELAXED,
        0.32,
        Subsurface_Weight=0.3,
        Subsurface_Radius=(1.0, 0.25, 0.15),
        Subsurface_Scale=0.006,
        Coat_Weight=0.4,
        Coat_Roughness=0.12,
    )
    # Per-object colour drives the muscle tint, so one material shows activation.
    nodes, links = muscle.node_tree.nodes, muscle.node_tree.links
    info = nodes.new("ShaderNodeObjectInfo")
    links.new(info.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])

    skin_objects = meta.get("skin_objects", [])
    opaque_skin = bool(skin_objects) and meta.get("skin_style") == "opaque"
    textured = bool(skin_objects) and bool(meta.get("skin_texture"))
    skin = material(
        "Skin",
        SKIN_COLOR,
        0.5,
        # A texture carries the skin tone itself; the reddish scattering would tint it.
        Subsurface_Weight=0.0 if textured else 0.6 if opaque_skin else 0.15,
        Subsurface_Radius=(1.0, 0.45, 0.3),
        Subsurface_Scale=0.008,
        Coat_Weight=0.1,
    )
    if textured:
        nodes, links = skin.node_tree.nodes, skin.node_tree.links
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(output / meta["skin_texture"]))
        links.new(image.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    if skin_objects and not opaque_skin:
        # X-ray look: clear where the skin faces the camera, denser at the
        # silhouette, so the outline reads and the anatomy stays visible.
        alpha = meta.get("skin_alpha", 0.3)
        nodes, links = skin.node_tree.nodes, skin.node_tree.links
        facing = nodes.new("ShaderNodeLayerWeight")
        facing.inputs["Blend"].default_value = 0.35
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value = 0.2 * alpha
        ramp.inputs["To Max"].default_value = min(1.0, 3 * alpha)
        links.new(facing.outputs["Facing"], ramp.inputs["Value"])
        links.new(ramp.outputs["Result"], nodes["Principled BSDF"].inputs["Alpha"])
    hidden = meta.get("scenery_objects", []) + meta.get("replaced_tendons", [])
    if meta.get("muscle_mesh"):
        hidden += meta.get("muscle_objects", [])
    if opaque_skin:
        # Occluded anyway; hiding them keeps bones from poking through the skin.
        hidden += meta.get("muscle_objects", []) + meta.get("bone_objects", [])
        hidden += meta.get("tendon_objects", [])
    if meta.get("scene", "studio") == "studio":
        hidden += meta.get("plane_objects", [])
    bones = meta.get("bone_objects", [])
    for obj in scene.objects:
        if any(name in obj.name for name in hidden):
            # Drop visibility keyframes, which would otherwise unhide it.
            obj.animation_data_clear()
            obj.hide_render = True
        if obj.type != "MESH":
            continue
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        if any(obj.name.startswith(name) for name in skin_objects):
            obj.data.materials.clear()
            obj.data.materials.append(skin)
        elif "_tendon" in obj.name or obj.name.startswith("Muscle_"):
            obj.data.materials.clear()
            obj.data.materials.append(muscle)
            obj.color = (
                MUSCLE_UNIFORM
                if meta.get("muscle_color") == "uniform"
                else _muscle_colour(0.0, meta.get("muscles") == "paths")
            )
        elif any(name in obj.name for name in bones):
            obj.data.materials.clear()
            obj.data.materials.append(bone)
    _animate_activation(output, meta, scene)
    if meta.get("muscle_mesh"):
        _add_muscle_mesh(output, meta, scene, muscle)

    low, high = (Vector(v) for v in meta["bounds"])
    center = (low + high) / 2
    span = max((high - low).length, 0.3)
    aspect = scene.render.resolution_x / scene.render.resolution_y
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Studio camera"
    camera.data.lens = 85
    camera.data.sensor_width = 36
    view = Vector((1.3, -1.7, 0.75)).normalized()
    camera.location = center + view * span * 3
    camera.rotation_euler = (
        (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.clip_end = max(100, span * 40)
    scene.camera = camera
    anatomy = [
        o
        for o in scene.objects
        if o.type == "MESH"
        and not any(name in o.name for name in meta.get("static_objects", []))
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
        # Fit the perspective camera: distance so both extents fit with a margin.
        vlow = Vector(tuple(min(p[i] for p in view_points) for i in range(3)))
        vhigh = Vector(tuple(max(p[i] for p in view_points) for i in range(3)))
        target = rotation @ ((vlow + vhigh) / 2)
        half_tan = camera.data.sensor_width / 2 / camera.data.lens
        tan_x, tan_y = (
            (half_tan, half_tan / aspect)
            if aspect >= 1
            else (
                half_tan * aspect,
                half_tan,
            )
        )
        distance = (
            1.18 * max((vhigh.x - vlow.x) / 2 / tan_x, (vhigh.y - vlow.y) / 2 / tan_y)
            + (vhigh.z - vlow.z) / 2
        )
        camera.location = target + view * distance
        center = target
    forward = (center - camera.location).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    for name, offset, power, color in STUDIO_LIGHTS:
        location = (
            center
            + (right * offset[0] + forward * offset[1] + Vector((0, 0, offset[2])))
            * span
        )
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = power * span**2
        light.data.color = color
        light.data.shape = "DISK"
        light.data.size = span * 1.1
        light.rotation_euler = (
            (center - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    floor_z = meta.get("floor_height")
    if floor_z is None:
        floor_z = low.z - 0.01 * span
    if meta.get("scene", "studio") == "studio":
        _add_cyclorama(
            center,
            forward,
            floor_z,
            span,
            material("Backdrop", SKIN_BACKDROP if skin_objects else BACKDROP, 0.8),
        )
    else:
        bpy.ops.mesh.primitive_plane_add(
            size=span * 200, location=(center.x, center.y, floor_z - span * 0.01)
        )
        bpy.context.object.name = "Studio floor"
        bpy.context.object.data.materials.append(
            material("Floor", (0.12, 0.14, 0.17, 1), 0.85)
        )
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(output / "frames" / "frame_")
    bpy.ops.file.pack_all()
    # relative_remap stores the USD cache path relative to scene.blend.
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output / "scene.blend"), compress=True, relative_remap=True
    )
    if meta["preview"]:
        scene.render.filepath = str(output / "preview.png")
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


def render_rollout(
    config: RenderConfig, export_only: bool = False, blender_log: Path | None = None
) -> dict[str, Any]:
    """Export a rollout, render it in a separate Blender process and write the video.

    Args:
        config: Render settings.
        export_only: Stop after the USD export.
        blender_log: File for Blender's console output (default: this process's).

    Returns:
        The export metadata (see :func:`export_rollout`).
    """
    if (config.output / "render.json").exists():
        raise FileExistsError("Choose a new output directory to avoid mixing runs.")
    meta = export_rollout(config)
    print(
        f"Exported {meta['samples']} samples, {meta['duration']:.3f}s to {meta['usd']}",
        flush=True,
    )
    if export_only:
        return meta
    command = [
        config.blender,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--blender-stage",
        str(config.output.resolve()),
    ]
    if blender_log is None:
        subprocess.run(command, check=True)
    else:
        with Path(blender_log).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not config.preview:
        import imageio.v3 as iio

        from myosuite.utils.video_io import write_video

        frames = sorted((config.output / "frames").glob("frame_*.png"))
        write_video(
            config.output / "video.mp4",
            [iio.imread(f)[..., :3] for f in frames],
            fps=config.fps,
        )
    result = config.output / ("preview.png" if config.preview else "video.mp4")
    print(f"Blender scene: {(config.output / 'scene.blend').resolve()}")
    print(f"{'Preview' if config.preview else 'Video'}: {result.resolve()}", flush=True)
    return meta


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (also the Blender stage, via ``--blender-stage``).

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "--blender-stage" in argv:
        build_blender_scene(Path(argv[argv.index("--blender-stage") + 1]))
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="SB3 .zip, RSL-RL .pt or run directory. Default: the newest local run of --env, "
        "else the published baseline from Hugging Face (find_checkpoint).",
    )
    source.add_argument(
        "--random", action="store_true", help="Explicit random-action demo."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--muscles",
        choices=["volumetric", "paths"],
        default="volumetric",
        help="Volume-scaled muscle bellies, or MuJoCo's thin tendon paths.",
    )
    parser.add_argument(
        "--muscle-color",
        choices=["activation", "uniform"],
        default="activation",
        help="Tint muscles by activation (MuJoCo colours on paths), or one colour.",
    )
    parser.add_argument(
        "--muscle-scale", type=float, default=1.0, help="Muscle radius multiplier."
    )
    parser.add_argument(
        "--scene",
        choices=["studio", "mujoco"],
        default="studio",
        help="Studio backdrop, or keep the task's own floor (e.g. a soccer pitch).",
    )
    parser.add_argument(
        "--skin", help='Body skin: a MuJoCo .skn file, or "fullbody" (bundled).'
    )
    parser.add_argument(
        "--skin-style", choices=["translucent", "opaque"], default="translucent"
    )
    parser.add_argument("--skin-alpha", type=float, default=0.3)
    parser.add_argument(
        "--skin-texture",
        type=Path,
        help="Colour image for the skin, in its UV layout (use with --skin-style opaque).",
    )
    parser.add_argument(
        "--skin-inflate", type=float, default=0.0, help="Skin normal offset (m)."
    )
    parser.add_argument(
        "--muscle-mesh",
        help='Anatomical muscle meshes instead of tubes: "atlas", "atlas-hd" '
        "(downloaded from Hugging Face, full body only) or a rest-pose .glb.",
    )
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples.")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    if checkpoint is None and not args.random:
        from myosuite.utils.checkpoint_utils import find_checkpoint  # noqa: PLC0415

        checkpoint = find_checkpoint(args.env)
        if checkpoint is None:
            parser.error(
                f"no checkpoint found for {args.env}: pass --checkpoint, or --random"
            )
        print(f"checkpoint: {checkpoint}")
    try:
        config = RenderConfig(
            env=args.env,
            output=args.output,
            checkpoint=checkpoint,
            seed=args.seed,
            seconds=args.seconds,
            fps=args.fps,
            resolution=tuple(args.resolution),
            preview=args.preview,
            muscles=args.muscles,
            muscle_color=args.muscle_color,
            muscle_scale=args.muscle_scale,
            scene=args.scene,
            skin=args.skin,
            skin_style=args.skin_style,
            skin_alpha=args.skin_alpha,
            skin_texture=args.skin_texture,
            skin_inflate=args.skin_inflate,
            muscle_mesh=args.muscle_mesh,
            samples=args.samples,
            blender=args.blender,
        )
    except ValueError as err:
        parser.error(str(err))
    if (config.output / "render.json").exists():
        parser.error("Choose a new output directory to avoid mixing runs.")
    render_rollout(config, export_only=args.export_only)


if __name__ == "__main__":
    main()
