# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Import compiled anatomy in Blender through the optional MuSkeMo API.

Executed by file path inside Blender; module imports use the standard library.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _matrix(position: list[float], quaternion: list[float]) -> Any:
    from mathutils import Quaternion

    matrix = Quaternion(quaternion).to_matrix().to_4x4()
    matrix.translation = position
    return matrix


def _six(matrix: Any) -> list[float]:
    return [
        matrix[0][0],
        matrix[1][1],
        matrix[2][2],
        matrix[0][1],
        matrix[0][2],
        matrix[1][2],
    ]


def _primitive(geom: dict[str, Any], name: str, collection: Any) -> Any:
    import bpy
    import bmesh
    from mathutils import Matrix

    mesh = bpy.data.meshes.new(name)
    if geom["type"] == "MESH":
        mesh.from_pydata(geom["vertices"], [], geom["faces"])
    else:
        bm = bmesh.new()
        size, kind = geom["size"], geom["type"]
        if kind in ("SPHERE", "ELLIPSOID"):
            vertices = bmesh.ops.create_uvsphere(
                bm, u_segments=24, v_segments=12, radius=1
            )["verts"]
            bmesh.ops.scale(
                bm, vec=size if kind == "ELLIPSOID" else [size[0]] * 3, verts=vertices
            )
        elif kind in ("CYLINDER", "CAPSULE"):
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=24,
                radius1=size[0],
                radius2=size[0],
                depth=2 * size[1],
            )
            if kind == "CAPSULE":
                for sign in (-1, 1):
                    bmesh.ops.create_uvsphere(
                        bm,
                        u_segments=24,
                        v_segments=12,
                        radius=size[0],
                        matrix=Matrix.Translation((0, 0, sign * size[1])),
                    )
        elif kind in ("BOX", "PLANE"):
            vertices = bmesh.ops.create_cube(bm, size=2)["verts"]
            scale = size if kind == "BOX" else [size[0] or 1, size[1] or 1, 0.001]
            bmesh.ops.scale(bm, vec=scale, verts=vertices)
        else:
            bm.free()
            raise ValueError(f"Unsupported geometry type {kind}")
        bm.to_mesh(mesh)
        bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.color = geom["rgba"]
    return obj


def import_model(
    filepath: str | Path,
    *,
    addon: str | Path | None = None,
    prefix: str = "MyoSuite",
) -> dict[str, Any]:
    """Import native MuSkeMo objects with MuJoCo-evaluated poses and metadata.

    Args:
        filepath: JSON from ``muskemo.export_model``.
        addon: Separately installed MuSkeMo directory, when not enabled.
        prefix: Namespace for imported objects and collections.

    Returns:
        Objects indexed by role and source ID. Joint equalities remain in source
        metadata and their evaluated behavior is preserved in baked poses.

    Raises:
        ValueError: Unsupported schema, invalid namespace or existing import.
        RuntimeError: MuSkeMo is unavailable.
    """
    import bpy

    document = json.loads(Path(filepath).read_text(encoding="utf-8"))
    if (
        document.get("schema") != 1
        or document.get("units") != "metres-radians-kilograms"
    ):
        raise ValueError("Unsupported anatomy schema or units")
    if not prefix or len(prefix) > 32 or any(c in prefix for c in "\\/:"):
        raise ValueError("Use a namespace of 1-32 characters without slashes/colon")
    if any(
        c.name.startswith(prefix + ":")
        for c in (*bpy.data.collections, *bpy.data.objects)
    ):
        raise ValueError(f"An import already exists in namespace {prefix!r}")
    if addon is not None:
        path = Path(addon).resolve()
        sys.path.insert(0, str(path.parent))
        package = importlib.import_module(path.name)
        if not hasattr(bpy.context.scene, "muskemo"):
            package.register()
    else:
        package = next(
            (
                sys.modules[n]
                for n in bpy.context.preferences.addons
                if n in sys.modules
                and getattr(sys.modules[n], "bl_info", {}).get("name") == "MuSkeMo"
            ),
            None,
        )
    if package is None or not hasattr(bpy.context.scene, "muskemo"):
        raise RuntimeError("Enable MuSkeMo or pass its separately installed directory")
    helpers = {
        kind: getattr(
            importlib.import_module(package.__name__ + ".scripts." + module), function
        )
        for kind, module, function in (
            ("body", "create_body_func", "create_body"),
            ("frame", "create_frame_func", "create_frame"),
            ("joint", "create_joint_func", "create_joint"),
            ("muscle", "create_muscle_func", "create_muscle"),
            ("wrap", "create_wrapgeom_func", "create_wrapgeom"),
        )
    }
    containers = [
        getattr(bpy.data, kind)
        for kind in (
            "objects",
            "collections",
            "meshes",
            "curves",
            "materials",
            "node_groups",
            "texts",
            "actions",
        )
    ]
    initial = [set(values) for values in containers]

    class Reporter:
        def report(self, level: set[str], message: str) -> None:
            if "ERROR" in level:
                raise RuntimeError(message)
            print(message)

    try:
        return _populate(document, helpers, Reporter(), prefix)
    except Exception:
        bpy.data.batch_remove(
            ids=set().union(
                *(set(values) - old for values, old in zip(containers, initial))
            )
        )
        raise


def _populate(
    document: dict[str, Any], helpers: dict[str, Any], reporter: Any, prefix: str
) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix, Vector

    objects: dict[str, dict[int, Any]] = {
        k: {} for k in ("body", "joint", "site", "geom", "muscle")
    }

    def name(role: str, identifier: int) -> str:
        return f"{prefix}:{role}:{identifier}"

    def collection(role: str) -> Any:
        key = f"{prefix}:{role}"
        if key not in bpy.data.collections:
            value = bpy.data.collections.new(key)
            bpy.context.scene.collection.children.link(value)
        return bpy.data.collections[key]

    def remember(obj: Any, role: str, item: dict[str, Any]) -> None:
        obj["mujoco_id"], obj["mujoco_name"] = item["id"], item["name"]
        obj["mujoco_role"] = role
        objects[role][item["id"]] = obj

    first = document["frames"][0]
    for body in document["bodies"]:
        i = body["id"]
        transform = _matrix(first["body_pos"][i], first["body_quat"][i])
        inertia = (
            transform.to_3x3()
            @ Matrix(body["inertia"])
            @ transform.to_3x3().transposed()
        )
        helpers["body"](
            name=name("body", i),
            size=0.025,
            self=reporter,
            mass=body["mass"],
            COM=list(transform @ Vector(body["com"])),
            inertia_COM=_six(inertia),
            COM_local=body["com"],
            inertia_COM_local=_six(body["inertia"]),
            local_frame=name("frame", i),
            collection_name=collection("Bodies").name,
        )
        obj = bpy.data.objects[name("body", i)]
        obj.rotation_mode = "QUATERNION"
        obj.matrix_world = transform
        obj["default_pose"] = [list(row) for row in transform]
        remember(obj, "body", body)
        helpers["frame"](
            name=name("frame", i),
            size=0.025,
            pos_in_global=first["body_pos"][i],
            gRb=transform.to_3x3(),
            parent_body=obj.name,
            collection_name=collection("Frames").name,
        )
    bpy.context.view_layer.update()
    for joint in document["joints"]:
        i, body = joint["id"], objects["body"][joint["body"]]
        helpers["joint"](
            name=name("joint", i),
            radius=0.01,
            collection_name=collection("Joints").name,
            parent_body=objects["body"][
                document["bodies"][joint["body"]]["parent"]
            ].name,
            child_body=body.name,
            pos_in_global=list(body.matrix_world @ Vector(joint["pos"])),
            or_in_global_quat=list(body.matrix_world.to_quaternion()),
        )
        obj = bpy.data.objects[name("joint", i)]
        obj["mujoco_joint"] = json.dumps(joint)
        remember(obj, "joint", joint)
    for i, obj in objects["body"].items():
        obj.parent = None
        obj.matrix_world = _matrix(first["body_pos"][i], first["body_quat"][i])
    bpy.context.view_layer.update()
    for site in document["sites"]:
        obj = bpy.data.objects.new(name("site", site["id"]), None)
        collection("Sites").objects.link(obj)
        obj.empty_display_type, obj.empty_display_size = "SPHERE", 0.003
        obj.parent = objects["body"][site["body"]]
        obj.location = site["pos"]
        remember(obj, "site", site)
    wrap_ids = {
        e["object"]
        for t in document["tendons"]
        for e in t["elements"]
        if e["type"] in ("SPHERE", "CYLINDER")
    }
    for geom in document["geoms"]:
        i = geom["id"]
        if i in wrap_ids:
            dimensions = dict(radius=geom["size"][0])
            if geom["type"] == "CYLINDER":
                dimensions["height"] = 2 * geom["size"][1]
            helpers["wrap"](
                name=name("geom", i),
                geomtype=geom["type"].title(),
                collection_name=collection("Wraps").name,
                parent_body=objects["body"][geom["body"]].name,
                pos_in_global=first["geom_pos"][i],
                or_in_global_quat=first["geom_quat"][i],
                dimensions=dimensions,
            )
            obj = bpy.data.objects[name("geom", i)]
        else:
            obj = _primitive(geom, name("geom", i), collection("Geometry"))
        transform = _matrix(first["geom_pos"][i], first["geom_quat"][i])
        obj.parent = objects["body"][geom["body"]]
        obj.rotation_mode = "QUATERNION"
        obj.matrix_parent_inverse.identity()
        obj.matrix_basis = obj.parent.matrix_world.inverted() @ transform
        remember(obj, "geom", geom)
    for muscle_index, muscle in enumerate(document["muscles"]):
        elements = document["tendons"][muscle["tendon"]]["elements"]
        sites = [
            document["sites"][e["object"]] for e in elements if e["type"] == "SITE"
        ]
        if len(sites) < 2 or any(e["type"] in ("JOINT", "PULLEY") for e in elements):
            raise ValueError(f"Unsupported muscle path {muscle['name']!r}")
        for site in sites:
            helpers["muscle"](
                muscle_name=name("muscle", muscle["id"]),
                point_position=first["site_pos"][site["id"]],
                body_name=objects["body"][site["body"]].name,
                collection_name=collection("Muscles").name,
                F_max=muscle["force"],
            )
        obj = bpy.data.objects[name("muscle", muscle["id"])]
        obj["mujoco_path"] = json.dumps(elements)
        obj["mujoco_parameters"] = json.dumps(muscle)
        obj["mujoco_site_ids"] = [s["id"] for s in sites]
        obj["mujoco_lengths"] = [
            f["muscle_length"][muscle_index] for f in document["frames"]
        ]
        pre_wrap = 0
        for element in elements:
            if element["type"] == "SITE":
                pre_wrap += 1
            elif element["type"] in ("SPHERE", "CYLINDER"):
                package = helpers["muscle"].__module__.rsplit(".scripts.", 1)[0]
                assign = importlib.import_module(
                    package + ".scripts.assign_muscle_wrap_func"
                )
                panel = importlib.import_module(package + ".scripts.muscle_panel")
                wrap_name = objects["geom"][element["object"]].name
                assign.assign_muscle_wrap(wrap_name, obj.name, reporter)
                modifier = obj.modifiers[obj.name + "_wrap_" + wrap_name]
                panel.set_socket(
                    modifier, "Index Of Pre Wrap Point Starting At 1", pre_wrap
                )
                panel.set_socket(modifier, "Shortest Wrap", True)
        obj["wrapping_status"] = (
            "MuSkeMo preview; validate changes against MuJoCo lengths and moment arms"
        )
        remember(obj, "muscle", muscle)
    for f, pose in enumerate(document["frames"], start=1):
        for body in document["bodies"]:
            i, obj = body["id"], objects["body"][body["id"]]
            obj.rotation_mode = "QUATERNION"
            obj.location, obj.rotation_quaternion = (
                pose["body_pos"][i],
                pose["body_quat"][i],
            )
            if len(document["frames"]) > 1:
                obj.keyframe_insert("location", frame=f)
                obj.keyframe_insert("rotation_quaternion", frame=f)
    bpy.context.scene.frame_end = len(document["frames"])
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    text = bpy.data.texts.new(prefix + ":source")
    text.write(
        json.dumps(
            {k: v for k, v in document.items() if k not in ("geoms", "frames")},
            indent=2,
        )
    )
    bpy.context.scene[prefix + ":fingerprint"] = document["fingerprint"]
    return objects


def register() -> None:
    """Add a compiled anatomy entry to Blender's File > Import menu."""
    import bpy
    from bpy_extras.io_utils import ImportHelper

    class ImportCompiledAnatomy(bpy.types.Operator, ImportHelper):
        bl_idname = "import_scene.myosuite_anatomy"
        bl_label = "MyoSuite compiled anatomy"
        bl_options = {"REGISTER", "UNDO"}
        filename_ext = ".json"

        def execute(self, context: Any) -> set[str]:
            try:
                import_model(self.filepath)
            except (ValueError, RuntimeError, ImportError, OSError) as error:
                self.report({"ERROR"}, str(error))
                return {"CANCELLED"}
            return {"FINISHED"}

    ImportCompiledAnatomy.__annotations__ = {
        "filter_glob": bpy.props.StringProperty(default="*.json", options={"HIDDEN"})
    }
    bpy.utils.register_class(ImportCompiledAnatomy)

    def menu(self: Any, context: Any) -> None:
        self.layout.operator(
            ImportCompiledAnatomy.bl_idname, text="MyoSuite anatomy (.json)"
        )

    bpy.types.TOPBAR_MT_file_import.append(menu)


def main() -> None:
    """Run inside Blender, optionally saving a new blend file."""
    import bpy

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--addon", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", default="MyoSuite")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    import_model(args.input, addon=args.addon, prefix=args.prefix)
    if args.output:
        bpy.ops.wm.save_as_mainfile(
            filepath=str(args.output.resolve()), check_existing=False
        )


if __name__ == "__main__":
    if "--" in sys.argv:
        main()
    else:
        register()
