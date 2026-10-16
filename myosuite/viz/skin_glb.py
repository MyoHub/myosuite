# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A body skin as a rigged glTF binary (``.glb``), to edit its blend weights by hand.

:func:`write_skin_glb` writes a skin posed at the model's rest pose (``qpos0``),
one joint per bone named after its body and placed at the body's rest frame.
Edit it in Blender (Weight Paint) and export it again as glTF binary, then read
it back with :func:`read_skin_glb` (or ``load_skin``) and save it with ``save_skn``.
Vertices, faces and UVs come from the file; each joint binds to the body of its
name at ``qpos0``, so the mesh must stay modelled in that pose. A ``.skn`` bound
at ``qpos0`` (like the bundled skin) survives the round trip unchanged.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mujoco
import numpy as np

from myosuite.viz.skin import Skin, SkinPose

# glTF is Y-up: MuJoCo (x, y, z) is glTF (x, z, -y).
_TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
_COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _rest(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    return data


def write_skin_glb(skin: Skin, model: mujoco.MjModel, path: Path | str) -> None:
    """Write *skin* as a rigged ``.glb`` in *model*'s rest pose.

    Args:
        skin: Parsed skin.
        model: Model whose bodies the skin bones are named after.
        path: Output ``.glb`` file.

    Raises:
        ValueError: Some bone names are not bodies of *model*.
    """
    pose = SkinPose.bind(skin, model)
    data = _rest(model)
    vert = pose.vertices(data).astype(np.float64)
    normal = pose.normals(vert)
    nvert, nbone = len(vert), len(skin.bone_names)
    weight = np.zeros((nvert, nbone))
    for b, (ids, ws) in enumerate(zip(skin.vertid, skin.vertweight)):
        np.add.at(weight[:, b], ids, ws)
    weight /= np.where(
        weight.sum(1, keepdims=True) > 0, weight.sum(1, keepdims=True), 1
    )
    sets = max(1, -(-int((weight > 0).sum(1).max()) // 4))
    order = np.argsort(-weight, axis=1)[:, : 4 * sets]
    weights = np.take_along_axis(weight, order, 1)
    pad = 4 * sets - order.shape[1]  # fewer bones than influence slots
    joints = np.pad(order, ((0, 0), (0, pad))).astype(np.uint16)
    weights = np.pad(weights, ((0, 0), (0, pad))).astype(np.float32)

    blob, views, accessors = bytearray(), [], []

    def add(
        array: np.ndarray, kind: str, component: int, target: int | None = None
    ) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes}
            | ({"target": target} if target else {})
        )
        blob.extend(array.tobytes())
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": len(array),
            "type": kind,
        }
        if kind == "VEC3" and component == 5126:
            accessor |= {"min": array.min(0).tolist(), "max": array.max(0).tolist()}
        accessors.append(accessor)
        return len(accessors) - 1

    attributes = {
        "POSITION": add((vert @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962),
        "NORMAL": add((normal @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962),
    }
    if len(skin.texcoord):
        uv = skin.texcoord.astype(np.float32) * [1, -1] + [
            0,
            1,
        ]  # glTF UVs start top-left
        attributes["TEXCOORD_0"] = add(uv.astype(np.float32), "VEC2", 5126, 34962)
    for s in range(sets):
        attributes[f"JOINTS_{s}"] = add(
            np.ascontiguousarray(joints[:, 4 * s : 4 * s + 4]), "VEC4", 5123, 34962
        )
        attributes[f"WEIGHTS_{s}"] = add(
            np.ascontiguousarray(weights[:, 4 * s : 4 * s + 4]), "VEC4", 5126, 34962
        )
    indices = add(skin.face.astype(np.uint32).ravel(), "SCALAR", 5125, 34963)

    ids = list(pose.body_ids)
    worlds, inverse_bind = [], []
    for b in ids:
        world = np.eye(4)
        world[:3, :3] = _TO_GLTF @ data.xmat[b].reshape(3, 3) @ _TO_GLTF.T
        world[:3, 3] = _TO_GLTF @ data.xpos[b]
        worlds.append(world)
        inverse_bind.append(np.linalg.inv(world).T.ravel())  # column-major
    ibm = add(np.array(inverse_bind, np.float32), "MAT4", 5126)
    # Chain the bones along the body tree, so the rig poses like a skeleton.
    parents = []
    for b in ids:
        p = model.body_parentid[b]
        while p > 0 and p not in ids:
            p = model.body_parentid[p]
        parents.append(ids.index(p) if p > 0 else -1)
    nodes = []
    for i, b in enumerate(ids):
        local = (
            worlds[i]
            if parents[i] < 0
            else np.linalg.inv(worlds[parents[i]]) @ worlds[i]
        )
        w = np.zeros(4)
        mujoco.mju_mat2Quat(w, local[:3, :3].ravel())
        nodes.append(
            {
                "name": model.body(
                    b
                ).name,  # the model's own name (phalanx aliases resolved)
                "translation": local[:3, 3].tolist(),
                "rotation": [*w[1:], w[0]],
            }
            | (
                {"children": [c for c, p in enumerate(parents) if p == i]}
                if i in parents
                else {}
            )
        )
    nodes.append(
        {"name": "Skeleton", "children": [i for i, p in enumerate(parents) if p < 0]}
    )
    nodes.append({"name": "Skin", "mesh": 0, "skin": 0})
    gltf = {
        "asset": {"version": "2.0", "generator": "myosuite.viz.skin_glb"},
        "scene": 0,
        "scenes": [{"nodes": [nbone, nbone + 1]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "Skin",
                "primitives": [{"attributes": attributes, "indices": indices}],
            }
        ],
        "skins": [
            {
                "name": "Skin",
                "joints": list(range(nbone)),
                "skeleton": nbone,
                "inverseBindMatrices": ibm,
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(blob))
    out += (
        struct.pack("<I4s", len(text), b"JSON")
        + text
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + bytes(blob)
    )
    Path(path).write_bytes(out)


def _node_matrices(gltf: dict) -> list[np.ndarray]:
    """World matrix of every node."""
    local = []
    for node in gltf["nodes"]:
        if "matrix" in node:
            local.append(np.array(node["matrix"], float).reshape(4, 4).T)
            continue
        m = np.eye(4)
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array([w, x, y, z], float))
        m[:3, :3] = rot.reshape(3, 3) * np.array(node.get("scale", [1, 1, 1]), float)
        m[:3, 3] = node.get("translation", [0, 0, 0])
        local.append(m)
    parent = {c: i for i, n in enumerate(gltf["nodes"]) for c in n.get("children", [])}

    def world(i: int) -> np.ndarray:
        return world(parent[i]) @ local[i] if i in parent else local[i]

    return [world(i) for i in range(len(local))]


def read_skin_glb(path: Path | str, model: mujoco.MjModel) -> Skin:
    """Read a rigged ``.glb`` skin (as written by :func:`write_skin_glb`, edited or not).

    Args:
        path: ``.glb`` file with one skinned mesh in *model*'s rest pose.
        model: Model whose bodies the joints are named after; each bone binds
            to its body's pose at ``qpos0``.

    Returns:
        The skin, with the file's vertices, faces, UVs and blend weights.

    Raises:
        ValueError: Not a glTF binary, or no skinned mesh, or a joint that is
            not a body of *model*.
    """
    buf = Path(path).read_bytes()
    magic, _, _ = struct.unpack_from("<4sII", buf)
    if magic != b"glTF":
        raise ValueError(f"Not a glTF binary: {path}")
    length, _ = struct.unpack_from("<I4s", buf, 12)
    gltf = json.loads(buf[20 : 20 + length])
    bin_start = 20 + length + 8
    blob = buf[bin_start:]

    def read(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        if "sparse" in acc or "bufferView" not in acc:
            raise ValueError(f"Unsupported glTF accessor in {path}")
        view = gltf["bufferViews"][acc["bufferView"]]
        dtype, width = np.dtype(_COMPONENTS[acc["componentType"]]), _WIDTH[acc["type"]]
        stride = view.get("byteStride") or dtype.itemsize * width
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(
            blob, np.uint8, (acc["count"] - 1) * stride + dtype.itemsize * width, start
        )
        rows = np.lib.stride_tricks.as_strided(
            raw, (acc["count"], dtype.itemsize * width), (stride, 1)
        )
        out = rows.copy().view(dtype).reshape(acc["count"], width).astype(np.float64)
        if acc.get("normalized"):
            out /= np.iinfo(dtype).max
        return out

    nodes = [n for n in gltf["nodes"] if "mesh" in n and "skin" in n]
    if not nodes:
        raise ValueError(f"No skinned mesh in {path}")
    node = nodes[0]
    joint_nodes = gltf["skins"][node["skin"]]["joints"]
    names = [gltf["nodes"][j].get("name", "") for j in joint_nodes]
    to_world = _node_matrices(gltf)[gltf["nodes"].index(node)]
    verts, uvs, faces, weight = [], [], [], []
    for prim in gltf["meshes"][node["mesh"]]["primitives"]:
        attr, base = prim["attributes"], sum(len(v) for v in verts)
        pos = read(attr["POSITION"])
        verts.append(pos @ to_world[:3, :3].T + to_world[:3, 3])
        uvs.append(
            read(attr["TEXCOORD_0"]) * [1, -1] + [0, 1]
            if "TEXCOORD_0" in attr
            else np.zeros((len(pos), 2))
        )
        faces.append(read(prim["indices"]).astype(np.int64).reshape(-1, 3) + base)
        w = np.zeros((len(pos), len(names)))
        s = 0
        while f"JOINTS_{s}" in attr:
            j, ws = read(attr[f"JOINTS_{s}"]).astype(int), read(attr[f"WEIGHTS_{s}"])
            for k in range(j.shape[1]):
                np.add.at(w, (np.arange(len(pos)), j[:, k]), ws[:, k])
            s += 1
        weight.append(w)
    vert = np.concatenate(verts) @ _TO_GLTF  # back to MuJoCo axes
    weight = np.concatenate(weight)
    weight /= np.where(
        weight.sum(1, keepdims=True) > 0, weight.sum(1, keepdims=True), 1
    )
    data = _rest(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    missing = [n for n, i in zip(names, ids) if i < 0]
    if missing:
        raise ValueError(f"Skin joints that are not bodies of the model: {missing}")
    used = [b for b in range(len(names)) if weight[:, b].any()]
    return Skin(
        vert.astype(np.float32),
        np.concatenate(uvs).astype(np.float32),
        np.concatenate(faces).astype(np.int32),
        [names[b] for b in used],
        data.xpos[[ids[b] for b in used]].astype(np.float32),
        data.xquat[[ids[b] for b in used]].astype(np.float32),
        [np.flatnonzero(weight[:, b]).astype(np.int32) for b in used],
        [weight[weight[:, b] > 0, b].astype(np.float32) for b in used],
    )
