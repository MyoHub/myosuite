# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Load a MuJoCo ``.skn`` body skin and pose it from simulation state.

The skin is purely visual: it is posed with MuJoCo's linear-blend skinning
(``mjv_updateSkin``) from body poses, so the simulated model is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# Bundled full-body skin; source and license in assets/CREDITS.md.
FULLBODY_SKIN = Path(__file__).parent / "assets" / "myofullbody.skn"
_NAME_BYTES = 40


@dataclass
class Skin:
    """A skin mesh with per-bone vertex weights, as stored in a ``.skn`` file.

    Attributes:
        vert: ``(V, 3)`` bind-pose vertices.
        texcoord: ``(V, 2)`` UVs, or ``(0, 2)``.
        face: ``(F, 3)`` triangle vertex indices.
        bone_names: Body name of each bone.
        bindpos: ``(B, 3)`` body position at bind time.
        bindquat: ``(B, 4)`` body orientation at bind time (w, x, y, z).
        vertid: Vertex indices influenced by each bone.
        vertweight: Weights matching ``vertid``.
    """

    vert: np.ndarray
    texcoord: np.ndarray
    face: np.ndarray
    bone_names: list[str]
    bindpos: np.ndarray
    bindquat: np.ndarray
    vertid: list[np.ndarray]
    vertweight: list[np.ndarray]


# Finger phalanges are "<phalanx><finger>_<side>" in the bundled skin (``midph2_r``) and
# "<finger><phalanx>_<side>" in musclemimic_models (``2midph_r``); bones bind under either.
_PHALANX = re.compile(
    r"^(proxph|midph|distph)(\d)(_[lr])$|^(\d)(proxph|midph|distph)(_[lr])$"
)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    """Id of the body *name*, else of its phalanx alias (``midph2_r`` <-> ``2midph_r``), else -1."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    match = _PHALANX.match(name)
    if body >= 0 or match is None:
        return body
    phalanx, finger, side = (
        (match[1], match[2], match[3]) if match[1] else (match[5], match[4], match[6])
    )
    alias = f"{finger}{phalanx}{side}" if match[1] else f"{phalanx}{finger}{side}"
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alias)


def load_skn(path: Path | str) -> Skin:
    """Read a MuJoCo binary skin file.

    Args:
        path: ``.skn`` file, or ``"fullbody"`` for the bundled MyoSuite body skin.

    Returns:
        The parsed skin.

    Raises:
        ValueError: The file is truncated or its indices are out of range.
    """
    path = FULLBODY_SKIN if str(path) == "fullbody" else Path(path)
    buf = path.read_bytes()
    offset = 0

    def take(dtype: type, count: int) -> np.ndarray:
        nonlocal offset
        size = np.dtype(dtype).itemsize * count
        if offset + size > len(buf):
            raise ValueError(f"Truncated skin file: {path}")
        out = np.frombuffer(buf, dtype, count, offset)
        offset += size
        return out

    nvert, ntex, nface, nbone = (int(n) for n in take(np.int32, 4))
    if min(nvert, ntex, nface, nbone) < 0 or ntex not in (0, nvert):
        raise ValueError(f"Invalid skin header in {path}")
    vert = take(np.float32, 3 * nvert).reshape(-1, 3)
    texcoord = take(np.float32, 2 * ntex).reshape(-1, 2)
    face = take(np.int32, 3 * nface).reshape(-1, 3)
    names, bindpos, bindquat, vertid, vertweight = [], [], [], [], []
    for _ in range(nbone):
        raw = take(np.uint8, _NAME_BYTES).tobytes()
        names.append(raw.split(b"\0")[0].decode())
        bindpos.append(take(np.float32, 3))
        bindquat.append(take(np.float32, 4))
        count = int(take(np.int32, 1)[0])
        vertid.append(take(np.int32, count))
        vertweight.append(take(np.float32, count))
    if offset != len(buf):
        raise ValueError(f"Unexpected trailing data in {path}")
    ids = np.concatenate([face.ravel(), *vertid]) if nbone else face.ravel()
    if ids.size and (ids.min() < 0 or ids.max() >= nvert):
        raise ValueError(f"Vertex index out of range in {path}")
    return Skin(
        vert,
        texcoord,
        face,
        names,
        np.array(bindpos).reshape(-1, 3),
        np.array(bindquat).reshape(-1, 4),
        vertid,
        vertweight,
    )


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)


class SkinPose:
    """A skin bound to a model's bodies; poses its vertices from ``MjData``."""

    def __init__(self, skin: Skin, body_ids: np.ndarray, inflate: float) -> None:
        self.skin, self.body_ids, self.inflate = skin, body_ids, inflate
        self._bone = np.concatenate(
            [np.full(len(v), b) for b, v in enumerate(skin.vertid)]
        )
        self._vert = np.concatenate(skin.vertid)
        weight = np.concatenate(skin.vertweight).astype(np.float64)
        # MuJoCo normalises each vertex's weights to sum to one.
        total = np.bincount(self._vert, weight, len(skin.vert))
        self._weight = weight / np.where(total > 0, total, 1)[self._vert]
        bind_rot = _quat_to_mat(skin.bindquat.astype(np.float64))
        # Each vertex in its bones' bind frames, so posing is one rotation each.
        self._local = np.einsum(
            "nji,nj->ni",
            bind_rot[self._bone],
            skin.vert[self._vert] - skin.bindpos[self._bone],
        )

    @classmethod
    def bind(cls, skin: Skin, model: mujoco.MjModel, inflate: float = 0.0) -> SkinPose:
        """Bind *skin* bones to the bodies of the same name.

        Args:
            skin: Parsed skin.
            model: Compiled model whose bodies drive the skin.
            inflate: Offset of every vertex along its normal (m).

        Returns:
            The bound skin.

        Raises:
            ValueError: Some bone names are not bodies of *model*.
        """
        ids = np.array([_body_id(model, n) for n in skin.bone_names], dtype=int)
        missing = [n for n, i in zip(skin.bone_names, ids) if i < 0]
        if missing:
            raise ValueError(f"Skin bones missing from the model: {missing}")
        return cls(skin, ids, inflate)

    def vertices(self, data: mujoco.MjData) -> np.ndarray:
        """Posed skin vertices after ``mj_forward`` / ``mj_step``.

        Args:
            data: Simulation state of the bound model.

        Returns:
            ``(V, 3)`` float32 world-frame vertices.
        """
        rot = _quat_to_mat(data.xquat[self.body_ids])[self._bone]
        world = np.einsum("nij,nj->ni", rot, self._local)
        world += data.xpos[self.body_ids][self._bone]
        vert = np.zeros((len(self.skin.vert), 3))
        np.add.at(vert, self._vert, self._weight[:, None] * world)
        if self.inflate:
            vert += self.inflate * self.normals(vert)
        return vert.astype(np.float32)

    def normals(self, vert: np.ndarray) -> np.ndarray:
        """Area-weighted unit vertex normals of the skin at *vert*.

        Args:
            vert: ``(V, 3)`` vertices.

        Returns:
            ``(V, 3)`` normals.
        """
        a, b, c = (vert[self.skin.face[:, i]] for i in range(3))
        face_normal = np.cross(b - a, c - a)
        normal = np.zeros_like(vert)
        for i in range(3):
            np.add.at(normal, self.skin.face[:, i], face_normal)
        return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


class ViserSkin:
    """A posed skin drawn in an mjviser scene, which does not draw MuJoCo skins.

    Example::

        skin = ViserSkin(scene, SkinPose.bind(load_skn("fullbody"), model), data)
        # each frame, after scene.update_from_mjdata(data):
        skin.update(data)
    """

    def __init__(
        self,
        scene: Any,
        pose: SkinPose,
        data: mujoco.MjData,
        *,
        color: tuple[int, int, int] = (217, 178, 158),
        opacity: float = 0.45,
        name: str = "skin",
    ) -> None:
        """Add the skin mesh to *scene*.

        Args:
            scene: ``mjviser.scene.ViserMujocoScene`` of the bound model.
            pose: Skin bound to the scene's model.
            data: Current simulation state.
            color: RGB colour, 0-255.
            opacity: Mesh opacity, 0-1.
            name: Node name under the scene's body frame.
        """
        self.pose = pose
        # Under mjviser's body frame, so the skin follows camera tracking.
        self.mesh = scene.server.scene.add_mesh_simple(
            f"{scene.fixed_bodies_frame.name}/{name}",
            pose.vertices(data),
            pose.skin.face,
            color=color,
            opacity=opacity,
            side="double",
        )

    def update(self, data: mujoco.MjData) -> None:
        """Re-pose the mesh from *data*.

        Args:
            data: Simulation state after ``mj_forward`` / ``mj_step``.
        """
        self.mesh.vertices = self.pose.vertices(data)
