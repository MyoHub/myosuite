# Copyright (c) MyoSuite Authors. All rights reserved.
# Portions adapted from robfiras/loco-mujoco (loco_mujoco/core/visuals/viser_viewer.py,
# Copyright (c) 2024 Al-Hafez, MIT License).
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Viser user-scene marker bridge, sky/floor styling, and multi-env grid layout.

Adapted from ``robfiras/loco-mujoco`` (``loco_mujoco/core/visuals/viser_viewer.py``,
MIT License, Copyright (c) 2024 Al-Hafez) for native MuJoCo ``MjvScene`` markers
and MyoSuite's ``mjviser`` scenes.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import mujoco
import numpy as np

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_ARROWS = frozenset(
    int(t)
    for t in (
        mujoco.mjtGeom.mjGEOM_ARROW,
        mujoco.mjtGeom.mjGEOM_ARROW1,
        mujoco.mjtGeom.mjGEOM_ARROW2,
    )
)

_UNIT_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SKY_PRESETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "light": ((180, 210, 240), (242, 246, 250)),
    "dark": ((20, 28, 42), (52, 64, 82)),
}
_FLOOR_PRESETS: dict[str, tuple[int, int, int]] = {
    "light": (220, 224, 230),
    "dark": (46, 50, 58),
}


def _get_unit_mesh(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return cached ``(vertices, faces)`` for a unit primitive *kind*."""
    cached = _UNIT_MESH_CACHE.get(kind)
    if cached is not None:
        return cached

    import trimesh  # noqa: PLC0415

    if kind in ("sphere", "ellipsoid"):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    elif kind == "box":
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=16)
    elif kind == "capsule":
        mesh = trimesh.creation.capsule(height=1.0, radius=1.0, count=[8, 16])
    elif kind == "arrow_shaft":
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
        mesh.apply_translation([0.0, 0.0, 0.5])
    elif kind == "arrow_head":
        mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=12)
    else:
        raise ValueError(f"Unsupported unit mesh kind: {kind!r}")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    _UNIT_MESH_CACHE[kind] = (verts, faces)
    return verts, faces


def _clamp_rgb(rgb: Sequence[int | float] | np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(rgb, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected 3-element RGB tuple, got shape {arr.shape}")
    clamped = np.clip(np.round(arr), 0.0, 255.0).astype(int)
    return (int(clamped[0]), int(clamped[1]), int(clamped[2]))


def _normalize_env_offsets(
    env_offsets: np.ndarray | None,
    n_scenes: int,
) -> np.ndarray:
    if env_offsets is None:
        return np.zeros((n_scenes, 3), dtype=np.float64)
    arr = np.asarray(env_offsets, dtype=np.float64)
    if arr.ndim == 1 and n_scenes == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] != n_scenes or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"env_offsets must have shape ({n_scenes}, 2) or ({n_scenes}, 3), "
            f"got {arr.shape}"
        )
    if arr.shape[1] == 2:
        return np.column_stack([arr, np.zeros(n_scenes, dtype=np.float64)])
    return arr


class ViserMarkers:
    """Batched ``MjvGeom`` user-scene marker bridge for ``mjviser`` scenes.

    Converts active ``MjvGeom`` entries from one or more ``mujoco.MjvScene``
    buffers into ``viser`` batched meshes parented under the scene's body frame
    so markers automatically follow camera tracking.

    Args:
        scene: ``mjviser`` ``ViserMujocoScene`` (or compatible scene wrapper).
        model: Compiled MuJoCo model used to resolve ``mjGEOM_MESH`` assets.
        root_name: Scene-tree prefix for batched marker handles.
    """

    def __init__(
        self,
        scene: Any,
        model: mujoco.MjModel,
        *,
        root_name: str = "/myosuite_markers",
    ) -> None:
        self.scene = scene
        self.server = scene.server
        self.model = model

        stripped = root_name.strip("/")
        clean = f"/{stripped}" if stripped else "/myosuite_markers"
        fixed_frame = getattr(scene, "fixed_bodies_frame", None)
        frame_name = getattr(fixed_frame, "name", None)
        if frame_name:
            prefix = "/" + str(frame_name).strip("/")
            self.root_path = (
                clean
                if clean == prefix or clean.startswith(f"{prefix}/")
                else f"{prefix}{clean}"
            )
            self._use_scene_offset = False
        else:
            self.root_path = clean
            self._use_scene_offset = True

        self.handles: dict[str, Any] = {}
        self._counts: dict[str, int] = {}
        self._mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._callback_scn: mujoco.MjvScene | None = None
        self._callback_maxgeom: int = 0

    def _model_mesh(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        cached = self._mesh_cache.get(mesh_id)
        if cached is not None:
            return cached
        if mesh_id < 0 or mesh_id >= self.model.nmesh:
            return None
        v_adr = int(self.model.mesh_vertadr[mesh_id])
        v_num = int(self.model.mesh_vertnum[mesh_id])
        f_adr = int(self.model.mesh_faceadr[mesh_id])
        f_num = int(self.model.mesh_facenum[mesh_id])
        if v_num <= 0 or f_num <= 0:
            return None
        verts = np.asarray(
            self.model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float32
        ).copy()
        faces = np.asarray(
            self.model.mesh_face[f_adr : f_adr + f_num], dtype=np.uint32
        ).copy()
        self._mesh_cache[mesh_id] = (verts, faces)
        return verts, faces

    def _resolve_mesh_id(self, dataid: int) -> int:
        if 0 <= dataid < self.model.nmesh:
            return dataid
        if self.model.nmesh <= dataid < 2 * self.model.nmesh:
            return dataid // 2
        return -1

    def _upsert_batch(
        self,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        scales: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ) -> None:
        count = int(positions.shape[0])
        handle = self.handles.get(key)
        if handle is not None and self._counts.get(key) == count:
            handle.batched_positions = positions
            handle.batched_wxyzs = orientations
            handle.batched_scales = scales
            handle.batched_colors = colors
            handle.batched_opacities = opacities
            handle.visible = True
            return

        if handle is not None:
            handle.remove()

        handle = self.server.scene.add_batched_meshes_simple(
            f"{self.root_path}/{key}",
            vertices,
            faces,
            batched_wxyzs=orientations,
            batched_positions=positions,
            batched_scales=scales,
            batched_colors=colors,
            batched_opacities=opacities,
            lod="off",
            cast_shadow=False,
            receive_shadow=False,
        )
        self.handles[key] = handle
        self._counts[key] = count

    def update(
        self,
        user_scn: mujoco.MjvScene | Sequence[mujoco.MjvScene] | None,
        *,
        env_offsets: np.ndarray | None = None,
    ) -> None:
        """Synchronize Viser batched marker handles with *user_scn*.

        Args:
            user_scn: Single ``MjvScene``, per-environment sequence of
                ``MjvScene`` instances, or ``None`` to hide all markers.
            env_offsets: Optional ``(n_envs, 2)`` or ``(n_envs, 3)`` world-space
                offsets applied per scene.
        """
        if user_scn is None:
            scenes: list[mujoco.MjvScene] = []
        elif isinstance(user_scn, mujoco.MjvScene):
            scenes = [user_scn]
        else:
            scenes = list(user_scn)

        offsets = (
            _normalize_env_offsets(env_offsets, len(scenes))
            if scenes
            else np.zeros((0, 3), dtype=np.float64)
        )
        scene_shift = (
            np.asarray(
                getattr(self.scene, "_scene_offset", np.zeros(3)), dtype=np.float64
            )
            if self._use_scene_offset
            else np.zeros(3, dtype=np.float64)
        )

        # Per-key accumulators: (pos, quat, scale, rgb, opacity)
        batches: dict[
            str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
        ] = {}
        mesh_ids_for_key: dict[str, int] = {}
        quat_buf = np.empty(4, dtype=np.float64)

        for scn, env_off in zip(scenes, offsets):
            shift = env_off + scene_shift
            ngeom = min(int(scn.ngeom), len(scn.geoms))
            for idx in range(ngeom):
                g = scn.geoms[idx]
                rgba = np.asarray(g.rgba, dtype=np.float64)
                if rgba[3] <= 0.0:
                    continue

                geom_type = int(g.type)
                if geom_type in (_PLANE, _HFIELD):
                    continue

                pos = np.asarray(g.pos, dtype=np.float64) + shift
                mat = np.asarray(g.mat, dtype=np.float64).reshape(3, 3)
                mujoco.mju_mat2Quat(quat_buf, mat.reshape(9))
                quat = quat_buf.astype(np.float32)
                size = np.asarray(g.size, dtype=np.float64)
                rgb = np.clip(np.round(rgba[:3] * 255.0), 0.0, 255.0).astype(np.uint8)
                alpha = float(np.clip(rgba[3], 0.0, 1.0))

                if geom_type == _SPHERE:
                    r = max(float(size[0]), 1e-6)
                    scale = np.array([r, r, r], dtype=np.float32)
                    batches.setdefault("sphere", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type == _ELLIPSOID:
                    scale = np.maximum(size[:3], 1e-6).astype(np.float32)
                    batches.setdefault("ellipsoid", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type == _BOX:
                    scale = np.maximum(size[:3], 1e-6).astype(np.float32)
                    batches.setdefault("box", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type == _CYLINDER:
                    r = max(float(size[0]), 1e-6)
                    height = max(2.0 * float(size[2]), 1e-6)
                    scale = np.array([r, r, height], dtype=np.float32)
                    batches.setdefault("cylinder", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type == _CAPSULE:
                    r = max(float(size[0]), 1e-6)
                    half_len = max(float(size[2]), 0.0)
                    scale_z = max(2.0 * (half_len + r) / 3.0, 1e-6)
                    scale = np.array([r, r, scale_z], dtype=np.float32)
                    batches.setdefault("capsule", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type in _ARROWS:
                    shaft_r = max(float(size[0]), 1e-6)
                    head_r = float(size[1]) if float(size[1]) > 0.0 else 2.0 * shaft_r
                    total_len = max(float(size[2]), 1e-6)
                    shaft_len = 0.8 * total_len
                    head_len = 0.2 * total_len

                    shaft_scale = np.array(
                        [shaft_r, shaft_r, shaft_len], dtype=np.float32
                    )
                    head_pos = (pos + mat[:, 2] * shaft_len).astype(np.float32)
                    head_scale = np.array(
                        [max(head_r, 1e-6), max(head_r, 1e-6), head_len],
                        dtype=np.float32,
                    )
                    batches.setdefault("arrow_shaft", []).append(
                        (pos.astype(np.float32), quat, shaft_scale, rgb, alpha)
                    )
                    batches.setdefault("arrow_head", []).append(
                        (head_pos, quat.copy(), head_scale, rgb.copy(), alpha)
                    )
                elif geom_type == _MESH:
                    mesh_id = self._resolve_mesh_id(int(g.dataid))
                    if self._model_mesh(mesh_id) is None:
                        continue
                    key = f"mesh_{mesh_id}"
                    mesh_ids_for_key[key] = mesh_id
                    scale = np.ones(3, dtype=np.float32)
                    batches.setdefault(key, []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )

        atomic_fn = getattr(self.server, "atomic", None)
        ctx = atomic_fn() if callable(atomic_fn) else contextlib.nullcontext()
        active_keys = set(batches.keys())

        with ctx:
            for key, items in batches.items():
                if key in mesh_ids_for_key:
                    mesh_pair = self._model_mesh(mesh_ids_for_key[key])
                    if mesh_pair is None:
                        active_keys.discard(key)
                        continue
                    verts, faces = mesh_pair
                else:
                    verts, faces = _get_unit_mesh(key)

                positions = np.stack([it[0] for it in items], axis=0)
                orientations = np.stack([it[1] for it in items], axis=0)
                scales = np.stack([it[2] for it in items], axis=0)
                colors = np.stack([it[3] for it in items], axis=0)
                opacities = np.array([it[4] for it in items], dtype=np.float32)
                self._upsert_batch(
                    key,
                    verts,
                    faces,
                    positions,
                    orientations,
                    scales,
                    colors,
                    opacities,
                )

            for key, handle in self.handles.items():
                if key not in active_keys:
                    handle.visible = False

    def update_from_callback(
        self,
        step_idx: int,
        data: mujoco.MjData,
        viz_fn: Callable[[int, mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None],
        *,
        maxgeom: int = 256,
    ) -> mujoco.MjvScene:
        """Populate an internal ``MjvScene`` via *viz_fn* and update Viser markers.

        Compatible directly with ``SiteMarkerViz.as_viz_fn()`` and
        ``GhostBodyViz.as_viz_fn()``.

        Args:
            step_idx: Current step or frame index passed to *viz_fn*.
            data: Current ``MjData`` passed to *viz_fn*.
            viz_fn: Callback ``(step_idx, model, data, user_scn) -> None``.
            maxgeom: Capacity of the internal ``MjvScene`` buffer.

        Returns:
            The populated ``MjvScene`` instance.
        """
        if self._callback_scn is None or self._callback_maxgeom != maxgeom:
            self._callback_scn = mujoco.MjvScene(self.model, maxgeom=maxgeom)
            self._callback_maxgeom = maxgeom
        self._callback_scn.ngeom = 0
        viz_fn(step_idx, self.model, data, self._callback_scn)
        self.update(self._callback_scn)
        return self._callback_scn

    def clear(self) -> None:
        """Remove all marker handles from the Viser scene and reset caches."""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()
        self._counts.clear()


def _extract_model_sky_colors(
    model: mujoco.MjModel,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    sky_type = int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    for tex_id in range(model.ntex):
        if int(model.tex_type[tex_id]) != sky_type:
            continue
        w = int(model.tex_width[tex_id])
        h = int(model.tex_height[tex_id])
        nc = int(model.tex_nchannel[tex_id])
        adr = int(model.tex_adr[tex_id])
        if w <= 0 or h <= 0 or nc <= 0:
            continue
        raw = np.asarray(model.tex_data[adr : adr + w * h * nc], dtype=np.uint8)
        img = raw.reshape(h, w, nc)
        if nc == 1:
            img = np.repeat(img, 3, axis=2)
        else:
            img = img[:, :, :3]

        if h >= 6 and h % 6 == 0:
            face_h = h // 6
            up_face = img[2 * face_h : 3 * face_h]
            down_face = img[3 * face_h : 4 * face_h]
            top_rgb = _clamp_rgb(up_face[face_h // 2, w // 2])
            bot_rgb = _clamp_rgb(down_face[face_h // 2, w // 2])
        else:
            top_rgb = _clamp_rgb(img[0].mean(axis=0))
            bot_rgb = _clamp_rgb(img[-1].mean(axis=0))
        return (top_rgb, bot_rgb)
    return None


def apply_viser_sky(
    server: Any,
    model: mujoco.MjModel,
    sky: str | tuple[tuple[int, int, int], tuple[int, int, int]] | None = "model",
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Apply a vertical skybox gradient to a Viser server scene.

    Args:
        server: ``viser.ViserServer`` instance.
        model: Compiled ``MjModel`` inspected when ``sky == "model"``.
        sky: ``"model"`` to extract zenith/nadir colours from ``mjTEXTURE_SKYBOX``,
            ``"light"`` / ``"dark"`` presets, an explicit
            ``((r1, g1, b1), (r2, g2, b2))`` top/bottom RGB tuple, or ``None``
            to clear the background image.

    Returns:
        Applied ``(top_rgb, bottom_rgb)`` 0-255 integer tuples, or ``None`` if
        cleared or no skybox texture exists in *model*.

    Raises:
        ValueError: If *sky* is an unrecognized preset name or malformed tuple.
    """
    if sky is None or (isinstance(sky, str) and sky.lower() in ("none", "off")):
        server.scene.set_background_image(None)
        return None

    colors: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    if isinstance(sky, str):
        key = sky.lower()
        if key == "model":
            colors = _extract_model_sky_colors(model)
            if colors is None:
                return None
        elif key in _SKY_PRESETS:
            colors = _SKY_PRESETS[key]
        else:
            raise ValueError(
                f"Unsupported sky mode {sky!r}; expected 'model', 'light', 'dark', "
                "None, or ((r1, g1, b1), (r2, g2, b2))."
            )
    else:
        if len(sky) != 2:
            raise ValueError(f"Expected (top_rgb, bottom_rgb) pair, got {sky!r}")
        colors = (_clamp_rgb(sky[0]), _clamp_rgb(sky[1]))

    top = np.array(colors[0], dtype=np.float64)
    bottom = np.array(colors[1], dtype=np.float64)
    gradient = np.linspace(top, bottom, 64, dtype=np.float64)
    image = np.repeat(
        np.clip(np.round(gradient), 0.0, 255.0).astype(np.uint8)[:, None, :],
        4,
        axis=1,
    )
    server.scene.set_background_image(image)
    return colors


def _extract_model_floor_colors(
    model: mujoco.MjModel,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None, float] | None:
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != _PLANE:
            continue
        geom_rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
        if geom_rgba[3] <= 0.0:
            continue

        mat_id = int(model.geom_matid[geom_id])
        if 0 <= mat_id < model.nmat:
            mat_rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float64)
            opacity = float(np.clip(mat_rgba[3], 0.0, 1.0))
            tex_id = int(
                model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]
            )
            if tex_id < 0:
                tex_id = int(
                    model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)]
                )
            if 0 <= tex_id < model.ntex:
                w = int(model.tex_width[tex_id])
                h = int(model.tex_height[tex_id])
                nc = int(model.tex_nchannel[tex_id])
                adr = int(model.tex_adr[tex_id])
                if w > 0 and h > 0 and nc > 0:
                    raw = np.asarray(
                        model.tex_data[adr : adr + w * h * nc], dtype=np.float64
                    ).reshape(h, w, nc)
                    rgb_img = np.repeat(raw, 3, axis=2) if nc == 1 else raw[:, :, :3]
                    tint = np.clip(mat_rgba[:3], 0.0, 1.0)
                    c0 = _clamp_rgb(rgb_img[0, 0] * tint)
                    c1 = _clamp_rgb(rgb_img[0, max(w // 2, 0)] * tint)
                    mean_rgb = _clamp_rgb(rgb_img.mean(axis=(0, 1)) * tint)
                    accent = c1 if c1 != c0 else None
                    primary = c0 if accent is not None else mean_rgb
                    return (primary, accent, opacity)
            return (_clamp_rgb(mat_rgba[:3] * 255.0), None, opacity)

        return (
            _clamp_rgb(geom_rgba[:3] * 255.0),
            None,
            float(np.clip(geom_rgba[3], 0.0, 1.0)),
        )
    return None


def _derive_grid_line_colors(
    base_rgb: tuple[int, int, int],
    accent_rgb: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    base = np.array(base_rgb, dtype=np.float64)
    if accent_rgb is not None:
        acc = np.array(accent_rgb, dtype=np.float64)
        return (_clamp_rgb(acc), _clamp_rgb(0.5 * (base + acc)))
    luminance = float(base.mean())
    delta = -28.0 if luminance > 127.0 else 28.0
    cell = _clamp_rgb(base + delta)
    section = _clamp_rgb(base + 1.8 * delta)
    return (cell, section)


def _find_floor_grid_handles(server: Any) -> list[Any]:
    handles_map = getattr(server.scene, "_handle_from_node_name", None)
    if not isinstance(handles_map, dict):
        return []
    return [
        h
        for h in handles_map.values()
        if hasattr(getattr(h, "_impl", None), "props")
        and hasattr(h._impl.props, "plane_color")
        and hasattr(h._impl.props, "cell_color")
    ]


def apply_viser_floor(
    server: Any,
    model: mujoco.MjModel,
    floor_color: str | tuple[int, int, int] | None = "model",
) -> tuple[int, int, int] | None:
    """Style the Viser floor grid from *model*'s plane material/texture or *floor_color*.

    Args:
        server: ``viser.ViserServer`` instance.
        model: Compiled ``MjModel`` inspected when ``floor_color == "model"``.
        floor_color: ``"model"`` to inspect ``mjGEOM_PLANE`` materials/checkerboard
            textures, ``"light"`` / ``"dark"`` presets, an explicit
            ``(r, g, b)`` 0-255 tuple, or ``None`` to hide floor grids.

    Returns:
        Applied ``(r, g, b)`` floor colour tuple, or ``None`` if hidden or no
        plane geom exists in *model* when ``floor_color == "model"``.

    Raises:
        ValueError: If *floor_color* is an unrecognized preset or malformed tuple.
    """
    grid_handles = _find_floor_grid_handles(server)

    if floor_color is None or (
        isinstance(floor_color, str) and floor_color.lower() in ("none", "off")
    ):
        for handle in grid_handles:
            handle.visible = False
        return None

    accent_rgb: tuple[int, int, int] | None = None
    opacity = 0.75
    if isinstance(floor_color, str):
        key = floor_color.lower()
        if key == "model":
            extracted = _extract_model_floor_colors(model)
            if extracted is None:
                return None
            rgb, accent_rgb, model_alpha = extracted
            opacity = float(np.clip(model_alpha, 0.4, 0.9))
        elif key in _FLOOR_PRESETS:
            rgb = _FLOOR_PRESETS[key]
        else:
            raise ValueError(
                f"Unsupported floor_color {floor_color!r}; expected 'model', "
                "'light', 'dark', None, or (r, g, b)."
            )
    else:
        rgb = _clamp_rgb(floor_color)

    cell_rgb, section_rgb = _derive_grid_line_colors(rgb, accent_rgb)
    if not grid_handles:
        server.scene.add_grid(
            "/fixed_bodies/world/floor",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_color=rgb,
            plane_opacity=opacity,
            cell_color=cell_rgb,
            section_color=section_rgb,
        )
    else:
        for handle in grid_handles:
            handle.plane_color = rgb
            handle.plane_opacity = opacity
            handle.cell_color = cell_rgb
            handle.section_color = section_rgb
            handle.visible = True
    return rgb


def square_grid_offsets(num_envs: int, spacing: float = 2.0) -> np.ndarray:
    """Compute centered ``(num_envs, 2)`` XY grid offsets for parallel environments.

    Args:
        num_envs: Number of parallel environments (``>= 1``).
        spacing: Centre-to-centre distance in metres between adjacent cells.

    Returns:
        ``(num_envs, 2)`` float64 array of XY offsets centred around ``(0, 0)``.

    Raises:
        ValueError: If ``num_envs < 1`` or ``spacing < 0``.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if spacing < 0.0:
        raise ValueError(f"spacing must be >= 0, got {spacing}")
    if num_envs == 1:
        return np.zeros((1, 2), dtype=np.float64)

    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    idx = np.arange(num_envs, dtype=np.float64)
    col = idx % cols
    row = idx // cols
    x = (col - 0.5 * (cols - 1)) * float(spacing)
    y = (row - 0.5 * (rows - 1)) * float(spacing)
    return np.column_stack([x, y])


class ViserMultiEnvScene:
    """Multi-environment XY grid wrapper around ``mjviser``'s ``ViserMujocoScene``.

    Supports both free-joint models (offsetting free-joint XY coordinates in
    ``qpos`` + ``mocap_pos``) and fixed-base models such as
    ``myoElbowPose1D6MRandom-v0`` (offsetting non-world ``body_xpos[:, 1:, :2]``
    + ``mocap_pos``), while synchronizing per-environment ``MjvScene`` markers
    on the same grid.

    Args:
        server: ``viser.ViserServer`` instance.
        model: Compiled MuJoCo model shared across environments.
        num_envs: Number of parallel environments.
        spacing: Grid cell spacing in metres.
        root_name: Scene-tree prefix for :class:`ViserMarkers`.
        camera_tracking: Override ``scene.camera_tracking_enabled``; defaults to
            ``True`` for a single environment and ``False`` for ``num_envs > 1``.
    """

    square_grid_offsets = staticmethod(square_grid_offsets)

    def __init__(
        self,
        server: Any,
        model: mujoco.MjModel,
        num_envs: int = 1,
        *,
        spacing: float = 2.0,
        root_name: str = "/myosuite_markers",
        camera_tracking: bool | None = None,
    ) -> None:
        from mjviser import ViserMujocoScene  # noqa: PLC0415

        self.server = server
        self.model = model
        self.num_envs = int(num_envs)
        self.spacing = float(spacing)
        self.grid_offsets = square_grid_offsets(self.num_envs, spacing=self.spacing)
        self.scene = ViserMujocoScene(server, model, num_envs=self.num_envs)
        if camera_tracking is None:
            camera_tracking = self.num_envs == 1
        self.scene.camera_tracking_enabled = bool(camera_tracking)

        self.markers = ViserMarkers(self.scene, model, root_name=root_name)
        self._data = mujoco.MjData(model)
        self._default_mocap_pos = self._data.mocap_pos.copy()
        self._default_mocap_quat = self._data.mocap_quat.copy()
        self._free_qpos_xy_adrs = tuple(
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        )

    def update(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
        *,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        user_scenes: mujoco.MjvScene | Sequence[mujoco.MjvScene] | None = None,
    ) -> None:
        """Advance forward kinematics for all environments and update Viser.

        Args:
            qpos: Batched joint positions ``(num_envs, nq)`` (or ``(nq,)`` when
                ``num_envs == 1``).
            qvel: Optional batched joint velocities ``(num_envs, nv)``.
            mocap_pos: Optional mocap positions ``(num_envs, nmocap, 3)``.
            mocap_quat: Optional mocap quaternions ``(num_envs, nmocap, 4)``.
            user_scenes: Optional single ``MjvScene`` or per-env sequence of
                ``MjvScene`` buffers rendered via :attr:`markers`.

        Raises:
            ValueError: If input array shapes do not match ``(num_envs, ...)``.
        """
        qpos_arr = np.array(qpos, dtype=np.float64, copy=True)
        if qpos_arr.ndim == 1 and self.num_envs == 1:
            qpos_arr = qpos_arr[None, :]
        if qpos_arr.shape != (self.num_envs, self.model.nq):
            raise ValueError(
                f"Expected qpos shape {(self.num_envs, self.model.nq)}, "
                f"got {qpos_arr.shape}"
            )

        if qvel is None:
            qvel_arr = np.zeros((self.num_envs, self.model.nv), dtype=np.float64)
        else:
            qvel_arr = np.array(qvel, dtype=np.float64, copy=True)
            if qvel_arr.ndim == 1 and self.num_envs == 1:
                qvel_arr = qvel_arr[None, :]
            if qvel_arr.shape != (self.num_envs, self.model.nv):
                raise ValueError(
                    f"Expected qvel shape {(self.num_envs, self.model.nv)}, "
                    f"got {qvel_arr.shape}"
                )

        nmocap = int(self.model.nmocap)
        if nmocap > 0:
            if mocap_pos is None:
                mpos_arr = np.repeat(
                    self._default_mocap_pos[None, :, :], self.num_envs, axis=0
                )
            else:
                mpos_arr = np.array(mocap_pos, dtype=np.float64, copy=True)
                if mpos_arr.shape != (self.num_envs, nmocap, 3):
                    raise ValueError(
                        f"Expected mocap_pos shape {(self.num_envs, nmocap, 3)}, "
                        f"got {mpos_arr.shape}"
                    )
            mpos_arr[:, :, :2] += self.grid_offsets[:, None, :]

            if mocap_quat is None:
                mquat_arr = np.repeat(
                    self._default_mocap_quat[None, :, :], self.num_envs, axis=0
                )
            else:
                mquat_arr = np.array(mocap_quat, dtype=np.float64, copy=True)
                if mquat_arr.shape != (self.num_envs, nmocap, 4):
                    raise ValueError(
                        f"Expected mocap_quat shape {(self.num_envs, nmocap, 4)}, "
                        f"got {mquat_arr.shape}"
                    )
        else:
            mpos_arr = np.zeros((self.num_envs, 0, 3), dtype=np.float64)
            mquat_arr = np.zeros((self.num_envs, 0, 4), dtype=np.float64)

        has_free_joint = bool(self._free_qpos_xy_adrs)
        if has_free_joint:
            for adr in self._free_qpos_xy_adrs:
                qpos_arr[:, adr : adr + 2] += self.grid_offsets

        nbody = int(self.model.nbody)
        body_xpos = np.empty((self.num_envs, nbody, 3), dtype=np.float64)
        body_xmat = np.empty((self.num_envs, nbody, 3, 3), dtype=np.float64)

        for env_i in range(self.num_envs):
            self._data.qpos[:] = qpos_arr[env_i]
            if self.model.nv > 0:
                self._data.qvel[:] = qvel_arr[env_i]
            if nmocap > 0:
                self._data.mocap_pos[:] = mpos_arr[env_i]
                self._data.mocap_quat[:] = mquat_arr[env_i]
            mujoco.mj_forward(self.model, self._data)
            body_xpos[env_i] = self._data.xpos
            body_xmat[env_i] = self._data.xmat.reshape(nbody, 3, 3)
            if not has_free_joint and nbody > 1:
                body_xpos[env_i, 1:, :2] += self.grid_offsets[env_i]

        self.scene.update_from_arrays(
            body_xpos,
            body_xmat,
            mocap_pos=mpos_arr,
            mocap_quat=mquat_arr,
            qpos=qpos_arr,
            qvel=qvel_arr,
        )

        if user_scenes is None:
            self.markers.update(None)
        elif isinstance(user_scenes, mujoco.MjvScene):
            single_off = self.grid_offsets[0] if self.num_envs == 1 else None
            self.markers.update(user_scenes, env_offsets=single_off)
        else:
            self.markers.update(user_scenes, env_offsets=self.grid_offsets)

    def clear_markers(self) -> None:
        """Clear all user-scene marker handles managed by :attr:`markers`."""
        self.markers.clear()


__all__ = [
    "ViserMarkers",
    "ViserMultiEnvScene",
    "apply_viser_floor",
    "apply_viser_sky",
    "square_grid_offsets",
]
