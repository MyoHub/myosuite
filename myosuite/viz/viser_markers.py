# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Thin ``mjviser`` adapter for MyoSuite ``VizCallback`` / ``MotionClip`` markers and XY grids.

Scope against ``mjviser>=0.0.14,<0.0.17``:
- Reuses ``mjviser.ViserMujocoScene(num_envs=...)``, ``update_from_arrays``,
  public ``scene.fixed_bodies_frame`` camera tracking,
  ``mjviser.conversions.create_primitive_mesh`` and
  ``mjviser.conversions.mujoco_mesh_to_trimesh``, and ``viser``'s public
  ``add_batched_meshes_simple`` / ``add_arrows`` APIs.
- ``ViserMujocoScene._update_decor_from_mjvscene`` is private and overwrites its
  internal ``_mjv_scene`` via ``mjv_updateScene`` each frame rather than
  accepting caller-populated ``MjvScene`` buffers (such as
  :class:`~myosuite.viz.site_marker_viz.SiteMarkerViz` or
  :class:`~myosuite.viz.ghost_body_viz.GhostBodyViz`). Until ``mjviser`` exposes
  a public external-``MjvScene`` hook (e.g. ``update_user_scene(mjv_scene)``),
  :class:`ViserMarkers` bridges MyoSuite ``VizCallback`` buffers using public
  ``mjviser.conversions`` and ``viser`` handles only.
- ``ViserMujocoScene(num_envs=...)`` overlays environments at the origin unless
  caller-supplied ``body_xpos`` / ``mocap_pos`` arrays include spatial offsets;
  :class:`ViserMultiEnvScene` isolates that grid-offset + ``mj_forward`` step.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from myosuite.core.trajectory_io import MotionClip

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

_PRIMITIVE_SPECS: dict[str, tuple[mujoco.mjtGeom, list[float]]] = {
    "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, [1.0, 0.0, 0.0]),
    "box": (mujoco.mjtGeom.mjGEOM_BOX, [1.0, 1.0, 1.0]),
    "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, [1.0, 0.5, 0.0]),
    "capsule": (mujoco.mjtGeom.mjGEOM_CAPSULE, [1.0, 0.5, 0.0]),
    "ellipsoid": (mujoco.mjtGeom.mjGEOM_ELLIPSOID, [1.0, 1.0, 1.0]),
}


@functools.lru_cache(maxsize=1)
def _unit_primitives() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build unit primitive ``(vertices, faces)`` once via ``mjviser.conversions``."""
    from mjviser.conversions import create_primitive_mesh  # noqa: PLC0415

    spec = mujoco.MjSpec()
    order = list(_PRIMITIVE_SPECS)
    for name in order:
        gtype, gsize = _PRIMITIVE_SPECS[name]
        geom = spec.worldbody.add_geom(name=name)
        geom.type = gtype
        geom.size = gsize
    unit_model = spec.compile()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, name in enumerate(order):
        mesh = create_primitive_mesh(unit_model, gid)
        out[name] = (
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
        )
    return out


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
    """Bridge MyoSuite ``VizCallback`` / ``MotionClip`` ``MjvGeom`` markers into ``mjviser``.

    Markers are parented under ``scene.fixed_bodies_frame`` so they track
    ``ViserMujocoScene`` camera offsets automatically through public handles.

    Args:
        scene: ``mjviser.ViserMujocoScene`` (or an object exposing ``server`` and
            optional ``fixed_bodies_frame``).
        model: Compiled MuJoCo model used to resolve ``mjGEOM_MESH`` geoms.
        root_name: Scene-tree suffix for owned marker handles.
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
        else:
            self.root_path = clean

        self.handles: dict[str, Any] = {}
        self.opacities: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._geom_for_mesh_id: dict[int, int] = {}
        for gid in range(model.ngeom):
            if int(model.geom_type[gid]) == _MESH:
                mid = int(model.geom_dataid[gid])
                self._geom_for_mesh_id.setdefault(mid, gid)
        self._callback_scn: mujoco.MjvScene | None = None
        self._callback_maxgeom: int = 0

    def _mesh_vertices_faces(
        self, mesh_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if mesh_id < 0 or mesh_id >= self.model.nmesh:
            return None
        gid = self._geom_for_mesh_id.get(mesh_id)
        if gid is not None:
            from mjviser.conversions import mujoco_mesh_to_trimesh  # noqa: PLC0415

            mesh = mujoco_mesh_to_trimesh(self.model, gid)
            return (
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.uint32),
            )
        v_adr = int(self.model.mesh_vertadr[mesh_id])
        v_num = int(self.model.mesh_vertnum[mesh_id])
        f_adr = int(self.model.mesh_faceadr[mesh_id])
        f_num = int(self.model.mesh_facenum[mesh_id])
        if v_num <= 0 or f_num <= 0:
            return None
        return (
            np.asarray(self.model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float32),
            np.asarray(self.model.mesh_face[f_adr : f_adr + f_num], dtype=np.uint32),
        )

    def _resolve_mesh_id(self, dataid: int) -> int:
        if 0 <= dataid < self.model.nmesh:
            return dataid
        if 0 <= dataid < 2 * self.model.nmesh:
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
            self.opacities[key] = opacities
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
        handle.batched_opacities = opacities
        self.handles[key] = handle
        self.opacities[key] = opacities
        self._counts[key] = count

    def update(
        self,
        user_scn: mujoco.MjvScene | Sequence[mujoco.MjvScene] | None,
        *,
        env_offsets: np.ndarray | None = None,
    ) -> None:
        """Synchronize owned Viser marker handles with *user_scn*."""
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

        batches: dict[
            str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
        ] = {}
        mesh_ids_for_key: dict[str, int] = {}
        arrow_points: list[np.ndarray] = []
        arrow_colors: list[np.ndarray] = []
        arrow_radii: list[tuple[float, float, float]] = []
        quat_buf = np.empty(4, dtype=np.float64)

        for scn, shift in zip(scenes, offsets, strict=True):
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
                quat = quat_buf.astype(np.float32).copy()
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
                    half_h = float(size[2]) if float(size[2]) > 0.0 else float(size[1])
                    height = max(2.0 * half_h, 1e-6)
                    scale = np.array([r, r, height], dtype=np.float32)
                    batches.setdefault("cylinder", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type == _CAPSULE:
                    r = max(float(size[0]), 1e-6)
                    half_len = (
                        float(size[2]) if float(size[2]) > 0.0 else float(size[1])
                    )
                    # Unit capsule from create_primitive_mesh(r=1, half=0.5) has total Z=3.
                    scale_z = max(2.0 * (max(half_len, 0.0) + r) / 3.0, 1e-6)
                    scale = np.array([r, r, scale_z], dtype=np.float32)
                    batches.setdefault("capsule", []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )
                elif geom_type in _ARROWS:
                    shaft_r = max(float(size[0]), 1e-6)
                    head_r = float(size[1]) if float(size[1]) > 0.0 else 2.0 * shaft_r
                    total_len = max(float(size[2]), 1e-6)
                    end = pos + mat[:, 2] * total_len
                    arrow_points.append(np.stack([pos, end], axis=0).astype(np.float32))
                    arrow_colors.append(rgb)
                    arrow_radii.append((shaft_r, max(head_r, 1e-6), 0.2 * total_len))
                elif geom_type == _MESH:
                    mesh_id = self._resolve_mesh_id(int(g.dataid))
                    if mesh_id < 0 or mesh_id >= self.model.nmesh:
                        continue
                    key = f"mesh_{mesh_id}"
                    mesh_ids_for_key[key] = mesh_id
                    scale = np.ones(3, dtype=np.float32)
                    batches.setdefault(key, []).append(
                        (pos.astype(np.float32), quat, scale, rgb, alpha)
                    )

        primitives = _unit_primitives() if batches else {}
        atomic_fn = getattr(self.server, "atomic", None)
        ctx = atomic_fn() if callable(atomic_fn) else contextlib.nullcontext()
        active_keys = set(batches.keys())

        with ctx:
            for key, items in batches.items():
                if key in mesh_ids_for_key:
                    mesh_pair = self._mesh_vertices_faces(mesh_ids_for_key[key])
                    if mesh_pair is None:
                        active_keys.discard(key)
                        continue
                    verts, faces = mesh_pair
                else:
                    verts, faces = primitives[key]

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

            if arrow_points:
                active_keys.add("arrows")
                prev_arrows = self.handles.get("arrows")
                if prev_arrows is not None:
                    prev_arrows.remove()
                shaft_r, head_r, head_len = arrow_radii[0]
                self.handles["arrows"] = self.server.scene.add_arrows(
                    f"{self.root_path}/arrows",
                    np.stack(arrow_points, axis=0),
                    np.stack(arrow_colors, axis=0),
                    shaft_radius=shaft_r,
                    head_radius=head_r,
                    head_length=head_len,
                )
                self._counts["arrows"] = len(arrow_points)

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
        """Populate an ``MjvScene`` via *viz_fn* (``SiteMarkerViz`` / ``GhostBodyViz``) and sync."""
        if self._callback_scn is None or self._callback_maxgeom != maxgeom:
            self._callback_scn = mujoco.MjvScene(self.model, maxgeom=maxgeom)
            self._callback_maxgeom = maxgeom
        self._callback_scn.ngeom = 0
        viz_fn(step_idx, self.model, data, self._callback_scn)
        self.update(self._callback_scn)
        return self._callback_scn

    def update_from_clip(
        self,
        clip: MotionClip,
        frame_idx: int,
        *,
        rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 0.6),
        sphere_radius: float = 0.03,
        maxgeom: int = 256,
    ) -> mujoco.MjvScene:
        """Render ``clip.site_xpos[frame_idx]`` directly via :class:`SiteMarkerViz`."""
        from myosuite.viz.site_marker_viz import SiteMarkerViz  # noqa: PLC0415

        site_names = tuple(clip.site_names or ())
        viz = SiteMarkerViz(
            clip=clip,
            site_names=site_names,
            model=self.model,
            rgba=rgba,
            sphere_radius=sphere_radius,
        )
        cap = max(
            maxgeom, int(clip.site_xpos.shape[1]) if clip.site_xpos is not None else 0
        )
        if self._callback_scn is None or self._callback_maxgeom != cap:
            self._callback_scn = mujoco.MjvScene(self.model, maxgeom=cap)
            self._callback_maxgeom = cap
        viz.draw(frame_idx, self._callback_scn)
        self.update(self._callback_scn)
        return self._callback_scn

    def clear(self) -> None:
        """Remove all owned marker handles from the Viser scene."""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()
        self.opacities.clear()
        self._counts.clear()


def square_grid_offsets(num_envs: int, spacing: float = 2.0) -> np.ndarray:
    """Compute centered ``(num_envs, 2)`` XY grid offsets for parallel environments."""
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
    """Thin composition adapter over ``mjviser.ViserMujocoScene(num_envs=...)``.

    Delegates scene rendering and camera tracking to ``ViserMujocoScene`` while
    isolating per-env ``mj_forward`` grid offsets and :class:`ViserMarkers` sync.
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
        self.last_body_xpos: np.ndarray | None = None
        self.last_mocap_pos: np.ndarray | None = None
        self._data = mujoco.MjData(model)
        self._default_mocap_pos = self._data.mocap_pos.copy()
        self._default_mocap_quat = self._data.mocap_quat.copy()
        self._free_qpos_xy_adrs = tuple(
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scene, name)

    def update(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
        *,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        user_scenes: mujoco.MjvScene | Sequence[mujoco.MjvScene] | None = None,
    ) -> None:
        """Compute grid-offset forward kinematics and forward arrays to ``ViserMujocoScene``."""
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

        self.last_body_xpos = body_xpos
        self.last_mocap_pos = mpos_arr
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
    "square_grid_offsets",
]
