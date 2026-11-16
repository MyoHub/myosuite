# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""3D muscle mesh reconstruction and glTF export over kinematic trajectories.

Reconstructs per-frame 3D muscle surface meshes and activations from any compiled
:class:`mujoco.MjModel` (nominal or anthropometrically scaled) and ``qpos`` trajectory:

* ``mesh_mode="volumetric"`` builds constant-volume fusiform muscle bellies and tendons
  along MuJoCo's wrapped tendon paths (:mod:`myosuite.viz.muscle_tubes`), requiring no
  external assets.
* ``mesh_mode="atlas"`` / ``"atlas-hd"`` (or a path to a rigged ``.glb``) deforms
  skinned anatomical muscle meshes by linear-blend skinning on the model's bone geoms,
  preserving anthropometric bone-scale ratios when the target model has been scaled.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.viz.blender_render import (
    MUSCLE_ACTIVE,
    MUSCLE_MESH_ENV,
    MUSCLE_MESHES,
    MUSCLE_RELAXED,
    _anatomy_bodies,
    _binding_targets,
    _resolve_muscle_mesh,
)
from myosuite.viz.muscle_tubes import (
    belly_radius,
    belly_volume,
    muscle_actuators,
    muscle_shape,
    radius_profile,
    resample_path,
    tube_mesh,
)
from myosuite.viz.skin_glb import _COMPONENTS, _TO_GLTF, _WIDTH, _node_matrices

BONE_COLOR = (0.83, 0.79, 0.72, 1.0)


@dataclass(frozen=True)
class MuscleMeshReconstruction:
    """Reconstructed 3D muscle meshes, tendon lengths and activations across a clip.

    Attributes:
        muscle_names: Names of the reconstructed muscles ``(M,)``.
        vertices: World-space vertices in metres (MuJoCo Z-up). Either a dense
            ``(T, M, V, 3)`` ``float32`` array (``mode="volumetric"``) or a tuple of
            ``M`` ``(T, V_m, 3)`` ``float32`` arrays (skinned ``.glb`` meshes).
        faces: Triangle indices ``(F, 3)`` ``int32`` (shared array for ``"volumetric"``,
            or a tuple of ``M`` ``(F_m, 3)`` arrays for skinned ``.glb`` meshes).
        tendon_length: ``(T, M_act)`` tendon path lengths in metres for
            :func:`~myosuite.viz.muscle_tubes.muscle_actuators`.
        activation: ``(T, M)`` muscle activations in ``[0, 1]`` aligned with
            ``muscle_names``.
        fps: Trajectory frame rate in Hz.
        mode: Mesh source identifier (``"volumetric"``, ``"atlas"``, ``"atlas-hd"``, or
            ``.glb`` path).
    """

    muscle_names: tuple[str, ...]
    vertices: np.ndarray | tuple[np.ndarray, ...]
    faces: np.ndarray | tuple[np.ndarray, ...]
    tendon_length: np.ndarray
    activation: np.ndarray
    fps: float
    mode: str = "volumetric"

    @property
    def num_frames(self) -> int:
        return int(self.activation.shape[0])

    @property
    def num_muscles(self) -> int:
        return len(self.muscle_names)

    def _index(self, muscle: int | str) -> int:
        if isinstance(muscle, str):
            return self.muscle_names.index(muscle)
        return int(muscle)

    def muscle_vertices(self, muscle: int | str) -> np.ndarray:
        """Return ``(T, V_m, 3)`` world-space vertices for *muscle*."""
        idx = self._index(muscle)
        if isinstance(self.vertices, np.ndarray):
            return self.vertices[:, idx]
        return self.vertices[idx]

    def muscle_faces(self, muscle: int | str) -> np.ndarray:
        """Return ``(F_m, 3)`` triangle indices for *muscle*."""
        idx = self._index(muscle)
        if isinstance(self.faces, np.ndarray):
            return self.faces
        return self.faces[idx]

    def write_glb(
        self,
        path: str | Path,
        *,
        frame: int = 0,
        model: mujoco.MjModel | None = None,
        qpos: np.ndarray | None = None,
        include_bones: bool = True,
    ) -> Path:
        """Export one posed frame as a glTF 2.0 binary (``.glb``)."""
        return write_muscle_meshes_glb(
            self,
            path,
            frame=frame,
            model=model,
            qpos=qpos,
            include_bones=include_bones,
        )

    def save_npz(self, path: str | Path) -> Path:
        """Save reconstructed meshes, tendon lengths and activations to ``.npz``."""
        return save_muscle_meshes_npz(self, path)


def _triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tris: list[list[int]] = []
    pos = 0
    for c in counts.tolist():
        poly = indices[pos : pos + c]
        for k in range(1, c - 1):
            tris.append([int(poly[0]), int(poly[k]), int(poly[k + 1])])
        pos += c
    return np.asarray(tris, dtype=np.int32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    out = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(out, faces[:, 0], fn)
    np.add.at(out, faces[:, 1], fn)
    np.add.at(out, faces[:, 2], fn)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, out / np.maximum(norm, 1e-12), [0.0, 0.0, 1.0])


def _differentiate_fallback(
    model: mujoco.MjModel, qpos: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(qpos)
    if n == 1:
        z = np.zeros((1, model.nv), dtype=np.float64)
        return z, z.copy()
    step = np.empty((n - 1, model.nv), dtype=np.float64)
    for t in range(n - 1):
        mujoco.mj_differentiatePos(model, step[t], dt, qpos[t], qpos[t + 1])
    qvel = np.empty((n, model.nv), dtype=np.float64)
    qvel[0], qvel[-1] = step[0], step[-1]
    if n == 3:
        qvel[1] = 0.5 * (step[0] + step[1])
        qacc = np.tile((step[1] - step[0]) / dt, (3, 1))
    else:
        qacc = np.zeros((n, model.nv), dtype=np.float64)
    return qvel, qacc


def _compute_activations(
    model: mujoco.MjModel,
    muscles: np.ndarray,
    tendon_length: np.ndarray,
    qpos: np.ndarray,
    fps: float,
    *,
    mode: str,
    qvel: np.ndarray | None,
    qacc: np.ndarray | None,
    so_config: Any | None,
) -> np.ndarray:
    T, M = len(qpos), len(muscles)
    if M == 0 or mode == "none":
        return np.zeros((T, M), dtype=np.float32)
    if mode == "shortening":
        act = np.zeros((T, M), dtype=np.float64)
        for t in range(1, T):
            speed = (
                (tendon_length[t - 1] - tendon_length[t])
                * fps
                / np.maximum(tendon_length[t], 1e-3)
            )
            act[t] = act[t - 1] + (np.clip(speed / 1.5, 0.0, 1.0) - act[t - 1]) * 0.35
        return act.astype(np.float32)
    if mode == "static_optimization":
        from myosuite.biomechanics.inverse_dynamics import clip_kinematics
        from myosuite.biomechanics.static_optimization import (
            StaticOptimizationConfig,
            static_optimization,
        )

        dt = 1.0 / fps
        if qvel is None or qacc is None:
            if T >= 4:
                kin = clip_kinematics(model, qpos, dt)
                qv, qa = kin.qvel, kin.qacc
            else:
                qv, qa = _differentiate_fallback(model, qpos, dt)
            qvel = qv if qvel is None else qvel
            qacc = qa if qacc is None else qacc
        if so_config is None:
            has_world_floor = any(
                model.geom_bodyid[g] == 0
                and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)
                for g in range(model.ngeom)
            )
            so_config = StaticOptimizationConfig(
                contact_forces=has_world_floor,
                contact_distance=0.02,
            )
        res = static_optimization(model, qpos, qvel, qacc, config=so_config)
        return np.clip(res.activation[:, muscles], 0.0, 1.0).astype(np.float32)
    raise ValueError(
        f"Unknown activation_mode {mode!r}; expected 'shortening', "
        "'static_optimization', or 'none'"
    )


def _default_atlas_reference() -> mujoco.MjModel:
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body

    return compiled_body("myofullbody")


def _read_glb_skinned_meshes(path: Path) -> list[dict[str, Any]]:
    buf = path.read_bytes()
    if len(buf) < 20:
        raise ValueError(f"Not a glTF binary: {path}")
    magic, _, _ = struct.unpack_from("<4sII", buf)
    if magic != b"glTF":
        raise ValueError(f"Not a glTF binary: {path}")
    length, _ = struct.unpack_from("<I4s", buf, 12)
    gltf = json.loads(buf[20 : 20 + length])
    blob = buf[20 + length + 8 :]

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

    node_worlds = _node_matrices(gltf)
    meshes_out: list[dict[str, Any]] = []
    for node_idx, node in enumerate(gltf.get("nodes", [])):
        if "mesh" not in node or "skin" not in node:
            continue
        skin = gltf["skins"][node["skin"]]
        joint_nodes = skin["joints"]
        joint_names = [gltf["nodes"][j].get("name", f"joint_{j}") for j in joint_nodes]
        joint_pos_mj = (
            np.array([node_worlds[j][:3, 3] for j in joint_nodes], dtype=np.float64)
            @ _TO_GLTF
        )
        to_world = node_worlds[node_idx]
        mesh_def = gltf["meshes"][node["mesh"]]
        name = node.get("name") or mesh_def.get("name") or f"muscle_{len(meshes_out)}"

        verts, faces, weights = [], [], []
        for prim in mesh_def["primitives"]:
            attr = prim["attributes"]
            base = sum(len(v) for v in verts)
            pos = read(attr["POSITION"])
            verts.append(pos @ to_world[:3, :3].T + to_world[:3, 3])
            if "indices" in prim:
                idx = read(prim["indices"]).astype(np.int32).reshape(-1, 3) + base
            else:
                idx = np.arange(base, base + len(pos), dtype=np.int32).reshape(-1, 3)
            faces.append(idx)
            w = np.zeros((len(pos), len(joint_nodes)), dtype=np.float64)
            s = 0
            while f"JOINTS_{s}" in attr and f"WEIGHTS_{s}" in attr:
                j = read(attr[f"JOINTS_{s}"]).astype(int)
                ws = read(attr[f"WEIGHTS_{s}"])
                for k in range(j.shape[1]):
                    np.add.at(w, (np.arange(len(pos)), j[:, k]), ws[:, k])
                s += 1
            weights.append(w)

        w_all = np.concatenate(weights, axis=0)
        w_sum = w_all.sum(axis=1, keepdims=True)
        if not np.all(w_sum > 0):
            raise ValueError(f"Muscle mesh {name!r} in {path} has unweighted vertices")
        w_all /= w_sum
        meshes_out.append(
            {
                "name": str(name),
                "vertices_rest": np.concatenate(verts, axis=0) @ _TO_GLTF,
                "faces": np.concatenate(faces, axis=0).astype(np.int32),
                "weights": w_all,
                "joint_names": joint_names,
                "joint_pos": joint_pos_mj,
            }
        )
    if not meshes_out:
        raise ValueError(f"No skinned muscle meshes found in {path}")
    return meshes_out


def _reconstruct_skinned_glb(
    glb_spec: str | Path,
    model: mujoco.MjModel,
    geom_xpos: np.ndarray,
    geom_xmat: np.ndarray,
    muscles: np.ndarray,
    act_tendon: np.ndarray,
    *,
    reference_model: mujoco.MjModel | None,
) -> tuple[
    tuple[str, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    spec_str = str(glb_spec)
    bundled = spec_str in MUSCLE_MESHES
    glb_path = _resolve_muscle_mesh(spec_str)
    authored = _read_glb_skinned_meshes(glb_path)

    scaled = not np.allclose(model.mesh_scale, 1.0)
    if reference_model is not None:
        ref = reference_model
    elif bundled or (scaled and MUSCLE_MESH_ENV == "myoMimicFullbody-v0"):
        ref = _default_atlas_reference()
    else:
        ref = model

    d_ref = mujoco.MjData(ref)
    d_ref.qpos[:] = ref.qpos0
    mujoco.mj_forward(ref, d_ref)
    ref_anatomy = _anatomy_bodies(ref) or set(range(1, ref.nbody))
    ref_geoms = {
        ref.mesh(ref.geom_dataid[g]).name: g
        for g in range(ref.ngeom)
        if ref.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and ref.geom_dataid[g] >= 0
        and ref.geom_bodyid[g] in ref_anatomy
    }
    by_len = sorted(ref_geoms, key=len, reverse=True)

    scene_anatomy = _anatomy_bodies(model) or set(range(1, model.nbody))
    matched_bones: dict[str, tuple[int, int]] = {}
    for g in range(model.ngeom):
        if (
            model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            or model.geom_dataid[g] < 0
            or model.geom_bodyid[g] not in scene_anatomy
        ):
            continue
        mname = model.mesh(model.geom_dataid[g]).name
        rmatch = next((r for r in by_len if mname.endswith(r)), None)
        if rmatch is not None:
            key = f"{g}:{mname}"
            matched_bones[key] = (g, ref_geoms[rmatch])
    if not matched_bones:
        raise ValueError(
            f"No bone mesh in target model matches reference for {glb_path}"
        )

    bone_rest_pos = {
        key: d_ref.geom_xpos[g_ref].copy() for key, (_, g_ref) in matched_bones.items()
    }

    act_by_name = {
        model.actuator(int(a)).name.lower(): idx for idx, a in enumerate(muscles)
    }
    act_bodies: list[set[int]] = []
    for a in muscles:
        t = int(model.actuator_trnid[int(a), 0])
        adr, num = int(model.tendon_adr[t]), int(model.tendon_num[t])
        bodies = {
            int(model.site_bodyid[model.wrap_objid[w]])
            for w in range(adr, adr + num)
            if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE
        }
        act_bodies.append(bodies)

    T = len(geom_xpos)
    out_names: list[str] = []
    out_verts: list[np.ndarray] = []
    out_faces: list[np.ndarray] = []
    out_acts: list[np.ndarray] = []

    for item in authored:
        joints = dict(zip(item["joint_names"], item["joint_pos"]))
        targets = _binding_targets(joints, bone_rest_pos)
        weights = item["weights"]
        used_j = [
            j for j, jname in enumerate(item["joint_names"]) if weights[:, j].any()
        ]
        if not all(item["joint_names"][j] in targets for j in used_j):
            continue

        v_rest = item["vertices_rest"]
        deformed = np.zeros((T, len(v_rest), 3), dtype=np.float64)
        spanned_bodies: set[int] = set()
        for j in used_j:
            jname = item["joint_names"][j]
            g_scene, g_ref = matched_bones[targets[jname]]
            spanned_bodies.add(int(model.geom_bodyid[g_scene]))
            v_idx = np.flatnonzero(weights[:, j] > 0)
            w_j = weights[v_idx, j]
            p_ref = d_ref.geom_xpos[g_ref]
            R_ref = d_ref.geom_xmat[g_ref].reshape(3, 3)
            s_ref = ref.mesh_scale[ref.geom_dataid[g_ref]]
            s_scene = model.mesh_scale[model.geom_dataid[g_scene]]
            s_ratio = s_scene / np.where(np.abs(s_ref) > 1e-12, s_ref, 1.0)

            local = ((v_rest[v_idx] - p_ref) @ R_ref) * s_ratio
            R_t = geom_xmat[:, g_scene].reshape(T, 3, 3)
            p_t = geom_xpos[:, g_scene]
            world_jt = p_t[:, None, :] + np.einsum("tij,vj->tvi", R_t, local)
            deformed[:, v_idx, :] += w_j[None, :, None] * world_jt

        mname = item["name"]
        out_names.append(mname)
        out_verts.append(deformed.astype(np.float32))
        out_faces.append(item["faces"])

        low_name = mname.lower()
        if low_name in act_by_name:
            out_acts.append(act_tendon[:, act_by_name[low_name]])
        else:
            min_shared = 2 if len(spanned_bodies) >= 2 else 1
            spanning = [
                idx
                for idx, bset in enumerate(act_bodies)
                if len(bset & spanned_bodies) >= min_shared
            ]
            if spanning:
                out_acts.append(act_tendon[:, spanning].mean(axis=1))
            else:
                out_acts.append(np.zeros(T, dtype=np.float32))

    if not out_names:
        raise ValueError(
            f"No muscle in {glb_path} had all of its weighted bones in the target model"
        )
    return (
        tuple(out_names),
        tuple(out_verts),
        tuple(out_faces),
        np.stack(out_acts, axis=1).astype(np.float32),
    )


def reconstruct_muscle_meshes(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float = 30.0,
    *,
    qvel: np.ndarray | None = None,
    qacc: np.ndarray | None = None,
    activation_mode: str = "shortening",
    mesh_mode: str | Path = "volumetric",
    muscle_scale: float = 1.0,
    reference_model: mujoco.MjModel | None = None,
    so_config: Any | None = None,
) -> MuscleMeshReconstruction:
    """Reconstruct 3D muscle meshes, tendon lengths and activations along *qpos*.

    Args:
        model: Compiled MuJoCo model (nominal or anthropometrically scaled).
        qpos: ``(T, nq)`` or ``(nq,)`` generalized coordinates.
        fps: Trajectory frame rate in Hz.
        qvel: Optional ``(T, nv)`` generalized velocities (computed on the manifold
            when needed by ``activation_mode="static_optimization"``).
        qacc: Optional ``(T, nv)`` generalized accelerations.
        activation_mode: ``"shortening"`` (fast kinematic proxy),
            ``"static_optimization"`` (inverse-dynamics OSQP solve), or ``"none"``.
        mesh_mode: ``"volumetric"`` (constant-volume fusiform muscle tubes),
            ``"atlas"`` / ``"atlas-hd"`` (BodyParts3D skinned meshes), or a path to a
            skinned ``.glb`` file.
        muscle_scale: Multiplier on volumetric muscle belly radii.
        reference_model: Optional unscaled reference model in whose ``qpos0`` a custom
            skinned ``.glb`` was authored.
        so_config: Optional :class:`~myosuite.biomechanics.static_optimization.StaticOptimizationConfig`.

    Returns:
        A :class:`MuscleMeshReconstruction` containing per-frame 3D meshes, tendon
        lengths and activations.
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != model.nq or len(q) < 1:
        raise ValueError(f"qpos must have shape (T, {model.nq}), got {q.shape}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not np.isfinite(muscle_scale) or muscle_scale <= 0:
        raise ValueError(f"muscle_scale must be positive, got {muscle_scale}")

    T = len(q)
    muscles = muscle_actuators(model)
    tendons = model.actuator_trnid[muscles, 0] if len(muscles) else np.zeros(0, int)
    tendon_length = np.empty((T, len(muscles)), dtype=np.float64)
    is_volumetric = str(mesh_mode) == "volumetric"

    paths: list[list[np.ndarray]] = []
    geom_xpos = np.empty((0, 0, 3)) if is_volumetric else np.empty((T, model.ngeom, 3))
    geom_xmat = np.empty((0, 0, 9)) if is_volumetric else np.empty((T, model.ngeom, 9))

    data = mujoco.MjData(model)
    for t in range(T):
        data.qpos[:] = q[t]
        mujoco.mj_forward(model, data)
        if len(muscles):
            tendon_length[t] = data.ten_length[tendons]
        if is_volumetric and len(muscles):
            wrap_pts = data.wrap_xpos.reshape(-1, 3)
            paths.append(
                [
                    wrap_pts[a : a + n].copy()
                    for a, n in zip(
                        data.ten_wrapadr[tendons], data.ten_wrapnum[tendons]
                    )
                ]
            )
        elif not is_volumetric:
            geom_xpos[t] = data.geom_xpos
            geom_xmat[t] = data.geom_xmat

    act_tendon = _compute_activations(
        model,
        muscles,
        tendon_length,
        q,
        fps,
        mode=activation_mode,
        qvel=qvel,
        qacc=qacc,
        so_config=so_config,
    )

    if is_volumetric:
        names = tuple(model.actuator(int(a)).name for a in muscles)
        if len(muscles) == 0:
            return MuscleMeshReconstruction(
                names,
                np.zeros((T, 0, 0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                tendon_length,
                act_tendon,
                float(fps),
                "volumetric",
            )
        resampled = [[resample_path(p) for p in frame] for frame in paths]
        centres = np.array([[c for c, _ in frame] for frame in resampled])
        lengths = np.array([[ln for _, ln in frame] for frame in resampled])
        peak0, fraction = muscle_shape(model, lengths[0])
        profile = radius_profile(fraction)
        volume = belly_volume(peak0, lengths[0], profile)
        peak = belly_radius(volume, lengths, profile) * muscle_scale
        points, counts, indices = tube_mesh(centres, peak[..., None] * profile)
        faces = _triangulate_faces(counts, indices)
        return MuscleMeshReconstruction(
            muscle_names=names,
            vertices=points.astype(np.float32),
            faces=faces,
            tendon_length=tendon_length,
            activation=act_tendon,
            fps=float(fps),
            mode="volumetric",
        )

    names, verts_tup, faces_tup, acts = _reconstruct_skinned_glb(
        mesh_mode,
        model,
        geom_xpos,
        geom_xmat,
        muscles,
        act_tendon,
        reference_model=reference_model,
    )
    return MuscleMeshReconstruction(
        muscle_names=names,
        vertices=verts_tup,
        faces=faces_tup,
        tendon_length=tendon_length,
        activation=acts,
        fps=float(fps),
        mode=str(mesh_mode),
    )


def _muscle_rgba(activation: float) -> list[float]:
    a = float(np.clip(activation, 0.0, 1.0))
    return [float(r + (q - r) * a) for r, q in zip(MUSCLE_RELAXED, MUSCLE_ACTIVE)]


def write_muscle_meshes_glb(
    recon: MuscleMeshReconstruction,
    path: str | Path,
    *,
    frame: int = 0,
    model: mujoco.MjModel | None = None,
    qpos: np.ndarray | None = None,
    include_bones: bool = True,
) -> Path:
    """Write one frame of *recon* (plus optional posed bones) to a glTF 2.0 ``.glb``."""
    if not 0 <= frame < recon.num_frames:
        raise IndexError(f"frame {frame} out of range [0, {recon.num_frames})")

    blob = bytearray()
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def add_accessor(
        array: np.ndarray, kind: str, component: int, target: int | None = None
    ) -> int:
        arr = np.ascontiguousarray(array)
        while len(blob) % 4:
            blob.append(0)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": len(blob),
            "byteLength": arr.nbytes,
        }
        if target is not None:
            view["target"] = target
        views.append(view)
        blob.extend(arr.tobytes())
        acc: dict[str, Any] = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": int(len(arr)),
            "type": kind,
        }
        if kind == "VEC3" and component == 5126 and len(arr):
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    shared_idx_acc: int | None = None
    if isinstance(recon.faces, np.ndarray) and len(recon.faces):
        shared_idx_acc = add_accessor(
            recon.faces.astype(np.uint32).ravel(), "SCALAR", 5125, 34963
        )

    muscle_children: list[int] = []
    for m, name in enumerate(recon.muscle_names):
        v_mj = np.asarray(recon.muscle_vertices(m)[frame], dtype=np.float64)
        f_m = recon.muscle_faces(m)
        if len(v_mj) == 0 or len(f_m) == 0:
            continue
        n_mj = _vertex_normals(v_mj, f_m)
        pos_acc = add_accessor(
            (v_mj @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962
        )
        nrm_acc = add_accessor(
            (n_mj @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962
        )
        idx_acc = (
            shared_idx_acc
            if shared_idx_acc is not None
            else add_accessor(f_m.astype(np.uint32).ravel(), "SCALAR", 5125, 34963)
        )
        mat_idx = len(materials)
        materials.append(
            {
                "name": f"mat_{name}",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": _muscle_rgba(float(recon.activation[frame, m])),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                },
            }
        )
        mesh_idx = len(meshes)
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": mat_idx,
                    }
                ],
            }
        )
        node_idx = len(nodes)
        nodes.append({"name": name, "mesh": mesh_idx})
        muscle_children.append(node_idx)

    root_nodes: list[int] = []
    if muscle_children:
        muscles_group = len(nodes)
        nodes.append({"name": "Muscles", "children": muscle_children})
        root_nodes.append(muscles_group)

    if include_bones and model is not None:
        d = mujoco.MjData(model)
        if qpos is not None:
            q = np.asarray(qpos, dtype=np.float64)
            d.qpos[:] = q if q.ndim == 1 else q[frame]
        else:
            d.qpos[:] = model.qpos0
        mujoco.mj_forward(model, d)
        anatomy = _anatomy_bodies(model) or set(range(1, model.nbody))
        bone_mat_idx = len(materials)
        materials.append(
            {
                "name": "Bone",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(BONE_COLOR),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.45,
                },
            }
        )
        bone_children: list[int] = []
        for g in range(model.ngeom):
            if (
                model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or model.geom_dataid[g] < 0
                or model.geom_bodyid[g] not in anatomy
            ):
                continue
            mid = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            if vn == 0 or fn == 0:
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            v_world = d.geom_xpos[g] + model.mesh_vert[va : va + vn] @ R.T
            f_bone = model.mesh_face[fa : fa + fn].astype(np.int32)
            n_world = _vertex_normals(v_world, f_bone)
            pos_acc = add_accessor(
                (v_world @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962
            )
            nrm_acc = add_accessor(
                (n_world @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126, 34962
            )
            idx_acc = add_accessor(
                f_bone.astype(np.uint32).ravel(), "SCALAR", 5125, 34963
            )
            bname = model.geom(g).name or f"{model.mesh(mid).name}_{g}"
            mesh_idx = len(meshes)
            meshes.append(
                {
                    "name": bname,
                    "primitives": [
                        {
                            "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                            "indices": idx_acc,
                            "material": bone_mat_idx,
                        }
                    ],
                }
            )
            node_idx = len(nodes)
            nodes.append({"name": bname, "mesh": mesh_idx})
            bone_children.append(node_idx)
        if bone_children:
            bones_group = len(nodes)
            nodes.append({"name": "Bones", "children": bone_children})
            root_nodes.append(bones_group)

    gltf = {
        "asset": {"version": "2.0", "generator": "myosuite.viz.muscle_meshes"},
        "scene": 0,
        "scenes": [{"nodes": root_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    text = json.dumps(gltf).encode("utf-8")
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    out_bytes = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(blob))
        + struct.pack("<I4s", len(text), b"JSON")
        + text
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + bytes(blob)
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(out_bytes)
    return out


def save_muscle_meshes_npz(recon: MuscleMeshReconstruction, path: str | Path) -> Path:
    """Write *recon* to a compressed ``.npz`` archive (pickle-free)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "muscle_names": np.asarray(recon.muscle_names, dtype=str),
        "tendon_length": recon.tendon_length.astype(np.float32),
        "activation": recon.activation.astype(np.float32),
        "fps": np.float64(recon.fps),
        "mode": np.array(recon.mode),
    }
    if isinstance(recon.vertices, np.ndarray) and isinstance(recon.faces, np.ndarray):
        arrays["vertices"] = recon.vertices.astype(np.float32)
        arrays["faces"] = recon.faces.astype(np.int32)
    else:
        v_counts = np.array([v.shape[1] for v in recon.vertices], dtype=np.int32)
        f_counts = np.array([len(f) for f in recon.faces], dtype=np.int32)
        arrays["vertex_counts"] = v_counts
        arrays["face_counts"] = f_counts
        arrays["vertices_packed"] = (
            np.concatenate(list(recon.vertices), axis=1).astype(np.float32)
            if recon.vertices
            else np.zeros((recon.num_frames, 0, 3), dtype=np.float32)
        )
        arrays["faces_packed"] = (
            np.concatenate(list(recon.faces), axis=0).astype(np.int32)
            if recon.faces
            else np.zeros((0, 3), dtype=np.int32)
        )
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)
    return out


__all__ = [
    "MuscleMeshReconstruction",
    "reconstruct_muscle_meshes",
    "save_muscle_meshes_npz",
    "write_muscle_meshes_glb",
]
