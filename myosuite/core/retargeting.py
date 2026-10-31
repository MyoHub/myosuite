# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Retarget labelled marker trajectories onto a MuJoCo model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.capture.markers import _read_markers
from myosuite.physics.inverse_kinematics import solve_ik

_SOLE_GEOMS = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
    int(mujoco.mjtGeom.mjGEOM_MESH),
}


def retarget(
    model: mujoco.MjModel,
    markers: str | Path | Mapping[str, ArrayLike],
    *,
    sites: Mapping[str, str] | None = None,
    fps: float | None = None,
    ground_sites: Sequence[str] | None = None,
    joints: Sequence[str] | None = None,
    qpos: ArrayLike | None = None,
    rotation: ArrayLike | None = None,
    translation: ArrayLike | None = None,
) -> MotionClip:
    """Fit a model to marker trajectories, one frame at a time.

    Each frame is a :func:`~myosuite.physics.inverse_kinematics.solve_ik` call
    with ``strict=False``, seeded with the previous frame's pose (only the
    first frame may retry from mid-range joint values). Markers that are NaN
    in a frame are left out of that frame; a frame without any mapped marker
    keeps the neighbouring pose.

    The marker error and the sole height are recorded for every frame. With a
    free root joint, the whole clip is then shifted vertically by one offset so
    that its lowest sole point touches ``z = 0``. The sole is the collision
    geometry of the root's tree (the feet of a standing model) or, if given,
    *ground_sites*. A fixed-base model is not shifted.

    Example::

        clip = retarget(model, "trial.trc")
        clip = retarget(model, "trial.trc", sites={"RHJC": "hip_r"})
        clip = retarget(model, {"hand": hand_xyz}, fps=100)

    Args:
        model: Compiled model; it is not modified.
        markers: A ``.trc``, ``.c3d`` (needs ``myosuite[capture]``) or
            ``label,x,y,z`` landmark ``.csv`` path, or a mapping of marker name
            to an array ``(T, 3)`` in metres. Files are converted to metres.
        sites: Marker name to model site, for markers whose names are not site
            names. Markers named like a site need no entry.
        fps: Frame rate in Hz. Read from ``.trc`` and ``.c3d`` files; required
            for arrays with more than one frame.
        ground_sites: Sites whose lowest point is placed on the ground instead
            of the collision geometry. An empty sequence keeps the fitted
            height and still records the sole height.
        joints: Joints that may move (see ``solve_ik``); all by default.
        qpos: Seed pose of the first frame. Defaults to ``model.qpos0``.
        rotation: Rotation matrix ``(3, 3)`` from the marker frame to the model
            world frame, e.g. ``[[1, 0, 0], [0, 0, -1], [0, 1, 0]]`` for the
            Y-up frame of OpenSim files. Identity by default.
        translation: Offset ``(3,)`` in metres added after *rotation*.

    Returns:
        A :class:`~myosuite.core.trajectory_io.MotionClip` with ``qpos``,
        central-difference ``qvel``, the fitted site positions of the mapped
        markers (``site_xpos``, ``site_names``) and ``frequency_hz``. Its
        ``metadata`` holds ``markers`` (names), ``marker_sites``,
        ``marker_xpos`` ``(T, M, 3)`` (measured, in the model frame),
        ``marker_error`` ``(T, M)`` (metres, NaN where missing),
        ``sole_height`` ``(T,)`` (metres, ``None`` without a free root),
        ``ground_offset`` (metres added to the root height), ``solved``
        ``(T,)`` (frames with markers) and ``source``.

    Raises:
        KeyError: If a mapped marker or site does not exist.
        ValueError: For invalid arrays, frame rate, frame transform or
            ground sites, or when no marker maps to a site.
    """
    names, positions, rate, source = _load(markers, fps)
    positions = positions @ _rotation(rotation).T + _translation(translation)
    mapping = _site_mapping(model, names, sites)
    columns = [names.index(marker) for marker in mapping]
    site_names = list(mapping.values())
    site_ids = np.array([_site_id(model, name) for name in site_names], dtype=int)
    roots = _free_roots(model, site_ids)
    sole = _Sole(model, roots, ground_sites)
    data = mujoco.MjData(model)
    frames = len(positions)
    qpos_out = np.zeros((frames, model.nq))
    site_xpos = np.zeros((frames, len(mapping), 3))
    heights = np.full(frames, np.nan)
    solved = np.zeros(frames, dtype=bool)
    pose = np.array(model.qpos0 if qpos is None else qpos, dtype=float)
    for frame in range(frames):
        targets = {
            name: positions[frame, column]
            for name, column in zip(site_names, columns)
            if np.all(np.isfinite(positions[frame, column]))
        }
        if targets:
            # Only the first solve is a cold start; later frames track the last pose.
            retry = not solved.any()
            pose = solve_ik(
                model, targets, joints=joints, qpos=pose, strict=False, retry=retry
            )
            solved[frame] = True
        qpos_out[frame] = pose
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        site_xpos[frame] = data.site_xpos[site_ids]
        heights[frame] = sole.height(data)
    if not solved.any():
        raise ValueError("No frame has a finite marker that maps to a site.")
    first = int(np.argmax(solved))  # frames before the first marker take its pose
    qpos_out[:first] = qpos_out[first]
    site_xpos[:first] = site_xpos[first]
    heights[:first] = heights[first]
    offset = 0.0
    if sole.place:
        offset = -float(np.min(heights))
        for joint in roots:
            qpos_out[:, model.jnt_qposadr[joint] + 2] += offset
        site_xpos[:, sole.in_roots(site_ids), 2] += offset
        heights += offset
    measured = positions[:, columns]
    metadata: dict[str, Any] = {
        "markers": list(mapping),
        "marker_sites": dict(mapping),
        "marker_xpos": measured,
        "marker_error": np.linalg.norm(site_xpos - measured, axis=2),
        "sole_height": heights if sole.active else None,
        "ground_offset": offset,
        "solved": solved,
        "source": source,
    }
    joint_names = [model.joint(j).name for j in range(model.njnt)]
    named = all(joint_names)
    return MotionClip(
        qpos=qpos_out,
        qvel=_velocities(model, qpos_out, rate),
        site_xpos=site_xpos,
        site_names=site_names,
        qpos_joint_names=joint_names if named else None,
        qvel_joint_names=joint_names if named else None,
        frequency_hz=rate,
        source_path=None if source is None else Path(source),
        metadata=metadata,
    )


def _load(
    markers: str | Path | Mapping[str, ArrayLike], fps: float | None
) -> tuple[list[str], np.ndarray, float | None, str | None]:
    """Marker names, positions ``(T, M, 3)`` in metres, frame rate and source path."""
    if isinstance(markers, str | Path):
        data = _read_markers(markers)
        names, positions, rate = list(data.names), data.positions, data.rate
        source: str | None = str(Path(markers).resolve())
    else:
        if not markers:
            raise ValueError("markers is empty.")
        names, source, rate = list(markers), None, None
        arrays = [np.asarray(markers[name], dtype=float) for name in names]
        if (
            any(a.ndim != 2 or a.shape[1] != 3 for a in arrays)
            or len({len(a) for a in arrays}) != 1
        ):
            raise ValueError("Every marker array must have the same shape (T, 3).")
        positions = np.stack(arrays, axis=1)
    if np.any(np.isinf(positions)) or not len(positions):
        raise ValueError(
            "Marker positions must be finite or NaN, with at least one frame."
        )
    rate = fps if fps is not None else rate
    if rate is None and len(positions) > 1:
        raise ValueError("fps is required for marker arrays with more than one frame.")
    if rate is not None and (not np.isfinite(rate) or rate <= 0):
        raise ValueError(f"fps must be finite and positive, got {rate}.")
    return names, positions, None if rate is None else float(rate), source


def _rotation(rotation: ArrayLike | None) -> np.ndarray:
    if rotation is None:
        return np.eye(3)
    matrix = np.asarray(rotation, dtype=float)
    proper = (
        matrix.shape == (3, 3)
        and np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)
    )
    if not proper:
        raise ValueError("rotation must be a proper 3x3 rotation matrix.")
    return matrix


def _translation(translation: ArrayLike | None) -> np.ndarray:
    offset = (
        np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    )
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("translation must be a finite 3-vector in metres.")
    return offset


def _site_mapping(
    model: mujoco.MjModel, names: Sequence[str], sites: Mapping[str, str] | None
) -> dict[str, str]:
    """Marker to site: explicit entries, plus markers named like an untargeted site."""
    explicit = dict(sites or {})
    unknown = sorted(set(explicit) - set(names))
    if unknown:
        raise KeyError(f"sites maps markers that are not in the data: {unknown}.")
    for site in explicit.values():
        _site_id(model, site)
    targeted = set(explicit.values())
    mapping = {
        name: name
        for name in names
        if name not in explicit
        and name not in targeted
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    }
    mapping.update(explicit)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each site can follow only one marker.")
    if not mapping:
        raise ValueError(
            "No marker is named like a model site; map them with sites={marker: site}."
        )
    return {name: mapping[name] for name in names if name in mapping}


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise KeyError(f"No site named {name!r} in the model.")
    return site


def _top_body(model: mujoco.MjModel, body: int) -> int:
    while body and model.body_parentid[body]:
        body = int(model.body_parentid[body])
    return body


def _free_roots(model: mujoco.MjModel, site_ids: np.ndarray) -> list[int]:
    """Free joints of the top-level bodies that carry the mapped sites."""
    tops = {_top_body(model, int(model.site_bodyid[s])) for s in site_ids}
    return sorted(
        joint
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        and int(model.jnt_bodyid[joint]) in tops
    )


class _Sole:
    """Lowest point of the sole geometry (or of the ground sites) in a pose."""

    def __init__(
        self,
        model: mujoco.MjModel,
        roots: Sequence[int],
        ground_sites: Sequence[str] | None,
    ) -> None:
        self.model = model
        self._tops = {int(model.jnt_bodyid[joint]) for joint in roots}
        self.place = bool(roots) and (ground_sites is None or len(ground_sites) > 0)
        self._sites = np.zeros(0, dtype=int)
        self._geoms: list[int] = []
        if ground_sites:
            if not roots:
                raise ValueError(
                    "ground_sites needs a free root joint above the marker sites; "
                    "a fixed-base model is not shifted."
                )
            self._sites = np.array([_site_id(model, name) for name in ground_sites])
            if not np.all(self.in_roots(self._sites)):
                raise ValueError(
                    "ground_sites must belong to the tree of the free root."
                )
        elif roots:
            # Imported here: body_scaling re-exports retarget, so a module-level
            # import would be circular.
            from myosuite.core.body_scaling import _geom_bottom

            self._bottom = _geom_bottom
            self._geoms = [
                geom
                for geom in range(model.ngeom)
                if (model.geom_contype[geom] or model.geom_conaffinity[geom])
                and int(model.geom_type[geom]) in _SOLE_GEOMS
                and _top_body(model, int(model.geom_bodyid[geom])) in self._tops
            ]
        self.active = bool(len(self._sites) or self._geoms)
        # Without collision geometry in the tree there is nothing to place.
        self.place = self.place and self.active

    def in_roots(self, site_ids: np.ndarray) -> np.ndarray:
        """Whether each site moves with a free root."""
        return np.array(
            [
                _top_body(self.model, int(self.model.site_bodyid[s])) in self._tops
                for s in site_ids
            ],
            dtype=bool,
        )

    def height(self, data: mujoco.MjData) -> float:
        """Lowest sole z after ``mj_kinematics``; NaN when there is no sole."""
        if len(self._sites):
            return float(np.min(data.site_xpos[self._sites, 2]))
        if not self._geoms:
            return float("nan")
        return min(self._bottom(self.model, data, geom) for geom in self._geoms)


def _velocities(
    model: mujoco.MjModel, qpos: np.ndarray, rate: float | None
) -> np.ndarray:
    """Central-difference ``qvel`` (one-sided at the ends) via ``mj_differentiatePos``."""
    frames = len(qpos)
    qvel = np.zeros((frames, model.nv))
    if frames < 2 or rate is None:
        return qvel
    dt = 1.0 / rate
    for frame in range(frames):
        before, after = max(frame - 1, 0), min(frame + 1, frames - 1)
        mujoco.mj_differentiatePos(
            model, qvel[frame], (after - before) * dt, qpos[before], qpos[after]
        )
    return qvel


__all__ = ["retarget"]
