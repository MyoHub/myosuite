# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Ghost-body visualisation for MuscleMimic motion imitation.

Draws a semi-transparent skeleton of the reference motion clip alongside the
policy-driven body in the MuJoCo passive viewer.  The ghost is rendered each
frame from the clip's precomputed ``xpos`` / ``xquat`` body-position arrays,
so no second physics simulation is required.

Usage::

    from myosuite.viz.ghost_body_viz import GhostBodyViz, make_ghost_viz_fn

    viz = GhostBodyViz.from_clip_and_model(clip, model)
    # viz.as_viz_fn() is a VizCallback for run_passive_viewer_loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    pass


# ── geometry helpers ────────────────────────────────────────────────────────


def _quat_to_mat9(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (w,x,y,z) to column-major 3×3 rotation matrix.

    Args:
        q: Shape ``(4,)`` quaternion ``[w, x, y, z]``.

    Returns:
        Shape ``(9,)`` row-major rotation matrix for ``mjv_initGeom``.
    """
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q.astype(np.float64))
    return mat


def _capsule_between(
    p0: np.ndarray, p1: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, mat, size) for a capsule spanning *p0* → *p1*.

    The capsule z-axis aligns with the segment direction.

    Args:
        p0: Start point, shape ``(3,)``.
        p1: End point, shape ``(3,)``.
        radius: Capsule radius in metres.

    Returns:
        Tuple of (centre pos, rotation mat9, size array ``[r, r, half_len]``).
    """
    centre = (p0 + p1) / 2.0
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-6:
        diff = np.array([0.0, 0.0, 1.0])
        length = 1.0
    z_axis = diff / length

    # Build rotation: align capsule z-axis with segment
    ref = np.array([0.0, 0.0, 1.0])
    cross = np.cross(ref, z_axis)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-6:
        # Already aligned (or anti-aligned)
        mat = np.eye(3) if z_axis[2] > 0 else -np.eye(3)
        mat = mat.flatten()
    else:
        axis = cross / cross_norm
        angle = float(np.arcsin(min(1.0, cross_norm)))
        q = np.empty(4)
        mujoco.mju_axisAngle2Quat(q, axis, angle)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, q)

    size = np.array([radius, radius, length / 2.0], dtype=np.float64)
    return centre.astype(np.float64), mat.astype(np.float64), size


# ── main class ───────────────────────────────────────────────────────────────


@dataclass
class GhostBodyViz:
    """Renders a semi-transparent reference-motion body from a motion clip.

    The ghost skeleton is drawn each frame by:

    1. Reading ``clip.xpos[frame]`` and ``clip.xquat[frame]`` (precomputed
       body world positions/orientations from the retargeting pipeline).
    2. Drawing a capsule between each body and its parent (using
       ``mjv_initGeom``), coloured with semi-transparent ghost RGBA.
    3. Optionally drawing a sphere at each body centre.

    All geometry is written into ``viewer.user_scn`` — no model changes.

    Args:
        xpos: Body positions from clip, shape ``(T, nbody, 3)``.
        xquat: Body orientations from clip, shape ``(T, nbody, 4)`` (w,x,y,z).
        parent_ids: Parent body id for each body, shape ``(nbody,)``.
        visible_body_ids: Indices of bodies to draw (bodies with geometry).
        skeleton_rgba: RGBA for capsule segments.
        joint_rgba: RGBA for joint spheres.
        capsule_radius: Radius of limb capsules in metres.
        joint_radius: Radius of joint spheres in metres.
        frame_offset: Frame offset applied to each ``step_idx`` lookup.
        ctrl_dt_ratio: Ratio of env ctrl_dt to clip frame_dt for frame indexing.
        position_offset: Optional ``(3,)`` world-space translation applied to all
            ghost positions.  Use e.g. ``np.array([0, 1, 0])`` to place the ghost
            1 m to the side of the physics body when doing reference replay (both
            bodies would otherwise occupy the same position and the ghost is
            completely occluded).
    """

    xpos: np.ndarray  # (T, nbody, 3)
    xquat: np.ndarray  # (T, nbody, 4)
    parent_ids: np.ndarray  # (nbody,)
    visible_body_ids: np.ndarray  # (V,)
    skeleton_rgba: tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.35)
    joint_rgba: tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5)
    capsule_radius: float = 0.018
    joint_radius: float = 0.022
    frame_offset: int = 0
    ctrl_dt_ratio: int = 1
    position_offset: np.ndarray | None = (
        None  # (3,) world-space shift for all ghost positions
    )

    # pre-built arrays (set in __post_init__)
    _skel_rgba: np.ndarray = field(init=False, repr=False)
    _joint_rgba: np.ndarray = field(init=False, repr=False)
    _T: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._skel_rgba = np.array(self.skeleton_rgba, dtype=np.float64)
        self._joint_rgba = np.array(self.joint_rgba, dtype=np.float64)
        self._T = int(self.xpos.shape[0])

    # ------------------------------------------------------------------ #

    @classmethod
    def from_clip_and_model(
        cls,
        clip_path_or_npz: str | np.lib.npyio.NpzFile,
        model: mujoco.MjModel,
        *,
        skeleton_rgba: tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.35),
        joint_rgba: tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5),
        capsule_radius: float = 0.018,
        joint_radius: float = 0.022,
        frame_offset: int = 0,
        ctrl_dt_ratio: int = 1,
        position_offset: np.ndarray | None = None,
    ) -> GhostBodyViz:
        """Build a :class:`GhostBodyViz` from a .npz clip file and MuJoCo model.

        The clip must contain ``xpos`` (T, nbody, 3) and ``xquat``
        (T, nbody, 4) arrays.  If the clip's nbody differs from the model's
        nbody, only the overlapping prefix is used.

        Args:
            clip_path_or_npz: Path to the .npz clip, or an already-loaded
                ``numpy.NpzFile``.
            model: Compiled MuJoCo model — used for body count and parent IDs.
            skeleton_rgba: Ghost capsule colour (R, G, B, A).
            joint_rgba: Ghost joint-sphere colour (R, G, B, A).
            capsule_radius: Limb capsule radius in metres.
            joint_radius: Joint sphere radius in metres.
            frame_offset: Starting frame offset (e.g. for random-start eval).
            ctrl_dt_ratio: Ratio of control dt to clip frame dt.
            position_offset: Optional ``(3,)`` world-space shift applied to all
                ghost positions.  Pass e.g. ``np.array([0, 1, 0])`` to place the
                ghost 1 m to the side for side-by-side reference replay.

        Returns:
            Configured :class:`GhostBodyViz` instance.
        """
        from pathlib import Path

        if isinstance(clip_path_or_npz, (str, Path)):
            npz = np.load(clip_path_or_npz, allow_pickle=True)
        else:
            npz = clip_path_or_npz

        xpos = np.asarray(npz["xpos"], dtype=np.float64)  # (T, nbody_clip, 3)
        xquat = np.asarray(npz["xquat"], dtype=np.float64)  # (T, nbody_clip, 4)

        # Align clip body count to model body count (both should be 102)
        n_clip = xpos.shape[1]
        n_model = model.nbody
        n = min(n_clip, n_model)
        xpos = xpos[:, :n, :]
        xquat = xquat[:, :n, :]
        parent_ids = model.body_parentid[:n].copy()

        # Visible bodies: those with at least one geom, skip world (id 0)
        visible = [i for i in range(1, n) if int(model.body_geomnum[i]) > 0]
        visible_body_ids = np.array(visible, dtype=np.int32)

        offset = (
            np.asarray(position_offset, dtype=np.float64)
            if position_offset is not None
            else None
        )

        return cls(
            xpos=xpos,
            xquat=xquat,
            parent_ids=parent_ids,
            visible_body_ids=visible_body_ids,
            skeleton_rgba=skeleton_rgba,
            joint_rgba=joint_rgba,
            capsule_radius=capsule_radius,
            joint_radius=joint_radius,
            frame_offset=frame_offset,
            ctrl_dt_ratio=ctrl_dt_ratio,
            position_offset=offset,
        )

    # ------------------------------------------------------------------ #

    def draw(self, step_idx: int, user_scn: mujoco.MjvScene) -> None:
        """Draw ghost skeleton geoms into *user_scn* for *step_idx*.

        Appends ghost geoms **after** any existing geoms in the scene so that
        site markers from :class:`~myosuite.viz.site_marker_viz.SiteMarkerViz`
        remain visible if used alongside this viz.

        Args:
            step_idx: Current control step index.
            user_scn: Viewer user scene (obtained inside ``viewer.lock()``).
        """
        frame = (self.frame_offset + step_idx * self.ctrl_dt_ratio) % self._T
        pos_frame = self.xpos[frame]  # (nbody, 3)
        if self.position_offset is not None:
            pos_frame = pos_frame + self.position_offset
        max_geoms = int(user_scn.maxgeom)
        geom_i = int(user_scn.ngeom)  # append after existing geoms

        for body_id in self.visible_body_ids:
            if geom_i >= max_geoms - 1:
                break
            parent_id = int(self.parent_ids[body_id])
            p_child = pos_frame[body_id]
            p_parent = pos_frame[parent_id] if parent_id != body_id else p_child

            # Draw capsule connecting parent to child
            if np.linalg.norm(p_child - p_parent) > 1e-4:
                centre, mat9, size = _capsule_between(
                    p_parent, p_child, self.capsule_radius
                )
                mujoco.mjv_initGeom(
                    user_scn.geoms[geom_i],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size,
                    centre,
                    mat9,
                    self._skel_rgba,
                )
                user_scn.ngeom += 1
                geom_i += 1

            # Draw joint sphere at body position
            if geom_i < max_geoms:
                sz = np.array(
                    [self.joint_radius, self.joint_radius, self.joint_radius],
                    dtype=np.float64,
                )
                mujoco.mjv_initGeom(
                    user_scn.geoms[geom_i],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    sz,
                    p_child.astype(np.float64),
                    np.eye(3).flatten(),
                    self._joint_rgba,
                )
                user_scn.ngeom += 1
                geom_i += 1

    def as_viz_fn(self) -> _GhostVizFn:
        """Return a VizCallback bound to this instance.

        Returns:
            Callable ``(step_idx, model, data, user_scn) -> None``.
        """
        return _GhostVizFn(self)


@dataclass
class _GhostVizFn:
    _viz: GhostBodyViz

    def __call__(
        self,
        step_idx: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        user_scn: mujoco.MjvScene,
    ) -> None:
        self._viz.draw(step_idx, user_scn)


# ── combined viz ─────────────────────────────────────────────────────────────


@dataclass
class CombinedViz:
    """Applies multiple VizCallbacks in sequence.

    Useful for combining :class:`GhostBodyViz` with
    :class:`~myosuite.viz.site_marker_viz.SiteMarkerViz`.

    Args:
        callbacks: Ordered list of viz callbacks to apply.
    """

    callbacks: list

    def __call__(
        self,
        step_idx: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        user_scn: mujoco.MjvScene,
    ) -> None:
        for cb in self.callbacks:
            cb(step_idx, model, data, user_scn)


# ── convenience factory ───────────────────────────────────────────────────────


def make_ghost_viz_fn(
    clip_path: str,
    model: mujoco.MjModel,
    skeleton_rgba: tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.35),
    joint_rgba: tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5),
    capsule_radius: float = 0.018,
    joint_radius: float = 0.022,
    frame_offset: int = 0,
    ctrl_dt_ratio: int = 1,
    position_offset: np.ndarray | None = None,
) -> _GhostVizFn:
    """Convenience factory: build a ghost VizCallback from a clip path.

    Args:
        clip_path: Path to retargeted .npz motion clip.
        model: Compiled MuJoCo model.
        skeleton_rgba: Ghost capsule colour.
        joint_rgba: Ghost joint sphere colour.
        capsule_radius: Limb radius in metres.
        joint_radius: Joint sphere radius in metres.
        frame_offset: Start frame offset.
        ctrl_dt_ratio: Control-dt to clip-frame-dt ratio.
        position_offset: Optional ``(3,)`` world-space shift for the ghost.

    Returns:
        VizCallback for ``run_passive_viewer_loop``.
    """
    viz = GhostBodyViz.from_clip_and_model(
        clip_path,
        model,
        skeleton_rgba=skeleton_rgba,
        joint_rgba=joint_rgba,
        capsule_radius=capsule_radius,
        joint_radius=joint_radius,
        frame_offset=frame_offset,
        ctrl_dt_ratio=ctrl_dt_ratio,
        position_offset=position_offset,
    )
    return viz.as_viz_fn()


__all__ = [
    "CombinedViz",
    "GhostBodyViz",
    "make_ghost_viz_fn",
]
