# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for rendering target-site markers in the MuJoCo passive viewer.

Usage::

    from myosuite.viz.site_marker_viz import SiteMarkerViz

    viz = SiteMarkerViz(clip, site_names, model, rgba=(1.0, 0.4, 0.0, 0.6))
    # pass viz.as_viz_fn() to run_passive_viewer_loop(viz_fn=...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.trajectory_io import MotionClip


# Identity rotation matrix (row-major, 9 floats) used for all sphere markers.
_IDENTITY_MAT = np.eye(3, dtype=np.float64).flatten()


@dataclass
class SiteMarkerViz:
    """Renders reference-clip target positions as spheres in the viewer scene.

    Each frame, the world-space site positions from ``clip.site_xpos[frame]``
    are drawn as small semi-transparent spheres using the viewer's ``user_scn``
    buffer.  Up to ``user_scn.maxgeom`` markers can be rendered; excess
    targets are silently skipped.

    Args:
        clip: Loaded motion clip with ``site_xpos`` array ``(T, n_sites, 3)``.
        site_names: Ordered site names corresponding to axis-1 of
                    ``clip.site_xpos``.  Used only for logging; order must
                    match the clip's site axis.
        model: Compiled MuJoCo model (used to look up current site positions
               for error annotation; optional tracking).
        rgba: RGBA colour for the target spheres.  Default is orange at 60 %
              opacity.
        sphere_radius: Radius of each marker sphere in metres.

    Raises:
        ValueError: If ``clip.site_xpos`` is ``None``.
    """

    clip: MotionClip
    site_names: tuple[str, ...]
    model: mujoco.MjModel
    rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 0.6)
    sphere_radius: float = 0.03

    _rgba_arr: np.ndarray = field(init=False, repr=False)
    _size_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.clip.site_xpos is None:
            raise ValueError(
                "SiteMarkerViz requires clip.site_xpos; the loaded MotionClip "
                "does not contain site positions."
            )
        self._rgba_arr = np.array(self.rgba, dtype=np.float64)
        r = float(self.sphere_radius)
        self._size_arr = np.array([r, r, r], dtype=np.float64)

    # ------------------------------------------------------------------ #

    def draw(
        self,
        frame_idx: int,
        user_scn: mujoco.MjvScene,
    ) -> None:
        """Write target-sphere geoms into *user_scn* for *frame_idx*.

        Resets ``user_scn.ngeom`` to 0 before writing so stale markers from
        the previous frame are cleared.

        Args:
            frame_idx: Index into ``clip.site_xpos`` (clamped to ``[0, T)``).
            user_scn: Viewer user scene obtained inside ``viewer.lock()``.
        """
        assert self.clip.site_xpos is not None
        n_frames, n_sites, _ = self.clip.site_xpos.shape
        frame_idx = max(0, min(int(frame_idx), n_frames - 1))
        targets = self.clip.site_xpos[frame_idx]  # (n_sites, 3)

        user_scn.ngeom = 0
        max_geoms = int(user_scn.maxgeom)
        for i in range(min(n_sites, max_geoms)):
            pos = np.asarray(targets[i], dtype=np.float64)
            mujoco.mjv_initGeom(
                user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                self._size_arr,
                pos,
                _IDENTITY_MAT,
                self._rgba_arr,
            )
            user_scn.ngeom += 1

    def as_viz_fn(
        self,
    ) -> SiteMarkerVizFn:
        """Return a ``VizCallback``-compatible callable bound to this instance.

        The returned object can be passed directly to
        :func:`~myosuite.core.mujoco_playback.run_passive_viewer_loop` as the
        ``viz_fn`` argument.

        Returns:
            A callable ``(step, model, data, user_scn) -> None``.
        """
        return SiteMarkerVizFn(self)


@dataclass
class SiteMarkerVizFn:
    """Callable adapter wrapping :class:`SiteMarkerViz` for the playback loop.

    Implements the ``VizCallback`` signature:
    ``(step_idx, model, data, user_scn) -> None``.
    """

    _viz: SiteMarkerViz

    def __call__(
        self,
        step_idx: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        user_scn: mujoco.MjvScene,
    ) -> None:
        self._viz.draw(step_idx, user_scn)


def make_site_marker_viz_fn(
    clip: MotionClip,
    site_names: tuple[str, ...],
    model: mujoco.MjModel,
    rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 0.6),
    sphere_radius: float = 0.03,
) -> SiteMarkerVizFn:
    """Convenience factory for a site-marker ``VizCallback``.

    Args:
        clip: Loaded motion clip with ``site_xpos`` populated.
        site_names: Ordered site names (axis-1 of ``clip.site_xpos``).
        model: Compiled MuJoCo model.
        rgba: Sphere colour (R, G, B, A).
        sphere_radius: Radius in metres.

    Returns:
        A ``VizCallback``-compatible callable ready for ``run_passive_viewer_loop``.
    """
    viz = SiteMarkerViz(
        clip=clip,
        site_names=site_names,
        model=model,
        rgba=rgba,
        sphere_radius=sphere_radius,
    )
    return viz.as_viz_fn()


__all__ = [
    "SiteMarkerViz",
    "SiteMarkerVizFn",
    "make_site_marker_viz_fn",
]
