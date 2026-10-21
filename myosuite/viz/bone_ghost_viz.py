# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Anatomical mesh ghost rendering for reference motion overlays."""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.viz.ghost_body_viz import GhostBodyViz, _quat_to_mat9


class BoneGhostBodyViz(GhostBodyViz):
    """GhostBodyViz variant that renders the model's anatomical mesh geoms.

    This keeps collision/wrapping primitives out of the ghost and uses MuJoCo's
    mesh data IDs directly, so the same overlay works in the native renderer and
    in USD/Blender exporters.
    """

    def draw(self, step_idx: int, user_scn: mujoco.MjvScene) -> None:
        if self.bone_model is None:
            super().draw(step_idx, user_scn)
            return

        frame = (self.frame_offset + step_idx * self.ctrl_dt_ratio) % self._T
        pos_frame = self.xpos[frame]
        if self.position_offset is not None:
            pos_frame = pos_frame + self.position_offset

        model = self.bone_model
        max_geoms = int(user_scn.maxgeom)
        start = self._first_slot(user_scn)
        geom_i = start
        position = np.empty(3)
        quaternion = np.empty(4)
        visible = set(map(int, self.visible_body_ids))

        for geom_id in range(model.ngeom):
            body = int(model.geom_bodyid[geom_id])
            if (
                model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
                or body not in visible
            ):
                continue
            if geom_i >= max_geoms:
                break
            mujoco.mju_mulPose(
                position,
                quaternion,
                pos_frame[body],
                self.xquat[frame, body],
                model.geom_pos[geom_id],
                model.geom_quat[geom_id],
            )
            geom = user_scn.geoms[geom_i]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_MESH,
                model.geom_size[geom_id],
                position,
                _quat_to_mat9(quaternion),
                self._skel_rgba,
            )
            geom.dataid = int(model.geom_dataid[geom_id])
            geom_i += 1

        user_scn.ngeom = geom_i
        self._drawn[user_scn] = (start, geom_i)
