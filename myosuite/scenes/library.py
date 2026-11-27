# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene library: named SceneSpec presets and re-exported scene specifications.

SceneSpec is backend-aware:
- CPU / MjSpec -> MJCF XML or direct MjSpec body/site attachment
- mjlab        -> EntityCfg entries in SceneCfg
"""

from __future__ import annotations

from myosuite.scenes.specs import (
    FixtureSpec,
    FloorSpec,
    ObjectSpec,
    SceneSpec,
    SiteMarkerSpec,
)
from myosuite.scenes.workspaces import WorkspaceConfig, make_workspace_scene

SCENES: dict[str, SceneSpec] = {
    "flat_floor": SceneSpec(
        floor=FloorSpec(kind="flat"),
        objects=[],
    ),
    "uneven_floor": SceneSpec(
        floor=FloorSpec(kind="uneven"),
        objects=[],
    ),
    "cube_reach": SceneSpec(
        floor=FloorSpec(kind="flat"),
        objects=[ObjectSpec(name="cube", shape="box", size=0.03)],
    ),
    "sphere_reach": SceneSpec(
        floor=FloorSpec(kind="flat"),
        objects=[ObjectSpec(name="sphere", shape="sphere", size=0.03)],
    ),
    "baoding_pair": SceneSpec(
        floor=FloorSpec(kind="flat"),
        objects=[
            ObjectSpec(
                name="ball_0",
                shape="sphere",
                size=0.028,
                mass=0.045,
                rgba=(0.8, 0.2, 0.2, 1.0),
            ),
            ObjectSpec(
                name="ball_1",
                shape="sphere",
                size=0.028,
                mass=0.045,
                rgba=(0.2, 0.2, 0.8, 1.0),
            ),
        ],
    ),
    "tabletop_workspace": make_workspace_scene(
        WorkspaceConfig(include_seat=False, include_panel=True)
    ),
    "seated_workspace": make_workspace_scene(
        WorkspaceConfig(
            include_seat=True,
            include_armrests=True,
            include_surface=True,
            include_panel=True,
            boundary_half_extents=(0.55, 0.45, 0.55),
            boundary_collide=False,
        )
    ),
}

__all__ = [
    "SCENES",
    "FixtureSpec",
    "FloorSpec",
    "ObjectSpec",
    "SceneSpec",
    "SiteMarkerSpec",
]
