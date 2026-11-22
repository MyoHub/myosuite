# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Parameterized workspace presets mapped onto generic :class:`SceneSpec` entries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from myosuite.scenes.library import (
    FixtureSpec,
    FloorSpec,
    SceneSpec,
    SiteMarkerSpec,
    quat_from_euler_xyz,
)


@dataclass(frozen=True)
class WorkspaceConfig:
    """Parameterized 3D workspace configuration for seated or standing tasks.

    Coordinates follow Z-up, +X forward (or lateral per anchor), +Y left.
    """

    include_seat: bool = True
    seat_pos: tuple[float, float, float] = (0.0, 0.0, 0.46)
    seat_size: tuple[float, float, float] = (0.24, 0.24, 0.03)
    backrest_height: float = 0.48
    backrest_recline_rad: float = 0.14
    include_armrests: bool = False
    armrest_height: float = 0.22
    include_surface: bool = True
    surface_pos: tuple[float, float, float] = (0.45, 0.0, 0.72)
    surface_size: tuple[float, float, float] = (0.25, 0.40, 0.018)
    include_panel: bool = True
    panel_pos: tuple[float, float, float] = (0.55, 0.0, 1.05)
    panel_size: tuple[float, float, float] = (0.01, 0.24, 0.16)
    panel_pitch_rad: float = -0.20
    panel_mocap: bool = False
    target_grid: tuple[int, int] = (2, 3)
    target_radius: float = 0.015
    boundary_half_extents: tuple[float, float, float] | None = None
    boundary_center: tuple[float, float, float] = (0.25, 0.0, 0.95)
    boundary_collide: bool = False


def make_workspace_scene(cfg: WorkspaceConfig | None = None) -> SceneSpec:
    """Construct a :class:`SceneSpec` from a :class:`WorkspaceConfig`."""
    c = WorkspaceConfig() if cfg is None else cfg
    fixtures: list[FixtureSpec] = []

    if c.include_seat:
        sx, sy, sz = c.seat_pos
        hx, hy, hz = c.seat_size
        fixtures.append(
            FixtureSpec(
                name="seat_base",
                shape="box",
                size=c.seat_size,
                pos=c.seat_pos,
                rgba=(0.32, 0.34, 0.38, 1.0),
            )
        )
        if c.backrest_height > 0.0:
            half_h = 0.5 * c.backrest_height
            recline = c.backrest_recline_rad
            bx = sx - hx - half_h * math.sin(recline)
            bz = sz + hz + half_h * math.cos(recline)
            fixtures.append(
                FixtureSpec(
                    name="seat_backrest",
                    shape="box",
                    size=(0.025, hy, half_h),
                    pos=(bx, sy, bz),
                    quat=quat_from_euler_xyz(0.0, -recline, 0.0),
                    rgba=(0.32, 0.34, 0.38, 1.0),
                )
            )
        if c.include_armrests:
            for side, sign in (("left", 1.0), ("right", -1.0)):
                fixtures.append(
                    FixtureSpec(
                        name=f"seat_armrest_{side}",
                        shape="box",
                        size=(0.16, 0.03, 0.015),
                        pos=(sx + 0.02, sy + sign * (hy + 0.03), sz + c.armrest_height),
                        rgba=(0.28, 0.30, 0.34, 1.0),
                    )
                )

    if c.include_surface:
        fixtures.append(
            FixtureSpec(
                name="work_surface",
                shape="box",
                size=c.surface_size,
                pos=c.surface_pos,
                rgba=(0.76, 0.72, 0.65, 1.0),
            )
        )

    if c.include_panel:
        rows, cols = c.target_grid
        px_half, py_half, pz_half = c.panel_size
        sites: list[SiteMarkerSpec] = []
        if rows > 0 and cols > 0:
            ys = (
                [0.0]
                if cols == 1
                else [(j / (cols - 1) - 0.5) * 1.5 * py_half for j in range(cols)]
            )
            zs = (
                [0.0]
                if rows == 1
                else [(0.5 - i / (rows - 1)) * 1.5 * pz_half for i in range(rows)]
            )
            r = c.target_radius
            for i, z_off in enumerate(zs):
                for j, y_off in enumerate(ys):
                    sites.append(
                        SiteMarkerSpec(
                            name=f"panel_target_{i}_{j}",
                            pos=(-px_half - 0.002, y_off, z_off),
                            size=(r, r, r),
                        )
                    )
        fixtures.append(
            FixtureSpec(
                name="interaction_panel",
                shape="box",
                size=c.panel_size,
                pos=c.panel_pos,
                quat=quat_from_euler_xyz(0.0, c.panel_pitch_rad, 0.0),
                mocap=c.panel_mocap,
                rgba=(0.16, 0.20, 0.26, 0.92),
                sites=tuple(sites),
            )
        )

    if c.boundary_half_extents is not None:
        ex, ey, ez = c.boundary_half_extents
        cx, cy, cz = c.boundary_center
        wall_t = 0.006
        wall_rgba = (0.2, 0.65, 0.9, 0.18)
        walls = (
            ("boundary_left", (ex, wall_t, ez), (cx, cy + ey, cz)),
            ("boundary_right", (ex, wall_t, ez), (cx, cy - ey, cz)),
            ("boundary_front", (wall_t, ey, ez), (cx + ex, cy, cz)),
            ("boundary_top", (ex, ey, wall_t), (cx, cy, cz + ez)),
        )
        for w_name, w_size, w_pos in walls:
            fixtures.append(
                FixtureSpec(
                    name=w_name,
                    shape="box",
                    size=w_size,
                    pos=w_pos,
                    collide=c.boundary_collide,
                    rgba=wall_rgba,
                )
            )

    return SceneSpec(floor=FloorSpec(kind="flat"), fixtures=fixtures)
