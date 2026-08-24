# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared arena loader for 1v1 chase-tag model builders.

Reads the canonical fence geometry from ``chasetag_arena.xml`` and adds it to
an existing :class:`mujoco.MjSpec`.  Both the myoLeg and full-body model
builders call :func:`add_arena` so the arena is defined exactly once.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

_ARENA_XML = Path(__file__).parents[3] / "assets" / "scene" / "chasetag_arena.xml"


def add_arena(spec: mujoco.MjSpec) -> None:
    """Parse ``chasetag_arena.xml`` and merge its meshes + geoms into *spec*.

    The XML is the single source of truth for fence geometry.  Materials are
    not copied; geoms receive their colour directly from the XML's material
    rgba so no material lookup is needed in the target spec.
    """
    tree = ET.parse(_ARENA_XML)
    root = tree.getroot()

    # ── meshes ────────────────────────────────────────────────────────────────
    mesh_rgba: dict[str, list[float]] = {}
    # Read fence material rgba (used below for geom colours)
    for mat in root.findall("./asset/material"):
        if mat.get("name") == "fence":
            rgba = [float(v) for v in mat.get("rgba", "0.27 0.25 0.25 1").split()]
            mesh_rgba["fence"] = rgba

    fence_rgba = mesh_rgba.get("fence", [0.27, 0.25, 0.25, 1.0])

    for mesh_el in root.findall("./asset/mesh"):
        name = mesh_el.get("name", "")
        vertex_str = mesh_el.get("vertex", "")
        verts = [float(v) for v in vertex_str.split()]
        m = spec.add_mesh(name=name)
        m.uservert = verts

    # ── geoms ─────────────────────────────────────────────────────────────────
    for geom_el in root.findall("./worldbody/geom"):
        name = geom_el.get("name", "")
        meshname = geom_el.get("mesh", "")
        pos = [float(v) for v in geom_el.get("pos", "0 0 0").split()]
        euler_deg = [float(v) for v in geom_el.get("euler", "0 0 0").split()]
        # Convert euler ZYX (as used in the XML) to quaternion [w, x, y, z]
        quat = _euler_to_quat_z(euler_deg[2])

        g = spec.worldbody.add_geom(name=name)
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = meshname
        g.pos = pos
        g.quat = quat
        g.rgba = fence_rgba
        g.contype = 1
        g.conaffinity = 1


def _euler_to_quat_z(angle_rad: float) -> list[float]:
    """Quaternion [w, x, y, z] for a pure Z-axis rotation by *angle_rad*."""
    return [math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2)]
