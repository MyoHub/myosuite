# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared boxing visual helpers: gloves, helmet, ring, and hand-mesh hiding.

All canonical visual constants live here.  Call sites inherit the defaults
and override only what genuinely differs for their task.
"""

from __future__ import annotations
from pathlib import Path

import mujoco
import numpy as np

from myosuite.physics.quat_math import intrinsic_euler2quat, euler2quat

_ASSETS = Path(__file__).resolve().parents[2] / "assets"

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
BOXING_GLOVE_MESH: Path = _ASSETS / "boxing_glove_r.stl"
BOXING_HELMET_MESH: Path = _ASSETS / "boxing_elmet.stl"
BOXING_RING_MESH: Path = _ASSETS / "ring.obj"

# ---------------------------------------------------------------------------
# Canonical visual defaults
# Edit these to adjust appearance across all boxing tasks at once.
# Override per-call only when a task genuinely needs different placement.
# ---------------------------------------------------------------------------

# Glove — anchored to ``lunate_{r|l}`` (wrist carpal body).
GLOVE_SCALE: float = 0.06
GLOVE_POS: list[float] = [0.0, -0.07, -0.02]
GLOVE_EULER_DEG: list[float] = [7.5, 0.0, 180.0]

# Helmet — anchored to ``head`` body.
# +z = upward, −z = toward neck; +y = anterior, −y = posterior.
HELMET_SCALE: float = 0.22
HELMET_POS: list[float] = [0.0, -0.05, 0.0]
HELMET_EULER_DEG: list[float] = [-90.0, 90.0, 0.0]

# Boxing ring — world-level decoration.
RING_SCALE: list[float] = [0.12, 0.12, 0.12]
RING_POS: list[float] = [0.0, -3.9, 0.0]

# ---------------------------------------------------------------------------
# Mesh-hiding lists — used by replace_hand_visuals_with_gloves
# ---------------------------------------------------------------------------

# Geom names whose mesh should be hidden (exact match).
HAND_VISUAL_GEOM_NAMES: tuple[str, ...] = (
    "lunate",
    "scaphoid",
    "pisiform",
    "triquetrum",
    "capitate",
    "trapezium",
    "trapezoid",
    "firstmc",
    "secondmc",
    "2proxph",
    "2distph",
    "3proxph",
    "3midph",
    "3distph",
    "4proxph",
    "5proxph",
)

# Geom name tokens — any geom whose name contains one of these is hidden.
HAND_GEOM_TOKENS: tuple[str, ...] = (
    "lunate",
    "scaphoid",
    "pisiform",
    "triquetrum",
    "capitate",
    "trapezium",
    "trapezoid",
    "firstmc",
    "secondmc",
    "1proxph",
    "1distph",
    "1mc",
    "2mc",
    "thumb",
    "2proxph",
    "2distph",
    "3proxph",
    "3midph",
    "3distph",
    "4proxph",
    "4distph",
    "5proxph",
    "5distph",
    "proxph",
    "midph",
    "distph",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quat_from_euler_xyz_deg(euler_deg: list[float]) -> list[float]:
    """Convert intrinsic XYZ (roll, pitch, yaw) Euler angles in degrees to quat.

    Uses ``intrinsic_euler2quat`` — the same convention as the historical
    half-angle composition used for glove/helmet mesh placement.  Do not use
    ``euler2quat`` here: that helper follows a different axis ordering used
    elsewhere in MyoSuite.

    Args:
        euler_deg: [roll_deg, pitch_deg, yaw_deg] about X, then Y, then Z.

    Returns:
        Quaternion as [w, x, y, z].
    """
    euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    return intrinsic_euler2quat(euler_rad).astype(np.float64).tolist()


def replace_hand_visuals_with_gloves(
    spec: mujoco.MjSpec,
    *,
    scale: float = GLOVE_SCALE,
    pos: list[float] = GLOVE_POS,
    euler_deg: list[float] = GLOVE_EULER_DEG,
    left_euler_deg: list[float] | None = None,
    rgba: list[float] | None = None,
    delete_hand: bool = True,
) -> None:
    """Hide hand/finger meshes and add boxing-glove visual meshes.

    Visual-only: collision geoms and body/joint topology are unchanged.

    Args:
        spec: The MjSpec to modify (per-agent spec or combined scene spec).
        scale: Uniform scale applied to the glove mesh on all three axes.
        pos: [x, y, z] offset in the ``lunate_r`` / ``lunate_l`` body frame.
            The z component is mirrored for the left hand.
        euler_deg: XYZ Euler angles in degrees for right glove orientation.
        left_euler_deg: Optional explicit left-glove XYZ Euler orientation.
            If None, a mirrored orientation plus 180 deg twist is derived from
            ``euler_deg`` as ``[rx, 180 - ry, -rz]``.
        rgba: Glove colour [r, g, b, a].  Defaults to classic red.
        delete_hand: Whether to delete hand/finger meshes, otherwise sets to transparent.
    """
    if rgba is None:
        rgba = [0.84, 0.12, 0.12, 1.0]

    glove_mesh_r = spec.add_mesh()
    glove_mesh_r.name = "boxing_glove_mesh_r"
    glove_mesh_r.file = str(BOXING_GLOVE_MESH)
    glove_mesh_r.scale = [scale, scale, scale]

    glove_mesh_l = spec.add_mesh()
    glove_mesh_l.name = "boxing_glove_mesh_l"
    glove_mesh_l.file = str(BOXING_GLOVE_MESH)
    glove_mesh_l.scale = [-scale, scale, scale]

    glove_mat = spec.add_material()
    glove_mat.name = "boxing_glove_mat"
    glove_mat.rgba = rgba
    glove_mat.specular = 0.15
    glove_mat.shininess = 0.35

    for geom in [g for g in spec.geoms if any(token in g.name for token in HAND_GEOM_TOKENS)]:
        if not delete_hand:
            geom.rgba = [0.0, 0.0, 0.0, 0.0]
        else:
            if geom.meshname:
                paired_mesh = next((m for m in spec.meshes if geom.meshname == m.name), None)
                if paired_mesh is not None:
                    spec.delete(paired_mesh)
            spec.delete(geom)


    right_quat = quat_from_euler_xyz_deg(euler_deg)
    if left_euler_deg is None:
        left_euler_deg = [euler_deg[0], 180.0 - euler_deg[1], -euler_deg[2]]
    left_quat = quat_from_euler_xyz_deg(left_euler_deg)
    for side in ("r", "l"):
        side_pos = [pos[0], pos[1], pos[2] if side == "r" else -pos[2]]
        glove_body = spec.body(f"lunate_{side}").add_body(name=f"boxing_glove_body_{side}",
                                                          pos=side_pos,
                                                          quat=right_quat if side == "r" else left_quat,)
        glove = glove_body.add_geom(
            name=f"boxing_glove_{side}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            rgba=rgba,
            contype=0,
            conaffinity=0,
            density=0,
            mass=0
        )
        collider_geom =glove_body.add_geom(
            name=f"boxing_glove_coll_{side}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[-0.034*scale, 0.9*scale, 0.0],
            group=4,
            size=[scale, scale, 0],
            mass=0.3,
            conaffinity=3,
            contype=3,
            quat=quat_from_euler_xyz_deg([0, 90, 0])
        )
        glove.meshname = "boxing_glove_mesh_r" if side == "r" else "boxing_glove_mesh_l"
        glove.material = "boxing_glove_mat"
    return spec


def add_helmet(
    spec: mujoco.MjSpec,
    *,
    scale: float = HELMET_SCALE,
    pos: list[float] = HELMET_POS,
    euler_deg: list[float] = HELMET_EULER_DEG,
    rgba: list[float] | None = None,
) -> None:
    """Attach a visual-only boxing helmet mesh to the ``head`` body.

    Must be called before the spec is attached to a combined scene so the
    body is still named ``"head"`` (not ``"{prefix}head"``).

    Args:
        spec: Per-agent MjSpec (before attachment).
        scale: Uniform scale applied to the helmet mesh.
        pos: [x, y, z] offset in the ``head`` body's local frame.
        euler_deg: XYZ Euler angles in degrees for helmet orientation.
        rgba: Helmet colour [r, g, b, a].  Defaults to dark grey.
    """
    if rgba is None:
        rgba = [0.15, 0.15, 0.15, 1.0]

    helmet_mesh = spec.add_mesh()
    helmet_mesh.name = "boxing_helmet_mesh"
    helmet_mesh.file = str(BOXING_HELMET_MESH)
    helmet_mesh.scale = [scale, scale, scale]

    helmet_mat = spec.add_material()
    helmet_mat.name = "boxing_helmet_mat"
    helmet_mat.rgba = rgba
    helmet_mat.specular = 0.3
    helmet_mat.shininess = 0.5

    geom = spec.body("head").add_geom(
        name="boxing_helmet",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=pos,
        quat=quat_from_euler_xyz_deg(euler_deg),
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )
    geom.meshname = "boxing_helmet_mesh"
    geom.material = "boxing_helmet_mat"
    return spec


def replace_floor_visual_with_boxing_ring(spec: mujoco.MjSpec) -> None:
    """Hide default floor visual and render boxing ring mesh instead."""
    ring_mesh = spec.add_mesh()
    ring_mesh.name = "boxing_ring_mesh"
    ring_mesh.file = str(BOXING_RING_MESH)
    ring_mesh.scale = RING_SCALE

    ring_material = spec.add_material()
    ring_material.name = "boxing_ring_mat"
    ring_material.rgba = [1.0, 1.0, 1.0, 1.0]
    ring_material.specular = 0.05
    ring_material.shininess = 0.05

    texture_ring = spec.add_texture()
    spec.compiler.texturedir = str(_ASSETS)
    texture_ring.name = f"ring_tex"
    texture_ring.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture_ring.file = "ring.png"
    ring_material.textures = [''] + ["ring_tex"] + [''] * 8

    # Keep floor plane collision, but hide its visual.
    for geom in spec.geoms:
        if (geom.name or "") == "floor":
            geom.rgba = [0.0, 0.0, 0.0, 0.0]
            break

    ring_geom = spec.worldbody.add_geom(
        name="boxing_ring_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=RING_POS,
        contype=0,
        conaffinity=0,
    )
    ring_geom.meshname = "boxing_ring_mesh"
    ring_geom.material = "boxing_ring_mat"
    return spec


def build_targets_spec() -> mujoco.MjSpec:
    """Build 6-target boxing machine spec."""
    def _quat_from_euler_deg(roll_pitch_yaw_deg: tuple[float, float, float]) -> list[float]:
        """Convert MuJoCo XML euler degrees to quat [w, x, y, z]."""
        quat = euler2quat(np.deg2rad(np.asarray(roll_pitch_yaw_deg, dtype=np.float64)))
        return quat.astype(np.float64).tolist()

    spec = mujoco.MjSpec()
    spec.modelname = "boxing_targets_agent"
    spec.compiler.meshdir = str(_ASSETS)
    spec.compiler.texturedir = str(_ASSETS)
    spec.compiler.autolimits = True
    spec.compiler.inertiafromgeom = True

    targets_mesh = spec.add_mesh()
    targets_mesh.name = "boxing_targets_mesh"
    targets_mesh.file = "dummy.obj"

    texture_dummy = spec.add_texture()
    texture_dummy.name = f"dummy_tex"
    texture_dummy.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture_dummy.file = "dummy.png"

    mesh_gray = spec.add_material()
    mesh_gray.name = "boxing_targets_mesh_gray"
    mesh_gray.textures = ['', 'dummy_tex', '', '', '', '', '', '', '', '']
    mesh_gray.shininess = 0.2
    spec.visual.map.fogstart = 0.002
    spec.visual.map.fogend = 0.004

    fill = spec.worldbody.add_light(name="scene_spot")
    fill.pos = [-2.0, -2.0, 4.0]
    fill.dir = [0.5, 0.5, -1]
    fill.specular = [0.5, 0.5, 0.5]
    fill.castshadow = True

    target_cyan = spec.add_material()
    target_cyan.name = "boxing_targets_cyan"
    target_cyan.rgba = [32/256, 138/256, 160/256, 1.0]

    machine = spec.worldbody.add_body(name="targets", pos=[0.0, 0.0, 0.0], mocap=True)
    machine_mesh = machine.add_geom(
        name="mannequin_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        contype=0,
        conaffinity=0,
    )
    machine_mesh.meshname = "boxing_targets_mesh"
    machine_mesh.material = "boxing_targets_mesh_gray"

    pad_data = (
        (
            "pad_0",
            [0.0, 0.08, 1.645],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.14, 0.02, 0.0],
            (60.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
        ),
        (
            "pad_1",
            [0.36, 0.0, 1.54],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.12, 0.01, 0.0],
            (69, 49, 0),
            (90.0, 0.0, -45.0),
        ),
        (
            "pad_2",
            [0.29, 0.23, 1.038],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.10, 0.012, 0.0],
            (53, 49, 0),
            (90.0, 0.0, -135.0),
        ),
        (
            "pad_3",
            [-0.29, 0.23, 1.038],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.10, 0.012, 0.0],
            (53, 49+90-6.5, 0),
            (90.0, 0.0, -225.0),
        ),
        (
            "pad_4",
            [-0.36, 0.0, 1.54],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.12, 0.012, 0.0],
            (69, 49+90-6.5, 0),
            (90.0, 0.0, -270.0),
        ),
    )
    for name, pos, geom_type, size, geom_euler, site_euler in pad_data:
        pad = machine.add_body(name=name, pos=pos)
        geom = pad.add_geom(
            name=f"{name}_geom",
            type=geom_type,
            size=size,
            material="boxing_targets_cyan",
            contype=2,
            conaffinity=2,
        )

        site = pad.add_site(
            name=f"{name}_site",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array(size) + 0.05,
            rgba=[0.6, 0.6, 0.6, 0.5],
            group=4
        )

        spec.add_sensor(name=f"punch_sensor_{name}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{name}_site")

        if geom_euler is not None:
            geom.quat = _quat_from_euler_deg(geom_euler)
            site.quat = _quat_from_euler_deg(geom_euler)

    return spec
