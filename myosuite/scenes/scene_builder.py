# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""SceneSpec to backend-specific scene representation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import mujoco

from myosuite.scenes.library import (
    SCENES,
    FixtureSpec,
    ObjectSpec,
    SceneSpec,
    SiteMarkerSpec,
)

_GEOM_TYPE_MAP: dict[str, mujoco.mjtGeom] = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def get_scene(name: str) -> SceneSpec:
    """Look up a copy of a named scene from the library.

    Args:
        name: Scene name (e.g. ``"flat_floor"``, ``"seated_workspace"``).

    Returns:
        A deep copy of the corresponding :class:`SceneSpec`.

    Raises:
        KeyError: If the scene name is not registered.
    """
    if name not in SCENES:
        raise KeyError(f"Unknown scene: {name!r}. Available: {list(SCENES)}")
    return copy.deepcopy(SCENES[name])


def _geom_type_enum(shape: str) -> mujoco.mjtGeom:
    if shape not in _GEOM_TYPE_MAP:
        raise ValueError(
            f"Unsupported shape {shape!r}. Expected one of {sorted(_GEOM_TYPE_MAP)}"
        )
    return _GEOM_TYPE_MAP[shape]


def _geom_size_xml(shape: str, size: tuple[float, float, float]) -> str:
    s0, s1, s2 = size
    if shape == "sphere":
        return f"{s0}"
    if shape in ("capsule", "cylinder"):
        return f"{s0} {s1}"
    return f"{s0} {s1} {s2}"


def _iter_object_instances(spec: SceneSpec) -> list[tuple[str, ObjectSpec]]:
    instances: list[tuple[str, ObjectSpec]] = []
    for obj in spec.objects:
        count = max(0, int(obj.count))
        if count == 1:
            instances.append((obj.name, obj))
        else:
            for idx in range(count):
                instances.append((f"{obj.name}_{idx}", obj))
    return instances


def _object_spawn_pos(
    obj: ObjectSpec, index: int, total: int
) -> tuple[float, float, float]:
    spacing = max(3.0 * float(obj.size), 0.06)
    x = (index - 0.5 * (total - 1)) * spacing if total > 1 else 0.0
    z = 2.0 * float(obj.size) if obj.shape == "capsule" else float(obj.size)
    return (x, 0.0, z)


def _site_to_mjcf_line(site: SiteMarkerSpec) -> str:
    sx, sy, sz = site.pos
    size_str = _geom_size_xml(site.shape, site.size)
    rgba_str = " ".join(str(c) for c in site.rgba)
    return (
        f'      <site name="{site.name}" type="{site.shape}"'
        f' pos="{sx} {sy} {sz}" size="{size_str}"'
        f' rgba="{rgba_str}" group="{site.group}"/>'
    )


def scene_to_mjcf_xml(spec: SceneSpec) -> str:
    """Render a :class:`SceneSpec` to compilable MJCF XML.

    Args:
        spec: Scene specification to render.

    Returns:
        MJCF XML string with ``<option>``, ``<asset>``, and ``<worldbody>``.
    """
    gx, gy, gz = spec.gravity
    lines = [
        "<mujoco>",
        f'  <option gravity="{gx} {gy} {gz}"/>',
    ]

    if spec.floor.kind == "uneven":
        lines.extend(
            [
                "  <asset>",
                f'    <hfield name="terrain" nrow="64" ncol="64"'
                f' size="{spec.floor.size} {spec.floor.size} 0.2 0.1"/>',
                "  </asset>",
            ]
        )

    lines.append("  <worldbody>")

    if spec.floor.kind == "flat":
        lines.append(
            f'    <geom name="floor" type="plane"'
            f' size="{spec.floor.size} {spec.floor.size} 0.1"'
            ' rgba="0.9 0.9 0.9 1" condim="3"/>'
        )
    elif spec.floor.kind == "uneven":
        lines.append(
            '    <geom name="floor" type="hfield" hfield="terrain"'
            ' rgba="0.7 0.7 0.5 1" condim="3"/>'
        )

    for fix in spec.fixtures:
        _geom_type_enum(fix.shape)
        px, py, pz = fix.pos
        qw, qx, qy, qz = fix.quat
        mocap_attr = ' mocap="true"' if fix.mocap else ""
        con_val = 1 if fix.collide else 0
        size_str = _geom_size_xml(fix.shape, fix.size)
        rgba_str = " ".join(str(c) for c in fix.rgba)
        lines.append(
            f'    <body name="{fix.name}" pos="{px} {py} {pz}"'
            f' quat="{qw} {qx} {qy} {qz}"{mocap_attr}>'
        )
        lines.append(
            f'      <geom name="{fix.name}_geom" type="{fix.shape}"'
            f' size="{size_str}" rgba="{rgba_str}"'
            f' contype="{con_val}" conaffinity="{con_val}"/>'
        )
        for site in fix.sites:
            lines.append(_site_to_mjcf_line(site))
        lines.append("    </body>")

    instances = _iter_object_instances(spec)
    n_total = len(instances)
    for idx, (body_name, obj) in enumerate(instances):
        if obj.shape == "box":
            geom_attrs = f'type="box" size="{obj.size} {obj.size} {obj.size}"'
        elif obj.shape in ("capsule", "cylinder"):
            geom_attrs = f'type="{obj.shape}" size="{obj.size} {obj.size}"'
        else:
            geom_attrs = f'type="sphere" size="{obj.size}"'

        px, py, pz = _object_spawn_pos(obj, idx, n_total)
        rgba_str = " ".join(str(c) for c in obj.rgba)
        lines.append(
            f'    <body name="{body_name}" pos="{px} {py} {pz}">'
            f"\n      <freejoint/>"
            f'\n      <geom {geom_attrs} rgba="{rgba_str}" mass="{obj.mass}"/>'
            f"\n    </body>"
        )

    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def _attach_fixture_to_spec(mj_spec: mujoco.MjSpec, fix: FixtureSpec) -> None:
    body = mj_spec.worldbody.add_body(
        name=fix.name,
        pos=list(fix.pos),
        quat=list(fix.quat),
        mocap=bool(fix.mocap),
    )
    geom = body.add_geom(name=f"{fix.name}_geom")
    geom.type = _geom_type_enum(fix.shape)
    geom.size = list(fix.size)
    geom.rgba = list(fix.rgba)
    con_val = 1 if fix.collide else 0
    geom.contype = con_val
    geom.conaffinity = con_val
    for s in fix.sites:
        site = body.add_site(name=s.name)
        site.type = _geom_type_enum(s.shape)
        site.pos = list(s.pos)
        site.size = list(s.size)
        site.rgba = list(s.rgba)
        site.group = int(s.group)


def apply_scene_to_spec(
    mj_spec: mujoco.MjSpec,
    spec: SceneSpec,
    *,
    include_floor: bool = False,
) -> mujoco.MjSpec:
    """Attach a :class:`SceneSpec`'s fixtures and objects onto an existing ``MjSpec``.

    Suitable as a :meth:`~myosuite.core.model_builder.ModelBuilder.apply_transform`
    stage when composing musculoskeletal models with parameterized workspaces.

    Args:
        mj_spec: Target MuJoCo specification modified in place.
        spec: Scene specification whose fixtures and objects are added.
        include_floor: Also add ``spec.floor`` when ``mj_spec`` has no floor geom.

    Returns:
        *mj_spec* for fluent chaining.
    """
    if include_floor and spec.floor.kind == "flat":
        floor = mj_spec.worldbody.add_geom(name="floor")
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [spec.floor.size, spec.floor.size, 0.1]
        floor.rgba = [0.9, 0.9, 0.9, 1.0]
        floor.condim = 3
        floor.contype = 1
        floor.conaffinity = 1

    for fix in spec.fixtures:
        _attach_fixture_to_spec(mj_spec, fix)

    instances = _iter_object_instances(spec)
    n_total = len(instances)
    for idx, (body_name, obj) in enumerate(instances):
        pos = _object_spawn_pos(obj, idx, n_total)
        body = mj_spec.worldbody.add_body(name=body_name, pos=list(pos))
        body.add_freejoint()
        geom = body.add_geom()
        geom.type = _geom_type_enum(obj.shape)
        geom.mass = float(obj.mass)
        geom.rgba = list(obj.rgba)
        geom.contype = 1
        geom.conaffinity = 1
        if obj.shape == "box":
            geom.size = [obj.size, obj.size, obj.size]
        elif obj.shape in ("capsule", "cylinder"):
            geom.size = [obj.size, obj.size, 0.0]
        else:
            geom.size = [obj.size, 0.0, 0.0]
    return mj_spec


def _make_object_spec_fn(
    body_name: str, obj: ObjectSpec, pos: tuple[float, float, float]
) -> Callable[[], mujoco.MjSpec]:
    def _build() -> mujoco.MjSpec:
        s = mujoco.MjSpec()
        body = s.worldbody.add_body(name=body_name, pos=list(pos))
        body.add_freejoint()
        geom = body.add_geom()
        geom.mass = float(obj.mass)
        geom.rgba = list(obj.rgba)
        geom.type = _geom_type_enum(obj.shape)
        if obj.shape == "box":
            geom.size = [obj.size, obj.size, obj.size]
        elif obj.shape in ("capsule", "cylinder"):
            geom.size = [obj.size, obj.size, 0.0]
        else:
            geom.size = [obj.size, 0.0, 0.0]
        return s

    return _build


def _make_fixture_spec_fn(fix: FixtureSpec) -> Callable[[], mujoco.MjSpec]:
    fix_copy = copy.deepcopy(fix)

    def _build() -> mujoco.MjSpec:
        s = mujoco.MjSpec()
        _attach_fixture_to_spec(s, fix_copy)
        return s

    return _build


def scene_to_mjlab_cfg(spec: SceneSpec) -> dict[str, Any]:
    """Render a :class:`SceneSpec` to mjlab ``EntityCfg`` entries.

    Args:
        spec: Scene specification to render.

    Returns:
        Dict of ``entity_name -> EntityCfg``, or ``{}`` if ``mjlab`` is unavailable.
    """
    try:
        from mjlab.entity import EntityCfg
    except ImportError:
        return {}

    cfg: dict[str, Any] = {}
    for fix in spec.fixtures:
        cfg[fix.name] = EntityCfg(
            spec_fn=_make_fixture_spec_fn(fix),
            init_state=EntityCfg.InitialStateCfg(pos=fix.pos, rot=fix.quat),
        )

    instances = _iter_object_instances(spec)
    n_total = len(instances)
    for idx, (body_name, obj) in enumerate(instances):
        pos = _object_spawn_pos(obj, idx, n_total)
        cfg[body_name] = EntityCfg(
            spec_fn=_make_object_spec_fn(body_name, obj, pos),
            init_state=EntityCfg.InitialStateCfg(pos=pos),
        )
    return cfg
