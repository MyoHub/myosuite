# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""SceneSpec to backend-specific scene representation via a single MjSpec emitter."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import mujoco

from myosuite.scenes.library import (
    SCENES,
    FixtureSpec,
    FloorSpec,
    ObjectSpec,
    SceneSpec,
)

_GEOM_TYPE_MAP: dict[str, mujoco.mjtGeom] = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}

_HFIELD_NROW = 64
_HFIELD_NCOL = 64


def get_scene(name: str) -> SceneSpec:
    """Look up a deep copy of a named scene from the library.

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


def _object_geom_size(obj: ObjectSpec) -> list[float]:
    s = float(obj.size)
    if obj.shape == "box":
        return [s, s, s]
    if obj.shape in ("capsule", "cylinder"):
        return [s, s, 0.0]
    return [s, 0.0, 0.0]


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


def _attach_floor_to_spec(mj_spec: mujoco.MjSpec, floor: FloorSpec) -> None:
    if floor.kind == "none":
        return
    if any(g.name == "floor" for g in mj_spec.geoms):
        return
    if floor.kind == "flat":
        geom = mj_spec.worldbody.add_geom(name="floor")
        geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        geom.size = [float(floor.size), float(floor.size), 0.1]
        geom.rgba = [0.9, 0.9, 0.9, 1.0]
        geom.condim = 3
        geom.contype = 1
        geom.conaffinity = 1
    elif floor.kind == "uneven":
        if not any(h.name == "terrain" for h in mj_spec.hfields):
            mj_spec.add_hfield(
                name="terrain",
                nrow=_HFIELD_NROW,
                ncol=_HFIELD_NCOL,
                size=[float(floor.size), float(floor.size), 0.2, 0.1],
                userdata=[0.0] * (_HFIELD_NROW * _HFIELD_NCOL),
            )
        geom = mj_spec.worldbody.add_geom(name="floor")
        geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
        geom.hfieldname = "terrain"
        geom.rgba = [0.7, 0.7, 0.5, 1.0]
        geom.condim = 3
        geom.contype = 1
        geom.conaffinity = 1
    else:
        raise ValueError(
            f"Unsupported floor kind {floor.kind!r}. Expected 'flat', 'uneven', or 'none'."
        )


def _attach_fixture_to_spec(
    mj_spec: mujoco.MjSpec,
    fix: FixtureSpec,
    *,
    root_origin: bool = False,
) -> None:
    pos = [0.0, 0.0, 0.0] if root_origin else list(fix.pos)
    quat = [1.0, 0.0, 0.0, 0.0] if root_origin else list(fix.quat)
    mocap = False if root_origin else bool(fix.mocap)
    body = mj_spec.worldbody.add_body(
        name=fix.name,
        pos=pos,
        quat=quat,
        mocap=mocap,
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


def _attach_object_to_spec(
    mj_spec: mujoco.MjSpec,
    body_name: str,
    obj: ObjectSpec,
    pos: tuple[float, float, float],
) -> None:
    body = mj_spec.worldbody.add_body(name=body_name, pos=list(pos))
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = _geom_type_enum(obj.shape)
    geom.size = _object_geom_size(obj)
    geom.mass = float(obj.mass)
    geom.rgba = list(obj.rgba)
    geom.contype = 1
    geom.conaffinity = 1


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
        include_floor: Add ``spec.floor`` when ``mj_spec`` does not already
            define a geom named ``"floor"``.

    Returns:
        *mj_spec* for fluent chaining.
    """
    if include_floor:
        _attach_floor_to_spec(mj_spec, spec.floor)

    for fix in spec.fixtures:
        _attach_fixture_to_spec(mj_spec, fix)

    instances = _iter_object_instances(spec)
    n_total = len(instances)
    for idx, (body_name, obj) in enumerate(instances):
        pos = _object_spawn_pos(obj, idx, n_total)
        _attach_object_to_spec(mj_spec, body_name, obj, pos)
    return mj_spec


def scene_to_mjcf_xml(spec: SceneSpec) -> str:
    """Render a :class:`SceneSpec` to compilable MJCF XML via ``MjSpec``.

    Args:
        spec: Scene specification to render.

    Returns:
        Serialized MJCF XML string compiled from :func:`apply_scene_to_spec`.
    """
    mj_spec = mujoco.MjSpec()
    mj_spec.option.gravity = [float(g) for g in spec.gravity]
    apply_scene_to_spec(mj_spec, spec, include_floor=True)
    mj_spec.compile()
    return mj_spec.to_xml()


def _make_object_spec_fn(
    body_name: str, obj: ObjectSpec, pos: tuple[float, float, float]
) -> Callable[[], mujoco.MjSpec]:
    obj_copy = copy.deepcopy(obj)

    def _build() -> mujoco.MjSpec:
        s = mujoco.MjSpec()
        _attach_object_to_spec(s, body_name, obj_copy, pos)
        return s

    return _build


def _make_fixture_spec_fn(fix: FixtureSpec) -> Callable[[], mujoco.MjSpec]:
    fix_copy = copy.deepcopy(fix)

    def _build() -> mujoco.MjSpec:
        s = mujoco.MjSpec()
        # mjlab mounts fixed-base entities on a mocap frame placed at InitialStateCfg(pos, rot).
        _attach_fixture_to_spec(s, fix_copy, root_origin=True)
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
