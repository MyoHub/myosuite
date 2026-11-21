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

from myosuite.scenes.library import SCENES, ObjectSpec, SceneSpec


def get_scene(name: str) -> SceneSpec:
    """Look up a copy of a named scene from the library.

    Args:
        name: Scene name (e.g. ``"flat_floor"``, ``"cube_reach"``).

    Returns:
        A deep copy of the corresponding :class:`SceneSpec`.

    Raises:
        KeyError: If the scene name is not registered.
    """
    if name not in SCENES:
        raise KeyError(f"Unknown scene: {name!r}. Available: {list(SCENES)}")
    return copy.deepcopy(SCENES[name])


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
        if obj.shape == "box":
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [obj.size, obj.size, obj.size]
        elif obj.shape == "capsule":
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.size = [obj.size, obj.size, 0.0]
        elif obj.shape == "cylinder":
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [obj.size, obj.size, 0.0]
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = [obj.size, 0.0, 0.0]
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
    instances = _iter_object_instances(spec)
    n_total = len(instances)
    for idx, (body_name, obj) in enumerate(instances):
        pos = _object_spawn_pos(obj, idx, n_total)
        cfg[body_name] = EntityCfg(
            spec_fn=_make_object_spec_fn(body_name, obj, pos),
            init_state=EntityCfg.InitialStateCfg(pos=pos),
        )
    return cfg
