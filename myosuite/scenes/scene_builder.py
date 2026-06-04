# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
SceneSpec → backend-specific scene representation.

CPU / MJX path: SceneSpec → MJCF XML string appended before MjSpec.compile()
mjlab path:     SceneSpec → dict of RigidObjectCfg / TerrainCfg entries
                            (populated into SceneCfg by the caller)
"""

from __future__ import annotations

from typing import Any

from myosuite.scenes.library import SCENES, SceneSpec


def get_scene(name: str) -> SceneSpec:
    """Look up a named scene from the library.

    Args:
        name: Scene name (e.g. "flat_floor", "cube_reach").

    Returns:
        The corresponding SceneSpec.

    Raises:
        KeyError: If the scene name is not registered.
    """
    if name not in SCENES:
        raise KeyError(f"Unknown scene: {name!r}. Available: {list(SCENES)}")
    return SCENES[name]


def scene_to_mjcf_xml(spec: SceneSpec) -> str:
    """Render a SceneSpec to MJCF XML for use with CPU / MJX paths.

    The returned XML fragment contains a ``<worldbody>`` element with
    a floor geom and any object bodies. It should be inserted into the
    model's ``<worldbody>`` before calling ``MjSpec.compile()``.

    Args:
        spec: Scene specification to render.

    Returns:
        MJCF XML string fragment.
    """
    lines = ["<worldbody>"]

    # Floor
    if spec.floor.kind == "flat":
        lines.append(
            f'  <geom name="floor" type="plane" size="{spec.floor.size} {spec.floor.size} 0.1"'
            ' rgba="0.9 0.9 0.9 1" condim="3"/>'
        )
    elif spec.floor.kind == "uneven":
        lines.append(
            '  <geom name="floor" type="hfield" hfield="terrain"'
            ' rgba="0.7 0.7 0.5 1" condim="3"/>'
        )

    # Objects
    for obj in spec.objects:
        if obj.shape == "box":
            geom_attrs = f'type="box" size="{obj.size} {obj.size} {obj.size}"'
        elif obj.shape == "sphere":
            geom_attrs = f'type="sphere" size="{obj.size}"'
        elif obj.shape in ("capsule", "cylinder"):
            geom_attrs = f'type="{obj.shape}" size="{obj.size} {obj.size}"'
        else:
            geom_attrs = f'type="sphere" size="{obj.size}"'

        rgba_str = " ".join(str(c) for c in obj.rgba)
        lines.append(
            f'  <body name="{obj.name}" pos="0 0 {obj.size}">'
            f"\n    <freejoint/>"
            f'\n    <geom {geom_attrs} rgba="{rgba_str}" mass="{obj.mass}"/>'
            f"\n  </body>"
        )

    lines.append("</worldbody>")
    return "\n".join(lines)


def scene_to_mjlab_cfg(spec: SceneSpec) -> dict[str, Any]:
    """Render a SceneSpec to mjlab scene config entries.

    Returns a dict mapping entity names to mjlab config objects.
    The caller is responsible for merging these into their ``SceneCfg``.

    Args:
        spec: Scene specification to render.

    Returns:
        Dict of entity_name → mjlab config object (RigidObjectCfg, etc.).
        Returns an empty dict if mjlab is not installed.
    """
    try:
        from mjlab.scene import RigidObjectCfg
    except ImportError:
        return {}

    cfg: dict[str, Any] = {}

    # Objects
    for obj in spec.objects:
        cfg[obj.name] = RigidObjectCfg(
            shape=obj.shape,
            size=obj.size,
            mass=obj.mass,
        )

    return cfg
