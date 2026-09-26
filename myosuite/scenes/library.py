# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene library: named SceneSpec entries used by TaskConfig.scene.

SceneSpec is backend-aware:
- CPU / MJX  → MJCF XML appended before MjSpec.compile()
- mjlab      → RigidObjectCfg / TerrainCfg entries in SceneCfg
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FloorSpec:
    """Floor geometry specification.

    Args:
        kind: Floor type — "flat" or "uneven".
        size: Half-size of the floor plane in metres.
    """

    kind: str = "flat"
    size: float = 10.0


@dataclass
class ObjectSpec:
    """Interactable rigid-body object specification.

    Args:
        name: Object identifier used in MJCF and mjlab.
        shape: Geometry type ("box", "sphere", "capsule", "cylinder").
        size: Primary size parameter in metres (radius or half-extent).
        count: Number of identical objects to place in the scene.
        mass: Object mass in kg.
        rgba: RGBA colour tuple.
    """

    name: str
    shape: str = "sphere"
    size: float = 0.03
    count: int = 1
    mass: float = 0.1
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)


@dataclass
class SceneSpec:
    """Full scene specification combining floor and objects.

    Args:
        floor: Floor geometry specification.
        objects: List of interactable objects in the scene.
        gravity: Gravity vector (m/s²).
    """

    floor: FloorSpec = field(default_factory=FloorSpec)
    objects: list[ObjectSpec] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


# ---------------------------------------------------------------------------
# Named scene library
# ---------------------------------------------------------------------------

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
}
