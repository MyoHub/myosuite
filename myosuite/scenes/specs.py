# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Generic scene data structures shared across presets and emitters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FloorSpec:
    """Floor geometry specification.

    Args:
        kind: Floor type — ``"flat"``, ``"uneven"``, or ``"none"``.
        size: Half-size of the floor plane in metres.
    """

    kind: str = "flat"
    size: float = 10.0


@dataclass
class ObjectSpec:
    """Interactable free-floating rigid-body object specification.

    Args:
        name: Object identifier used in MJCF and mjlab.
        shape: Geometry type (``"box"``, ``"sphere"``, ``"capsule"``, ``"cylinder"``).
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


@dataclass(frozen=True)
class SiteMarkerSpec:
    """Named site attached to a fixture body (e.g. target or reference anchor).

    Args:
        name: Unique site identifier in MJCF.
        pos: Local position ``(x, y, z)`` [m] in the parent fixture frame.
        size: Site geom half-extents or radius ``(s0, s1, s2)`` [m].
        shape: Site visualization type (``"sphere"``, ``"box"``, ``"cylinder"``, ``"ellipsoid"``).
        rgba: RGBA colour tuple.
        group: MuJoCo visualization group index.
    """

    name: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.015, 0.015, 0.015)
    shape: str = "sphere"
    rgba: tuple[float, float, float, float] = (0.1, 0.8, 0.3, 0.85)
    group: int = 0


@dataclass
class FixtureSpec:
    """Static or mocap scene fixture (surface, seat, panel, or spatial boundary).

    Args:
        name: Body and primary geom name in MJCF / mjlab.
        shape: Geometry type (``"box"``, ``"sphere"``, ``"capsule"``, ``"cylinder"``, ``"ellipsoid"``).
        size: 3-tuple MuJoCo geom size parameters in metres.
        pos: World position ``(x, y, z)`` [m].
        quat: World orientation ``(w, x, y, z)`` unit quaternion.
        mocap: Create a mocap body controllable via ``data.mocap_pos`` / ``data.mocap_quat``.
        collide: Enable physical contacts (``False`` creates a visual-only boundary/zone).
        rgba: RGBA colour tuple.
        sites: Optional sites rigidly attached in the fixture's local frame.
    """

    name: str
    shape: str = "box"
    size: tuple[float, float, float] = (0.30, 0.30, 0.02)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mocap: bool = False
    collide: bool = True
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.78, 1.0)
    sites: tuple[SiteMarkerSpec, ...] = ()


@dataclass
class SceneSpec:
    """Full scene specification combining floor, free objects, gravity, and fixtures.

    Args:
        floor: Floor geometry specification.
        objects: List of free-floating interactable objects.
        gravity: Gravity vector ``(gx, gy, gz)`` in m/s^2.
        fixtures: List of static or mocap workspace fixtures and boundaries.
    """

    floor: FloorSpec = field(default_factory=FloorSpec)
    objects: list[ObjectSpec] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    fixtures: list[FixtureSpec] = field(default_factory=list)
