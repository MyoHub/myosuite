# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared utilities for building two-agent MuJoCo competitive scenes.

Both ``boxing_vs_model`` and ``saber_vs_model`` use frame attachment to compose
two independent agent specs into a single combined scene.  This module extracts
the common scaffolding so each model file only needs to implement its own
agent-specific geometry (fists/sabers, scoring zones, sensors).
"""

from __future__ import annotations

import logging

import mujoco

logger = logging.getLogger(__name__)

# 180° rotation around Z: places agent 1 facing agent 0 (MJCF: w first).
_QUAT_180Z: list[float] = [0.0, 0.0, 0.0, 1.0]


def build_combined_spec(
    agent0_spec: mujoco.MjSpec,
    agent1_spec: mujoco.MjSpec,
    *,
    separation_m: float,
    sim_dt: float,
    model_name: str = "combined",
    prefixes: tuple[str, str] = ("a0_", "a1_"),
    floor_rgba: list[float] | None = None,
    floor_collision: bool = True,
) -> mujoco.MjSpec:
    """Attach two agent specs into a combined scene and return the combined MjSpec.

    Agent 0 is placed at ``[0, +separation_m/2, 0]`` in its default orientation.
    Agent 1 is placed at ``[0, -separation_m/2, 0]`` and rotated 180° around Z
    so both agents face each other along the Y axis.

    A floor plane and a fill light are added to the combined spec.  Each agent
    spec is attached under a named frame so MuJoCo scopes all names automatically
    via the supplied prefixes.

    Args:
        agent0_spec: MjSpec for agent 0 (not yet attached to any parent).
        agent1_spec: MjSpec for agent 1 (not yet attached to any parent).
        separation_m: Frame-to-frame Y-axis distance (metres).
        sim_dt: Simulation timestep forwarded to ``spec.option.timestep``.
        model_name: Value assigned to ``spec.modelname``.
        prefixes: Name prefixes for agent 0 and agent 1 respectively.
        floor_rgba: RGBA colour of the floor plane.  Defaults to light grey.
        floor_collision: Whether the floor participates in collisions.

    Returns:
        Combined ``mujoco.MjSpec`` with both agents attached.
    """
    if floor_rgba is None:
        floor_rgba = [0.75, 0.75, 0.75, 1.0]

    spec = mujoco.MjSpec()
    spec.modelname = model_name
    spec.option.timestep = sim_dt

    floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [8.0, 8.0, 0.1]
    floor.rgba = floor_rgba
    floor.contype = 1 if floor_collision else 0
    floor.conaffinity = 1 if floor_collision else 0

    fill = spec.worldbody.add_light(name="scene_fill")
    fill.pos = [0.0, 0.0, 4.0]
    fill.diffuse = [0.5, 0.5, 0.55]
    fill.castshadow = False

    frame0 = spec.worldbody.add_frame()
    frame0.pos = [0.0, separation_m / 2.0, 0.0]
    frame0.quat = [1.0, 0.0, 0.0, 0.0]
    spec.attach(agent0_spec, prefix=prefixes[0], suffix="", frame=frame0)

    frame1 = spec.worldbody.add_frame()
    frame1.pos = [0.0, -separation_m / 2.0, 0.0]
    frame1.quat = _QUAT_180Z
    spec.attach(agent1_spec, prefix=prefixes[1], suffix="", frame=frame1)

    return spec


def mj_name2id_strict(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    """Return the MuJoCo ID for *name*, raising ``RuntimeError`` if not found.

    Args:
        model: Compiled MjModel.
        obj_type: MuJoCo object type constant (e.g. ``mujoco.mjtObj.mjOBJ_SITE``).
        name: Fully-qualified object name (including agent prefix).

    Returns:
        Non-negative integer ID.

    Raises:
        RuntimeError: If the object is not found in the model.
    """
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx == -1:
        raise RuntimeError(f"MuJoCo object not found: type={obj_type} name={name!r}")
    return idx


def extract_prefix_indices(
    model: mujoco.MjModel,
    prefix: str,
    *,
    arm_joint_tokens: tuple[str, ...] = (),
) -> tuple[list[int], list[int], list[int]]:
    """Extract actuator indices, joint ids, and arm-joint subset for one agent prefix.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        arm_joint_tokens: Short joint names (without prefix) that constitute the
            arm-only subset used for opponent observations.  If empty, the arm
            subset is also empty.

    Returns:
        Tuple of ``(act_indices, jnt_ids, arm_jnt_ids)``:
        - ``act_indices``: All actuator indices whose name starts with *prefix*.
        - ``jnt_ids``: All joint ids whose name starts with *prefix*.
        - ``arm_jnt_ids``: Subset of *jnt_ids* whose full name is in
          ``{prefix + tok for tok in arm_joint_tokens}``.
    """
    act_indices = [
        i
        for i in range(model.nu)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(
            prefix
        )
    ]

    jnt_ids = [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
            prefix
        )
    ]

    arm_names = {f"{prefix}{tok}" for tok in arm_joint_tokens}
    arm_jnt_ids = [
        j
        for j in jnt_ids
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "") in arm_names
    ]

    return act_indices, jnt_ids, arm_jnt_ids


def extract_site_ids(
    model: mujoco.MjModel,
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, int]:
    """Return ``{suffix: site_id}`` for each suffix in *suffixes*.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        suffixes: Short site name suffixes (without prefix) to look up.

    Returns:
        Dict mapping each suffix to its site id.

    Raises:
        RuntimeError: If any site is not found.
    """
    return {
        suffix: mj_name2id_strict(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{suffix}")
        for suffix in suffixes
    }


def extract_sensor_addrs(
    model: mujoco.MjModel,
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, int]:
    """Return ``{suffix_vel: sensor_adr}`` for each suffix, looking up ``{prefix}{suffix}_vel``.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        suffixes: Short site/sensor name suffixes (without prefix or ``_vel``).
            The sensor name looked up is ``{prefix}{suffix}_vel``.

    Returns:
        Dict mapping ``"{suffix}_vel"`` to the first sensordata index.

    Raises:
        RuntimeError: If any sensor is not found.
    """
    result: dict[str, int] = {}
    for suffix in suffixes:
        sensor_name = f"{prefix}{suffix}_vel"
        sid = mj_name2id_strict(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        result[f"{suffix}_vel"] = int(model.sensor_adr[sid])
    return result
