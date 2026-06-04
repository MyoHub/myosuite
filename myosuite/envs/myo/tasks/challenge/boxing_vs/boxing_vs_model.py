# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model builder for the two-agent competitive boxing MuJoCo scene.

Combines two MuscleMimic full-body models placed facing each other on a flat
floor and adds all boxing-specific sites and velocity sensors.

The resulting :class:`BoxingVsModelMeta` stores every index needed by the env
and obs/reward term functions so there is no name-based lookup at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mujoco

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.combat_model_meta import CombatModelMetaBase
from myosuite.envs.myo.tasks.challenge.boxing_visuals import (
    BOXING_RING_MESH,
    RING_POS,
    RING_SCALE,
    add_helmet,
    replace_hand_visuals_with_gloves,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.two_agent_scene import (
    build_combined_spec,
    extract_prefix_indices,
    extract_sensor_addrs,
    extract_site_ids,
    mj_name2id_strict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent constants
# ---------------------------------------------------------------------------

AGENTS: tuple[str, str] = ("agent_0", "agent_1")

# MuJoCo prefix used for each agent's named elements after attachment.
_AGENT_PREFIXES: dict[str, str] = {"agent_0": "a0_", "agent_1": "a1_"}

# Per-agent visual colours: agent 0 = red, agent 1 = blue.
_AGENT_COLORS: dict[int, dict[str, list[float]]] = {
    0: {"glove": [0.84, 0.12, 0.12, 1.0], "helmet": [0.80, 0.10, 0.10, 1.0]},
    1: {"glove": [0.12, 0.12, 0.84, 1.0], "helmet": [0.10, 0.10, 0.80, 1.0]},
}

# Head and body zone heights in the *agent's local frame* (metres).
# These match the P0 boxing task values so pre-trained P0 policies transfer.
_HEAD_ZONE_Z: float = 1.25
_BODY_ZONE_Z: float = 0.95

# Arm-level joint names used for the "opponent joint subset" observation.
# These are joint names as they appear *before* prefix attachment.
_ARM_JOINT_TOKENS: tuple[str, ...] = (
    "shoulder_elv_l",
    "shoulder_elv_r",
    "elv_angle_l",
    "elv_angle_r",
    "shoulder_rot_l",
    "shoulder_rot_r",
    "elbow_flex_l",
    "elbow_flex_r",
    "pro_sup_l",
    "pro_sup_r",
    "deviation_l",
    "deviation_r",
    "flexion_l",
    "flexion_r",
)


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class BoxingVsModelMeta(CombatModelMetaBase):
    """Pre-computed indices into MjModel/MjData arrays for both agents.

    All index/address fields are lists of integers so that slicing a
    concatenated vector (e.g. ``qpos``) is a one-liner:
    ``np.concatenate([data.qpos[adr:adr+w] for adr, w in zip(adr_list, width_list)])``.

    Attributes:
        n_act: Total number of actuators in the combined model.
        act_indices: Per-agent actuator indices into ``data.ctrl`` / ``data.act``.
        jnt_ids: Per-agent joint ids (used to compute qpos/qvel slices).
        arm_jnt_ids: Per-agent subset of arm joints for the opponent obs.
        pelvis_body_id: Per-agent pelvis body id (for COM / fall detection).
        site_ids: Per-agent dict of {name_suffix → site_id}; suffixes are
            ``"fist_r"``, ``"fist_l"``, ``"head_zone"``, ``"body_zone"``,
            ``"pelvis_site"``.
        sensor_adr: Per-agent dict of {name_suffix → first sensordata index};
            suffixes are ``"fist_r_vel"``, ``"fist_l_vel"``, ``"pelvis_site_vel"``.
    """

    pelvis_body_id: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    """Return the MuJoCo ID for *name*, raising if not found."""
    return mj_name2id_strict(model, obj_type, name)


def _add_agent_sites(spec: mujoco.MjSpec, prefix: str) -> None:
    """Add fist, scoring-zone, and pelvis tracking sites for one agent.

    Args:
        spec: The combined MjSpec (post-attachment).
        prefix: Name prefix of the agent (``"a0_"`` or ``"a1_"``).
    """
    # Right fist — proximal phalange of middle finger
    spec.body(f"{prefix}3proxph_r").add_site(
        name=f"{prefix}fist_r",
        pos=[0.0, 0.0, 0.0],
        size=[0.015, 0.0, 0.0],
        rgba=[1.0, 0.2, 0.2, 0.4],
    )
    # Left fist
    spec.body(f"{prefix}3proxph_l").add_site(
        name=f"{prefix}fist_l",
        pos=[0.0, 0.0, 0.0],
        size=[0.015, 0.0, 0.0],
        rgba=[0.2, 0.2, 1.0, 0.4],
    )
    # Head scoring zone
    spec.body(f"{prefix}head").add_site(
        name=f"{prefix}head_zone",
        pos=[0.0, 0.0, 0.0],
        size=[0.02, 0.0, 0.0],
        rgba=[1.0, 1.0, 0.0, 0.4],
    )
    # Body scoring zone (lumbar / solar-plexus)
    spec.body(f"{prefix}lumbar1").add_site(
        name=f"{prefix}body_zone",
        pos=[0.0, 0.0, 0.0],
        size=[0.02, 0.0, 0.0],
        rgba=[0.0, 1.0, 0.5, 0.4],
    )
    # Pelvis reference site (for COM / velocity tracking)
    spec.body(f"{prefix}pelvis").add_site(
        name=f"{prefix}pelvis_site",
        pos=[0.0, 0.0, 0.0],
        size=[0.01, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )


def _add_agent_sensors(spec: mujoco.MjSpec, prefix: str) -> None:
    """Add framelinvel sensors for fist and pelvis sites of one agent.

    Args:
        spec: The combined MjSpec.
        prefix: Name prefix of the agent.
    """
    for suffix in ("fist_r", "fist_l", "pelvis_site"):
        sensor_name = f"{prefix}{suffix}_vel"
        spec.add_sensor(
            name=sensor_name,
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{prefix}{suffix}",
        )


def _extract_meta(
    model: mujoco.MjModel,
    config: BoxingVsConfig,
) -> BoxingVsModelMeta:
    """Extract all runtime indices from the compiled model.

    Args:
        model: Compiled MjModel for the boxing-vs scene.
        config: Task configuration (unused currently but kept for future use).

    Returns:
        Fully populated :class:`BoxingVsModelMeta`.
    """
    meta = BoxingVsModelMeta(n_act=model.nu)

    for agent_id, prefix in _AGENT_PREFIXES.items():
        act_idx, jnt_ids, arm_jnt_ids = extract_prefix_indices(
            model, prefix, arm_joint_tokens=_ARM_JOINT_TOKENS
        )
        meta.act_indices[agent_id] = act_idx
        meta.jnt_ids[agent_id] = jnt_ids
        meta.arm_jnt_ids[agent_id] = arm_jnt_ids
        meta.pelvis_body_id[agent_id] = mj_name2id_strict(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis"
        )
        meta.site_ids[agent_id] = extract_site_ids(
            model, prefix, ("fist_r", "fist_l", "head_zone", "body_zone", "pelvis_site")
        )
        meta.sensor_adr[agent_id] = extract_sensor_addrs(
            model, prefix, ("fist_r", "fist_l", "pelvis_site")
        )

    return meta


def _replace_floor_visual_with_boxing_ring(spec: mujoco.MjSpec) -> None:
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


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def _build_boxing_vs_spec(config: BoxingVsConfig) -> mujoco.MjSpec:
    """Build and return the combined two-agent boxing MjSpec."""
    fullbody_cfg = default_mimic_fullbody_config()
    fullbody_cfg.disable_fingers = config.disable_fingers
    fullbody_cfg.sim_dt = config.sim_dt

    agent0_spec, _ = build_mimic_fullbody_spec(fullbody_cfg)
    agent1_spec, _ = build_mimic_fullbody_spec(fullbody_cfg)
    for idx, agent_spec in enumerate((agent0_spec, agent1_spec)):
        colors = _AGENT_COLORS[idx]
        replace_hand_visuals_with_gloves(agent_spec, rgba=colors["glove"])
        add_helmet(agent_spec, rgba=colors["helmet"])

    spec = build_combined_spec(
        agent0_spec,
        agent1_spec,
        separation_m=config.agent_separation_m,
        sim_dt=config.sim_dt,
        model_name="boxing_vs",
        floor_rgba=[0.75, 0.75, 0.75, 1.0],
        floor_collision=True,
    )
    _replace_floor_visual_with_boxing_ring(spec)

    for prefix in ("a0_", "a1_"):
        _add_agent_sites(spec, prefix)
        _add_agent_sensors(spec, prefix)

    return spec


def build_boxing_vs_model(
    config: BoxingVsConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData, BoxingVsModelMeta]:
    """Build the combined two-agent boxing MuJoCo model.

    Places two MuscleMimic full-body agents facing each other on a flat floor.
    Both agents are stripped of scene includes, have fingers optionally removed,
    and have boxing-specific fist/scoring sites and velocity sensors added.

    Placement convention:
        Agent 0 is at positive Y, facing −Y (default model orientation).
        Agent 1 is at negative Y, rotated 180° around Z so it also faces its
        opponent.  Separation is ``config.agent_separation_m``.

    Args:
        config: Task configuration (timing, placement, model toggles).

    Returns:
        Tuple of (MjModel, MjData, BoxingVsModelMeta).

    Raises:
        ImportError: If ``musclemimic_models`` is not installed.
        RuntimeError: If a required body/site/sensor is missing after compilation.
    """
    spec = _build_boxing_vs_spec(config)

    # --- Compile ---
    model = spec.compile()
    model.opt.timestep = config.sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    logger.info(
        "Boxing-vs model compiled: nq=%d nv=%d nu=%d ngeom=%d",
        model.nq,
        model.nv,
        model.nu,
        model.ngeom,
    )

    meta = _extract_meta(model, config)
    logger.info(
        "Agent 0: %d actuators, %d joints | Agent 1: %d actuators, %d joints",
        len(meta.act_indices["agent_0"]),
        len(meta.jnt_ids["agent_0"]),
        len(meta.act_indices["agent_1"]),
        len(meta.jnt_ids["agent_1"]),
    )

    return model, data, meta
