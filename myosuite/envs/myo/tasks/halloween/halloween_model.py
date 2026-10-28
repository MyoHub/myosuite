# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoFullBody + free-joint pumpkin scene for MyoHalloween.

Visual mesh is Kenney Food Kit ``pumpkin`` (CC0). Collision uses a sphere so
contact is stable under MuJoCo muscle simulation. The pumpkin is a free body
appended to the mimic full-body spec (robot free joint addresses stay at 0).
"""

from __future__ import annotations

from pathlib import Path
import json
from functools import partial

import mujoco
import numpy as np
from ml_collections import config_dict

from myosuite.envs.myo.tasks.halloween.config import HALLOWEEN
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "halloween"
PUMPKIN_OBJ = ASSET_DIR / "pumpkin.obj"
PUMPKIN_BODY = "pumpkin"
PUMPKIN_JOINT = "pumpkin_free"
PUMPKIN_COLLISION_GEOM = "pumpkin_collision"
PUMPKIN_VISUAL_GEOM = "pumpkin_visual"
PUMPKIN_MESH = "pumpkin_mesh"

# Kenney Food Kit OBJ is Y-up, ~0.42 m wide. Scale + rotate into MuJoCo Z-up.
_MESH_SCALE = 0.55
_COLLISION_RADIUS = HALLOWEEN.pumpkin_radius
_PARK_POS = np.array(HALLOWEEN.park_pos, dtype=np.float64)


def add_pumpkin_contact_pairs(
    spec: mujoco.MjSpec,
    *,
    pumpkin_geom: str = PUMPKIN_COLLISION_GEOM,
    robot_prefix: str = "",
) -> None:
    """Enable projectile contacts with every collidable actor geom, including explicit-only pairs."""
    paired = {name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)}
    for body in spec.bodies:
        if (
            not body.name
            or body.name == PUMPKIN_BODY
            or (robot_prefix and not body.name.startswith(robot_prefix))
        ):
            continue
        for geom in body.geoms:
            if (
                geom.name != pumpkin_geom
                and geom.name
                and (geom.name in paired or geom.contype or geom.conaffinity)
            ):
                spec.add_pair(geomname1=pumpkin_geom, geomname2=geom.name)


def _add_pumpkin(spec: mujoco.MjSpec, *, terrain: bool = False) -> None:
    """Append a free-joint pumpkin with sphere collision and Kenney visual mesh."""
    if not terrain and not PUMPKIN_OBJ.is_file():
        raise FileNotFoundError(
            f"Missing Kenney CC0 pumpkin mesh at {PUMPKIN_OBJ}. "
            "Expected myosuite/envs/myo/assets/halloween/pumpkin.obj"
        )

    if not terrain:
        spec.add_mesh(
            name=PUMPKIN_MESH,
            file=str(PUMPKIN_OBJ.resolve()),
            scale=[_MESH_SCALE, _MESH_SCALE, _MESH_SCALE],
        )

    radius = 0.065 if terrain else _COLLISION_RADIUS
    position = [8, 0, 0.5] if terrain else list(_PARK_POS)
    body = spec.worldbody.add_body(
        name=PUMPKIN_BODY, pos=position, gravcomp=int(terrain)
    )
    body.add_freejoint(name=PUMPKIN_JOINT)

    collision = body.add_geom(
        name=PUMPKIN_COLLISION_GEOM,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0.0, 0.0],
        rgba=[1, 0.28, 0.015, 1] if terrain else [1.0, 0.45, 0.05, 1.0],
        mass=0.25 if terrain else HALLOWEEN.pumpkin_mass,
    )
    collision.contype = 1
    collision.conaffinity = 1
    if terrain:
        collision.friction = [0.7, 0.01, 0.001]
        for j in range(8):
            angle = j * np.pi / 4
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                pos=[0.28 * radius * np.cos(angle), 0.28 * radius * np.sin(angle), 0],
                size=[0.78 * radius, 0.78 * radius, 0.92 * radius],
                mass=0,
                contype=0,
                conaffinity=0,
                rgba=[0.9, 0.19, 0.008, 1],
            )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=[0, 0, radius + 0.005],
            size=[0.009, 0.018, 0],
            mass=0,
            contype=0,
            conaffinity=0,
            rgba=[0.18, 0.28, 0.05, 1],
        )
    else:
        collision.condim = 3
        collision.friction = [0.9, 0.1, 0.01]
        collision.solref = [0.02, 1.0]
        collision.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

        # OBJ Y-up → MuJoCo Z-up (rotate −90° about X) on the visual geom.
        visual = body.add_geom(
            name=PUMPKIN_VISUAL_GEOM,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=PUMPKIN_MESH,
            contype=0,
            conaffinity=0,
            group=1,
            rgba=[1.0, 0.45, 0.05, 1.0],
            quat=[0.70710678, 0.70710678, 0.0, 0.0],
        )
        visual.density = 0.0


def build_halloween_spec(
    config: config_dict.ConfigDict | None = None,
    *,
    profile: str = "benchmark",
) -> mujoco.MjSpec:
    """Build uncompiled MyoFullBody + pumpkin ``MjSpec``."""
    if profile not in ("benchmark", "terra"):
        raise ValueError("profile must be 'benchmark' or 'terra'")
    if profile == "terra":
        if config is not None:
            raise ValueError("The terra profile uses the calibrated TERRA model config")
        from myosuite.integrations.musclemimic import build_terrain_fullbody_spec

        spec = build_terrain_fullbody_spec(partial(_add_pumpkin, terrain=True))
        add_pumpkin_contact_pairs(spec)
        state = json.loads((ASSET_DIR / "terra_initial_state.json").read_text())
        spec.add_key(
            name="halloween_ready",
            qpos=state["qpos"] + [8, 0, 0.5, 1, 0, 0, 0],
            qvel=state["qvel"] + [0] * 6,
            act=state["act"],
        )
        return spec
    cfg = config if config is not None else default_mimic_fullbody_config()
    spec, _xml_path = build_mimic_fullbody_spec(cfg)
    spec.option.timestep = float(cfg.sim_dt)
    _add_pumpkin(spec)
    add_pumpkin_contact_pairs(spec)
    return spec


def build_default_halloween_spec() -> mujoco.MjSpec:
    """Zero-arg wrapper for picklable ``model_spec_fn`` / EntityCfg.spec_fn."""
    return build_halloween_spec()


def compile_halloween_model(
    config: config_dict.ConfigDict | None = None,
    *,
    profile: str = "benchmark",
) -> tuple[mujoco.MjModel, mujoco.MjSpec, str]:
    """Compile the Halloween scene.

    Returns:
        ``(model, spec, label)``.
    """
    spec = build_halloween_spec(config, profile=profile)
    return spec.compile(), spec, "myohalloween:in_memory"


def pumpkin_qposadr(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
    if jid < 0:
        raise RuntimeError(f"Joint {PUMPKIN_JOINT!r} missing from Halloween model")
    return int(model.jnt_qposadr[jid])


def pumpkin_dofadr(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
    if jid < 0:
        raise RuntimeError(f"Joint {PUMPKIN_JOINT!r} missing from Halloween model")
    return int(model.jnt_dofadr[jid])


def park_pumpkin(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Park the pumpkin far from the character with zero velocity."""
    adr = pumpkin_qposadr(model)
    dadr = pumpkin_dofadr(model)
    data.qpos[adr : adr + 3] = _PARK_POS
    data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr : dadr + 6] = 0.0


def throw_pumpkin(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: float,
    speed: float = 1.0,
    x_offset: float = 0.0,
    height: float = 0.4,
    flight: float = 0.65,
) -> None:
    """Launch the free pumpkin from the character's left (``side<0``) or right at its upper body.

    Lateral axis is world **Y** (same as MuscleMimic dodge goal injection). The pumpkin starts 1.6 m to the side and
    reaches the point ``x_offset`` ahead of and ``height`` above the pelvis after ``flight / speed`` seconds (the mjlab
    task draws ``x_offset`` in [-0.08, 0.08], ``height`` in [0.25, 0.55] and ``flight`` in [0.55, 0.75]).
    """
    adr = pumpkin_qposadr(model)
    dadr = pumpkin_dofadr(model)
    s = -1.0 if side < 0 else 1.0
    t = flight / float(np.clip(speed, 0.5, 2.0))
    target = np.array([data.qpos[0] + x_offset, data.qpos[1], data.qpos[2] + height])
    start = target + [0.0, s * 1.6, 0.0]
    data.qpos[adr : adr + 3] = start
    data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr : dadr + 3] = (target - start) / t + [0.0, 0.0, 9.81 * t / 2]
    data.qvel[dadr + 3 : dadr + 6] = [0.0, s * 2.5, 0.0]


def pumpkin_body_contact_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    steps: int = 500,
    side: float = -1.0,
    speed: float = 1.0,
) -> dict[str, int]:
    """Roll out a throw and count MuJoCo contacts involving the pumpkin."""
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    else:
        data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    throw_pumpkin(model, data, side=side, speed=speed)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PUMPKIN_BODY)
    hits = 0
    body_hits = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        touched = False
        for c in range(data.ncon):
            g1 = int(model.geom_bodyid[data.contact[c].geom1])
            g2 = int(model.geom_bodyid[data.contact[c].geom2])
            if bid not in (g1, g2):
                continue
            touched = True
            other = g2 if g1 == bid else g1
            if other > 0:
                body_hits += 1
        if touched:
            hits += 1
    return {
        "contact_frames": hits,
        "body_contact_frames": body_hits,
        "nq": int(model.nq),
        "nv": int(model.nv),
    }


__all__ = [
    "ASSET_DIR",
    "PUMPKIN_BODY",
    "PUMPKIN_JOINT",
    "build_default_halloween_spec",
    "build_halloween_spec",
    "compile_halloween_model",
    "park_pumpkin",
    "pumpkin_body_contact_frames",
    "throw_pumpkin",
]
