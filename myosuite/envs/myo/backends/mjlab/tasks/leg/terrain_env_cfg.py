# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Leg terrain walking (mjlab twin of CPU ``LegTerrainEnvV0``).

Observations, rewards and actions are those of the ``myoLegWalk-v0`` twin; the
task adds an height-field terrain and the CPU env's knee termination signal. The
CPU env rewrites the terrain at every reset. MuJoCo Warp shares one height field
between all envs, so the twin bakes one terrain (fixed seed for ``rough``) into the
model instead; the ``hilly`` and ``stairs`` CPU registrations use their fixed variant
and are reproduced exactly.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref
from myosuite.envs.myo.backends.mjlab.tasks.mdp import reset_to_cpu_state

ENTITY = "walk_robot"  # entity key the walk observation/reward terms look up
_TERRAIN_SEED = 0  # the CPU env draws a new rough terrain at every reset
_N = 10_000  # height-field samples (100 x 100)


def terrain_heights(
    terrain: str, variant: str | None, rng: np.random.Generator
) -> np.ndarray:
    """Height-field elevations (metres, flat ``(10000,)``) as ``LegTerrainEnvV0.reset``.

    Args:
        terrain: ``"rough"``, ``"hilly"`` or ``"stairs"``.
        variant: ``"fixed"`` for the fixed hilly/stairs parameters, else sampled.
        rng: Random generator for the rough terrain and non-fixed variants.

    Returns:
        The elevations the CPU env writes into ``model.hfield_data``.
    """
    if terrain == "rough":
        rough = rng.uniform(low=-0.5, high=0.5, size=(_N,))
        normalized = (rough - rough.min()) / (rough.max() - rough.min())
        return normalized * 0.08 - 0.02
    if terrain == "hilly":
        flat_length, frequency = 3000, 3
        scalar = 0.63 if variant == "fixed" else float(rng.uniform(0.53, 0.73))
        combined = np.concatenate(
            (
                -2 * np.ones(flat_length),
                -2
                + 0.5
                * (
                    np.sin(
                        np.linspace(0, frequency * np.pi, _N - flat_length) + np.pi / 2
                    )
                    - 1
                ),
            )
        )
        normalized = (combined - combined.min()) / (combined.max() - combined.min())
        return np.flip(normalized.reshape(100, 100) * scalar, [0, 1]).reshape(_N)
    if terrain == "stairs":
        num_stairs, stair_height = 12, 0.1
        flat = 5200 - (_N - 5200) % num_stairs
        stairs_width = (_N - flat) // num_stairs
        scalar = 2.5 if variant == "fixed" else float(rng.uniform(1.5, 3.5))
        parts = [
            np.full((int(stairs_width // 100), 100), -2 + stair_height * j)
            for j in range(num_stairs)
        ]
        data = np.concatenate([np.full((int(flat // 100), 100), -2)] + parts, axis=0)
        normalized = (data + 2) / (2 + stair_height * num_stairs)
        return np.flip(normalized.reshape(100, 100) * scalar, [0, 1]).reshape(_N)
    raise ValueError(f"Unknown terrain {terrain!r}")


def add_terrain(spec: mujoco.MjSpec, terrain: str, variant: str | None) -> None:
    """Bake the terrain into the entity spec, visible and colliding as after a CPU reset.

    MuJoCo normalises height-field data to ``[0, 1]`` at compile time, so the elevation
    range goes into the field's z size and its minimum into the geom's z position.

    Args:
        spec: Entity spec of the leg model (with its ``terrain`` height field).
        terrain: ``"rough"``, ``"hilly"`` or ``"stairs"``.
        variant: CPU ``variant`` of the registration.
    """
    heights = terrain_heights(terrain, variant, np.random.default_rng(_TERRAIN_SEED))
    field = spec.hfield("terrain")
    field.userdata = heights
    rx, ry, _, base = field.size
    field.size = [rx, ry, float(heights.max() - heights.min()), base]
    geom = spec.geom("terrain")
    geom.pos = np.array([0.0, 0.0, float(heights.min())])
    geom.rgba[3] = 1.0
    geom.contype = geom.conaffinity = 1


def make_leg_terrain_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``LegTerrainEnvV0`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no noise or DR).

    Returns:
        The env config.
    """
    del play
    # Imported here: the legacy walk module imports this package's helpers.
    from myosuite.envs.myo.backends.mjlab import register_mjlab_tasks as walk

    task = ref.cpu_task_spec(env_id)
    kw = task.kwargs
    info = ref.compiled_info(task)
    cfg = walk._make_walk_env_cfg()
    entity = cfg.scene.entities[walk._WALK_ENTITY_NAME]
    edit = functools.partial(add_terrain, terrain=kw["terrain"], variant=kw["variant"])
    entity.spec_fn = functools.partial(ref._entity_spec, task, (edit,))
    # CPU reset ("init"): keyframe 2, positions and velocities.
    entity.init_state = ref.init_state_from_model(info, info.key_qpos[2])
    cfg.events = {
        "reset_scene_to_default": EventTermCfg(
            func=reset_to_cpu_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(walk._WALK_ENTITY_NAME),
                "qpos": info.key_qpos[2],
                "qvel": info.key_qvel[2],
            },
        ),
    }
    # The CPU ``done`` of the terrain env also fires on the knee condition.
    cfg.rewards["done"] = RewardTermCfg(func=walk._terrain_done_signal, weight=-100.0)
    cfg.metrics = {
        "success": MetricsTermCfg(
            func=walk._walk_solved,
            params={
                "target_y_vel": float(kw["target_y_vel"]),
                "target_x_vel": float(kw["target_x_vel"]),
                "terrain": True,
            },
            reduce="last",
        )
    }
    return cfg
