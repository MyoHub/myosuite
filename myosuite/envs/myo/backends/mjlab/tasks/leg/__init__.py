"""Leg tasks (stand-and-reach, terrain walking); importing registers the twins."""

import functools

from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .rl_cfg import leg_stand_ppo_runner_cfg, leg_terrain_ppo_runner_cfg
from .stand_env_cfg import make_leg_stand_env_cfg
from .terrain_env_cfg import make_leg_terrain_env_cfg

LEG_STAND_IDS = ("myoLegStandRandom-v0",)
# CPU terrain env id -> experiment (log directory) name.
LEG_TERRAIN_EXPERIMENTS = {
    "myoLegRoughTerrainWalk-v0": "myo_leg_rough",
    "myoLegHillyTerrainWalk-v0": "myo_leg_hilly",
    "myoLegStairTerrainWalk-v0": "myo_leg_stairs",
}

register_cpu_twins(LEG_STAND_IDS, make_leg_stand_env_cfg, leg_stand_ppo_runner_cfg)
for _env_id, _experiment in LEG_TERRAIN_EXPERIMENTS.items():
    register_cpu_twins(
        (_env_id,),
        make_leg_terrain_env_cfg,
        functools.partial(leg_terrain_ppo_runner_cfg, _experiment),
    )
