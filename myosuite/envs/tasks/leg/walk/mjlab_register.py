"""Leg tasks (stand-and-reach, terrain walking); importing registers the twins."""

import functools

from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.leg.walk.rl_cfg import (
    leg_directional_ppo_runner_cfg,
    leg_stand_ppo_runner_cfg,
    leg_terrain_ppo_runner_cfg,
    leg_walk_ppo_runner_cfg,
)
from myosuite.envs.tasks.leg.directional.mjlab_cfg import make_leg_directional_env_cfg
from myosuite.envs.tasks.leg.stand.mjlab_cfg import make_leg_stand_env_cfg
from myosuite.envs.tasks.leg.walk.mjlab_cfg import make_leg_walk_env_cfg

LEG_STAND_IDS = ("myoLegStandRandom-v0",)
LEG_WALK_IDS = ("myoLegWalk-v0",)
# CPU directional env id -> experiment (log directory) name.
LEG_DIRECTIONAL_EXPERIMENTS = {
    "myoLegDirectionalForward-v0": "myo_leg_directional_fwd",
    "myoLegDirectionalBackward-v0": "myo_leg_directional_bwd",
    "myoLegDirectionalRandom-v0": "myo_leg_directional_rand",
}
# CPU terrain env id -> experiment (log directory) name.
LEG_TERRAIN_EXPERIMENTS = {
    "myoLegRoughTerrainWalk-v0": "myo_leg_rough",
    "myoLegHillyTerrainWalk-v0": "myo_leg_hilly",
    "myoLegStairTerrainWalk-v0": "myo_leg_stairs",
}

register_cpu_twins(LEG_STAND_IDS, make_leg_stand_env_cfg, leg_stand_ppo_runner_cfg)
register_cpu_twins(LEG_WALK_IDS, make_leg_walk_env_cfg, leg_walk_ppo_runner_cfg)
for _env_id, _experiment in LEG_TERRAIN_EXPERIMENTS.items():
    register_cpu_twins(
        (_env_id,),
        make_leg_walk_env_cfg,
        functools.partial(leg_terrain_ppo_runner_cfg, _experiment),
    )
for _env_id, _experiment in LEG_DIRECTIONAL_EXPERIMENTS.items():
    register_cpu_twins(
        (_env_id,),
        make_leg_directional_env_cfg,
        functools.partial(leg_directional_ppo_runner_cfg, _experiment),
    )

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
