"""Hand reach (``myoHandReach*``) environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.backends.mjlab.tasks.reach.reach_env_cfg import (
    make_reach_env_cfg,
)

HAND_REACH_IDS = ("myoHandReachFixed-v0", "myoHandReachRandom-v0")


def hand_reach_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Hand reach (``myoHandReach*``) task matching the CPU registration of *env_id*."""
    return make_reach_env_cfg(env_id, play=play)


from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
