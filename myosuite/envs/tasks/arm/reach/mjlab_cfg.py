"""Arm reach (``myoArmReach*``) environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.backends.mjlab.tasks.reach.reach_env_cfg import (
    make_reach_env_cfg,
)

ARM_REACH_IDS = ("myoArmReachFixed-v0", "myoArmReachRandom-v0")


def arm_reach_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Arm reach (``myoArmReach*``) task matching the CPU registration of *env_id*."""
    return make_reach_env_cfg(env_id, play=play)


from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
