"""Finger reach (``myoFingerReach*`` / ``motorFingerReach*``) environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.backends.mjlab.tasks.reach.reach_env_cfg import (
    make_reach_env_cfg,
)

FINGER_REACH_IDS = (
    "myoFingerReachFixed-v0",
    "myoFingerReachRandom-v0",
    "motorFingerReachFixed-v0",
    "motorFingerReachRandom-v0",
)


def finger_reach_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Finger reach (``myoFingerReach*`` / ``motorFingerReach*``) task matching the CPU registration of *env_id*."""
    return make_reach_env_cfg(env_id, play=play)


from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
