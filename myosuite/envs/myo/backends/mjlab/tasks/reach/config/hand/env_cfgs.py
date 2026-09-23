"""Hand reach (``myoHandReach*``) environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.myo.backends.mjlab.tasks.reach.reach_env_cfg import make_reach_env_cfg

HAND_REACH_IDS = ("myoHandReachFixed-v0", "myoHandReachRandom-v0")


def hand_reach_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Hand reach (``myoHandReach*``) task matching the CPU registration of *env_id*."""
    return make_reach_env_cfg(env_id, play=play)
