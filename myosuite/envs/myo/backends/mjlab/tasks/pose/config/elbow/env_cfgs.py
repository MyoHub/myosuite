"""Elbow pose environment configurations (``myoElbowPose1D6M*``, incl. exoskeleton)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.myo.backends.mjlab.tasks.pose.pose_env_cfg import make_pose_env_cfg

ELBOW_POSE_IDS = (
    "myoElbowPose1D6MFixed-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoElbowPose1D6MExoFixed-v0",
    "myoElbowPose1D6MExoRandom-v0",
)


def elbow_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Elbow pose task matching the CPU registration of *env_id*."""
    return make_pose_env_cfg(env_id, play=play)
