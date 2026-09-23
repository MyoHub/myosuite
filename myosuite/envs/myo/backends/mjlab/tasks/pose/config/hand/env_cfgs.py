"""Hand pose environment configurations (``myoHandPose*``, incl. ASL numerals)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.myo.backends.mjlab.tasks.pose.pose_env_cfg import make_pose_env_cfg

HAND_POSE_IDS = (
    "myoHandPoseFixed-v0",
    "myoHandPoseRandom-v0",
    *(f"myoHandPose{k}Fixed-v0" for k in range(10)),
)


def hand_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Hand pose task matching the CPU registration of *env_id*."""
    return make_pose_env_cfg(env_id, play=play)
