"""Finger pose environment configurations (``myoFingerPose*`` / ``motorFingerPose*``)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.myo.backends.mjlab.tasks.pose.pose_env_cfg import make_pose_env_cfg

MYO_FINGER_POSE_IDS = ("myoFingerPoseFixed-v0", "myoFingerPoseRandom-v0")
MOTOR_FINGER_POSE_IDS = ("motorFingerPoseFixed-v0", "motorFingerPoseRandom-v0")


def finger_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Finger pose task matching the CPU registration of *env_id*."""
    return make_pose_env_cfg(env_id, play=play)
