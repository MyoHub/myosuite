from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import MOTOR_FINGER_POSE_IDS, MYO_FINGER_POSE_IDS, finger_pose_env_cfg
from .rl_cfg import finger_pose_ppo_runner_cfg

register_cpu_twins(
    MYO_FINGER_POSE_IDS + MOTOR_FINGER_POSE_IDS,
    finger_pose_env_cfg,
    finger_pose_ppo_runner_cfg,
)
