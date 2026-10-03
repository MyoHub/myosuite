from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import HAND_POSE_IDS, hand_pose_env_cfg
from .rl_cfg import hand_pose_ppo_runner_cfg

register_cpu_twins(HAND_POSE_IDS, hand_pose_env_cfg, hand_pose_ppo_runner_cfg)
