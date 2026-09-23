from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import TORSO_POSE_IDS, torso_pose_env_cfg
from .rl_cfg import torso_pose_ppo_runner_cfg

register_cpu_twins(TORSO_POSE_IDS, torso_pose_env_cfg, torso_pose_ppo_runner_cfg)
