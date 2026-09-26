from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import ELBOW_POSE_IDS, elbow_pose_env_cfg
from .rl_cfg import elbow_pose_ppo_runner_cfg

register_cpu_twins(ELBOW_POSE_IDS, elbow_pose_env_cfg, elbow_pose_ppo_runner_cfg)
