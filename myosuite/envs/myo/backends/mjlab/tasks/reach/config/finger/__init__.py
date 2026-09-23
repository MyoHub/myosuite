from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import FINGER_REACH_IDS, finger_reach_env_cfg
from .rl_cfg import finger_reach_ppo_runner_cfg

register_cpu_twins(FINGER_REACH_IDS, finger_reach_env_cfg, finger_reach_ppo_runner_cfg)
