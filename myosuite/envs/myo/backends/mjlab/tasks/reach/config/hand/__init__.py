from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import HAND_REACH_IDS, hand_reach_env_cfg
from .rl_cfg import hand_reach_ppo_runner_cfg

register_cpu_twins(HAND_REACH_IDS, hand_reach_env_cfg, hand_reach_ppo_runner_cfg)
