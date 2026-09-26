from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfgs import ARM_REACH_IDS, arm_reach_env_cfg
from .rl_cfg import arm_reach_ppo_runner_cfg

register_cpu_twins(ARM_REACH_IDS, arm_reach_env_cfg, arm_reach_ppo_runner_cfg)
