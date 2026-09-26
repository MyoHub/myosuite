"""RL configuration for the leg tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def leg_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the leg stand-and-reach task (80 muscles)."""
    return myo_ppo_runner_cfg("myo_leg_stand")


def leg_terrain_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """PPO config for a leg terrain walking task (80 muscles)."""
    return myo_ppo_runner_cfg(experiment_name)
