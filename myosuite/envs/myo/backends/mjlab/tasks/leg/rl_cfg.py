"""RL configuration for the leg tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def leg_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the leg stand-and-reach task (80 muscles)."""
    return myo_ppo_runner_cfg("myo_leg_stand")


def leg_terrain_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """PPO config for a leg terrain walking task (80 muscles, 1000-step episodes)."""
    return myo_ppo_runner_cfg(experiment_name, gamma=0.99, num_steps_per_env=48)


def leg_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the leg walking task (80 muscles, 1000-step episodes).

    The shared defaults (gamma 0.97, 24-step rollouts) look ahead about 0.7 s, too short for
    gait learning on these long episodes; use the horizon of the earlier walk runs
    (gamma 0.99, 48-step rollouts).
    """
    return myo_ppo_runner_cfg("myo_leg_walk", gamma=0.99, num_steps_per_env=48)
