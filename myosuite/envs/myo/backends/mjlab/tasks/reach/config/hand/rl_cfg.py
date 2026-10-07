"""RL configuration for the hand reach tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def hand_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the hand reach tasks."""
    return myo_ppo_runner_cfg("myo_hand_reach")
