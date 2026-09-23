"""RL configuration for the finger reach tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def finger_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the finger reach tasks."""
    return myo_ppo_runner_cfg("myo_finger_reach", hidden_dims=(128, 128))
