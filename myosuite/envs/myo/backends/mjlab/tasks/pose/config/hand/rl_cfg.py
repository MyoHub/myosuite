"""RL configuration for the hand pose tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def hand_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the hand pose tasks."""
    return myo_ppo_runner_cfg("myo_hand_pose", hidden_dims=(256, 256))
