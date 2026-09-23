"""RL configuration for the elbow pose tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def elbow_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the elbow pose tasks."""
    return myo_ppo_runner_cfg("myo_elbow_pose", hidden_dims=(64, 64))
