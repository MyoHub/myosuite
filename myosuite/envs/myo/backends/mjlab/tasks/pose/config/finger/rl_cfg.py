"""RL configuration for the finger pose tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def finger_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the finger pose tasks."""
    return myo_ppo_runner_cfg("myo_finger_pose")
