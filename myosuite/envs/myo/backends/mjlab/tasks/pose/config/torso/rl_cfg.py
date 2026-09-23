"""RL configuration for the torso pose tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def torso_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the torso pose tasks (216 muscles)."""
    return myo_ppo_runner_cfg("myo_torso_pose", hidden_dims=(512, 256, 128))
