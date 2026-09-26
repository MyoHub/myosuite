"""RL configuration for the arm reach tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def arm_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the arm reach tasks (shared myoInteract-style defaults)."""
    return myo_ppo_runner_cfg("myo_arm_reach")
