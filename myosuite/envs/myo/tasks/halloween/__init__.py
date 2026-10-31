"""Register the CPU Halloween muscle-action environment."""

from myosuite.core.registry import register_env

register_env(
    env_id="myoHalloween-v0",
    entry_point="myosuite.envs.myo.tasks.halloween.cpu:HalloweenEnv",
    max_episode_steps=600,
    kwargs={"frame_skip": 5},
)
