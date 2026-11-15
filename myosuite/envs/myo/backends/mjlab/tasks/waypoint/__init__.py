"""Waypoint tasks; importing registers the mjlab twins of the CPU waypoint envs."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins
from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg

from .env_cfg import make_waypoint_env_cfg

WAYPOINT_IDS = ("myoFullBodyWaypoint-v0",)


def waypoint_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the full-body waypoint task (354 muscles, 1500-step episodes)."""
    return myo_ppo_runner_cfg(
        "myo_fullbody_waypoint",
        hidden_dims=(512, 256, 256),
        critic_hidden_dims=(512, 256, 256),
        gamma=0.99,
        num_steps_per_env=48,
    )


register_cpu_twins(WAYPOINT_IDS, make_waypoint_env_cfg, waypoint_ppo_runner_cfg)
