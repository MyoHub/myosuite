"""RL configuration for the finger pose tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.backends.mjlab.tasks.rl import myo_ppo_runner_cfg


def finger_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the finger pose tasks."""
    return myo_ppo_runner_cfg("myo_finger_pose")


from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
