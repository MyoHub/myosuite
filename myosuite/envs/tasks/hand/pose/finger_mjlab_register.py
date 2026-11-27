from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.hand.pose.finger_mjlab_cfg import (
    MOTOR_FINGER_POSE_IDS,
    MYO_FINGER_POSE_IDS,
    finger_pose_env_cfg,
)
from myosuite.envs.tasks.hand.pose.finger_rl_cfg import finger_pose_ppo_runner_cfg

register_cpu_twins(
    MYO_FINGER_POSE_IDS + MOTOR_FINGER_POSE_IDS,
    finger_pose_env_cfg,
    finger_pose_ppo_runner_cfg,
)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
