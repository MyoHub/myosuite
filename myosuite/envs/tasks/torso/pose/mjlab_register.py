from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.torso.pose.mjlab_cfg import TORSO_POSE_IDS, torso_pose_env_cfg
from myosuite.envs.tasks.torso.pose.rl_cfg import torso_pose_ppo_runner_cfg

register_cpu_twins(TORSO_POSE_IDS, torso_pose_env_cfg, torso_pose_ppo_runner_cfg)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
