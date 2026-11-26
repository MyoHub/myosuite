from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.arm.reach.mjlab_cfg import ARM_REACH_IDS, arm_reach_env_cfg
from myosuite.envs.tasks.arm.reach.rl_cfg import arm_reach_ppo_runner_cfg

register_cpu_twins(ARM_REACH_IDS, arm_reach_env_cfg, arm_reach_ppo_runner_cfg)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
