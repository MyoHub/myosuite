from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.hand.reach.finger_mjlab_cfg import (
    FINGER_REACH_IDS,
    finger_reach_env_cfg,
)
from myosuite.envs.tasks.hand.reach.finger_rl_cfg import finger_reach_ppo_runner_cfg

register_cpu_twins(FINGER_REACH_IDS, finger_reach_env_cfg, finger_reach_ppo_runner_cfg)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
