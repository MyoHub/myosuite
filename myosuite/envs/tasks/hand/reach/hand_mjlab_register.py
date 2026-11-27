from myosuite.envs.backends.mjlab.tasks.registration import register_cpu_twins

from myosuite.envs.tasks.hand.reach.hand_mjlab_cfg import (
    HAND_REACH_IDS,
    hand_reach_env_cfg,
)
from myosuite.envs.tasks.hand.reach.hand_rl_cfg import hand_reach_ppo_runner_cfg

register_cpu_twins(HAND_REACH_IDS, hand_reach_env_cfg, hand_reach_ppo_runner_cfg)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
