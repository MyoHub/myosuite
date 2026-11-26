"""Reach-task MDP terms (on top of the shared MyoSuite mjlab terms)."""

from myosuite.envs.backends.mjlab.tasks.mdp import *  # noqa: F401, F403

from myosuite.envs.backends.mjlab.tasks.reach.mdp.commands import (  # noqa: F401
    ReachTargetCommand,
    ReachTargetCommandCfg,
    RelativeReachTargetCommand,
    RelativeReachTargetCommandCfg,
    WorkspaceReachTargetCommand,
    WorkspaceReachTargetCommandCfg,
)
from myosuite.envs.backends.mjlab.tasks.reach.mdp.observations import reach_err, tip_pos  # noqa: F401
from myosuite.envs.backends.mjlab.tasks.reach.mdp.rewards import (  # noqa: F401
    leg_reach_components,
    leg_reach_term,
    reach_components,
    reach_term,
)
from myosuite.envs.backends.mjlab.tasks.reach.mdp.terminations import (  # noqa: F401
    leg_reach_failed,
    reach_failed,
)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
