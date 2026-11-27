"""Pose-task MDP terms (on top of the shared MyoSuite mjlab terms)."""

from myosuite.envs.backends.mjlab.tasks.mdp import *  # noqa: F401, F403

from myosuite.envs.backends.mjlab.tasks.pose.mdp.commands import (  # noqa: F401
    JointPoseCommand,
    JointPoseCommandCfg,
)
from myosuite.envs.backends.mjlab.tasks.pose.mdp.observations import pose_err  # noqa: F401
from myosuite.envs.backends.mjlab.tasks.pose.mdp.rewards import (  # noqa: F401
    act_norm,
    pose_bonus,
    pose_dist,
    pose_done,
    pose_penalty,
    pose_solved,
)
from myosuite.envs.backends.mjlab.tasks.pose.mdp.terminations import pose_diverged  # noqa: F401

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
