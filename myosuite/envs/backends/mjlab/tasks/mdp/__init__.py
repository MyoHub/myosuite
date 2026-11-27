"""Shared MDP terms for MyoSuite mjlab tasks (mirrors ``mjlab.tasks.*.mdp``)."""

from mjlab.envs.mdp import *  # noqa: F401, F403

from myosuite.envs.backends.mjlab.tasks.mdp.actions import MyoAction, MyoActionCfg  # noqa: F401
from myosuite.envs.backends.mjlab.tasks.mdp.commands import (  # noqa: F401
    HeadingCommand,
    HeadingCommandCfg,
    UniformVectorCommand,
    UniformVectorCommandCfg,
    site_position_command_cfg,
)
from myosuite.envs.backends.mjlab.tasks.mdp.events import (  # noqa: F401
    randomize_carry_weight,
    reset_joints_uniform_in_range,
    reset_to_cpu_state,
    torch_joint_couplings,
    write_cpu_state,
)
from myosuite.envs.backends.mjlab.tasks.mdp.observations import (  # noqa: F401
    act,
    qpos,
    qpos_chains,
    qvel,
    qvel_chains,
)
from myosuite.envs.backends.mjlab.tasks.mdp.terminations import SYNC_TERM, sync_forward  # noqa: F401

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
