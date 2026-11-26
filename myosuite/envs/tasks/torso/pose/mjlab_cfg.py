"""Torso pose environment configurations (``myoTorsoPoseFixed`` / ``myoTorsoExoPoseFixed``)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from myosuite.envs.backends.mjlab.tasks.pose.pose_env_cfg import (
    make_torso_pose_env_cfg,
)

TORSO_POSE_IDS = (
    "myoTorsoPoseFixed-v0",
    "myoTorsoExoPoseFixed-v0",
)


def torso_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Torso pose task matching the CPU registration of *env_id*."""
    return make_torso_pose_env_cfg(env_id, play=play)


from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
